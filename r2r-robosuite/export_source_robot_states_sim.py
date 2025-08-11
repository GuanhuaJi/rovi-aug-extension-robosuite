import os, argparse, random, json, datetime, sys, contextlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from multiprocessing import get_context

import numpy as np
import imageio.v3 as iio
import h5py
from tqdm import tqdm

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from envs import SourceEnvWrapper
from config.dataset_poses_dict import ROBOT_CAMERA_POSES_DICT


# ───────────────────────── helpers ─────────────────────────
def _worker_init(gpu_id: int | str | None) -> None:
    """Isolate each child process to a (logical) GPU id (or disable)."""
    if gpu_id is None or gpu_id == "":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)


class SuppressOutput(contextlib.AbstractContextManager):
    """
    Silence *all* stdout/stderr in this process (Python prints + C libraries).
    Safe to use in child processes.
    """
    def __enter__(self):
        self._devnull = os.open(os.devnull, os.O_WRONLY)
        self._stdout_fd = os.dup(1)
        self._stderr_fd = os.dup(2)
        os.dup2(self._devnull, 1)
        os.dup2(self._devnull, 2)
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            os.dup2(self._stdout_fd, 1)
            os.dup2(self._stderr_fd, 2)
        finally:
            os.close(self._devnull)
            os.close(self._stdout_fd)
            os.close(self._stderr_fd)


def process_one_episode(
    ep_id: int,
    joints: np.ndarray,
    gripper: np.ndarray,
    meta: dict,
    out_dir: str,
    verbose: bool = False,   # ignored in subprocess: always quiet
) -> None:
    """Process a single episode through SourceEnvWrapper (quiet child)."""
    with SuppressOutput():
        wrapper = SourceEnvWrapper(
            source_name=meta["robot"],
            source_gripper=meta["gripper"],
            robot_dataset=meta["dataset_name"],
            verbose=False,
        )
        wrapper.get_source_robot_states(
            save_source_robot_states_path=out_dir,
            episode=ep_id,
            joint_angles=joints,
            gripper_states=gripper,
        )
        wrapper.source_env.env.close_renderer()


# ───────────────────── episode dispatcher ─────────────────────
def dispatch_episodes(
    robot_dataset: str,
    hdf5_path: str,
    workers: int = os.cpu_count(),
    seed: int = 0,
    chunksize: int = 100,
    verbose: bool = False,
    start: int | None = None,
    end: int | None = None,
) -> None:

    random.seed(seed)
    np.random.seed(seed)

    meta = ROBOT_CAMERA_POSES_DICT[robot_dataset].copy()
    meta["dataset_name"] = robot_dataset

    # -------- output folders -------------------------------------------------
    oxe_videos_dir = Path(meta["replay_path"]) / "source_replays"
    oxe_videos_dir.mkdir(parents=True, exist_ok=True)
    src_states_dir = Path(meta["replay_path"]) / "source_robot_states"
    src_states_dir.mkdir(parents=True, exist_ok=True)

    # -------- open HDF5 ------------------------------------------------------
    with h5py.File(hdf5_path, "r") as h5:
        demos_grp = h5["data"]  # contains demo_0, demo_1, …
        demo_names = sorted(
            (name for name in demos_grp.keys() if name.startswith("demo_")),
            key=lambda s: int(s.split("_")[-1]),
        )
        if not demo_names:
            raise RuntimeError("No demo_* groups found in /data")

        # choose episodes by *demo number* (inclusive)
        all_pairs = [(int(n.split("_")[-1]), n) for n in demo_names]
        min_id = min(e for e, _ in all_pairs)
        max_id = max(e for e, _ in all_pairs)
        s = min_id if start is None else start
        e = max_id if end   is None else end
        if s > e:
            raise ValueError(f"--start ({s}) must be <= --end ({e})")
        selected = [(ep_id, name) for ep_id, name in all_pairs if s <= ep_id <= e]
        if not selected:
            raise RuntimeError(f"No episodes in requested range [{s}, {e}]")

        # inspect first selected demo to get image size
        sample_demo = demos_grp[selected[0][1]]
        img_h, img_w = sample_demo["obs/agentview_image"].shape[2:4]

        # ---------- metadata file --------------------------------------------
        meta_path = Path(meta["replay_path"]) / "dataset_metadata.json"
        meta_json = {
            "dataset": Path(hdf5_path).name,
            "num_episodes_total": int(len(demo_names)),
            "num_episodes_selected": int(len(selected)),
            "selected_range_inclusive": [int(s), int(e)],
            "image_height": int(img_h),
            "image_width": int(img_w),
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        }
        meta_path.write_text(json.dumps(meta_json, indent=2))
        if verbose:
            print(f"📄 metadata written to {meta_path}")

        # ---------- multiprocessing pool + single tqdm -----------------------
        ctx = get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=ctx,
            initializer=_worker_init,
            initargs=(0,),  # set to None/"" if you want children CPU-only
        ) as pool, tqdm(
            total=len(selected),
            desc="Episodes",
            dynamic_ncols=True,
            leave=True,
        ) as pbar:

            pending = []
            # submit tasks; write MP4s in the main process; keep children quiet
            for ep_id, demo in selected:
                obs_path = f"data/{demo}/obs"
                joints = np.asarray(h5[f"{obs_path}/robot0_joint_pos"])        # (T, J)
                gripper_raw = np.asarray(h5[f"{obs_path}/robot0_gripper_qpos"])# (T,), (T,1) or (T,2)
                finger0 = gripper_raw[:, 0] if gripper_raw.ndim == 2 else gripper_raw
                gripper = (finger0 > 0.03).astype(np.float32)
                frames = np.asarray(h5[f"{obs_path}/agentview_image"])         # (T, H, W, 3)

                # save RGB stream (main proc)
                mp4_path = oxe_videos_dir / f"{ep_id}.mp4"
                iio.imwrite(
                    mp4_path,
                    frames,
                    fps=30,
                    codec="libx264",
                    macro_block_size=1,    # disable 16px padding
                    pixelformat="yuv420p",
                )

                # enqueue SourceEnvWrapper job (child is fully silent)
                fut = pool.submit(
                    process_one_episode, ep_id, joints, gripper, meta, str(src_states_dir), False
                )
                pending.append(fut)

                # apply back-pressure; also drive the global progress bar
                if len(pending) >= chunksize:
                    done, still = wait(pending, return_when=FIRST_COMPLETED)
                    for f in done:
                        f.result()          # surface errors promptly
                    pbar.update(len(done))
                    pending = list(still)

            # finish remaining jobs and progress
            while pending:
                done, still = wait(pending, return_when=FIRST_COMPLETED)
                for f in done:
                    f.result()
                pbar.update(len(done))
                pending = list(still)

    print("🎉 all episodes exported")


# ──────────────────────────── CLI ────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--robot_dataset",
        required=True,
        help="Key into ROBOT_CAMERA_POSES_DICT (e.g. ucsd_kitchen_rlds)",
    )
    ap.add_argument("--hdf5_path", required=True, help="Path to the input .hdf5 file")
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunksize", type=int, default=40,
                    help="Number of pending futures before we wait")
    ap.add_argument("--verbose", action="store_true")

    # NEW: range by demo number (inclusive). If omitted, process all.
    ap.add_argument("--start", type=int, default=None,
                    help="Inclusive start demo number, e.g. 20 (uses demo_20)")
    ap.add_argument("--end", type=int, default=None,
                    help="Inclusive end demo number, e.g. 30 (uses demo_30)")

    args = ap.parse_args()

    dispatch_episodes(
        args.robot_dataset,
        hdf5_path=args.hdf5_path,
        workers=args.workers,
        seed=args.seed,
        chunksize=args.chunksize,
        verbose=args.verbose,
        start=args.start,
        end=args.end,
    )

'''
python /home/guanhuaji/OXE-Aug/r2r-robosuite/export_source_robot_states_sim.py --robot_dataset=can \
         --hdf5_path=/home/harshapolavaram/mirage/image84/can/image_84.hdf5 \
            --workers=10 --chunksize=40
'''