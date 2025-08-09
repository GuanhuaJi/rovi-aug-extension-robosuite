import os, argparse, random, json, datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from multiprocessing import get_context

import numpy as np
import imageio.v3 as iio
import h5py

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from envs import SourceEnvWrapper
from config.dataset_poses_dict import ROBOT_CAMERA_POSES_DICT


# ───────────────────────── helpers ─────────────────────────
def _worker_init(gpu_id: int) -> None:
    """Isolate each child process to a (logical) GPU id."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)


def process_one_episode(
    idx: int,
    joints: np.ndarray,
    gripper: np.ndarray,
    meta: dict,
    out_dir: str,
    verbose: bool = False,
) -> None:
    """Process a single episode through SourceEnvWrapper."""
    wrapper = SourceEnvWrapper(
        source_name=meta["robot"],
        source_gripper=meta["gripper"],
        robot_dataset=meta["dataset_name"],
        verbose=verbose,
    )
    wrapper.get_source_robot_states(
        save_source_robot_states_path=out_dir,
        episode=idx,
        joint_angles=joints,
        gripper_states=gripper,
    )
    wrapper.source_env.env.close_renderer()
    if verbose:
        print(f"✓ episode {idx} done")


# ───────────────────── episode dispatcher ─────────────────────
def dispatch_episodes(
    robot_dataset: str,
    hdf5_path: str,
    workers: int = os.cpu_count(),
    seed: int = 0,
    chunksize: int = 100,
    verbose: bool = False,
) -> None:

    random.seed(seed)
    np.random.seed(seed)

    meta = ROBOT_CAMERA_POSES_DICT[robot_dataset]
    meta["dataset_name"] = robot_dataset

    # -------- output folders -------------------------------------------------
    oxe_videos_dir = Path(meta["replay_path"]) / "original_oxe_videos"
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
        num_eps = len(demo_names)
        if num_eps == 0:
            raise RuntimeError("No demo_* groups found in /data")

        # inspect first demo to get image size
        sample_demo = demos_grp[demo_names[0]]
        img_h, img_w = sample_demo["obs/agentview_image"].shape[2:4]

        # ---------- metadata file --------------------------------------------
        meta_path = Path(meta["replay_path"]) / "dataset_metadata.json"
        meta_json = {
            "dataset": Path(hdf5_path).name,
            "num_episodes": int(num_eps),
            "image_height": int(img_h),
            "image_width": int(img_w),
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        }
        meta_path.write_text(json.dumps(meta_json, indent=2))
        if verbose:
            print(f"📄 metadata written to {meta_path}")

        # ---------- multiprocessing pool -------------------------------------
        ctx = get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=ctx,
            initializer=_worker_init,
            initargs=(0,),
        ) as pool:
            pending = []
            for idx, demo in enumerate(demo_names):
                print(f"Processing episode {idx + 1}/{num_eps}: {demo}")
                obs_path = f"data/{demo}/obs"
                joints = np.asarray(h5[f"{obs_path}/robot0_joint_pos"])          # (T, J)
                gripper_raw = np.asarray(h5[f"{obs_path}/robot0_gripper_qpos"])   # (T,), (T,1) or (T,2)
                if gripper_raw.ndim == 2:                 # (T, K) → pick finger-0
                    finger0 = gripper_raw[:, 0]
                else:                                     # already (T,)
                    finger0 = gripper_raw

                gripper = (finger0 > 0.03).astype(np.float32)
                frames = np.asarray(h5[f"{obs_path}/agentview_image"])           # (T, H, W, 3)

                # ---- save RGB stream ----------------------------------------
                mp4_path = oxe_videos_dir / f"{idx}.mp4"
                iio.imwrite(
                    mp4_path,
                    frames,
                    fps=30,
                    codec="libx264",
                    macro_block_size=1,      # ← disable the 16-pixel padding
                    pixelformat="yuv420p"    # keeps the file widely playable
                )
                if verbose:
                    print(f"🎞  saved {mp4_path}")

                # ---- enqueue SourceEnvWrapper job ---------------------------
                fut = pool.submit(
                    process_one_episode,
                    idx, joints, gripper, meta, str(src_states_dir), verbose
                )
                pending.append(fut)

                # ---- back-pressure once pending hits chunksize -------------
                if len(pending) >= chunksize:
                    done, pending_set = wait(pending, return_when=FIRST_COMPLETED)
                    pending = list(pending_set)

            # finish remaining jobs
            for fut in pending:
                fut.result()

    print("🎉 all episodes exported")


# ──────────────────────────── CLI ────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--robot_dataset",
        required=True,
        help="Key into ROBOT_CAMERA_POSES_DICT (e.g. ucsd_kitchen_rlds)",
    )
    ap.add_argument(
        "--hdf5_path",
        required=True,
        help="Path to the input .hdf5 file",
    )
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--chunksize",
        type=int,
        default=100,
        help="Number of pending futures before we wait",
    )
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    dispatch_episodes(
        args.robot_dataset,
        hdf5_path=args.hdf5_path,
        workers=args.workers,
        seed=args.seed,
        chunksize=args.chunksize,
        verbose=args.verbose,
    )


'''
python /home/guanhuaji/mirage/robot2robot/rendering/export_source_robot_states_sim.py --robot_dataset=can \
         --hdf5_path=/home/harshapolavaram/mirage/image84/can/image_84.hdf5 \
            --workers=10 --chunksize=40

python /home/guanhuaji/mirage/robot2robot/rendering/export_source_robot_states_sim.py --robot_dataset=lift \
         --hdf5_path=/home/harshapolavaram/mirage/image84/lift/image_84.hdf5 \
            --workers=10 --chunksize=40
'''