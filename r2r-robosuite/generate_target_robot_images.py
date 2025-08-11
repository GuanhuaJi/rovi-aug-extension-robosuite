import argparse
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

from core import pick_best_gpu, locked_json
from config.dataset_poses_dict import ROBOT_CAMERA_POSES_DICT
from config.robot_pose_dict import ROBOT_POSE_DICT


# ───────────────────────────── helpers ──────────────────────────────
def select_gripper(robot: str) -> str:
    if robot == "Sawyer":
        return "RethinkGripper"
    if robot == "Jaco":
        return "JacoThreeFingerGripper"
    if robot in {"IIWA", "UR5e", "Kinova3"}:
        return "Robotiq85Gripper"
    if robot == "Panda":
        return "PandaGripper"
    raise ValueError(f"Unknown robot {robot!r}")


def log_offsets(
    out_root: Path,
    robot: str,
    episode: int,
    tried: list[np.ndarray],
    working: np.ndarray,
) -> None:
    log_path = out_root / "target_robot_states" / f"{robot}_displacement.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not log_path.exists():
        log_path.write_text("[]", encoding="utf-8")
    with locked_json(log_path, default=list) as hist:
        hist.append(
            {
                "episode": int(episode),
                "tried_offsets": [o.tolist() for o in tried],
                "working_offset": working.tolist(),
            }
        )


# Completely silence child processes (Python prints + C-level stdout/stderr)
class SuppressOutput:
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


# ───────────────────────── single-episode worker ────────────────────
def generate_one_episode(
    robot_dataset: str,
    robot: str,
    episode: int,
    camera_hw: tuple[int, int],
    out_root: str | os.PathLike,
    unlimited: bool = False,
    load_displacement: bool = False,
    autosearch: bool = False,
) -> tuple[str, int, bool]:
    """
    Render one episode; return (robot, episode, success).
    Child process is completely silent.
    """
    try:
        with SuppressOutput():
            pick_best_gpu()
            os.environ["MUJOCO_GL"] = "egl"

            # Lazy import to avoid the main process polluting stdout/stderr early
            from envs import TargetEnvWrapper

            H, W = camera_hw
            gripper = select_gripper(robot)

            # ───────────── decide starting displacement ─────────────
            if autosearch:
                displacement = np.array([0.2, 0.0, 0.0], dtype=np.float32)
            else:
                if load_displacement:
                    off_file = (
                        Path(out_root)
                        / "source_robot_states"
                        / robot
                        / "offsets"
                        / f"{episode}.npy"
                    )
                    displacement = (
                        np.load(off_file) if off_file.is_file() else np.zeros(3, np.float32)
                    )
                else:
                    displacement = ROBOT_POSE_DICT[robot_dataset][robot]

            # ───────────── helper for dry-run image gen ─────────────
            def _try_disp(disp: np.ndarray):
                wrapper = TargetEnvWrapper(
                    robot,
                    gripper,
                    robot_dataset,
                    camera_height=H,
                    camera_width=W,
                )
                ok, _suggested, steps = wrapper.generate_image(
                    save_paired_images_folder_path=out_root,
                    source_robot_states_path=out_root,
                    robot_dataset=robot_dataset,
                    robot_disp=disp,
                    episode=episode,
                    unlimited=unlimited,
                    dry_run=True,
                )
                wrapper.target_env.env.close_renderer()
                return ok, steps

            # ───────────── optional grid search ─────────────
            tried: list[np.ndarray] = []
            best_disp = displacement.copy()
            if autosearch:
                best_steps = -1
                for step in [0.03, 0.1, 0.3]:
                    found_success = False
                    offsets = np.array(
                        [
                            [dx, dy, dz]
                            for dx in (0, -step, step)
                            for dy in (0, -step, step)
                            for dz in (0, -step, step)
                        ],
                        dtype=np.float32,
                    )
                    for offset in offsets:
                        cand = best_disp + offset
                        tried.append(cand.copy())
                        ok, steps = _try_disp(cand)
                        if ok:
                            best_disp = cand
                            best_steps = steps
                            found_success = True
                            break
                        if steps > best_steps:
                            best_steps = steps
                            best_disp = cand
                    if found_success:
                        break
            else:
                tried.append(best_disp.copy())

            # ───────────── final (non-dry) render ─────────────
            wrapper = TargetEnvWrapper(
                robot,
                gripper,
                robot_dataset,
                camera_height=H,
                camera_width=W,
            )
            success, _suggested, _ = wrapper.generate_image(
                save_paired_images_folder_path=out_root,
                source_robot_states_path=out_root,
                robot_dataset=robot_dataset,
                robot_disp=best_disp,
                episode=episode,          # store using 0/1/2/...
                unlimited=unlimited,
                dry_run=False,
            )
            wrapper.target_env.env.close_renderer()

            log_offsets(Path(out_root), robot, episode, tried, best_disp)

            return robot, episode, bool(success)
    except Exception:
        # Child process errors also stay silent; main process treats as failure
        return robot, episode, False


# ───────────────────────────── dispatcher ────────────────────────────
def _ensure_json_file(path: Path) -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--robot_dataset", required=True)
    p.add_argument("--target_robots", nargs="+", required=True)
    p.add_argument("--num_workers", type=int, default=20)
    p.add_argument("--unlimited", action="store_true")
    p.add_argument("--load_displacement", action="store_true")
    p.add_argument(
        "--autosearch",
        action="store_true",
        help="Perform coarse grid search on displacement; use default when disabled",
    )
    # New: select by episode index (inclusive range)
    p.add_argument("--start", type=int, default=None, help="Start episode index (inclusive)")
    p.add_argument("--end", type=int, default=None, help="End episode index (inclusive)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    meta = ROBOT_CAMERA_POSES_DICT[args.robot_dataset]
    out_root = Path(meta["replay_path"])

    # Read metadata to determine total episode count (default 0..N-1)
    dmeta = json.loads((out_root / "dataset_metadata.json").read_text(encoding="utf-8"))
    num_eps = int(dmeta.get("num_episodes") or dmeta["num_episodes_total"])
    H, W = int(dmeta["image_height"]), int(dmeta["image_width"])

    # Compute selection range (closed interval)
    s = 0 if args.start is None else max(0, args.start)
    e = (num_eps - 1) if args.end is None else min(num_eps - 1, args.end)
    if s > e:
        raise ValueError(f"--start ({s}) must be ≤ --end ({e})")
    episodes = range(s, e + 1)

    mp_ctx = mp.get_context("spawn")

    # Task list: generate one item for each target robot × episode
    tasks: list[tuple] = []
    for robot in args.target_robots:
        wl_path = out_root / robot / "whitelist.json"
        bl_path = out_root / robot / "blacklist.json"
        _ensure_json_file(wl_path)
        _ensure_json_file(bl_path)
        with locked_json(wl_path) as wl:
            done_eps = set(wl.get(robot, []))
        for ep in episodes:
            if ep not in done_eps:
                tasks.append(
                    (
                        args.robot_dataset,
                        robot,
                        ep,
                        (H, W),
                        str(out_root),
                        args.unlimited,
                        args.load_displacement,
                        args.autosearch,
                    )
                )

    if not tasks:
        print("Nothing to do – all selected episodes already processed.")
        return

    # Submit tasks & single progress bar; child processes are completely silent
    with ProcessPoolExecutor(max_workers=args.num_workers, mp_context=mp_ctx) as pool:
        futures = []
        fut2tag: dict = {}
        for t in tasks:
            f = pool.submit(generate_one_episode, *t)
            futures.append(f)
            fut2tag[f] = (t[1], t[2])  # (robot, episode)

        with tqdm(total=len(futures), desc="Rendering", unit="job", dynamic_ncols=True) as pbar:
            for fut in as_completed(futures):
                robot, ep = fut2tag[fut]
                ok = False
                try:
                    _robot_r, _ep_r, ok = fut.result()
                except Exception:
                    ok = False

                wl_path = out_root / robot / "whitelist.json"
                bl_path = out_root / robot / "blacklist.json"

                if ok:
                    with locked_json(wl_path) as wl:
                        eps = set(wl.setdefault(robot, []))
                        eps.add(ep)
                        wl[robot] = sorted(eps)
                    with locked_json(bl_path) as bl:
                        eps = set(bl.setdefault(robot, []))
                        if ep in eps:
                            eps.remove(ep)
                            bl[robot] = sorted(eps)
                else:
                    with locked_json(bl_path) as bl:
                        eps = set(bl.setdefault(robot, []))
                        eps.add(ep)
                        bl[robot] = sorted(eps)
                    with locked_json(wl_path) as wl:
                        eps = set(wl.setdefault(robot, []))
                        if ep in eps:
                            eps.remove(ep)
                            wl[robot] = sorted(eps)

                pbar.update(1)

    print("✓ all dispatched episodes finished")


# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()