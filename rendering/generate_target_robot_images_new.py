#!/usr/bin/env python3
# file: generate_target_robot_images_mp.py
import argparse, json, os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import os, argparse, json, multiprocessing as mp

import numpy as np

from core import pick_best_gpu, locked_json
from config.dataset_poses_dict import ROBOT_CAMERA_POSES_DICT
from config.robot_pose_dict import ROBOT_POSE_DICT


# ───────────────────────────── helpers ──────────────────────────────
def select_gripper(robot: str) -> str:
    """Return the proper gripper name for a given robot type."""
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
    """Append displacement search info for one episode to a JSON file."""
    log_path = out_root / "target_robot_states" / f"{robot}_displacement.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not log_path.exists():
        log_path.write_text("[]")
    with locked_json(log_path, default=list) as hist:
        hist.append(
            {
                "episode": int(episode),
                "tried_offsets": [o.tolist() for o in tried],
                "working_offset": working.tolist(),
            }
        )


def generate_one_episode(
    robot_dataset: str,
    robot: str,
    episode: int,
    camera_hw: tuple[int, int],
    out_root: str | os.PathLike,
    unlimited: bool = False,
    load_displacement: bool = False,
) -> tuple[str, int, bool]:
    pick_best_gpu()
    os.environ["MUJOCO_GL"] = "egl"
    from envs import TargetEnvWrapper
    H, W = camera_hw
    gripper = select_gripper(robot)
    wrapper = TargetEnvWrapper(
        robot,
        gripper,
        robot_dataset,
        camera_height=H,
        camera_width=W,
    )

    if load_displacement:
        off_file = Path(out_root) / "source_robot_states" / robot / "offsets" / f"{episode}.npy"
        if off_file.is_file():
            displacement = np.load(off_file)
        else:
            displacement = np.zeros(3, dtype=np.float32)
            print(f"WARNING: displacement file not found → {off_file}; using default [0, 0, 0].")
    else:
        displacement = ROBOT_POSE_DICT[robot_dataset][robot]

    def _try_disp(disp: np.ndarray):
        """Return (ok, steps) for a candidate displacement."""
        ok, _sug, steps = wrapper.generate_image(
            save_paired_images_folder_path=out_root,
            source_robot_states_path=out_root,
            robot_dataset=robot_dataset,
            robot_disp=disp,
            episode=episode,
            unlimited=unlimited,
            dry_run=True,
        )
        return ok, steps

    scales = [0.03, 0.1, 0.3]
    best_disp = displacement.copy()
    best_steps = -1
    tried: list[np.ndarray] = []

    for step in scales:
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

        np.random.shuffle(offsets)

        for offset in offsets:
            cand = best_disp + offset
            tried.append(cand.copy())
            print(
                f"Testing displacement {cand} (step={step}) for episode {episode} robot {robot}",
                flush=True,
            )
            ok, steps = _try_disp(cand)
            if ok:
                best_disp = cand
                best_steps = steps
                found_success = True
                break
            if steps > best_steps:
                best_steps = steps
                best_disp = cand

        # After scanning current grid, either we succeeded (so continue
        # refining around the new centre) or we failed (so shrink step around
        # the current best guess).
        displacement = best_disp.copy()
        if not found_success:
            # Failed on this scale – keep current centre but continue with
            # smaller step to search a finer neighbourhood.
            continue
        # If success, simply go to next (smaller) scale and search around the
        # new centre (already updated above).

    success, _sug, _ = wrapper.generate_image(
        save_paired_images_folder_path=out_root,
        source_robot_states_path=out_root,
        robot_dataset=robot_dataset,
        robot_disp=best_disp,
        episode=episode,
        unlimited=unlimited,
        dry_run=False,
    )

    log_offsets(Path(out_root), robot, episode, tried, best_disp)

    return robot, episode, bool(success)


# ───────────────────────────── dispatcher ────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--robot_dataset", required=True)
    p.add_argument("--target_robot", nargs="+", required=True)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--unlimited", action="store_true")
    p.add_argument("--load_displacement", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    meta = ROBOT_CAMERA_POSES_DICT[args.robot_dataset]
    with open(Path(meta["replay_path"]) / "dataset_metadata.json", encoding="utf-8") as f:
        dmeta = json.load(f)
    H, W = dmeta["image_height"], dmeta["image_width"]
    out_root = Path(meta["replay_path"])
    num_eps = dmeta["num_episodes"]
    episodes = range(num_eps)

    mp_ctx = mp.get_context("spawn")

    tasks = []
    for robot in args.target_robot:
        wl_path = out_root / robot / "whitelist.json"
        wl_path.parent.mkdir(parents=True, exist_ok=True)

        bl_path = out_root / robot / "blacklist.json"
        bl_path.parent.mkdir(parents=True, exist_ok=True)
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
                    )
                )

    if not tasks:
        print("Nothing to do – all episodes already processed.")
        return

    with ProcessPoolExecutor(max_workers=args.num_workers, mp_context=mp_ctx) as pool:
        futures = [pool.submit(generate_one_episode, *t) for t in tasks]
        with locked_json(wl_path) as wl:
            done_eps = set(wl.setdefault(robot, []))

        for fut in as_completed(futures):
            robot, ep, ok = fut.result()

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

    print("✓ all dispatched episodes finished")

'''
python /home/guanhuaji/mirage/robot2robot/rendering/generate_target_robot_images_new.py --robot_dataset nyu_franka --target_robot Jaco --num_workers 10 --load_displacement
'''

if __name__ == "__main__":
    main()
