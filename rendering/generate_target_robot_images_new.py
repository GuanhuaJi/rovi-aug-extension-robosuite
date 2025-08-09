#!/usr/bin/env python3
# file: generate_target_robot_images_mp.py
import argparse
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

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
        log_path.write_text("[]", encoding="utf-8")
    with locked_json(log_path, default=list) as hist:
        hist.append(
            {
                "episode": int(episode),
                "tried_offsets": [o.tolist() for o in tried],
                "working_offset": working.tolist(),
            }
        )


# ───────────────────────── single-episode worker ────────────────────
def generate_one_episode(
    robot_dataset: str,
    robot: str,
    episode: int,
    camera_hw: tuple[int, int],
    out_root: str | os.PathLike,
    unlimited: bool = False,
    load_displacement: bool = False,
    autosearch: bool = False,  # NEW
) -> tuple[str, int, bool]:
    """
    Render one episode for a target robot, optionally searching over
    displacement. Returns (robot, episode, success).
    """
    pick_best_gpu()
    os.environ["MUJOCO_GL"] = "egl"

    from envs import TargetEnvWrapper  # local import to keep fork-safety

    H, W = camera_hw
    gripper = select_gripper(robot)

    # ───────────── decide starting displacement ─────────────
    if autosearch:
        displacement = np.zeros(3, dtype=np.float32)
        displacement = np.array([0.2, 0.0, 0.0])
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
            if not off_file.is_file():
                print(
                    f"WARNING: displacement file not found → {off_file}; "
                    "using default [0, 0, 0]."
                )
        else:
            displacement = ROBOT_POSE_DICT[robot_dataset][robot]

    # ───────────── helper for dry-run image gen ─────────────
    def _try_disp(disp: np.ndarray):
        """Return (ok, steps) for a candidate displacement."""
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
        print(f"Displacement {disp} for episode {episode} robot {robot} → ok={ok}")
        wrapper.target_env.env.close_renderer()
        return ok, steps

    # ───────────── optional grid search ─────────────
    tried: list[np.ndarray] = []
    best_disp = displacement.copy()

    if autosearch:
        best_steps = -1
        scales = [0.03, 0.1, 0.3]

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

            for offset in offsets:
                cand = best_disp + offset
                tried.append(cand.copy())
                print(
                    f"Testing displacement {cand} (step={step}) "
                    f"for episode {episode} robot {robot}",
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

            # refine or stop depending on success
            displacement = best_disp.copy()
            if found_success:
                break
    else:
        # No search: just record the single attempt
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
        episode=episode,
        unlimited=unlimited,
        dry_run=False,
    )
    wrapper.target_env.env.close_renderer()

    log_offsets(Path(out_root), robot, episode, tried, best_disp)

    return robot, episode, bool(success)


# ───────────────────────────── dispatcher ────────────────────────────
def _ensure_json_file(path: Path) -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--robot_dataset", required=True)
    p.add_argument("--target_robot", nargs="+", required=True)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--unlimited", action="store_true")
    p.add_argument("--load_displacement", action="store_true")
    p.add_argument(
        "--autosearch",
        action="store_true",
        help="Enable grid-search for best displacement. "
        "If omitted, the script renders with (0,0,0) displacement only.",
    )
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

    # Build task list
    tasks = []
    for robot in args.target_robot:
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
                        args.autosearch,  # NEW
                    )
                )

    if not tasks:
        print("Nothing to do – all episodes already processed.")
        return

    # Submit to process pool
    with ProcessPoolExecutor(max_workers=args.num_workers, mp_context=mp_ctx) as pool:
        futures = [pool.submit(generate_one_episode, *t) for t in tasks]

        # We update whitelist/blacklist incrementally as tasks finish
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


# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()


'''
python /home/guanhuaji/mirage/robot2robot/rendering/generate_target_robot_images_new.py --robot_dataset ucsd_kitchen_rlds --target_robot IIWA --num_workers 10
python /home/guanhuaji/mirage/robot2robot/rendering/generate_target_robot_images_new.py --robot_dataset nyu_franka --target_robot Panda --num_workers 10
python /home/guanhuaji/mirage/robot2robot/rendering/generate_target_robot_images_new.py --robot_dataset toto --target_robot Panda --num_workers 20
python /home/guanhuaji/mirage/robot2robot/rendering/generate_target_robot_images_new.py --robot_dataset autolab_ur5 --target_robot UR5e --num_workers 20
python /home/guanhuaji/mirage/robot2robot/rendering/generate_target_robot_images_new.py --robot_dataset viola --target_robot Panda --num_workers 20
python /home/guanhuaji/mirage/robot2robot/rendering/generate_target_robot_images_new.py --robot_dataset austin_mutex --target_robot Panda --num_workers 20

python /home/guanhuaji/mirage/robot2robot/rendering/generate_target_robot_images_new.py --robot_dataset can --target_robot IIWA --num_workers 1

python /home/guanhuaji/mirage/robot2robot/rendering/generate_target_robot_images_new.py --robot_dataset lift --target_robot IIWA Sawyer Kinova3 Jaco UR5e Panda --num_workers 10
python /home/guanhuaji/mirage/robot2robot/rendering/generate_target_robot_images_new.py --robot_dataset lift --target_robot Panda --num_workers 10

python /home/guanhuaji/mirage/robot2robot/rendering/generate_target_robot_images_new.py --robot_dataset square --target_robot IIWA Sawyer Kinova3 Jaco UR5e Panda --num_workers 10
python /home/guanhuaji/mirage/robot2robot/rendering/generate_target_robot_images_new.py --robot_dataset stack --target_robot Panda UR5e --num_workers 30

'''

if __name__ == "__main__":
    main()
