#!/usr/bin/env python3
"""
Export source-robot EE-poses & gripper states for a dataset partition.

Example
-------
python /home/guanhuaji/mirage/robot2robot/rendering/export_source_robot_states_min.py \
        --robot_dataset austin_buds --partition 1
"""

import argparse
import random
import numpy as np

from pathlib import Path
from envs import SourceEnvWrapper
from core import pick_best_gpu
from config.dataset_poses_dict import ROBOT_CAMERA_POSES_DICT   # still external

# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seed",        type=int, default=0)
    p.add_argument("--robot_dataset", required=True)
    p.add_argument("--partition",   type=int, default=0)
    p.add_argument("--source_robot",   default=None)
    p.add_argument("--source_gripper", default=None)
    p.add_argument("--verbose",     action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── deterministic & GPU selection (optional) ───────────────────────
    random.seed(args.seed)
    np.random.seed(args.seed)
    pick_best_gpu()

    # ── dataset-specific meta info ────────────────────────────────────
    meta          = ROBOT_CAMERA_POSES_DICT[args.robot_dataset]
    replay_path   = Path(meta["replay_path"])
    out_dir       = replay_path / "source_robot_states"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── episode range for this partition ──────────────────────────────
    num_ep       = meta["num_episodes"]
    NUM_PARTS    = 5
    episodes     = range(num_ep * args.partition // NUM_PARTS,
                         num_ep * (args.partition + 1) // NUM_PARTS)

    # ── create env wrapper once per episode ───────────────────────────
    for ep in episodes:
        wrapper = SourceEnvWrapper(
            source_name=meta["robot"],
            source_gripper=meta["gripper"],
            robot_dataset=args.robot_dataset,
            camera_height=meta["camera_height"],
            camera_width=meta["camera_width"],
            verbose=args.verbose
        )
        wrapper.get_source_robot_states(save_source_robot_states_path=out_dir, episode=ep)

    print("✓ All done.")


if __name__ == "__main__":
    main()
