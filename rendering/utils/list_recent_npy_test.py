#!/usr/bin/env python3
"""
list_recent_npy.py
------------------
Print every .npy file under the given root directory that was
modified on or after 2025-06-13 00:00 (local timezone).

Usage
-----
    python /home/guanhuaji/mirage/robot2robot/rendering/list_recent_npy.py /home/guanhuaji/mirage/robot2robot/rendering/paired_images/autolab_ur5/source_robot_states
"""

from pathlib import Path
import datetime as dt
from zoneinfo import ZoneInfo   # Python ≥3.9
import numpy as np
import os
import sys

PACIFIC = ZoneInfo("America/Los_Angeles")
CUTOFF   = dt.datetime(2025, 6, 12, 0, 0, tzinfo=PACIFIC)   # 2025-06-13 00:00 PT
CUTOFF_TS = CUTOFF.timestamp()

# offset_dict = {
#     "Panda": {
#         "recent": np.array([0, 0.2, 0]),
#         "old": np.array([0, 0.2, 0]),
#     },
#     "IIWA": {
#         "recent": np.array([0.05, 0.05, 0]),
#         "old": np.array([0.05, 0.05, 0]),
#     },
#     "Sawyer": {
#         "recent": np.array([0, 0, 0]),
#         "old": np.array([0.02, 0.02, -0.02]),
#     },
#     "Jaco": {
#         "recent": np.array([0, 0, 0]),
#         "old": np.array([0, 0, 0]),
#     },
#     "UR5e": {
#         "recent": np.array([0, 0, 0]),
#         "old": np.array([0, 0, 0]),
#     },
#     "Kinova3": {
#         "recent": np.array([0.02, 0.02, -0.02]),
#         "old": np.array([0, 0, 0]),
#     },
# }


offset_dict = {
    "Panda": {
        "recent": np.array([0, 0, 0]),
        "old": np.array([0, 0, 0]),
    },
    "IIWA": {
        "recent": np.array([0.01, 0, 0]),
        "old": np.array([0.01, 0, 0]),
    },
    "Sawyer": {
        "recent": np.array([0, 0, 0]),
        "old": np.array([0, 0, 0]),
    },
    "Jaco": {
        "recent": np.array([0.1, 0, 0]),
        "old": np.array([0.1, 0, 0]),
    },
    "UR5e": {
        "recent": np.array([0, 0, 0]),
        "old": np.array([0, 0, 0]),
    },
    "Kinova3": {
        "recent": np.array([0.1, 0, 0]),
        "old": np.array([0.1, 0, 0]),
    },
}

ROBOTS = ["Panda", "IIWA", "Sawyer", "Jaco", "UR5e", "Kinova3"]        # add more robot names here if needed

def main(root: Path):
    for robot in ROBOTS:
        eef_path    = root / robot / "end_effector"
        offset_path = root / robot / "offsets"

        if not eef_path.is_dir():
            sys.exit(f"error: {eef_path} is not a directory")

        print(f"Scanning {eef_path} …")

        all_npy = list(eef_path.rglob("*.npy"))
        if not all_npy:
            print("  (no .npy files found)")
            continue

        # ensure the offset directory exists
        os.makedirs(offset_path, exist_ok=True)

        for p in all_npy:
            is_recent = p.stat().st_mtime >= CUTOFF_TS
            offset_npy_path = offset_path / p.name
            expected  = offset_dict[robot]["recent"] if is_recent else offset_dict[robot]["old"]
            actual = np.load(offset_npy_path)
            if not np.array_equal(actual, expected):
                tag = "recent" if is_recent else "old"
                print(f"  ERROR: {p.name} expected {tag} {expected} but found {actual}")
        print(f"all tested .npy files under {offset_path} are valid.")

if __name__ == "__main__":
    dataset = "toto"  # Change this to the desired dataset
    main(Path(f"/home/guanhuaji/mirage/robot2robot/rendering/paired_images/{dataset}/source_robot_states"))