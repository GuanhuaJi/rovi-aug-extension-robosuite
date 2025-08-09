#!/usr/bin/env python3
"""
Update an image_84.hdf5 file by attaching overlay videos and end‑effector state arrays
for multiple robots.

For every subgroup (episode) in /data of the HDF5 file, this script:
  • Inserts/overwrites the agent‑view overlay video for each robot as a 1‑D uint8
    dataset named  "agentview_image_<ROBOT>".
  • Inserts/overwrites the end‑effector target pose for each robot as a 2‑D float
    dataset  "<ROBOT>_eef_states"  directly from the corresponding .npz file.

Paths are built from three inputs:
  --dataset          Name of the dataset folder (e.g. train, val, real_robot)
  --h5-root          Root containing <dataset>/image_84.hdf5
                     (default: /home/harshapolavaram/mirage/image84)
  --paired-root      Root containing paired_images/<dataset>/...
                     (default: /home/guanhuaji/mirage/robot2robot/rendering/paired_images)

Usage
-----
python /home/guanhuaji/mirage/robot2robot/rendering/process_h5.py --dataset three_piece_assembly
"""
from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Sequence

import cv2  # OpenCV‑Python
import h5py
import numpy as np

ROBOTS: Sequence[str] = ["UR5e", "Jaco", "Sawyer", "Kinova3", "IIWA"]
TARGET_SIZE = (84, 84)  # (width, height)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def _decode_video(path: Path, size: tuple[int, int] = TARGET_SIZE) -> np.ndarray:
    """Load *path* and return RGB frames as uint8 array of shape (T,H,W,3)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video {path}")

    frames: list[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if (frame.shape[1], frame.shape[0]) != size:  # width, height
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        frames.append(frame.astype(np.uint8))
    cap.release()

    if not frames:
        raise ValueError(f"No frames decoded from {path}")
    return np.stack(frames, axis=0)  # (T, H, W, 3)


def _replace_dataset(group: h5py.Group, name: str, data: np.ndarray, **kw) -> None:
    """Delete *name* if it exists and recreate with *data*."""
    if name in group:
        del group[name]
    group.create_dataset(name, data=data, **kw)

# -----------------------------------------------------------------------------
# Core processing
# -----------------------------------------------------------------------------

def _episode_id_from_group(group_name: str) -> str:
    """Extract numeric ID from 'demo_###' → '###'. Return original if none."""
    m = re.search(r"(\d+)$", group_name)
    return m.group(1) if m else group_name


def process_episode(ep_group: h5py.Group, ep_name: str, paired_root: Path) -> None:
    """Attach / overwrite all robot datasets for one episode inside *obs/*."""
    file_id = _episode_id_from_group(ep_name)
    state_root = paired_root / "target_robot_states"

    obs_grp = ep_group.require_group("obs")

    for robot in ROBOTS:
        video_path = paired_root / f"{robot}_overlay" / f"{file_id}.mp4"
        state_path = state_root / robot / f"{file_id}.npz"

        # ---------------- Video frames ----------------
        if video_path.exists():
            try:
                frames = _decode_video(video_path)
                _replace_dataset(
                    obs_grp,
                    f"agentview_image_{robot}",
                    frames,
                    compression="gzip",
                    compression_opts=4,
                    chunks=(1, *TARGET_SIZE, 3),  # slow‑but‑safe chunking
                )
                logging.debug("Stored %s frames for %s", frames.shape[0], robot)
            except Exception as exc:
                logging.warning("Failed processing %s: %s", video_path, exc)
        else:
            logging.warning("Missing video: %s", video_path)

        # -------------- End‑effector pose -------------
        if state_path.exists():
            try:
                with np.load(state_path) as npz:
                    if "target_pose" not in npz.files:
                        raise KeyError("'target_pose' missing in " + str(state_path))
                    eef = npz["target_pose"].astype(np.float32)
                _replace_dataset(obs_grp, f"{robot}_eef_states", eef)
                logging.debug("Stored pose for %s", robot)
            except Exception as exc:
                logging.warning("Failed loading %s: %s", state_path, exc)
        else:
            logging.warning("Missing state: %s", state_path)

# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Augment image_84.hdf5 with R2R data")
    p.add_argument("--dataset", required=True, help="Dataset folder name")
    p.add_argument(
        "--h5-root",
        default="/home/abrashid/sim/",
        help="Root dir containing <dataset>/image_84.hdf5",
    )
    p.add_argument(
        "--paired-root",
        default="/home/guanhuaji/mirage/robot2robot/rendering/paired_images",
        help="Root dir of paired_images/<dataset>",
    )
    args = p.parse_args()

    h5_path = Path(args.h5_root) / args.dataset / "image_84.hdf5"
    paired_root = Path(args.paired_root) / args.dataset

    if not h5_path.exists():
        raise FileNotFoundError(h5_path)

    logging.info("Opening %s", h5_path)
    with h5py.File(h5_path, "a") as f:
        if "data" not in f:
            raise KeyError("'/data' group missing in HDF5 file")
        episodes = list(f["data"].keys())
        logging.info("%d episode group(s) detected", len(episodes))
        for ep in episodes:
            logging.info("Processing %s", ep)
            process_episode(f["data"][ep], ep, paired_root)

    logging.info("✔ All done.")

if __name__ == "__main__":
    main()
