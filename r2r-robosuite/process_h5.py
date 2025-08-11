#!/usr/bin/env python3
"""
Update an image_84.hdf5 file by attaching overlay videos and end-effector state arrays.
"""
from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Sequence

import cv2
import h5py
import numpy as np

ROBOTS: Sequence[str] = ["UR5e", "Jaco", "Sawyer", "Kinova3", "IIWA"]
TARGET_SIZE = (84, 84)  # (width, height)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

_num_re = re.compile(r"(\d+)$")

def _decode_video(path: Path, size: tuple[int, int] = TARGET_SIZE) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video {path}")
    frames: list[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if (frame.shape[1], frame.shape[0]) != size:
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        frames.append(frame.astype(np.uint8))
    cap.release()
    if not frames:
        raise ValueError(f"No frames decoded from {path}")
    return np.stack(frames, axis=0)

def _replace_dataset(group: h5py.Group, name: str, data: np.ndarray, **kw) -> None:
    if name in group:
        del group[name]
    group.create_dataset(name, data=data, **kw)

def _episode_id_from_group(group_name: str) -> str:
    m = _num_re.search(group_name)
    return m.group(1) if m else group_name  # string form for filenames

def _episode_num(group_name: str) -> int | None:
    m = _num_re.search(group_name)
    return int(m.group(1)) if m else None

# -----------------------------------------------------------------------------
# Core
# -----------------------------------------------------------------------------

def process_episode(ep_group: h5py.Group, ep_name: str, paired_root: Path) -> None:
    file_id = _episode_id_from_group(ep_name)  # e.g. "73"
    state_root = paired_root / "target_robot_states"
    obs_grp = ep_group.require_group("obs")

    for robot in ROBOTS:
        video_path = paired_root / f"{robot}_overlay" / f"{file_id}.mp4"
        state_path = state_root / robot / f"{file_id}.npz"

        # Video
        if video_path.exists():
            try:
                frames = _decode_video(video_path)
                _replace_dataset(
                    obs_grp,
                    f"agentview_image_{robot}",
                    frames,
                    compression="gzip",
                    compression_opts=4,
                    chunks=(1, *TARGET_SIZE, 3),
                )
                logging.debug("Stored %s frames for %s", frames.shape[0], robot)
            except Exception as exc:
                logging.warning("Failed processing %s: %s", video_path, exc)
        else:
            logging.warning("Missing video: %s", video_path)

        # EEF states
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
# Entry
# -----------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Augment image_84.hdf5 with R2R data")
    p.add_argument("--dataset", required=True, help="Dataset folder name")
    p.add_argument("--h5-root", default="/home/abrashid/sim/",
                   help="Root dir containing <dataset>/image_84.hdf5")
    p.add_argument("--paired-root",
                   default="/home/guanhuaji/mirage/robot2robot/rendering/paired_images",
                   help="Root dir of paired_images/<dataset>")
    # NEW: numeric episode range, inclusive, by demo suffix
    p.add_argument("--start", type=int, default=None,
                   help="Process from demo_<start> (inclusive)")
    p.add_argument("--end", type=int, default=None,
                   help="Process through demo_<end> (inclusive)")
    args = p.parse_args()

    h5_path = Path(args.h5_root) / args.dataset / "image_84.hdf5"
    paired_root = Path(args.paired_root) / args.dataset
    if not h5_path.exists():
        raise FileNotFoundError(h5_path)

    logging.info("Opening %s", h5_path)
    with h5py.File(h5_path, "a") as f:
        if "data" not in f:
            raise KeyError("'/data' group missing in HDF5 file")

        # Collect (num, name), keep only demo_* that have a numeric suffix
        items: list[tuple[int, str]] = []
        for name in f["data"].keys():
            n = _episode_num(name)
            if n is not None:
                items.append((n, name))
        if not items:
            raise RuntimeError("No demo_* groups with numeric suffix found under /data")

        # Sort by numeric ID
        items.sort(key=lambda x: x[0])

        # Apply numeric range filter (inclusive)
        min_id, max_id = items[0][0], items[-1][0]
        s = min_id if args.start is None else args.start
        e = max_id if args.end   is None else args.end
        if s > e:
            raise ValueError(f"--start ({s}) must be <= --end ({e})")

        selected = [(n, name) for (n, name) in items if s <= n <= e]
        logging.info("%d episode group(s) detected; %d selected in [%d, %d]",
                     len(items), len(selected), s, e)

        for n, ep in selected:
            logging.info("Processing demo_%d (%s)", n, ep)
            process_episode(f["data"][ep], ep, paired_root)

    logging.info("✔ All done.")

if __name__ == "__main__":
    main()

