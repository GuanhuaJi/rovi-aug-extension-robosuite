#!/usr/bin/env python3
"""
Overlay masked RGB frames onto background videos for multiple robots & episodes.

Directory layout (under `--source-dir`)
  inpaint/{episode}/inpaint_e2fgvi.mp4            – background
  {robot}_replay_mask/{episode}.mp4               – mask (white = keep RGB)
  {robot}_replay_video/{episode}.mp4              – RGB
Output
  {robot}_overlay/{episode}.mp4                   – composited video
"""

from pathlib import Path
import argparse
import cv2
import numpy as np
import imageio.v3 as iio          # v3 unified API
from tqdm import tqdm


def overlay_episode(source_dir: Path, robot: str, episode: int,
                    mask_thresh: int = 127, overwrite: bool = False) -> None:
    """Create `{robot}_overlay/{episode}.mp4` using `iio.imwrite`."""
    bg_path   = source_dir / "inpaint" / str(episode) / "inpaint_e2fgvi.mp4"
    mask_path = source_dir / f"{robot}_replay_mask"  / f"{episode}.mp4"
    rgb_path  = source_dir / f"{robot}_replay_video" / f"{episode}.mp4"
    out_dir   = source_dir / f"{robot}_overlay"
    out_path  = out_dir / f"{episode}.mp4"

    # ── sanity checks ────────────────────────────────────────────────────────
    if not (bg_path.exists() and mask_path.exists() and rgb_path.exists()):
        print(f"[{robot} ep{episode}] missing file(s) – skipping")
        return
    if out_path.exists() and not overwrite:
        # already done
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── open inputs ─────────────────────────────────────────────────────────
    bg_cap, mask_cap, rgb_cap = (cv2.VideoCapture(str(p)) for p in (bg_path, mask_path, rgb_path))

    fps = bg_cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-3:
        fps = 30  # fallback

    frames = []  # will hold RGB frames for final write

    # get target size from background
    w = int(bg_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(bg_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    target_size = (w, h)

    while True:
        ret_bg,   bg   = bg_cap.read()
        ret_mask, mask = mask_cap.read()
        ret_rgb,  rgb  = rgb_cap.read()
        if not (ret_bg and ret_mask and ret_rgb):
            break

        # Resize mask/rgb if needed to match background
        if (mask.shape[1], mask.shape[0]) != target_size:
            mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
        if (rgb.shape[1], rgb.shape[0]) != target_size:
            rgb = cv2.resize(rgb, target_size, interpolation=cv2.INTER_LINEAR)

        # Threshold mask → binary
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, mask_bin = cv2.threshold(mask_gray, mask_thresh, 255, cv2.THRESH_BINARY)

        # Composite (OpenCV is BGR)
        comp = bg.copy()
        comp[mask_bin == 255] = rgb[mask_bin == 255]

        # imageio expects RGB
        frames.append(cv2.cvtColor(comp, cv2.COLOR_BGR2RGB))

    # clean up caps
    for cap in (bg_cap, mask_cap, rgb_cap):
        cap.release()

    if not frames:                 # nothing decoded → nothing to write
        print(f"[{robot} ep{episode}] no frames – skipping")
        return

    # Stack to (T, H, W, 3) and write once with iio.imwrite
    video_np = np.stack(frames, axis=0).astype(np.uint8, copy=False)

    iio.imwrite(
        out_path,
        video_np,
        fps=float(fps),
        codec="libx264",
        macro_block_size=1,        # disable 16-pixel padding
        pixelformat="yuv420p",     # 3-channel output
    )


def parse_args():
    p = argparse.ArgumentParser(description="Overlay masked RGB onto background videos.")
    p.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Root folder containing inpaint/, {robot}_replay_mask/, {robot}_replay_video/."
    )
    p.add_argument(
        "--robots",
        nargs="+",
        required=True,
        help="One or more robot names, e.g. --robots Jaco Sawyer (comma-separated also allowed)."
    )
    p.add_argument("--start", type=int, required=True, help="First episode index (inclusive).")
    p.add_argument("--end",   type=int, required=True, help="Last episode index (inclusive).")
    p.add_argument("--mask-thresh", type=int, default=127, help="Threshold 0-255 for mask binarization.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite outputs if they already exist.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.end < args.start:
        raise SystemExit("--end must be >= --start")

    # allow comma-separated robots too
    robots = []
    for tok in args.robots:
        robots.extend([r.strip() for r in tok.split(",") if r.strip()])
    # de-dupe while preserving order
    seen = set()
    robots = [r for r in robots if not (r in seen or seen.add(r))]

    episodes = range(args.start, args.end + 1)

    for robot in robots:
        print(f"\n=== Processing {robot} ===")
        for ep in tqdm(episodes, desc=robot, ncols=80):
            overlay_episode(
                args.source_dir, robot, ep,
                mask_thresh=args.mask_thresh,
                overwrite=args.overwrite
            )


if __name__ == "__main__":
    main()

'''
python /home/guanhuaji/OXE-Aug/r2r-robosuite/overlay_sim.py --source-dir /home/guanhuaji/OXE-Aug/replay_videos/can \
  --robots Sawyer IIWA --start 50 --end 120 --overwrite
'''