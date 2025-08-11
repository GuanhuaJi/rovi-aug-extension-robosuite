#!/usr/bin/env python3
"""
Interactively segment & track objects in a video with SAM-2
— imageio / PyAV I/O edition —

Requirements
------------
conda create -n sam2 python=3.10 pytorch=2.5.1 torchvision=0.20.1 \
             pytorch-cuda=12.4 -c pytorch -c nvidia
conda activate sam2
pip install git+https://github.com/facebookresearch/sam2.git \
            imageio[pyav] matplotlib ipympl tqdm av
"""

import argparse, torch, numpy as np, matplotlib.pyplot as plt
import imageio.v3 as iio
from pathlib import Path
from tqdm import tqdm
from sam2.build_sam import build_sam2_video_predictor


# ─────────────────────────────────  I/O  ──────────────────────────────────
def read_video(path: Path) -> np.ndarray:
    """Read video -> (T,H,W,3) RGB uint8 using imageio/pyav."""
    return iio.imread(path, plugin="pyav")        # lazy-loads, then stacks

class VideoWriter:
    """Stream-writer that encodes on-the-fly via imageio/pyav."""
    def __init__(self, path: Path, fps: int, shape_hw: tuple[int, int]):
        self.writer = iio.get_writer(
            path,
            format="pyav",        # ← 指定后端
            fps=fps,
            codec="libx264",
            pixelformat="yuv420p",
            macro_block_size=None,
        )
        self._H, self._W = shape_hw

    def add(self, frame_rgb: np.ndarray):
        assert frame_rgb.shape[:2] == (self._H, self._W)
        self.writer.append_data(frame_rgb)

    def close(self):
        self.writer.close()


# ──────────────────────────────────  utils  ──────────────────────────────
def onclick_factory(clicks):
    def _onclick(ev):
        if ev.inaxes and ev.button == 1:
            clicks.append((ev.xdata, ev.ydata))
            ev.inaxes.scatter([ev.xdata], [ev.ydata], c="r", s=16)
            plt.draw()
    return _onclick


# ──────────────────────────────────  main  ───────────────────────────────
def main(args):
    vid = read_video(args.video)                  # (T,H,W,3) uint8 RGB
    H, W = vid.shape[1:3]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = build_sam2_video_predictor(
        args.config, args.checkpoint, device=device
    )

    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
        state = predictor.init_state(str(args.video))

    # ---------- click first frame ----------
    clicks = []
    plt.figure("Left-click object → close when done")
    plt.imshow(vid[0]); plt.axis("off")
    cid = plt.gcf().canvas.mpl_connect(
        "button_press_event", onclick_factory(clicks)
    )
    plt.show(); plt.gcf().canvas.mpl_disconnect(cid)

    if not clicks:
        print("⚠️  No clicks – exiting.")
        return
    points = np.asarray(clicks, np.float32)
    labels = np.ones(len(points), np.int32)

    _, obj_ids, _ = predictor.add_new_points_or_box(
        state, frame_idx=0, points=points, labels=labels, obj_id=-1
    )
    track_id = int(obj_ids[0])

    vw = VideoWriter(args.output, args.fps, (H, W))
    print("⏳ Propagating …")

    for f_idx, obj_ids, mask_logits in tqdm(
            predictor.propagate_in_video(state), total=len(vid)):
        m = (mask_logits[obj_ids.tolist().index(track_id)] > 0)
        overlay = vid[f_idx].copy()
        overlay[m] = [255, 0, 0]                  # red overlay
        vw.add(overlay)
    vw.close()
    print(f"✅ Saved {args.output}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video",   required=True, type=Path)
    ap.add_argument("--checkpoint",
                    default="/home/guanhuaji/oxeplusplus/sam2/checkpoints/checkpoint_150.pt")
    ap.add_argument("--config",
                    default="configs/sam2.1/sam2.1_hiera_b+.yaml")
    ap.add_argument("--output",  default="tracked.mp4", type=Path)
    ap.add_argument("--fps",     default=30, type=int)
    main(ap.parse_args())

'''
python /home/guanhuaji/oxeplusplus/sam2/sam2_click_track_io.py --video /home/guanhuaji/mirage/robot2robot/rendering/paired_images/kaist/Original_oxe_videos/0.mp4

'''