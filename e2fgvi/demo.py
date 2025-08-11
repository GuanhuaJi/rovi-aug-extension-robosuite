# -*- coding: utf-8 -*-
import os, re, math, argparse
import cv2, torch, imageio
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision.utils import save_image
from core.utils import to_tensors
import importlib
import warnings
import itertools
from tqdm import tqdm
from torch.amp import autocast
import sys 

import time, contextlib

import os
TQDM_POS = int(os.getenv("TQDM_POS", 0)) 

@contextlib.contextmanager
def timer(name="block"):
    start = time.perf_counter()
    yield
    print(f"[{name}] {time.perf_counter() - start:.3f}s")

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="torch.meshgrid: in an upcoming release.*",
)
warnings.filterwarnings(
    "ignore",                       # action: “ignore”, “error”, “once”, etc.
    category=DeprecationWarning,    # only this warning type
    module=r"torch\.distributed\.optim",  # regex matching the *origin* module
)

warnings.filterwarnings(
    action="ignore",
    message=r"`TorchScript` support for functional optimizers is deprecated.*",
    category=DeprecationWarning,
    module=r"mmengine\.optim\.optimizer\.zero_optimizer"
)

MAX_FRAMES = 100

# ───────────────────────── CLI ─────────────────────────
parser = argparse.ArgumentParser(description="E2FGVI")
parser.add_argument("-v",  "--video",       type=str, required=True)
parser.add_argument("-m",  "--mask",        type=str, required=True)
parser.add_argument("-c",  "--ckpt",        type=str, required=True)
parser.add_argument("-o",  "--save_frame",  type=str, required=True,
                    help="Path of output MP4 (H.264, yuv420p)")
parser.add_argument("--model", choices=["e2fgvi", "e2fgvi_hq"], required=True)
parser.add_argument("--step",             type=int, default=10)
parser.add_argument("--num_ref",          type=int, default=-1)
parser.add_argument("--neighbor_stride",  type=int, default=5)
parser.add_argument("--savefps",          type=int, default=24)

# HQ mode / arbitrary resolution
parser.add_argument("--set_size", action="store_true", default=False)
parser.add_argument("--width",    type=int)
parser.add_argument("--height",   type=int)

# NEW: mask dilation strength
parser.add_argument("--dilution", type=int, default=0,
                    help="Iterations for cv2.dilate on masks")

args = parser.parse_args()

ref_length      = args.step
num_ref         = args.num_ref
neighbor_stride = args.neighbor_stride
frame_save_path = args.save_frame  # now an MP4


# ────────────────── helpers ──────────────────
def natural_sort_key(s: str):
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r"(\d+)", s)]

def get_ref_index(f, neighbor_ids, length):
    """Reference-frame sampling (unchanged)"""
    if num_ref == -1:
        return [i for i in range(0, length, ref_length)
                if i not in neighbor_ids]
    ref = []
    start = max(0, f - ref_length * (num_ref // 2))
    end   = min(length, f + ref_length * (num_ref // 2))
    for i in range(start, end + 1, ref_length):
        if i not in neighbor_ids:
            if len(ref) >= num_ref:
                break
            ref.append(i)
    return ref

def read_mask(mask_path: str, size: tuple[int, int], dilation_iter: int):
    """
    Read masks, supporting both a directory (one image per frame) and mp4 video.
    Returns: List[PIL.Image], each a single-channel mask (0/255) already resized.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    masks  = []

    if mask_path.endswith(".mp4"):
        # ---------- Read MP4 frame by frame ----------
        cap, ok = cv2.VideoCapture(mask_path), True
        while ok:
            ok, frame = cap.read()
            if not ok:
                break
            # If mask video is color or grayscale, convert to single channel
            m = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            m = cv2.resize(m, size, interpolation=cv2.INTER_NEAREST)
            m = (m > 127).astype(np.uint8)          # binarize
            if dilation_iter:
                m = cv2.dilate(m, kernel, iterations=dilation_iter)
            masks.append(Image.fromarray(m * 255))
        cap.release()
    else:
        # ---------- Original directory logic ----------
        names = sorted(os.listdir(mask_path), key=natural_sort_key)
        for name in names:
            m = Image.open(os.path.join(mask_path, name)).resize(size, Image.NEAREST)
            m = (np.array(m.convert("L")) > 127).astype(np.uint8)
            if dilation_iter:
                m = cv2.dilate(m, kernel, iterations=dilation_iter)
            masks.append(Image.fromarray(m * 255))

    if not masks:
        raise RuntimeError(f"No masks found in {mask_path}")
    return masks

def read_frames(inp):
    """Load either mp4 or directory of images into PIL list"""
    frames = []
    if inp.endswith(".mp4"):
        cap, ok = cv2.VideoCapture(inp), True
        while ok:
            ok, frame = cap.read()
            if not ok: break
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    else:
        for fn in sorted(os.listdir(inp), key=natural_sort_key):
            img = cv2.imread(os.path.join(inp, fn))
            frames.append(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    return frames

def make_overlay(frame_np: np.ndarray,
                 mask_np: np.ndarray,
                 color=(0, 255, 0),
                 alpha: float = 0.45) -> np.ndarray:
    """
    Frame + mask ⇒ semi-transparent visualization
    frame_np : H×W×3, uint8, RGB
    mask_np  : H×W, bool / {0,1}
    color    : BGR color; default green for distinction
    alpha    : mask opacity
    """
    overlay = frame_np.copy()
    color_layer = np.zeros_like(overlay)
    color_layer[mask_np > 0.5] = color
    cv2.addWeighted(color_layer, alpha, overlay, 1 - alpha, 0, overlay)
    return overlay

# ────────────────── main worker ──────────────────
def main_worker():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Size preset
    size = (432, 240) if args.model == "e2fgvi" else None
    if args.set_size:
        size = (args.width, args.height)

    # Model
    net   = importlib.import_module(f"model.{args.model}")
    model = net.InpaintGenerator().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=True))
    model.eval()

    # Data
    raw_frames       = read_frames(args.video)
    raw_frames, size = ([f.resize(size) for f in raw_frames] if size else raw_frames,
                        raw_frames[0].size if size is None else size)
    h, w             = size[1], size[0]
    video_len        = len(raw_frames)

    frames_np = np.stack([np.array(f) for f in raw_frames])          # (T, H, W, 3)  uint8
    imgs = torch.from_numpy(frames_np)                               # to torch CPU
    imgs = imgs.permute(0, 3, 1, 2).unsqueeze(0).float() / 127.5 - 1
    imgs = imgs.to(device, non_blocking=True)
    masks_img = read_mask(args.mask, size, args.dilution)
    mask_np = np.stack([np.array(m) for m in masks_img])        # (T, H, W)  uint8

    # if len(mask_np) < video_len:
    #     last = mask_np[-1]
    #     pad  = np.repeat(last[None, ...], video_len - len(mask_np), axis=0)
    #     mask_np = np.concatenate([mask_np, pad], axis=0)
    # elif len(mask_np) > video_len:
    #     raise ValueError(f"More masks ({len(mask_np)}) than frames ({video_len}).")

    masks   = torch.from_numpy(mask_np)                         # CPU tensor
    masks   = masks.unsqueeze(0).unsqueeze(2).float() / 255.0   # (1, T, 1, H, W) in {0,1}
    masks   = masks.to(device, non_blocking=True)

    # Binary masks for compositing
    binary_masks = (mask_np[..., None] > 0.5).astype(np.uint8)
    comp_frames  = [None] * video_len
    overlay_frames = [None] * video_len

    # ─── optional chunking (long videos) ───
    if video_len > MAX_FRAMES:
        num_chunks  = math.ceil(video_len / MAX_FRAMES)
        chunk_size  = math.ceil(video_len / num_chunks)
    else:
        num_chunks, chunk_size = 1, video_len

    # Inference loop (chunk-wise)
    for k in tqdm(range(num_chunks), position=TQDM_POS, desc="Processing chunks"):
        s, e = k * chunk_size, min(video_len, (k + 1) * chunk_size)
        _process_segment(s, e, imgs, masks, raw_frames, comp_frames, overlay_frames,
                         device, h, w, model, binary_masks, video_len)

    # ─── write MP4 ───
    os.makedirs(os.path.dirname(frame_save_path) or ".", exist_ok=True)
    writer = imageio.get_writer(frame_save_path,
                                fps=args.savefps,
                                codec="libx264",
                                quality=8,
                                pixelformat="yuv420p",
                                macro_block_size=None)
    for frame in comp_frames:
        writer.append_data(frame.astype(np.uint8))
    writer.close()
    tqdm.write(f"✅ Done! MP4 saved to: {frame_save_path}")

    overlay_path = os.path.join(
        os.path.dirname(frame_save_path) or ".", "mask_in_e2fgvi.mp4"
    )
    ov_writer = imageio.get_writer(
        overlay_path,
        fps=args.savefps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=None,
    )
    for frame in overlay_frames:
        ov_writer.append_data(frame.astype(np.uint8))
    ov_writer.close()
    tqdm.write(f"✅ Mask visual saved to: {overlay_path}")

# ────────────────── segment processing (refactored) ──────────────────
def _process_segment(s, e, imgs, masks, frames, comp_frames, overlay_frames,
                     device, h, w, model, binary_masks, length):
    seg_imgs   = imgs[:, s:e]
    seg_masks  = masks[:, s:e]
    seg_len    = e - s

    for loc in range(0, seg_len, neighbor_stride):
        f_global    = s + loc
        neighbor_ids = list(range(max(s, f_global - neighbor_stride),
                                  min(e, f_global + neighbor_stride + 1)))
        ref_ids      = [rid for rid in get_ref_index(f_global, neighbor_ids, length)
                        if s <= rid < e]

        n_local = [i - s for i in neighbor_ids]
        r_local = [i - s for i in ref_ids]

        sel_imgs  = seg_imgs[:, n_local + r_local]

        idx = n_local + r_local                       # Python list of ints
        # turn it into a tensor on the same device as the data
        idx_t = torch.as_tensor(idx, device=seg_imgs.device)

        # ---------------------- DEBUG PRINTS ------------------------
        min_idx, max_idx = idx_t.min().item(), idx_t.max().item()

        # ---------------------- BOUNDS CHECK ------------------------
        seg_T = seg_masks.size(1)
        if min_idx < 0 or max_idx >= seg_T:
            raise ValueError(
                f"Index out of bounds for seg_masks (T={seg_T}): {idx}"
            )

        sel_masks = seg_masks[:, n_local + r_local]

        with torch.no_grad(), autocast(device_type='cuda', dtype=torch.bfloat16):
            masked    = sel_imgs * (1 - sel_masks)
            mod_h, mod_w = 60, 108
            h_pad = (mod_h - h % mod_h) % mod_h
            w_pad = (mod_w - w % mod_w) % mod_w
            masked = torch.cat([masked, torch.flip(masked, [3])], 3)[..., :h + h_pad, :]
            masked = torch.cat([masked, torch.flip(masked, [4])], 4)[..., :w + w_pad]
            pred, _ = model(masked, len(n_local))
            pred    = (pred[..., :h, :w] + 1) / 2
            pred = pred.to(torch.float32)
            pred    = pred.cpu().permute(0, 2, 3, 1).numpy() * 255

        for i, gid in enumerate(neighbor_ids):
            comp = (pred[i].astype(np.uint8) * binary_masks[gid] +
                    frames[gid] * (1 - binary_masks[gid]))
            comp_frames[gid] = comp if comp_frames[gid] is None else (
                (comp_frames[gid].astype(np.float32) + comp.astype(np.float32)) * 0.5
            )
            if overlay_frames[gid] is None:
                overlay_frames[gid] = make_overlay(
                    np.array(frames[gid]),
                    binary_masks[gid].squeeze()  # H×W bool
                )

# ────────────────────────────
if __name__ == "__main__":
    with timer("E2FGVI inference"):
        main_worker()
