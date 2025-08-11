#!/usr/bin/env python3
# chunk_inference.py  –  multi‑GPU SAM‑2 mask extractor with 500‑frame chunks
"""
For every episode:
  • Read RGB frames (or super‑resolution frames)
  • Slice the sequence into ≤500‑frame chunks
  • Run SAM‑2 segmentation per chunk
  • Write 0/255 binary masks to   {episode}/mask_frames/{frame_id}.jpg
No composite/background MP4s are generated; those were unused.
"""

import time
import multiprocessing as mp
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

import os, shutil, tempfile
from pathlib import Path
import numpy as np
import torch
import cv2
from PIL import Image
import argparse

import matplotlib.pyplot as plt  # still imported in case you keep plotting
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# ───────────────────────── Global settings ──────────────────────────
CHUNK_SIZE = 500                         # max frames per SAM‑2 pass
MODEL_CHECKPOINT = "./checkpoints/checkpoint_150.pt"
MODEL_CONFIG     = "configs/sam2.1/sam2.1_hiera_b+.yaml"

device = torch.device("cuda")
if device.type == "cuda":
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:   # Ampere+
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32  = True


# ───────────────────────── Core helpers ─────────────────────────────
def best_image_to_video_validation(video_dir: str,
                                   start_idx: int,
                                   end_idx: int,
                                   mask_dir: str,
                                   vid_predictor,
                                   img_predictor):
    """
    Segment frames [start_idx, end_idx) inside <video_dir> and save masks
    as JPEGs in <mask_dir>/<frame_id>.jpg (0/255, single‑channel).
    """

    os.makedirs(mask_dir, exist_ok=True)

    # 0) gather frame names and pick the slice we need ------------------------
    frame_names = [p for p in os.listdir(video_dir)
                   if os.path.splitext(p)[-1].lower() in (".jpg", ".jpeg", ".png")]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    slice_names = frame_names[start_idx:end_idx]

    # 1) copy that slice into a temp dir so SAM‑2 sees a trimmed sequence -----
    with tempfile.TemporaryDirectory() as tmpdir:
        for fn in slice_names:
            shutil.copy2(os.path.join(video_dir, fn), os.path.join(tmpdir, fn))

        # 2) initialise SAM‑2 state ------------------------------------------
        inference_state = vid_predictor.init_state(video_path=tmpdir)

        # pick one ‘annotation’ frame inside the slice (motion heuristic)
        n_local  = len(slice_names)
        weights  = [0.5 + 0 * abs(i - n_local/2) / (n_local/2) for i in range(n_local)]
        diffs    = []
        for i in range(1, n_local):
            prev = np.array(Image.open(os.path.join(tmpdir, slice_names[i-1]))).astype(np.float32)
            curr = np.array(Image.open(os.path.join(tmpdir, slice_names[i  ]))).astype(np.float32)
            diffs.append(np.sum(np.abs(curr - prev) > 50) * weights[i])
        ann_local = np.argmax(diffs) + 1  # frame index within slice

        # 3) prompt‑less mask on that frame -----------------------------------
        img_predictor.set_image(Image.open(os.path.join(tmpdir, slice_names[ann_local])))

        sparse_emb, dense_emb = img_predictor.model.sam_prompt_encoder(
            points=None, boxes=None, masks=None)
        hr_feats = [f[-1].unsqueeze(0) for f in img_predictor._features["high_res_feats"]]
        low_res, _, _, _ = img_predictor.model.sam_mask_decoder(
            image_embeddings=img_predictor._features["image_embed"][-1].unsqueeze(0),
            image_pe=img_predictor.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=True, repeat_image=False,
            high_res_features=hr_feats)

        prd_mask = torch.sigmoid(
            img_predictor._transforms.postprocess_masks(
                low_res, img_predictor._orig_hw[-1])[:, 0])
        clean = (prd_mask / prd_mask.max() >= 0.5).float().squeeze(0)

        vid_predictor.add_new_mask(inference_state, ann_local, 1, clean)

        # 4) propagate through the slice --------------------------------------
        segs = {}
        for rev in (True, False):
            for fidx, obj_ids, logits in vid_predictor.propagate_in_video(
                    inference_state, start_frame_idx=ann_local, reverse=rev):
                segs[fidx] = (logits[0] > 0).cpu().numpy()

        # 5) save 0/255 masks --------------------------------------------------
        for local_idx, mask in segs.items():
            global_name = slice_names[local_idx]              # same file name
            dst = os.path.join(mask_dir, global_name)
            cv2.imwrite(dst, mask.squeeze().astype(np.uint8) * 255)


# ───────────────────────── Worker (per GPU) ─────────────────────────
def worker_fn(episodes, directory, gpu_idx, super_resolution):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
    device = torch.device("cuda:0")
    print(f"[GPU {gpu_idx}] started – episodes {episodes}")

    if device.type == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32  = True

    vid_predictor = build_sam2_video_predictor(MODEL_CONFIG, MODEL_CHECKPOINT, device=device)
    sam2_model    = build_sam2(MODEL_CONFIG, MODEL_CHECKPOINT, device=device)
    img_predictor = SAM2ImagePredictor(sam2_model)

    # ── iterate over assigned episodes ───────────────────────────────────────
    for ep in episodes:
        ep_dir = Path(directory) / str(ep)
        frames_dir = ep_dir / ("super_resolution_frames" if super_resolution else "frames")
        mask_dir   = ep_dir / ("super_resolution_mask_frames" if super_resolution else "mask_frames")
        mask_dir.mkdir(parents=True, exist_ok=True)
        if not frames_dir.exists():
            print(f"[GPU {gpu_idx}] ⚠️  {frames_dir} missing – skip")
            continue

        # slice into CHUNK_SIZE pieces ----------------------------------------
        frame_list = sorted(f for f in os.listdir(frames_dir)
                            if f.lower().endswith(('.jpg', '.jpeg', '.png')))
        n_frames = len(frame_list)
        print(f"[GPU {gpu_idx}] ▶ Episode {ep} – {n_frames} frames")

        for s in range(0, n_frames, CHUNK_SIZE):
            e = min(s + CHUNK_SIZE, n_frames)
            best_image_to_video_validation(str(frames_dir), s, e,
                                           str(mask_dir),
                                           vid_predictor, img_predictor)

        mask_list = [f for f in os.listdir(mask_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if len(mask_list) == n_frames:
            print(f"[GPU {gpu_idx}] ✅ Episode {ep}: masks OK ({n_frames})")
        else:
            print(f"[GPU {gpu_idx}] ❌ Episode {ep}: "
                  f"frames={n_frames}, masks={len(mask_list)}  (check!)")

    print(f"[GPU {gpu_idx}] done 🎉")


# ────────────────────────── CLI entrypoint ──────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end",   type=int, required=True)
    ap.add_argument("--directory", type=str, required=True)
    ap.add_argument("--num_workers", type=int, default=8,
                    help="number of parallel GPU workers")
    ap.add_argument("--super_resolution", action="store_true",
                    help="use *_super_resolution_frames as input")
    args = ap.parse_args()

    total_gpus = torch.cuda.device_count()
    if args.num_workers > total_gpus:
        raise ValueError(f"Asked for {args.num_workers} workers but only {total_gpus} GPUs visible")

    # episodes list (replace with your fixed list or range)
    episodes = list(range(args.start, args.end))  # simple contiguous range
    # episodes = [...]  # ← if you prefer the long fixed list, paste it here

    # round‑robin split
    chunks = [[] for _ in range(args.num_workers)]
    for idx, ep in enumerate(episodes):
        chunks[idx % args.num_workers].append(ep)

    procs = []
    for gpu_idx, ep_chunk in enumerate(chunks):
        if not ep_chunk:
            continue
        p = mp.Process(target=worker_fn,
                       args=(ep_chunk, args.directory, gpu_idx, args.super_resolution),
                       daemon=False)
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    print("✅ all processes finished")


if __name__ == "__main__":
    main()

'''
python /home/guanhuaji/oxeplusplus/sam2/sam2/chunk_inference.py \
    --start 0 --end 1000 --directory /home/guanhuaji/load_datasets/berkeley_autolab_ur5 \
    --num_workers 8
'''