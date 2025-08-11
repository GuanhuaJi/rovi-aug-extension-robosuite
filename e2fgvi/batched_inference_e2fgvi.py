#!/usr/bin/env python3
# multi_gpu_e2fgvi.py
# -*- coding: utf-8 -*-
"""
Multi-GPU E2FGVI-HQ runner (process version)
- Single tqdm bar, subprocess stdout suppressed
- --skip_if_exist to avoid redoing existing outputs
- --num_workers to run multiple workers per GPU (even split)
"""
import argparse, os, sys, subprocess
from multiprocessing import Process, JoinableQueue, Queue
from typing import List, Tuple
from pathlib import Path

import cv2, numpy as np  # noqa: E402
from tqdm import tqdm

CKPT_PATH = "release_model/E2FGVI-HQ-CVPR22.pth"
IMG_EXTS  = (".jpg", ".jpeg", ".png", ".bmp")

# ───────────────────────── discover ──────────────────────────────────
def make_pair(ep: str, dir_root: str, output_root: str, dilution: int
              ) -> Tuple[str, str, str, int]:
    """Return (video_dir, mask_dir, save_path, dilution) for one episode."""
    return (
        os.path.join(dir_root,  ep, "frames"),
        os.path.join(dir_root,  ep, "mask_frames"),
        os.path.join(output_root, ep, "inpaint_e2fgvi.mp4"),
        dilution,
    )

def discover_range(start: int, end: int, dir_root: str,
                   output_root: str, dilution: int) -> List[Tuple[str, str, str, int]]:
    return [make_pair(str(i), dir_root, output_root, dilution)
            for i in range(start, end + 1)]

# ───────────────────────── subprocess wrapper ────────────────────────
def run_subprocess(video_dir, mask_dir, save_path, dilution, row):
    cmd = [
        sys.executable, "demo.py",
        "--model",  "e2fgvi_hq",
        "--video",  video_dir,
        "--mask",   mask_dir,
        "--ckpt",   CKPT_PATH,
        "--save_frame", save_path,
        "--dilution", str(dilution),
    ]
    env = os.environ.copy()
    env["TQDM_POS"] = str(row)  # mostly irrelevant since we suppress stdout
    env["PYTHONWARNINGS"] = "ignore::DeprecationWarning"

    # Silence child process
    subprocess.run(cmd, check=True, env=env,
                   stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

# ───────────────────────── episode driver ────────────────────────────
def run_episode(video_dir, mask_dir, save_path, dilution, row):
    # (kept in case you want local checks, but we silence child output)
    _ = sum(
        f.lower().endswith(IMG_EXTS) and
        os.path.exists(os.path.join(mask_dir, f))
        for f in os.listdir(video_dir)
    )
    run_subprocess(video_dir, mask_dir, save_path, dilution, row)

# ───────────────────────── multi-GPU worker ──────────────────────────
def worker_proc(gpu_id: int, worker_slot: int, job_q: JoinableQueue, done_q: Queue, failed_file: str) -> None:
    """Consumes jobs from `job_q` using GPU `gpu_id`.
       Always notifies `done_q` when a job finishes (success or fail)."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    cv2.setNumThreads(0)

    # Give each worker a stable row line
    row = gpu_id * 10 + worker_slot

    while True:
        item = job_q.get()
        if item is None:
            job_q.task_done()
            break

        video_dir, mask_dir, save_path, dilution = item
        try:
            run_episode(video_dir, mask_dir, save_path, dilution, row)
        except Exception as e:
            ep_id = Path(video_dir).parent.name
            print(f"[GPU {gpu_id} W{worker_slot}] ❌ Episode {ep_id}: {e}")
            try:
                with open(failed_file, "a", encoding="utf-8") as ff:
                    ff.write(f"{ep_id}\n")
            except Exception as fe:
                print(f"[GPU {gpu_id} W{worker_slot}] ⚠️  Could not write to {failed_file}: {fe}")
        finally:
            done_q.put(1)   # notify main progress bar
            job_q.task_done()

# ───────────────────────── helpers ───────────────────────────────────
def filter_existing_jobs(jobs: List[Tuple[str, str, str, int]]) -> Tuple[List[Tuple[str, str, str, int]], int]:
    """Return (remaining_jobs, skipped_count) by checking if save_path exists."""
    remaining = []
    skipped = 0
    for job in jobs:
        save_path = job[2]
        if os.path.isfile(save_path):
            print(f"Skipping existing: {save_path}")
            skipped += 1
            continue
        remaining.append(job)
    return remaining, skipped

def split_workers_evenly(num_workers: int, gpus: List[int]) -> List[int]:
    """Evenly split num_workers across GPUs; earlier GPUs get +1 if remainder."""
    if num_workers < 1:
        raise ValueError("--num_workers must be >= 1")
    if len(gpus) < 1:
        raise ValueError("At least one GPU id must be provided")
    base = num_workers // len(gpus)
    rem  = num_workers %  len(gpus)
    return [base + (1 if i < rem else 0) for i in range(len(gpus))]

# ───────────────────────── main ──────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser("Multi-GPU E2FGVI-HQ runner (process version)")
    ap.add_argument("--dir_root",    required=True, help="Dataset root with episode subfolders")
    ap.add_argument("--output_root", default=None, help="Where to write outputs (default: same as dir_root)")
    ap.add_argument("--start",  type=int, help="First episode id (inclusive)")
    ap.add_argument("--end",    type=int, help="Last  episode id (inclusive)")
    ap.add_argument("--gpus",   nargs="+", type=int, default=[0],
                    help="GPU IDs, e.g. --gpus 0 1 2 3")
    ap.add_argument("--dilution", type=int, default=0, help="Mask dilation factor (default: 0)")
    ap.add_argument("--redo_failed", action="store_true",
                    help="Only re-run episodes listed in failed_episodes.txt (from a previous run)")
    ap.add_argument("--list-file", action="store_true",
                    help="Use <dir_root>/needs_update.txt (one episode id per line)")
    ap.add_argument("--skip_if_exist", action="store_true",
                    help="Skip episodes whose target MP4 already exists")
    ap.add_argument("--num_workers", type=int, default=None,
                    help="Total worker processes across all GPUs (evenly split). Default: len(--gpus)")
    args = ap.parse_args()

    if not args.output_root:
        args.output_root = args.dir_root  # default same as input

    # ── build job list ────────────────────────────────────────────────
    if args.redo_failed:
        failed_file_prev = os.path.join(args.output_root, "failed_episodes.txt")
        if not os.path.exists(failed_file_prev):
            sys.exit(f"❌ {failed_file_prev} not found (nothing to redo)")
        with open(failed_file_prev, "r", encoding="utf-8") as ff:
            failed_eps = [ln.strip() for ln in ff if ln.strip()]
        if not failed_eps:
            print("🎉 No failed episodes recorded, nothing to redo.")
            return
        os.remove(failed_file_prev)
        jobs = [make_pair(ep, args.dir_root, args.output_root, args.dilution)
                for ep in failed_eps]
    elif args.list_file:
        txt_path = Path(args.dir_root) / "needs_update.txt"
        if not txt_path.is_file():
            raise SystemExit(f"❌ 未找到列表文件 {txt_path}")

        episodes = sorted({
            int(line.strip())
            for line in txt_path.read_text().splitlines()
            if line.strip().isdigit()
        })
        if not episodes:
            print(f"🎉 {txt_path} 中没有合法的 episode ID，已退出。")
            return
        jobs = [make_pair(str(ep), args.dir_root, args.output_root, args.dilution)
                for ep in episodes]
    else:
        if args.start is not None and args.end is not None:
            jobs = discover_range(args.start, args.end,
                                  args.dir_root, args.output_root, args.dilution)
        else:
            root = Path(args.dir_root)
            episode_ids = [p.name for p in root.iterdir()
                           if p.is_dir() and p.name.isdigit()]
            if not episode_ids:
                raise ValueError(f"No episode dirs under {root}")
            jobs = [make_pair(ep, args.dir_root, args.output_root, args.dilution)
                    for ep in sorted(episode_ids, key=int)]

    # ── optional skip-if-exist filter ─────────────────────────────────
    skipped_existing = 0
    if args.skip_if_exist:
        jobs, skipped_existing = filter_existing_jobs(jobs)

    print(f"[INFO] Dispatching {len(jobs)} episode(s) across GPUs {args.gpus}"
          + (f" (skipped {skipped_existing} existing)" if skipped_existing else ""))

    if len(jobs) == 0:
        print("✅ Nothing to do. Exiting.")
        return

    # ── prepare fresh failed-episode log ──────────────────────────────
    failed_file = os.path.join(args.output_root, "failed_episodes.txt")
    Path(args.output_root).mkdir(parents=True, exist_ok=True)
    open(failed_file, "w").close()

    # ── queues ───────────────────────────────────────────────────────
    job_q:  JoinableQueue = JoinableQueue()
    done_q: Queue         = Queue()

    for j in jobs:
        job_q.put(j)

    # ── worker allocation ────────────────────────────────────────────
    if args.num_workers is None:
        args.num_workers = len(args.gpus)

    per_gpu_counts = split_workers_evenly(args.num_workers, args.gpus)
    total_workers = sum(per_gpu_counts)

    # one poison pill per worker
    for _ in range(total_workers):
        job_q.put(None)

    # ── launch workers ───────────────────────────────────────────────
    procs: List[Process] = []
    for gi, gid in enumerate(args.gpus):
        for w in range(per_gpu_counts[gi]):
            p = Process(target=worker_proc, args=(gid, w, job_q, done_q, failed_file))
            p.start()
            procs.append(p)

    # helpful log
    alloc_str = ", ".join(f"GPU{gid}:{per_gpu_counts[i]}" for i, gid in enumerate(args.gpus))
    print(f"[INFO] Worker allocation → {alloc_str} (total {total_workers})")

    # ── global progress bar ──────────────────────────────────────────
    with tqdm(total=len(jobs), desc="Episodes", unit="ep") as pbar:
        completed = 0
        while completed < len(jobs):
            done_q.get()       # blocks until a worker reports completion
            completed += 1
            pbar.update(1)

    # ── cleanup ──────────────────────────────────────────────────────
    job_q.join()
    for p in procs:
        p.join()

    print("✅ All episodes finished.")
    print(f"📄 Failed episodes (if any) recorded in: {failed_file}")

if __name__ == "__main__":
    main()





'''
python /home/guanhuaji/diffusers-robotic-inpainting/batched_inference_e2fgvi.py \
    --dir_root /home/guanhuaji/load_datasets/language_table/train \
    --gpus 1 2 3 4 5 6 7 --num_workers 14 --dilution 8 --skip_if_exist

python /home/guanhuaji/diffusers-robotic-inpainting/batched_inference_e2fgvi.py \
    --dir_root /home/guanhuaji/load_datasets/toto/train \
    --start 100 --end 110 --output_root /home/guanhuaji/load_datasets/toto/train --gpus 0 1 2 3 4 5 6 7 --dilution 3

python /home/guanhuaji/diffusers-robotic-inpainting/batched_inference_e2fgvi.py \
    --dir_root /home/guanhuaji/load_datasets/toto/train \
    --output_root /home/guanhuaji/load_datasets/toto/train --gpus 0 1 2 3 4 5 6 7 --dilution 3

python /home/guanhuaji/diffusers-robotic-inpainting/batched_inference_e2fgvi.py \
    --dir_root /home/guanhuaji/load_datasets/berkeley_autolab_ur5/test \
    --gpus 0 1 2 3 4 5 6 7 --dilution 10


python /home/guanhuaji/diffusers-robotic-inpainting/batched_inference_e2fgvi.py \
    --dir_root /home/guanhuaji/load_datasets/utaustin_mutex/train \
    --gpus 0 1 2 3 4 5 6 7 --dilution 10

python /home/guanhuaji/diffusers-robotic-inpainting/batched_inference_e2fgvi.py \
    --dir_root /home/guanhuaji/load_datasets/austin_sailor_dataset_converted_externally_to_rlds/train \
    --gpus 0 1 2 3 4 5 6 7 --dilution 1

python /home/guanhuaji/diffusers-robotic-inpainting/batched_inference_e2fgvi.py     --dir_root /home/guanhuaji/load_datasets/utokyo_xarm_pick_and_place_converted_externally_to_rlds/train     --gpus 0 1 2 3 4 5 6 7 --dilution 15 --start 80 --end 85

python /home/guanhuaji/diffusers-robotic-inpainting/batched_inference_e2fgvi.py \
    --dir_root /home/guanhuaji/load_datasets/austin_buds_dataset_converted_externally_to_rlds/train \
    --gpus 0 1 2 3 4 5 6 7 --dilution 2

python /home/guanhuaji/diffusers-robotic-inpainting/batched_inference_e2fgvi.py \
    --dir_root /home/guanhuaji/load_datasets/ucsd_kitchen_dataset_converted_externally_to_rlds/train \
    --gpus 0 1 2 3 4 5 6 7 --dilution 20

python /home/guanhuaji/diffusers-robotic-inpainting/batched_inference_e2fgvi.py \
    --dir_root /home/guanhuaji/load_datasets/taco_play/train \
    --gpus 0 1 2 3 4 5 6 7 --dilution 2

python /home/guanhuaji/diffusers-robotic-inpainting/batched_inference_e2fgvi.py \
    --dir_root /home/guanhuaji/load_datasets/iamlab_cmu_pickup_insert_converted_externally_to_rlds/train \
    --gpus 0 1 2 3 4 5 6 7 --dilution 5
'''

'''
python /home/guanhuaji/diffusers-robotic-inpainting/batched_inference_e2fgvi.py \
    --dir_root /home/guanhuaji/load_datasets/nyu_franka_play_dataset_converted_externally_to_rlds/train \
    --gpus 0 1 2 3 4 5 6 7 --dilution 2

python /home/guanhuaji/diffusers-robotic-inpainting/batched_inference_e2fgvi.py \
    --dir_root /home/guanhuaji/load_datasets/nyu_franka_play_dataset_converted_externally_to_rlds/val \
    --gpus 0 1 2 3 4 5 6 7 --dilution 2

python /home/guanhuaji/diffusers-robotic-inpainting/batched_inference_e2fgvi.py \
    --dir_root /home/guanhuaji/load_datasets/austin_sailor_dataset_converted_externally_to_rlds/train \
    --gpus 0 1 2 3 4 5 6 7 --dilution 1

python /home/guanhuaji/diffusers-robotic-inpainting/batched_inference_e2fgvi.py \
    --dir_root /home/guanhuaji/load_datasets/austin_buds_dataset_converted_externally_to_rlds/train \
    --gpus 0 1 2 3 4 5 6 7 --dilution 2

python /home/guanhuaji/diffusers-robotic-inpainting/batched_inference_e2fgvi.py \
    --dir_root /home/guanhuaji/load_datasets/toto/train \
    --gpus 0 1 2 3 4 5 6 7 --dilution 3

python /home/guanhuaji/diffusers-robotic-inpainting/batched_inference_e2fgvi.py \
    --dir_root /home/guanhuaji/load_datasets/utaustin_mutex/train \
    --gpus 0 1 2 3 4 5 6 7 --dilution 10 --list-file 

python /home/guanhuaji/diffusers-robotic-inpainting/batched_inference_e2fgvi.py \
    --dir_root /home/guanhuaji/load_datasets/taco_play/train \
    --gpus 0 1 2 3 4 5 6 7 --dilution 2 --list-file

python /home/guanhuaji/diffusers-robotic-inpainting/batched_inference_e2fgvi.py \
    --dir_root /home/guanhuaji/load_datasets/taco_play/test \
    --gpus 0 1 2 3 4 5 6 7 --dilution 2 --list-file

python /home/guanhuaji/diffusers-robotic-inpainting/batched_inference_e2fgvi.py \
    --dir_root /home/guanhuaji/load_datasets/viola/test \
    --gpus 0 1 2 3 4 5 6 7 --dilution 5

python /home/guanhuaji/diffusers-robotic-inpainting/batched_inference_e2fgvi.py \
    --dir_root /home/guanhuaji/load_datasets/bridge/train \
    --gpus 0 1 2 3 4 5 6 7 --dilution 13

python /home/guanhuaji/diffusers-robotic-inpainting/batched_inference_e2fgvi.py \
    --dir_root /home/abrashid/OXE_inpainting/fractal20220817_data/train \
    --gpus 0 1 2 3 4 5 6 7 --num_workers 16 --dilution 3 --start 75000 --end 87212
'''

'''
path = /home/guanhuaji/load_datasets/austin_buds_dataset_converted_externally_to_rlds
for each sub dir:
    remove all files or folders beside frames(folder), mask_frames(folder), mask_frames_merged(folder), inpaint_e2fgvi.mp4, mask_in_e2fgvi.mp4, trajectory_background.mp4, and trajectory_replay.mp4
'''