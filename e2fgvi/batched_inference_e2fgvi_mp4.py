#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
multi_gpu_e2fgvi_mp4.py

Multi-GPU E2FGVI-HQ runner (process version, tqdm 全局进度).

用法示例
--------
# 静默成功输出，失败时打印错误
python multi_gpu_e2fgvi_mp4.py \
    --bg_root   /path/to/bg_videos \
    --mask_root /path/to/mask_videos \
    --output_root /path/to/out \
    --gpus 0 1 --dilution 1

# 调试模式：始终打印 demo.py 输出
python multi_gpu_e2fgvi_mp4.py ... -v
"""
from __future__ import annotations
import argparse, os, sys, subprocess
from multiprocessing import Process, JoinableQueue, Queue
from pathlib import Path
from typing import List, Tuple

import cv2           # noqa: E402
from tqdm import tqdm # noqa: E402

CKPT_PATH = "release_model/E2FGVI-HQ-CVPR22.pth"
# ───────────────────────── helpers ──────────────────────────────────
def make_pair(ep: str, bg_root: str, mask_root: str,
              output_root: str, dilution: int
              ) -> Tuple[str, str, str, int]:
    """Return (bg_video, mask_video, save_path, dilution) for one episode."""
    return (
        os.path.join(bg_root,  f"{ep}.mp4"),
        os.path.join(mask_root, f"{ep}.mp4"),
        os.path.join(output_root, ep, "inpaint_e2fgvi.mp4"),
        dilution,
    )

def discover_range(start: int, end: int, bg_root: str, mask_root: str,
                   output_root: str, dilution: int
                   ) -> List[Tuple[str, str, str, int]]:
    return [
        make_pair(str(i), bg_root, mask_root, output_root, dilution)
        for i in range(start, end + 1)
    ]

# ───────────────────────── subprocess wrapper ────────────────────────
def run_subprocess(bg_video: str, mask_video: str,
                   save_path: str, dilution: int,
                   row: int, verbose: bool) -> None:
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "demo.py",
        "--model", "e2fgvi_hq",
        "--video", bg_video,
        "--mask",  mask_video,
        "--ckpt",  CKPT_PATH,
        "--save_frame", save_path,
        "--dilution", str(dilution),
    ]
    env = os.environ.copy()
    env["TQDM_POS"] = str(row)
    env["PYTHONWARNINGS"] = "ignore::DeprecationWarning"

    if verbose:
        # 把 demo.py 输出实时打到终端
        subprocess.run(cmd, check=True, env=env)
    else:
        # 捕获输出；失败时抛出并带上日志
        result = subprocess.run(
            cmd, env=env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        if result.returncode != 0:
            raise RuntimeError(result.stdout)

# ───────────────────────── episode driver ────────────────────────────
def run_episode(bg_video: str, mask_video: str,
                save_path: str, dilution: int,
                row: int, verbose: bool) -> None:
    run_subprocess(bg_video, mask_video, save_path, dilution, row, verbose)

# ───────────────────────── worker proc ───────────────────────────────
def worker_proc(gpu_id: int, job_q: JoinableQueue,
                done_q: Queue, failed_file: str, verbose: bool) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    cv2.setNumThreads(0)

    row_base, job_idx = gpu_id * 10, 0
    while True:
        item = job_q.get()
        if item is None:  # poison pill
            job_q.task_done()
            break

        bg_video, mask_video, save_path, dilution = item
        ep_id = Path(bg_video).stem
        try:
            run_episode(bg_video, mask_video, save_path,
                        dilution, row_base + job_idx, verbose)
            job_idx += 1
        except Exception as e:
            # 将错误/日志打印出来
            print(f"[GPU {gpu_id}] ❌ Episode {ep_id} failed:\n{e}\n{'─'*60}")
            try:
                with open(failed_file, "a", encoding="utf-8") as ff:
                    ff.write(f"{ep_id}\n")
            except Exception as fe:
                print(f"[GPU {gpu_id}] ⚠️  Could not write log: {fe}")
        finally:
            done_q.put(1)
            job_q.task_done()

# ───────────────────────── main ──────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser("Multi-GPU E2FGVI-HQ runner (mp4 input)")
    ap.add_argument("--bg_root",    required=True,
                    help="Folder with background videos 0.mp4 1.mp4 …")
    ap.add_argument("--mask_root",  required=True,
                    help="Folder with mask videos 0.mp4 1.mp4 …")
    ap.add_argument("--output_root", default=None,
                    help="Destination root; default = bg_root")
    ap.add_argument("--start",  type=int,
                    help="First episode id (inclusive)")
    ap.add_argument("--end",    type=int,
                    help="Last  episode id (inclusive)")
    ap.add_argument("--gpus",   nargs="+", type=int, default=[0, 1, 2, 3, 4],
                    help="GPU IDs, e.g. --gpus 0 1 2 3")
    ap.add_argument("--dilution", type=int, default=0,
                    help="Mask dilation factor")
    ap.add_argument("--redo_failed", action="store_true",
                    help="Re-run episodes listed in failed_episodes.txt")
    ap.add_argument("--list_file", action="store_true",
                    help="Re-run episodes listed in needs_update.txt under bg_root")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="Show demo.py stdout/stderr (default: silent, "
                         "but prints on error)")
    args = ap.parse_args()

    if not args.output_root:
        args.output_root = args.bg_root

    # ── build job list ────────────────────────────────────────────────
    if args.redo_failed:
        fp_prev = os.path.join(args.output_root, "failed_episodes.txt")
        if not os.path.exists(fp_prev):
            sys.exit(f"❌ {fp_prev} not found (nothing to redo)")
        eps = [ln.strip() for ln in Path(fp_prev).read_text().splitlines() if ln.strip()]
        if not eps:
            print("🎉 No failed episodes recorded, exiting.")
            return
        Path(fp_prev).unlink()
        jobs = [make_pair(ep, args.bg_root, args.mask_root,
                          args.output_root, args.dilution) for ep in eps]

    elif args.list_file:
        txt_path = Path(args.bg_root) / "needs_update.txt"
        if not txt_path.is_file():
            sys.exit(f"❌ 列表文件 {txt_path} 不存在")
        episodes = sorted({int(x) for x in txt_path.read_text().splitlines()
                           if x.strip().isdigit()})
        if not episodes:
            print(f"🎉 {txt_path} 中没有合法 episode id, 已退出。")
            return
        jobs = [make_pair(str(ep), args.bg_root, args.mask_root,
                          args.output_root, args.dilution) for ep in episodes]

    else:
        if args.start is not None and args.end is not None:
            jobs = discover_range(args.start, args.end,
                                  args.bg_root, args.mask_root,
                                  args.output_root, args.dilution)
        else:
            eps = [p.stem for p in Path(args.bg_root).glob("*.mp4")
                   if p.stem.isdigit()]
            if not eps:
                raise ValueError(f"No *.mp4 episode files in {args.bg_root}")
            jobs = [make_pair(ep, args.bg_root, args.mask_root,
                              args.output_root, args.dilution)
                    for ep in sorted(eps, key=int)]

    print(f"[INFO] Dispatching {len(jobs)} episode(s) across GPUs {args.gpus}")

    # ── failed-episode log ───────────────────────────────────────────
    failed_file = os.path.join(args.output_root, "failed_episodes.txt")
    Path(args.output_root).mkdir(parents=True, exist_ok=True)
    open(failed_file, "w").close()

    # ── queues & workers ─────────────────────────────────────────────
    job_q:  JoinableQueue = JoinableQueue()
    done_q: Queue         = Queue()
    for j in jobs:
        job_q.put(j)
    for _ in args.gpus:
        job_q.put(None)            # poison pill

    procs: List[Process] = []
    for gid in args.gpus:
        p = Process(target=worker_proc,
                    args=(gid, job_q, done_q, failed_file, args.verbose))
        p.start()
        procs.append(p)

    # ── global progress ──────────────────────────────────────────────
    with tqdm(total=len(jobs), desc="Episodes", unit="ep") as pbar:
        for _ in range(len(jobs)):
            done_q.get()
            pbar.update(1)

    job_q.join()
    for p in procs:
        p.join()

    print("✅ All episodes finished.")
    print(f"📄 Failed episodes (if any) recorded in: {failed_file}")

if __name__ == "__main__":
    main()

'''
conda create -n e2fgvi python=3.9 -y
conda activate e2fgvi
conda config --env --set channel_priority strict
conda install pytorch==2.4.1 torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html
conda install tensorboard matplotlib scikit-image
conda install tqdm
pip install -U imageio imageio-ffmpeg
'''