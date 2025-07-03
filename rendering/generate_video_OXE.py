#!/usr/bin/env python3
# generate_video_parallel.py
"""
并行调用 overlay.py 生成合成视频；
Ctrl-C 可随时打断，子进程会被统一杀掉；
失败的 (dataset, robot, episode) 组合追加写入 failed_jobs.txt。

python /home/guanhuaji/mirage/robot2robot/rendering/generate_video_OXE.py
"""

import os, sys, signal, atexit, threading, subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ──────────────────────────── 路径 & 常量 ────────────────────────────
ROOT         = Path("/home/guanhuaji/mirage/robot2robot/rendering/inpaint_utils")
OVERLAY      = ROOT / "overlay.py"
FPS          = 30
MAX_WORKERS  = 20                       # 并发线程数，自行调整
FAILED_FILE  = "failed_jobs.txt"        # 失败记录文件

# 如需删掉上一轮失败记录，取消下一行注释
# Path(FAILED_FILE).write_text("")

# ──────────────────────────── 数据集信息 ────────────────────────────
from config.dataset_pair_location import dataset_path, inpainting_path
from config.dataset_poses_dict import ROBOT_CAMERA_POSES_DICT

# ──────────────────────────── 进程组管理 ────────────────────────────
PROCS: list[subprocess.Popen] = []      # 所有子进程句柄
_PROCS_LOCK = threading.Lock()

def _register(p: subprocess.Popen):
    with _PROCS_LOCK:
        PROCS.append(p)

def _unregister(p: subprocess.Popen):
    with _PROCS_LOCK:
        if p in PROCS:
            PROCS.remove(p)

def _kill_children(sig=signal.SIGTERM):
    """给每个子进程组发送信号（默认 SIGTERM）"""
    with _PROCS_LOCK:
        for p in PROCS:
            if p.poll() is None:
                try:
                    os.killpg(p.pid, sig)
                except ProcessLookupError:
                    pass
        PROCS.clear()

def _sig_handler(signum, frame):
    print("\n⏹  收到中断信号，正在终止所有子进程 ...", file=sys.stderr, flush=True)
    _kill_children(signal.SIGTERM)
    # 如果用户再按一次 Ctrl-C，让默认处理器直接终止主进程
    signal.signal(signal.SIGINT, signal.SIG_DFL)

signal.signal(signal.SIGINT,  _sig_handler)
signal.signal(signal.SIGTERM, _sig_handler)
atexit.register(_kill_children)

# ──────────────────────────── 失败记录 ────────────────────────────
FAILED: list[tuple[str, str, int]] = []
_FAIL_LOCK = threading.Lock()

def record_fail(ds: str, robot: str, ep: int):
    with _FAIL_LOCK:
        FAILED.append((ds, robot, ep))
        with open(FAILED_FILE, "a") as fh:
            fh.write(f"{ds},{robot},{ep}\n")

# ──────────────────────────── 子进程 & 命令封装 ────────────────────────────
def overlay_cmd(dataset: str, robot: str, ep: int) -> list[str]:
    paired_path = Path(dataset_path[dataset])
    inpaint_dir = Path(inpainting_path[dataset])
    out_root    = Path("/home/abrashid/cross_inpainting")


    if dataset == "taco_play":
        inpaint_dir = inpaint_dir / f"episode_{ep}" / "frames" / "inpaint_out.mp4"
    else:
        inpaint_dir = (inpaint_dir / f"{ep}.mp4"
                    if "shared" in inpainting_path[dataset]
                    else inpaint_dir / f"{ep}")

    return [
        sys.executable, str(OVERLAY),
        "--original_path", str(inpaint_dir),
        "--mask_folder",    str(paired_path / dataset / f"{robot}_mask" / f"{ep}"),
        "--overlay_folder", str(paired_path / dataset / f"{robot}_rgb"  / f"{ep}"),
        "--output_folder",  str(out_root / dataset / robot / f"{ep}")
    ]

def run_cmd(cmd: list[str], ds: str, robot: str, ep: int) -> None:
    """启动外部脚本并等待；异常返回时记录失败。"""
    p = subprocess.Popen(cmd, start_new_session=True)
    _register(p)
    try:
        ret = p.wait()
        if ret != 0:
            record_fail(ds, robot, ep)
    finally:
        _unregister(p)

# ──────────────────────────── Episode 任务 ────────────────────────────

ROBOTS = {
    "viola": ["Sawyer", "IIWA", "Jaco", "Kinova3", "UR5e"],
    "austin_sailor": ["Sawyer", "IIWA", "Jaco", "Kinova3"],
    "austin_buds": ["Sawyer", "IIWA", "Jaco", "Kinova3"],
    "toto": ["Sawyer", "IIWA", "Jaco", "Kinova3", "UR5e"],
    "furniture_bench": ["Sawyer", "IIWA", "Jaco", "Kinova3"],
    "taco_play": ["Sawyer", "IIWA", "Jaco", "Kinova3"],
    "iamlab_cmu": ["Sawyer", "IIWA", "Jaco", "Kinova3", "UR5e"],
    "austin_mutex": ["Sawyer", "IIWA", "Jaco", "Kinova3"],
    "kaist": ["Sawyer", "IIWA", "Jaco", "Kinova3"],
    "nyu_franka": ["Sawyer", "IIWA", "Jaco", "Kinova3"],
    "asu_table_top_rlds": ["Sawyer", "IIWA", "Jaco", "Kinova3", "Panda"],
    "autolab_ur5": ["Sawyer", "IIWA", "Jaco", "Kinova3"],
    "ucsd_kitchen_rlds": ["Sawyer", "IIWA", "Jaco", "Kinova3"],
    "utokyo_pick_and_place": ["Sawyer"],
}
def episode_task(dataset: str, ep: int):
    # 针对数据集选择机器人列表
    # robots = (["UR5e", "Sawyer", "IIWA", "Jaco", "Kinova3"]
    #           if dataset in {"viola", "furniture_bench", "taco_play", "iamlab_cmu"}
    #           else ["Sawyer", "IIWA", "Jaco", "Kinova3"])

    robots = ROBOTS[dataset]
    

    for robot in robots:
        run_cmd(overlay_cmd(dataset, robot, ep), dataset, robot, ep)

# ──────────────────────────── 主流程 ────────────────────────────
def main():
    # robot_datasets = [
    #     "toto", "furniture_bench", "kaist",
    #     "taco_play", "iamlab_cmu", "austin_mutex", "austin_sailor", "viola"
    # ]
    robot_datasets = [
        "utokyo_pick_and_place"
    ]


    try:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = [
                pool.submit(episode_task, ds, ep)
                for ds in robot_datasets
                #for ep in range(ROBOT_CAMERA_POSES_DICT[ds]["num_episodes"])
                for ep in range(5)
            ]

            # 逐个 wait；内部异常不会抛到这里
            for _ in as_completed(futures):
                pass

    except KeyboardInterrupt:
        print("🛑 用户中断 (Ctrl-C)")
    finally:
        _kill_children(signal.SIGKILL)  # 保险起见

    # ----- 结束汇总 -----
    if FAILED:
        print(f"⚠️  共 {len(FAILED)} 个任务失败，已写入 {FAILED_FILE}")
    else:
        print("🎉 所有任务完成且无失败！")

# ──────────────────────────── 入口 ────────────────────────────
if __name__ == "__main__":
    main()