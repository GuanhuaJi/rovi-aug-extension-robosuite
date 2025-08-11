#!/usr/bin/env python3
# export_episode_pool_light.py
import os, argparse, random
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from multiprocessing import get_context
import numpy as np
import imageio.v3 as iio 
import json, datetime

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import tensorflow_datasets as tfds
import tensorflow as tf
tf.get_logger().setLevel("ERROR") 

from envs import SourceEnvWrapper
from config.dataset_poses_dict import ROBOT_CAMERA_POSES_DICT

def _worker_init(gpu_id: int):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

def process_one_episode(idx: int,
                        joints: np.ndarray,
                        gripper: np.ndarray,
                        meta: dict,
                        out_dir: str,
                        verbose: bool = False):
    wrapper = SourceEnvWrapper(
        source_name    = meta["robot"],
        source_gripper = meta["gripper"],
        robot_dataset  = meta["dataset_name"],
        verbose        = verbose,
    )
    wrapper.get_source_robot_states(
        save_source_robot_states_path = out_dir,
        episode                       = idx,
        joint_angles                  = joints,
        gripper_states                = gripper,
    )
    wrapper.source_env.env.close_renderer()
    if verbose:
        print(f"✓ episode {idx} done")


import collections
import numpy as np
import tensorflow_datasets as tfds

def _stack_dict_list(list_of_dicts):
    """
    list_of_dicts : [ {k1: v1_step0, k2: v2_step0, ...},
                      {k1: v1_step1, ...}, ... ]
    return        : {k1: stacked_v1(T,...), k2: ...}
                   （若 value 仍是 dict，则递归地继续堆叠）
    """
    out = {}
    sample = list_of_dicts[0]
    for k, v0 in sample.items():
        if isinstance(v0, collections.abc.Mapping):
            # 递归处理子 dict
            out[k] = _stack_dict_list([d[k] for d in list_of_dicts])
        else:
            out[k] = np.stack([d[k] for d in list_of_dicts], axis=0)
    return out

def stack_steps(steps_ds):
    """
    steps_ds : episode['steps']  (tf.data.Dataset)
    return   : dict whose leaves are (T, …) np.ndarray
    """
    steps_iter = steps_ds.as_numpy_iterator()   # 正规 iterator
    list_of_dicts = list(steps_iter)            # 全部搬到内存
    return _stack_dict_list(list_of_dicts)


def dispatch_episodes(robot_dataset: str,
                      workers: int = 20,
                      seed: int = 0,
                      chunksize: int = 100,
                      verbose: bool = False):

    random.seed(seed); np.random.seed(seed)

    meta = ROBOT_CAMERA_POSES_DICT[robot_dataset]
    meta["dataset_name"] = robot_dataset
    oxe_videos_dir = Path(meta["replay_path"]) / "original_oxe_videos"
    oxe_videos_dir.mkdir(parents=True, exist_ok=True)
    src_states_dir = Path(meta["replay_path"]) / "source_robot_states"
    src_states_dir.mkdir(parents=True, exist_ok=True)

    builder = tfds.builder_from_directory(meta["GCS_path"])
    info    = builder.info                # <tfds.core.DatasetInfo ...>

    dataset_name = info.name              # 真实 TFDS 名

    # ---------- 再创建 dataset ----------------------------------------
    ds = builder.as_dataset(
            split="train",
            shuffle_files=False,
            read_config=tfds.ReadConfig(try_autocache=False),
        )

    first_ex = next(iter(ds))
    if robot_dataset == "viola":
        img_example = first_ex["steps"].element_spec["observation"]["agentview_rgb"]
    else:
        img_example = first_ex["steps"].element_spec["observation"]["image"]
    img_h, img_w = img_example.shape[:2]
    num_eps = tf.data.experimental.cardinality(ds).numpy()

    meta_path = Path(meta["replay_path"]) / "dataset_metadata.json"
    meta_json = {
        "dataset": dataset_name,
        "num_episodes": int(num_eps),
        "image_height": int(img_h),
        "image_width": int(img_w),
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta_json, f, indent=2)
    if verbose:
        print(f"📄 metadata written to {meta_path}")

    proc_fn = meta["processing_function"]

    ctx = get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, 
                                mp_context=ctx,                                 
                                initializer=_worker_init,
                                initargs=(0,)) as pool:
        pending = []
        for idx, ex in enumerate(ds):
            batched_steps = stack_steps(ex["steps"])
            states, frames = proc_fn({"steps": batched_steps})
            states = states.numpy()
            if robot_dataset == "autolab_ur5":
                joints, grip = states[:, :6], states[:, 6]
            else:
                joints, grip = states[:, :7], states[:, 7]
            mp4_path = oxe_videos_dir / f"{idx}.mp4"
            iio.imwrite(mp4_path, frames, fps=30, codec="libx264")
            if verbose:
                print(f"🎞  saved {mp4_path}")
            fut = pool.submit(process_one_episode,
                               idx, joints, grip, meta, str(src_states_dir), verbose)
            pending.append(fut)
            if len(pending) >= chunksize:
                done, pending_set = wait(pending, return_when=FIRST_COMPLETED)
                pending = list(pending_set)

        for fut in pending:
            fut.result()

    print("🎉 all episodes exported")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot_dataset", required=True)
    ap.add_argument("--workers",   type=int, default=os.cpu_count())
    ap.add_argument("--seed",      type=int, default=0)
    ap.add_argument("--chunksize", type=int, default=100)
    ap.add_argument("--verbose",   action="store_true")
    args = ap.parse_args()

    dispatch_episodes(args.robot_dataset,
                      workers=args.workers,
                      seed=args.seed,
                      chunksize=args.chunksize,
                      verbose=args.verbose)