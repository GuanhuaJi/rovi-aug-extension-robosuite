import argparse
import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import re
from pathlib import Path


_num_re = re.compile(r"\d+")

def numeric_key(p: Path) -> tuple[int, str]:
    """
    Return a key that sorts first by the first integer found in the stem,
    falling back to the full stem for non‑numeric or duplicate names.
    """
    m = _num_re.search(p.stem)
    return (int(m.group()) if m else float("inf"), p.stem)

#TODO: USE PROCESSING STEP FROM ROVI-AUG INSTEAD
def process_step_toto(step):
    arm_joints = step['observation']['state']
    gripper_joints = tf.cond(
        step['action']['open_gripper'],
        lambda: tf.constant([0.05, 0.05], dtype=tf.float32),
        lambda: tf.constant([0.02, 0.02], dtype=tf.float32),
    )
    image = step['observation']['image'] # Extract the image from the dataset
    return tf.concat([arm_joints, gripper_joints], axis=0), image

def process_step_nyu(step):
    arm_joints = step['observation']['state']
    binary_gripper = step['action'][13]
    bg = tf.cond(
        binary_gripper > 0,
        lambda: tf.constant([0.05, 0.05], dtype=tf.float32),
        lambda: tf.constant([0.0, 0.0], dtype=tf.float32),
    )
    
    image = step['observation']['image'] # Extract the image from the dataset
    return tf.concat([arm_joints[:7], bg], axis = 0), image

def process_step_berkeley_ur5(step):
    # Need to output [j0 -> j5, 0, 0, 0, 0, 0, 0, 0, 0]
    arm_joints = step['observation']['robot_state'][:14] # First 6 are joints
    is_gripper_closed = step['observation']['robot_state'][-2]
    is_gripper_closed = tf.cast(is_gripper_closed > 0.5, tf.bool)  # Assuming threshold 0.5 for True/False
    is_gripper_closed = tf.cond(is_gripper_closed, 
                                lambda: tf.constant([1.0, 0.025, 0.80, -0.80, 1.0, 0.0252768, 0.80, -0.80], dtype=tf.float32), 
                                lambda: tf.constant([0, 0, 0, 0, 0, 0, 0, 0], dtype=tf.float32))
    # zeros = tf.stack([is_gripper_closed, 0, 0, 0, is_gripper_closed, 0, 0, 0])
    joints = tf.concat([arm_joints[:6], is_gripper_closed], axis = 0)

    # zeros = tf.constant([0, 0, 0, 0, 0, 0, 0], dtype=tf.float32)
    # is_gripper_closed = step['observation']['robot_state'][-2]
    # bg = tf.cond(
    #     is_gripper_closed < 0,
    #     lambda: tf.constant([0.05, 0.05], dtype=tf.float32),
    #     lambda: tf.constant([0.025, 0.025], dtype=tf.float32),
    # )
    image = step['observation']['image'] # Extract the image from the dataset
    return joints, image

def process_step_ucsd_kitchen(step):
    arm_joints = step['observation']['state'][:7] # First seven joints
    gripper_open = step["action"][6]
    is_gripper_closed = tf.cast(gripper_open < 0.5, tf.bool)
    grip_control = tf.cond(
        is_gripper_closed,
        lambda: tf.constant([0.85, 0.844, 0.853, 0.85, 0.844, 0.853], dtype=tf.float32),
        lambda: tf.constant([0.185,0.185,0.188,0.185,0.185, 0.188], dtype=tf.float32)
    )

    image = step['observation']['image'] # Extract the image from the dataset
    return tf.concat([arm_joints, grip_control], axis = 0), image

def process_step_utokyo_xarm_pick_place(step):
    arm_joints = step['observation']['joint_state'][:7] # First seven joints
    is_gripper_closed = tf.cast(step["action"][-1] > 0.5, tf.bool)
    grip_control = tf.cond(
        is_gripper_closed,
        lambda: tf.constant([0.85, 0.844, 0.853, 0.85, 0.844, 0.853], dtype=tf.float32),
        lambda: tf.constant([0, 0, 0, 0, 0, 0], dtype=tf.float32)
    )

    image = step['observation']['image'] # Extract the image from the dataset
    return tf.concat([arm_joints, grip_control], axis = 0), image

def process_step_ucsd_pick_and_place(step):
    arm_joints = step['observation']['state'] # First seven joints
    is_gripper_closed = tf.cast(step["action"][-1] < 0, tf.bool)
    grip_control = tf.cond(
        is_gripper_closed,
        lambda: tf.constant([1.0,  0.75,  0.9,  1.0,  0.3, 1], dtype=tf.float32),
        lambda: tf.constant([0, 0, 0, 0, 0, 0], dtype=tf.float32)
    )

    image = step['observation']['image'] # Extract the image from the dataset
    return tf.concat([arm_joints, grip_control], axis = 0), image


def process_step_asu_table_top(step):
    arm_joints = step['observation']['state'][:6] # First six joints
    is_gripper_closed = tf.cast(step['observation']['state'][-1] > 0.2, tf.bool)
    is_gripper_closed = tf.cond(is_gripper_closed, 
                                lambda: tf.constant([0.498, 0.00155, 0.5, -0.482, 0.498,0.00154, 0.5,-0.482], dtype=tf.float32), 
                                lambda: tf.constant([0, 0, 0, 0, 0, 0, 0, 0], dtype=tf.float32))

    image = step['observation']['image'] # Extract the image from the dataset
    return tf.concat([arm_joints, is_gripper_closed], axis = 0), image

def process_step_kaist_nonprehensile(step):
    arm_joints = step['observation']['state'][:14:2] # First six joints
    is_gripper_open = tf.cast(False, tf.bool) # Gripper does not open
    gripper_joints = tf.cond(
        is_gripper_open,
        lambda: tf.constant([0.05, 0.05], dtype=tf.float32),
        lambda: tf.constant([0.0, 0.0], dtype=tf.float32),
    )
    image = step['observation']['image'] # Extract the image from the dataset
    return tf.concat([arm_joints, gripper_joints], axis = 0), image

def process_step_cmu_play_fusion(step):
    arm_joints = step['observation']['state'] # franka 
    gripper_dist = arm_joints[-1] 
    is_gripper_open = tf.cast(gripper_dist > 0.04, tf.bool)
    gripper_joints = tf.cond(
        is_gripper_open,
        lambda: tf.constant([0.05, 0.05], dtype=tf.float32),
        lambda: tf.constant([0.0, 0.0], dtype=tf.float32),
    )
    image = step['observation']['image'] # Extract the image from the dataset
    return tf.concat([arm_joints[:7], gripper_joints], axis = 0), image

def process_step_austin_buds(step):
    arm_joints = step['observation']['state'][:7] # franka 
    gripper_pos = step["observation"]["state"][7]
    is_gripper_open = tf.cast(gripper_pos > 0.04, tf.bool)
    gripper_joints = tf.cond(
        is_gripper_open,
        lambda: tf.constant([0.05, 0.05], dtype=tf.float32),
        lambda: tf.constant([0.0, 0.0], dtype=tf.float32),
    )
    image = step['observation']['image'] # Extract the image from the dataset
    return tf.concat([arm_joints[:7], gripper_joints], axis = 0), image

def process_step_austin_sailor(step):
    arm_joints = step['observation']['state_joint'] # franka 
    gripper_pos = step["observation"]["state"][-1]
    is_gripper_open = tf.cast(gripper_pos > 0.04, tf.bool)
    gripper_joints = tf.cond(
        is_gripper_open,
        lambda: tf.constant([0.05, 0.05], dtype=tf.float32),
        lambda: tf.constant([0.0, 0.0], dtype=tf.float32),
    )
    image = step['observation']['image'] # Extract the image from the dataset
    return tf.concat([arm_joints[:7], gripper_joints], axis = 0), image#, gripper_pos

def process_step_austin_mutex(step):
    arm_joints = step['observation']['state'][:7] # franka 
    gripper_pos = step["observation"]["state"][7:8]
    is_gripper_open = tf.cast(gripper_pos > 0.05, tf.bool)
    gripper_joints = tf.cond(
        is_gripper_open,
        lambda: tf.constant([0.05, 0.05], dtype=tf.float32),
        lambda: tf.constant([0.0, 0.0], dtype=tf.float32),
    )
    image = step['observation']['image'] # Extract the image from the dataset
    return tf.concat([arm_joints[:7], gripper_joints], axis = 0), image

def process_step_austin_sirius(step):
    arm_joints = step['observation']['state'][:7] # franka 
    gripper_pos = step["observation"]["state"][7:8]
    is_gripper_open = tf.cast(gripper_pos > 0.05, tf.bool)
    gripper_joints = tf.cond(
        is_gripper_open,
        lambda: tf.constant([0.05, 0.05], dtype=tf.float32),
        lambda: tf.constant([0.0, 0.0], dtype=tf.float32),
    )
    image = step['observation']['image'] # Extract the image from the dataset
    return tf.concat([arm_joints[:7], gripper_joints], axis = 0), image

def process_step_viola(step):
    arm_joints = step['observation']['joint_states']  # franka
    gripper_width = step['observation']['gripper_states'][0]  
    is_gripper_open = tf.cast(gripper_width > 0.04, tf.bool)
    gripper_joints = tf.cond(
        is_gripper_open,
        lambda: tf.constant([0.05, 0.05], dtype=tf.float32),
        lambda: tf.constant([0.0, 0.0], dtype=tf.float32),
    )
    full_joint_state = tf.concat([arm_joints, gripper_joints], axis=0)
    image = step['observation']['agentview_rgb']
    #agentview_rgb: a fixed camera showing the workspace (like a third-person view)
    #eye_in_hand_rgb: a camera mounted on the robot arm (first-person)
    return full_joint_state, image

def process_step_taco_play(step):
    robot_obs = step['observation']['robot_obs']  # franka
    joint_positions = robot_obs[7:14]  
    gripper_width = robot_obs[2]       
    is_gripper_open = tf.cast(gripper_width > 0.04, tf.bool)
    gripper_joints = tf.cond(
        is_gripper_open,
        lambda: tf.constant([0.05, 0.05], dtype=tf.float32),
        lambda: tf.constant([0.0, 0.0], dtype=tf.float32),
    )
    full_joint_state = tf.concat([joint_positions, gripper_joints], axis=0)
    image = step['observation']['rgb_static'] #swap rgb_static with rgb_gripper if want eye-in-hand camera
    return full_joint_state, image

def process_step_iamlab_cmu_pickup_insert(step):
    state = step['observation']['state']  
    arm_joints = state[:7]                
    gripper_width = state[-1]             
    is_gripper_open = tf.cast(gripper_width > 0.04, tf.bool)
    gripper_joints = tf.cond(
        is_gripper_open,
        lambda: tf.constant([0.05, 0.05], dtype=tf.float32),
        lambda: tf.constant([0.0, 0.0], dtype=tf.float32),
    )
    full_joint_state = tf.concat([arm_joints, gripper_joints], axis=0)
    image = step['observation']['image']  
    return full_joint_state, image

def process_step_bridge(step):
    arm_joints = step['observation']['state']
    image = step['observation']['image'] # Extract the image from the dataset

    return tf.concat([arm_joints, arm_joints[-1:]], axis = 0), image

def process_step_stanford_hydra(step):
    arm_joints = step['observation']['state'][10:17]
    image = step['observation']['image'] # Extract the image from the dataset

    return tf.concat([arm_joints, arm_joints[-1:]], axis = 0), image

def process_step_stanford_hydra(step):
    arm_joints = step['observation']['state']
    image = step['observation']['image'] # Extract the image from the dataset
    gripper_pos = step["observation"]["state"][-1]
    is_gripper_open = tf.cast(gripper_pos > 0.5, tf.bool)
    gripper_joints = tf.cond(
        is_gripper_open,
        lambda: tf.constant([0.05, 0.05], dtype=tf.float32),
        lambda: tf.constant([0.0, 0.0], dtype=tf.float32),
    )
    return tf.concat([arm_joints[:7], gripper_joints], axis = 0), image

def process_step_stanford_hydra(step):
    arm_joints = step['observation']['joint_pos']
    image = step['observation']['image_side_1'] # Extract the image from the dataset
    gripper_pos = step["observation"]["state_gripper_pose"]
    is_gripper_open = tf.cast(gripper_pos > 0.5, tf.bool)
    gripper_joints = tf.cond(
        is_gripper_open,
        lambda: tf.constant([0.05, 0.05], dtype=tf.float32),
        lambda: tf.constant([0.0, 0.0], dtype=tf.float32),
    )
    return tf.concat([arm_joints[:7], gripper_joints], axis = 0), image

def process_step_language_table(step):
    image = step['observation']['rgb'] # Extract the image from the dataset
    return None, image

def process_step_fractal(step):
    image = step['observation']['image'] # Extract the image from the dataset
    return None, image

def process_step_fmb(step):
    image = step['observation']['rgb'] # Extract the image from the dataset
    return None, image

dataset_to_processing_function = {
    'toto': process_step_toto,
    'nyu_franka_play_dataset_converted_externally_to_rlds': process_step_nyu,
    'berkeley_autolab_ur5': process_step_berkeley_ur5,
    'ucsd_kitchen_dataset_converted_externally_to_rlds': process_step_ucsd_kitchen,
    'utokyo_xarm_pick_and_place_converted_externally_to_rlds': process_step_utokyo_xarm_pick_place,
    'kaist_nonprehensile_converted_externally_to_rlds': process_step_kaist_nonprehensile,
    'asu_table_top_converted_externally_to_rlds': process_step_asu_table_top,
    'austin_buds_dataset_converted_externally_to_rlds': process_step_austin_buds,
    'utaustin_mutex': process_step_austin_mutex,
    'austin_sailor_dataset_converted_externally_to_rlds': process_step_austin_sailor,
    'bridge': process_step_bridge,
    'fractal20220817_data': process_step_fractal,
    'iamlab_cmu_pickup_insert_converted_externally_to_rlds': process_step_iamlab_cmu_pickup_insert,
    'viola': process_step_viola,
    'taco_play': process_step_taco_play,
    'language_table': process_step_language_table,
    'fmb': process_step_fmb,
}


import os, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"        # use CPU only
os.environ["TFDS_DISABLE_PROGRESSBAR"] = "1"     # concise logs

import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor

# ---------- Your dataset_to_processing_function must be defined before this ----------
# from processing_fns import dataset_to_processing_function  # example only

def main():
    # --------- Parse command line ----------
    parser = argparse.ArgumentParser(
        description="Load episodes from the Google Research Robotics datasets"
    )
    parser.add_argument("--dataset",  required=True,
                        help="e.g. language_table, austin_sailor_dataset, …")
    parser.add_argument("--start",    type=int,
                        help="episode start index (inclusive, 0‑based)")
    parser.add_argument("--end",      type=int,
                        help="episode end index   (inclusive, 0‑based)")
    parser.add_argument("--split",    type=str, default="train",
                        help="split name (default: train)")
    parser.add_argument("--save_dir", required=True,
                        help="root folder to save extracted frames")
    args = parser.parse_args()

    # --------- Parameter consistency check ----------
    if args.start is not None and args.start < 0:
        parser.error("--start must not be negative.")
    if args.end   is not None and args.end   < 0:
        parser.error("--end must not be negative.")
    if args.start is not None and args.end is not None and args.start > args.end:
        parser.error("--start cannot be greater than --end.")

    selecting_by_range = (args.start is not None or args.end is not None)

    # --------- Prepare TFDS builder ----------
    builder = tfds.builder_from_directory(
        f"gs://gresearch/robotics/{args.dataset}/0.1.0/"
    )

    # --------- Determine list of splits to process ----------
    if selecting_by_range:                           # process only specified split
        splits_to_process = [args.split]
    else:                                            # process all splits
        splits_to_process = list(builder.info.splits.keys())

    print(f"\n===== {args.dataset} =====")

    # --------- Iterate over each split ----------
    for split_name, split_info in builder.info.splits.items():
        if split_name not in splits_to_process:
            continue                                # skip splits not in target list

        # ------------- Stats and output -------------
        num_episodes_total = split_info.num_examples
        is_target_split    = (split_name == args.split)

        if selecting_by_range and is_target_split:
            if args.start is not None and args.end is not None:
                # start AND end
                start_idx, end_idx = args.start, args.end
                split_spec = f"{split_name}[{start_idx}:{end_idx + 1}]"
                num_episodes_plan = end_idx - start_idx + 1
                range_note        = f"[{start_idx}–{end_idx}]"
                ep_idx_offset     = start_idx
            elif args.start is not None:
                # start ONLY → to the end
                start_idx = args.start
                split_spec = f"{split_name}[{start_idx}:]"
                num_episodes_plan = num_episodes_total - start_idx
                range_note        = f"[{start_idx}–{num_episodes_total - 1}]"
                ep_idx_offset     = start_idx
            else:
                # end ONLY → from 0 to end
                end_idx = args.end
                split_spec = f"{split_name}[:{end_idx + 1}]"
                num_episodes_plan = end_idx + 1
                range_note        = f"[0–{end_idx}]"
                ep_idx_offset     = 0
        else:
            # full split
            split_spec        = split_name
            num_episodes_plan = num_episodes_total
            range_note        = "(all)"
            ep_idx_offset     = 0

        print(f"{split_name:<15}: {num_episodes_total:6,d} episodes "
            f"→ planned {num_episodes_plan:6,d} {range_note}")

        # --------- Read dataset ----------
        read_config = tfds.ReadConfig(
            interleave_cycle_length=1,     # read shards one by one
            shuffle_seed=None,             # no file shuffling
        )

        ds = builder.as_dataset(split=split_spec,
                                shuffle_files=False,
                                read_config=read_config)

        save_root = f"{args.save_dir}/{args.dataset}"
        os.makedirs(save_root, exist_ok=True)

        # --------- Multithreaded JPEG writing ----------
        needs_update = []
        with ThreadPoolExecutor(max_workers=os.cpu_count() * 2) as pool:
            for ep_idx, ep in enumerate(ds,    # ep_idx is already the sliced local index
                                      start=(args.start or 0)
                                      if selecting_by_range and split_name == args.split
                                      else 0):
                # ----------------------------------------------------------
                # ① Extract steps → apply custom processing function
                # ----------------------------------------------------------
                step_ds = (
                    ep["steps"]
                    .map(dataset_to_processing_function[args.dataset],
                         num_parallel_calls=tf.data.AUTOTUNE)
                    .prefetch(tf.data.AUTOTUNE)
                )

                # ② Convert to numpy
                episode_states, episode_imgs = [], []
                for s, img in tfds.as_numpy(step_ds):
                    episode_states.append(s)
                    episode_imgs.append(img)

                # ③ Create directory and write frames in parallel
                frames_dir = f"{save_root}/{split_name}/{ep_idx}/frames"
                frame_paths = sorted(Path(frames_dir).glob("*.jpg"), key=numeric_key)

                # ---------- Check if disk write can be skipped ----------
                tol_mean   = 3.0                 # mean difference < 3 considered identical
                n_compare  = 5 if len(episode_imgs) >= 10 else len(episode_imgs)
                skip_write = False

                def _load_rgb(idx: int) -> np.ndarray:
                    p = frame_paths[idx]
                    return cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)

                def _mean_error(a: np.ndarray, b: np.ndarray) -> float:
                    return np.abs(a.astype(np.int16) - b.astype(np.int16)).mean()

                if len(frame_paths) == len(episode_imgs):   # only compare pixels if lengths match
                    mean_errors = [
                        _mean_error(_load_rgb(i), episode_imgs[i])
                        for i in range(n_compare)
                    ] + [
                        _mean_error(_load_rgb(-(i + 1)),
                                    episode_imgs[-(i + 1)])
                        for i in range(n_compare)
                    ]
                    skip_write = all(err < tol_mean for err in mean_errors)

                if skip_write:
                    print(f"{split_name} ep {ep_idx}: already up‑to‑date (Δ < {tol_mean})")
                    continue

                # ---------- Write to disk ----------
                # clear old frames first
                for old_jpg in frame_paths:
                    old_jpg.unlink()
                Path(frames_dir).mkdir(parents=True, exist_ok=True)

                print(f"{split_name} ep {ep_idx}: saving {len(episode_imgs)} frames")
                needs_update.append(ep_idx)

                def _write_one(k_img):
                    k, img_arr = k_img
                    cv2.imwrite(f"{frames_dir}/{k}.jpg",
                                cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR))

                pool.map(_write_one, enumerate(episode_imgs))

                print(f"{split_name} ep {ep_idx}: saved {len(episode_imgs)} frames")

        # ---------- Write needs_update.txt ----------
        list_path = Path(save_root) / split_name / "needs_update.txt"
        list_path.parent.mkdir(parents=True, exist_ok=True)
        list_path.write_text("\n".join(map(str, needs_update)))
        print(f"📝 {len(needs_update)} episode(s) needing update → {list_path}")

    print("✅ All requested episodes saved successfully!")

if __name__ == "__main__":
    main()


'''
CUDA_VISIBLE_DEVICES="" python /home/guanhuaji/oxeplusplus/sam2/load_dataset.py --dataset toto --start 0 --end 901 --save_dir /home/guanhuaji/load_datasets

python /home/guanhuaji/oxeplusplus/sam2/load_dataset.py --dataset berkeley_autolab_ur5 --start 0 --end 895 --save_dir /home/guanhuaji/load_datasets
python /home/guanhuaji/oxeplusplus/sam2/load_dataset.py --dataset austin_buds_dataset_converted_externally_to_rlds --start 0 --end 134 --save_dir /home/guanhuaji/load_datasets
python /home/guanhuaji/oxeplusplus/sam2/load_dataset.py --dataset utokyo_xarm_pick_and_place_converted_externally_to_rlds --start 0 --end 134 --save_dir /home/guanhuaji/load_datasets
python /home/guanhuaji/oxeplusplus/sam2/load_dataset.py --dataset utaustin_mutex --start 0 --end 100 --save_dir /home/guanhuaji/load_datasets
python /home/guanhuaji/oxeplusplus/sam2/load_dataset.py --dataset austin_sailor_dataset_converted_externally_to_rlds --save_dir /home/guanhuaji/load_datasets

python /home/guanhuaji/oxeplusplus/sam2/load_dataset.py --dataset language_table --start 18 --end 18 --save_dir /home/guanhuaji/load_datasets

python /home/guanhuaji/oxeplusplus/sam2/load_dataset.py --dataset toto --start 0 --end 100 --save_dir /home/guanhuaji/load_datasets

python /home/guanhuaji/oxeplusplus/sam2/load_dataset.py --dataset viola --start 0 --end 100 --save_dir /home/guanhuaji/load_datasets
python /home/guanhuaji/oxeplusplus/sam2/load_dataset.py --dataset nyu_franka_play_dataset_converted_externally_to_rlds --start 0 --end 100 --save_dir /home/guanhuaji/load_datasets

python /home/guanhuaji/oxeplusplus/sam2/load_dataset.py --dataset ucsd_kitchen_dataset_converted_externally_to_rlds --start 0 --end 100 --save_dir /home/guanhuaji/load_datasets

python /home/guanhuaji/oxeplusplus/sam2/load_dataset.py --dataset fmb --start 0 --end 100 --save_dir /home/guanhuaji/load_datasets

python /home/guanhuaji/oxeplusplus/sam2/load_dataset.py --dataset taco_play --start 0 --end 100 --save_dir /home/guanhuaji/load_datasets

python /home/guanhuaji/oxeplusplus/sam2/load_dataset.py --dataset iamlab_cmu_pickup_insert_converted_externally_to_rlds --start 0 --end 100 --save_dir /home/guanhuaji/load_datasets


'''
'''
python /home/guanhuaji/oxeplusplus/sam2/load_dataset.py --dataset austin_sailor_dataset_converted_externally_to_rlds --save_dir /home/guanhuaji/load_datasets --start 0 --end 100
python /home/guanhuaji/oxeplusplus/sam2/load_dataset.py --dataset austin_buds_dataset_converted_externally_to_rlds --start 0 --end 134 --save_dir /home/guanhuaji/load_datasets
python /home/guanhuaji/oxeplusplus/sam2/load_dataset.py --dataset toto --start 0 --end 134 --save_dir /home/guanhuaji/load_datasets
python /home/guanhuaji/oxeplusplus/sam2/load_dataset.py --dataset berkeley_autolab_ur5 --start 0 --end 134 --save_dir /home/guanhuaji/load_datasets
python /home/guanhuaji/oxeplusplus/sam2/load_dataset.py --dataset utaustin_mutex --start 0 --end 134 --save_dir /home/guanhuaji/load_datasets
python /home/guanhuaji/oxeplusplus/sam2/load_dataset.py --dataset taco_play --start 0 --end 134 --save_dir /home/guanhuaji/load_datasets
python /home/guanhuaji/oxeplusplus/sam2/load_dataset.py --dataset language_table --start 0 --end 134 --save_dir /home/guanhuaji/load_datasets
python /home/guanhuaji/oxeplusplus/sam2/load_dataset.py --dataset viola --start 0 --end 134 --save_dir /home/guanhuaji/load_datasets
python /home/guanhuaji/oxeplusplus/sam2/load_dataset.py --dataset iamlab_cmu_pickup_insert_converted_externally_to_rlds --save_dir /home/guanhuaji/load_datasets
python /home/guanhuaji/oxeplusplus/sam2/load_dataset.py --dataset utokyo_xarm_pick_and_place_converted_externally_to_rlds --save_dir /home/guanhuaji/load_datasets
python /home/guanhuaji/oxeplusplus/sam2/load_dataset.py --dataset ucsd_kitchen_dataset_converted_externally_to_rlds --save_dir /home/guanhuaji/load_datasets
python /home/guanhuaji/oxeplusplus/sam2/load_dataset.py --dataset kaist_nonprehensile_converted_externally_to_rlds --save_dir /home/guanhuaji/load_datasets
python /home/guanhuaji/oxeplusplus/sam2/load_dataset.py --dataset bridge --save_dir /home/guanhuaji/load_datasets
python /home/guanhuaji/oxeplusplus/sam2/load_dataset.py --dataset fractal20220817_data --save_dir /home/abrashid/OXE_inpainting

'''