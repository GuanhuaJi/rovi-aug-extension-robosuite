
'''
    python /home/guanhuaji/mirage/robot2robot/rendering/generate_target_robot_displacements.py \
           --robot_dataset "autolab_ur5" \
           --target_robot  "Sawyer" \
           --partition     "0"

'''

import csv
import argparse
import json
import time
import os
import cv2
import socket, pickle, struct
import numpy as np
import matplotlib.pyplot as plt
import robosuite as suite
import robosuite.macros as macros
macros.IMAGE_CONVENTION = "opencv"
from export_source_robot_states import RobotCameraWrapper, change_brightness
from robot_pose_dict import ROBOT_POSE_DICT
from tqdm import tqdm
from transforms3d.quaternions import quat2mat
import json
from pathlib import Path
import pynvml
from contextlib import contextmanager
import json, os, tempfile, portalocker, pathlib

@contextmanager
def locked_json(path: pathlib.Path, mode="r+", default=lambda: {}):
    # 1️⃣ acquire an **exclusive lock**
    with portalocker.Lock(str(path), mode, timeout=30) as fp:    # ← blocks here
        try:
            data = json.load(fp)
        except json.JSONDecodeError:
            data = default()
        yield data                       # 🔒  work with the dict while locked
        fp.seek(0), fp.truncate()        # 2️⃣ rewind
        json.dump(data, fp, indent=2)    # 3️⃣ write
        fp.flush(), os.fsync(fp.fileno())  # 4️⃣ durability
    # ➡ lock is released automatically

def atomic_write_json(obj, path: pathlib.Path):
    with tempfile.NamedTemporaryFile(
            dir=path.parent, delete=False, mode="w") as tmp:
        json.dump(obj, tmp, indent=2)
        tmp.flush()
        os.fsync(tmp.fileno())           # make sure it’s on disk
    os.replace(tmp.name, path) 

def pick_best_gpu(policy="free-mem"):
    """
    Return the index of the “least busy” NVIDIA GPU and set CUDA_VISIBLE_DEVICES
    so frameworks (PyTorch, TensorFlow, JAX…) will automatically use it.

    policy
    ------
    "free-mem"   – prefer the card with the most free memory
    "low-util"   – prefer the card with the lowest compute utilisation
    "hybrid"     – most free mem, break ties with lowest utilisation
    """
    pynvml.nvmlInit()
    n = pynvml.nvmlDeviceGetCount()

    best_idx, best_score = None, None
    for i in range(n):
        h = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)          # bytes
        util = pynvml.nvmlDeviceGetUtilizationRates(h)   # %
        if policy == "free-mem":
            score = mem.free
        elif policy == "low-util":
            score = -util.gpu                            # negative ⇒ lower is better
        else:  # hybrid
            score = (mem.free, -util.gpu)                # tuple is fine for max()

        if best_score is None or score > best_score:
            best_idx, best_score = i, score

    os.environ["CUDA_VISIBLE_DEVICES"] = str(best_idx)   # frameworks see *only* this GPU
    print(f"👉  Selected GPU {best_idx}")
    return best_idx

def find_offset(displacement_csv_path, episode):
    with open(displacement_csv_path, "r") as f:
        reader = csv.reader(f)
        offset = np.zeros(3)
        offset = np.asarray(offset, dtype=np.float64)
        error = np.zeros(3)
        error = np.asarray(error, dtype=np.float64)
        for row in reader:
            if row[1] == str(episode):
                prev_offset = np.array([float(row[2]), float(row[3]), float(row[4])])
                error = np.array([float(row[5]), float(row[6]), float(row[7])])
                step = error / 3
                step = np.clip(step, -0.1, 0.1)
                offset = prev_offset + step
            if int(row[1]) > episode:
                return offset + step
        return np.asarray(offset, dtype=np.float64)

class TargetEnvWrapper:
    def __init__(self, target_name, target_gripper, robot_dataset, camera_height=256, camera_width=256):
        self.target_env = RobotCameraWrapper(robotname=target_name, grippername=target_gripper, robot_dataset=robot_dataset, camera_height=camera_height, camera_width=camera_width)
        self.target_name = target_name
    
    
    def generate_image(self, 
                        save_paired_images_folder_path="paired_images", 
                        displacement_csv_path=None,
                        source_robot_states_path="paired_images",
                        reference_joint_angles_path=None, 
                        reference_ee_states_path=None, 
                        robot_dataset=None, 
                        episode=0, 
                        camera_height=256, 
                        camera_width=256):
        data = np.load(os.path.join(source_robot_states_path, "source_robot_states", f"{episode}.npz"), allow_pickle=True)
        gripper_count = 0
        target_pose_array = data['pos']
        gripper_array = data['grip']
        num_robot_poses = target_pose_array.shape[0]
        success = True

        if ROBOT_POSE_DICT[robot_dataset][self.target_name]['safe_angle'] is not None:
            self.target_env.set_robot_joint_positions(ROBOT_POSE_DICT[robot_dataset][self.target_name]['safe_angle'])
        
        success = False
        while success == False:
            self.target_env.env.reset()
            offset = find_offset(displacement_csv_path, episode)
            target_reached_pose, target_reached, target_reached_error = np.zeros(3, dtype=np.float64), None, np.zeros(3, dtype=np.float64)
            for pose_index in tqdm(range(num_robot_poses), desc=f'{self.target_name} Pose Generation'):
            #for pose_index in range(num_robot_poses):
                target_pose=target_pose_array[pose_index]
                target_pose[:3] -= offset
                self.target_env.open_close_gripper(gripper_open=gripper_array[pose_index])
                target_reached, target_reached_pose = self.target_env.drive_robot_to_target_pose(target_pose=target_pose)

                if not target_reached:
                    target_reached_error = target_pose[:3] - target_reached_pose[:3]
                    break

            with open(displacement_csv_path, "a") as f:
                writer = csv.writer(f)
                if target_reached:
                    print(f"Target pose reached: {target_pose}, target_reached_error: {target_reached_error}")
                    writer.writerow([self.target_name, episode, target_pose[0], target_pose[1], target_pose[2], target_reached_error[0], target_reached_error[1], target_reached_error[2], "success"])
                    success = True
                else:
                    writer.writerow([self.target_name, episode, target_pose[0], target_pose[1], target_pose[2], target_reached_error[0], target_reached_error[1], target_reached_error[2], "failed"])




        
        

if __name__ == "__main__":

    """
    Registered environments: Lift, Stack, NutAssembly, NutAssemblySingle, NutAssemblySquare, NutAssemblyRound,
                             PickPlace, PickPlaceSingle, PickPlaceMilk, PickPlaceBread, PickPlaceCereal,
                             PickPlaceCan, Door, Wipe, TwoArmLift, TwoArmPegInHole, TwoArmHandover

    Possible robots: Baxter, IIWA, Jaco, Kinova3, Panda, Sawyer, UR5e
    """

    # print welcome info
    # print("Welcome to robosuite v{}!".format(suite.__version__))
    # print(suite.__logo__)

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="(optional) (optional) set seed")
    parser.add_argument("--target_gripper", type=str, default="Robotiq85Gripper", help="PandaGripper or Robotiq85Gripper")
    parser.add_argument("--num_robot_poses", type=int, default=5, help="(optional) (optional) set seed")
    parser.add_argument("--target_robot", type=str, default="IIWA", help="(optional) (optional) set seed")
    parser.add_argument("--save_paired_images_folder_path", type=str, default="paired_images", help="(optional) folder path to save the paired images")
    parser.add_argument("--robot_dataset", type=str, help="(optional) to match the robot poses from a dataset, provide the dataset name")
    parser.add_argument("--reference_joint_angles_path", type=str, help="(optional) to match the robot poses from a dataset, provide the path to the joint angles file (np.savetxt)")
    parser.add_argument("--reference_ee_states_path", type=str, help="(optional) to match the robot poses from a dataset, provide the path to the ee state file (np.savetxt)")
    parser.add_argument("--blacklist", type=bool, default=False, help="If set, prints extra debug/warning information")
    parser.add_argument("--partition", type=int, default=0, help="If set, prints extra debug/warning information")
    args = parser.parse_args()
    
    
    target_name = args.target_robot

    if target_name in ["Sawyer", "Jaco", "IIWA", "UR5e", "Kinova3", "Panda"]:
        robotlist = [target_name]
    if target_name == "all":
        robotlist = ["Sawyer", "Jaco", "IIWA", "UR5e", "Kinova3", "Panda"]
    if target_name == "1":
        robotlist = ["Sawyer", "Jaco", "IIWA"]
    if target_name == "2": 
        robotlist = ["UR5e", "Kinova3", "Panda"]

    for target_name in robotlist:

        if target_name == "Sawyer":
            target_gripper = "RethinkGripper"
        elif target_name == "Jaco":
            target_gripper = "JacoThreeFingerGripper"
        elif target_name == "IIWA":
            target_gripper = "Robotiq85Gripper"
        elif target_name == "UR5e":
            target_gripper = "Robotiq85Gripper"
        elif target_name == "Kinova3":
            target_gripper = "Robotiq85Gripper"
        elif target_name == "Panda":
            target_gripper = "PandaGripper"

        save_paired_images_folder_path = os.path.join("/home/guanhuaji/mirage/robot2robot/rendering/paired_images", args.robot_dataset)
        source_robot_states_path = os.path.join("/home/guanhuaji/mirage/robot2robot/rendering/paired_images", args.robot_dataset)
        
        if args.robot_dataset is not None:
            from dataset_poses_dict import ROBOT_CAMERA_POSES_DICT
            robot_dataset_info = ROBOT_CAMERA_POSES_DICT[args.robot_dataset]
            camera_height = robot_dataset_info["camera_heights"]
            camera_width = robot_dataset_info["camera_widths"]
        else:
            camera_height = 256
            camera_width = 256

        pick_best_gpu()
        NUM_PARTITIONS = 20

        episodes = []
        if args.blacklist:
            blacklist_path = Path(f"{save_paired_images_folder_path}/{target_name}/blacklist.json")
            with locked_json(blacklist_path) as blk:                      # 🔒 进入临界区
                robot_list = blk.get(target_name, [])
                if robot_list:
                    episodes = robot_list
                    print(f"\033[93m[BLACKLIST] Found {target_name} – episode {episodes}\033[0m")
                else:
                    print(f"\033[92m[BLACKLIST] No blacklisted episodes for {target_name}\033[0m")
        else:
            num_episode = ROBOT_CAMERA_POSES_DICT[args.robot_dataset]['num_episodes']
            episodes = range(num_episode * args.partition // NUM_PARTITIONS, num_episode * (args.partition + 1) // NUM_PARTITIONS)


        # Create displacement.csv if it does not exist
        csv_path = os.path.join(save_paired_images_folder_path, target_name, "displacement.csv")
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        if not os.path.isfile(csv_path):
            with open(csv_path, "w", newline="") as f:
                pass

        whitelist_path = Path(f"{save_paired_images_folder_path}/{target_name}/whitelist.json")
        if not whitelist_path.exists():
            whitelist_path.parent.mkdir(parents=True, exist_ok=True)   # create any missing folders
            whitelist_path.write_text(json.dumps({}))
        blacklist_path = Path(f"{save_paired_images_folder_path}/{target_name}/blacklist.json")
        if not blacklist_path.exists():
            blacklist_path.parent.mkdir(parents=True, exist_ok=True)
            blacklist_path.write_text(json.dumps({}))

        for episode in episodes:
            # if episode in whitelist, then skip
            whitelist_path = Path(f"{save_paired_images_folder_path}/{target_name}/whitelist.json")
            with locked_json(whitelist_path) as wl:
                robot_list = wl.get(target_name, [])
                if robot_list and episode in robot_list:
                    print(f"[WHITELIST] Skipping {target_name} – episode {episode}")
                    continue
            print(f"[INFO] Processing {target_name} – episode {episode}")
            target_env = TargetEnvWrapper(target_name, target_gripper, args.robot_dataset, camera_height, camera_width)
            env = target_env.target_env.env

            target_env.generate_image(
                save_paired_images_folder_path=save_paired_images_folder_path, 
                displacement_csv_path=csv_path,
                source_robot_states_path=source_robot_states_path,
                reference_joint_angles_path=args.reference_joint_angles_path, 
                reference_ee_states_path=args.reference_ee_states_path, 
                robot_dataset=args.robot_dataset, 
                episode=episode,
                camera_height=camera_height,
                camera_width=camera_width
            )
            target_env.target_env.env.close_renderer()