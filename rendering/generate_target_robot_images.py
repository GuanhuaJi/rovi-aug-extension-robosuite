'''
python /home/jiguanhua/mirage/robot2robot/rendering/generate_pairs.py --robot_dataset "stack" --target_robot "Sawyer"
python /home/jiguanhua/mirage/robot2robot/rendering/generate_pairs.py --robot_dataset "stack" --target_robot "Panda"
python /home/jiguanhua/mirage/robot2robot/rendering/generate_pairs.py --robot_dataset "autolab_ur5" --target_robot "Kinova3"
python /home/jiguanhua/mirage/robot2robot/rendering/generate_pairs.py --robot_dataset "kaist" --target_robot "Panda"
python /home/jiguanhua/mirage/robot2robot/rendering/generate_pairs.py --robot_dataset "toto" --target_robot "Panda"


python /home/jiguanhua/mirage/robot2robot/rendering/generate_target_robot_images.py --robot_dataset "autolab_ur5" --target_robot "1"
python /home/jiguanhua/mirage/robot2robot/rendering/generate_target_robot_images.py --robot_dataset "autolab_ur5" --target_robot "2"
python /home/jiguanhua/mirage/robot2robot/rendering/generate_target_robot_images.py --robot_dataset "austin_buds" --target_robot "Kinova3"
python /home/jiguanhua/mirage/robot2robot/rendering/generate_target_robot_images.py --robot_dataset "austin_sailor" --target_robot "Kinova3"
python /home/jiguanhua/mirage/robot2robot/rendering/generate_target_robot_images.py --robot_dataset "iamlab_cmu" --target_robot "2"
python /home/jiguanhua/mirage/robot2robot/rendering/generate_target_robot_images.py --robot_dataset "austin_mutex" --target_robot "2"
python /home/jiguanhua/mirage/robot2robot/rendering/generate_target_robot_images.py --robot_dataset "austin_sailor" --target_robot "Panda"
python /home/jiguanhua/mirage/robot2robot/rendering/generate_target_robot_images.py --robot_dataset "austin_sailor" --target_robot "Sawyer"
python /home/jiguanhua/mirage/robot2robot/rendering/generate_target_robot_images.py --robot_dataset "austin_sailor" --target_robot "Kinova3"
python /home/jiguanhua/mirage/robot2robot/rendering/generate_target_robot_images.py --robot_dataset "austin_sailor" --target_robot "UR5e"
python /home/jiguanhua/mirage/robot2robot/rendering/generate_target_robot_images.py --robot_dataset "austin_sailor" --target_robot "Jaco"
python /home/jiguanhua/mirage/robot2robot/rendering/generate_target_robot_images.py --robot_dataset "austin_sailor" --target_robot "IIWA"

python /home/jiguanhua/mirage/robot2robot/rendering/generate_target_robot_images.py --robot_dataset "autolab_ur5" --target_robot "Panda"
python /home/jiguanhua/mirage/robot2robot/rendering/generate_target_robot_images.py --robot_dataset "autolab_ur5" --target_robot "Sawyer"
python /home/jiguanhua/mirage/robot2robot/rendering/generate_target_robot_images.py --robot_dataset "autolab_ur5" --target_robot "Kinova3"
python /home/jiguanhua/mirage/robot2robot/rendering/generate_target_robot_images.py --robot_dataset "autolab_ur5" --target_robot "UR5e"
python /home/jiguanhua/mirage/robot2robot/rendering/generate_target_robot_images.py --robot_dataset "autolab_ur5" --target_robot "Jaco"
python /home/jiguanhua/mirage/robot2robot/rendering/generate_target_robot_images.py --robot_dataset "autolab_ur5" --target_robot "IIWA"

conda activate mirage

python /home/guanhuaji/mirage/robot2robot/rendering/generate_target_robot_images.py --robot_dataset "furniture_bench" --target_robot "Panda"
python /home/guanhuaji/mirage/robot2robot/rendering/generate_target_robot_images.py --robot_dataset "furniture_bench" --target_robot "Sawyer"
python /home/guanhuaji/mirage/robot2robot/rendering/generate_target_robot_images.py --robot_dataset "furniture_bench" --target_robot "Jaco"
python /home/guanhuaji/mirage/robot2robot/rendering/generate_target_robot_images.py --robot_dataset "furniture_bench" --target_robot "Kinova3"
python /home/guanhuaji/mirage/robot2robot/rendering/generate_target_robot_images.py --robot_dataset "furniture_bench" --target_robot "IIWA"
python /home/guanhuaji/mirage/robot2robot/rendering/generate_target_robot_images.py --robot_dataset "furniture_bench" --target_robot "UR5e"

python /home/guanhuaji/mirage/robot2robot/rendering/generate_target_robot_images.py --robot_dataset "taco_play" --target_robot "Panda"


python /home/jiguanhua/mirage/robot2robot/rendering/generate_target_robot_images.py --robot_dataset "three_piece_assembly" --target_robot "Jaco" --blacklist True
datasets: 
austin_buds, austin_mutex, austin_sailor, 
autolab_ur5, can, furniture_bench, iamlab_cmu, 
lift, nyu_franka, square, stack, three_piece_assembly, 
taco_play, ucsd_kitchen_rlds, viola
'''



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

GRIPPER_OPEN = {
    "Panda": [10, -10],
    "IIWA": [10, -10],
    "Sawyer": [10, -10],
    "Jaco": [10, -10],
    "UR5e": [5, -5],
    "Kinova3": [5, -5],
}

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

# def load_list(list_path) -> dict:
#     if list_path.exists() and list_path.stat().st_size > 0:
#         with list_path.open("r", encoding="utf-8") as f:
#             return json.load(f)
#     return {}                      # start empty the first time

# def save_list(list_path, l: dict) -> None:
#     with list_path.open("w", encoding="utf-8") as f:
#         json.dump(l, f, indent=2)

def load_list(path):
    with locked_json(path, "a+") as data:
        return data.copy()

def save_list(path, data):
    atomic_write_json(data, pathlib.Path(path))

def offset_in_quaternion_direction_batch(positions, quaternions, offset_dist=0.05, local_direction=None):
    """
    批量处理:
      - positions: shape = (N, 3) 的 numpy 数组，代表 N 个末端执行器的位置
      - quaternions: shape = (N, 4) 或 (4,) 的 numpy 数组，代表与每个位置对应的四元数 (w, x, y, z)
         * 如果 quaternions 的 shape = (4,) 表示所有位置都用同一个四元数
      - offset_dist: 沿着 local_direction 在世界坐标系中偏移的距离 (默认 0.05)
      - local_direction: 末端执行器在自身坐标系下的“前向”向量 (默认 [0, 0, -1])
    
    返回：
      - new_positions: shape = (N, 3) 的 numpy 数组，偏移后的世界坐标位置
    """
    # 如果用户没有指定末端执行器在自身坐标系中的“指向”，默认用 -Z
    if local_direction is None:
        local_direction = np.array([0, 0, -1], dtype=float)
    else:
        local_direction = np.array(local_direction, dtype=float)

    # 确保 positions 和 quaternions 都是 numpy 数组
    positions = np.array(positions, dtype=float)
    quaternions = np.array(quaternions, dtype=float)

    # 如果只有一个四元数，则对所有 positions 都使用这个四元数
    if quaternions.ndim == 1:  
        # shape = (4,)
        R = quat2mat(quaternions)  # 1 个旋转矩阵
        world_dir = R.dot(local_direction)  # 在世界坐标系中的方向
        new_positions = positions + offset_dist * world_dir
        return new_positions
    else:
        # shape = (N, 4) -> 每个 position 对应一个 quaternion
        new_positions = []
        for pos, quat in zip(positions, quaternions):
            R = quat2mat(quat)
            world_dir = R.dot(local_direction)
            new_pos = pos + offset_dist * world_dir
            new_positions.append(new_pos)
        return np.vstack(new_positions)


class TargetEnvWrapper:
    def __init__(self, target_name, target_gripper, robot_dataset, camera_height=256, camera_width=256):
        self.target_env = RobotCameraWrapper(robotname=target_name, grippername=target_gripper, robot_dataset=robot_dataset, camera_height=camera_height, camera_width=camera_width)
        self.target_name = target_name
    
    
    def generate_image(self, save_paired_images_folder_path="paired_images", reference_joint_angles_path=None, reference_ee_states_path=None, robot_dataset=None, episode=0, camera_height=256, camera_width=256):
        data = np.load(os.path.join(save_paired_images_folder_path, "source_robot_states", f"{episode}.npz"), allow_pickle=True)
        camera_pose = data['camera']
        camera_pose[:3] -= ROBOT_POSE_DICT[robot_dataset][self.target_name]['displacement']
        self.target_env.camera_wrapper.set_camera_pose(pos=camera_pose[:3], quat=camera_pose[3:])
        if "fov" in data:
            fov = data["fov"]
            self.target_env.camera_wrapper.set_camera_fov(fov)
        self.target_env.update_camera()

        os.makedirs(os.path.join(save_paired_images_folder_path, "{}_rgb".format(target_name), str(episode)), exist_ok=True)
        os.makedirs(os.path.join(save_paired_images_folder_path, "{}_mask".format(target_name), str(episode)), exist_ok=True)
        gripper_count = 0
        npz_path = os.path.join(save_paired_images_folder_path, "source_robot_states", f"{episode}.npz")
        data = np.load(npz_path, allow_pickle=True)
        target_pose_array = data['pos']
        gripper_array = data['grip']
        num_robot_poses = target_pose_array.shape[0]
        target_pose_list = []
        joint_angles_list = []
        gripper_width_list = []
        success = True

        if ROBOT_POSE_DICT[robot_dataset][self.target_name]['safe_angle'] is not None:
            self.target_env.set_robot_joint_positions(ROBOT_POSE_DICT[robot_dataset][self.target_name]['safe_angle'])
        
        for pose_index in range(num_robot_poses):
        #for pose_index in tqdm(range(num_robot_poses), desc=f'{self.target_name} Pose Generation'):
            target_pose=target_pose_array[pose_index]
            target_pose[:3] -= ROBOT_POSE_DICT[robot_dataset][self.target_name]['displacement']
            self.target_env.open_close_gripper(gripper_open=gripper_array[pose_index])
            target_reached, target_reached_pose = self.target_env.drive_robot_to_target_pose(target_pose=target_pose)
            if not target_reached:
                blacklist_path = Path(save_paired_images_folder_path, "blacklist.json")
                with locked_json(blacklist_path) as blk:          # 🔒 独占锁
                    # 取得/初始化当前机器人的列表
                    robot_list = blk.setdefault(self.target_name, [])
                    if episode not in robot_list:                 # 只有首次才追加
                        robot_list.append(episode)
                        robot_list.sort()                         # 保持升序，方便人工查看
                        # 写回由 locked_json 完成；此时其它进程仍在阻塞中
                        RED   = "\033[91m"   # bright red
                        RESET = "\033[0m"
                        print(f"{RED}[BLACKLIST] Added {self.target_name} – episode {episode}{RESET}")
                success = False
                break
            reached_pose = self.target_env.compute_eef_pose()
            reached_pose[:3] += ROBOT_POSE_DICT[robot_dataset][self.target_name]['displacement']
            target_pose_list.append(reached_pose)
            gripper_width_list.append(self.target_env.get_gripper_width_from_qpos())
            #print("TARGET_REACHED_POSE:", ppose)
            
            joint_indices = self.target_env.env.robots[0]._ref_joint_pos_indexes
            joint_angles = self.target_env.env.sim.data.qpos[joint_indices]
            joint_angles_list.append(joint_angles)


            target_robot_img, target_robot_seg_img = self.target_env.get_observation_fast(white_background=True, width=camera_width, height=camera_height)
            cv2.imwrite(os.path.join(save_paired_images_folder_path, f"{target_name}_rgb", f"{episode}/{pose_index}.jpg"), cv2.cvtColor(target_robot_img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(save_paired_images_folder_path, f"{target_name}_mask", f"{episode}/{pose_index}.jpg"), target_robot_seg_img * 255)
        
        if success:
            # remove the episode from the blacklist if it exists
            # === 1) 从黑名单里移除 episode =========================================
            blacklist_path = Path(save_paired_images_folder_path, "blacklist.json")
            with locked_json(blacklist_path) as blk:            # 🔒 独占锁
                robot_list = blk.get(self.target_name, [])
                if episode in robot_list:                       # ← 安全修改
                    robot_list.remove(episode)
                    if robot_list:
                        blk[self.target_name] = robot_list
                    else:
                        blk.pop(self.target_name)
                    print(f"\033[92m[BLACKLIST] Removed {self.target_name} – episode {episode}\033[0m")
            # 离开 with 时 locked_json 会写回并释放锁

            # === 2) 加入白名单（若尚未存在） =======================================
            whitelist_path = Path(save_paired_images_folder_path, "whitelist.json")
            with locked_json(whitelist_path) as wl:             # 🔒 独占锁
                robot_list = wl.get(self.target_name, [])
                if episode not in robot_list:
                    robot_list.append(episode)
                    robot_list.sort()
                    wl[self.target_name] = robot_list
                    print(f"\033[92m[WHITELIST] Added {self.target_name} – episode {episode}\033[0m")
            # 同样在退出 with 时安全写回

            target_pose_array = np.vstack(target_pose_list)
            joint_angles_array = np.vstack(joint_angles_list)
            gripper_width_array = np.array(gripper_width_list)
            eef_npy_path = os.path.join(save_paired_images_folder_path, "source_robot_states", f"{self.target_name}", "end_effector", f"{episode}.npy")
            np.save(eef_npy_path, target_pose_array)
            gripper_npy_path = os.path.join(save_paired_images_folder_path, "source_robot_states", f"{self.target_name}", "gripper_distance", f"{episode}.npy")
            np.save(gripper_npy_path, gripper_width_array)
            joint_angles_npy_path = os.path.join(save_paired_images_folder_path, "source_robot_states", f"{self.target_name}", "joint_angles", f"{episode}.npy")
            np.save(joint_angles_npy_path, joint_angles_array)



        
        

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

        # Save the captured images
        save_paired_images_folder_path = os.path.join("/home/guanhuaji/mirage/robot2robot/rendering/paired_images", args.robot_dataset)
        
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
            blacklist_path = Path(os.path.join(save_paired_images_folder_path, "blacklist.json"))
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


            '''
            python /home/guanhuaji/mirage/robot2robot/rendering/generate_target_robot_images.py --robot_dataset "furniture_bench" --target_robot "Panda" --partition 0 &
            python /home/guanhuaji/mirage/robot2robot/rendering/generate_target_robot_images.py --robot_dataset "furniture_bench" --target_robot "Sawyer" --partition 0 &
            python /home/guanhuaji/mirage/robot2robot/rendering/generate_target_robot_images.py --robot_dataset "furniture_bench" --target_robot "Jaco" --partition 0 &
            python /home/guanhuaji/mirage/robot2robot/rendering/generate_target_robot_images.py --robot_dataset "furniture_bench" --target_robot "Kinova3" --partition 0 &
            python /home/guanhuaji/mirage/robot2robot/rendering/generate_target_robot_images.py --robot_dataset "furniture_bench" --target_robot "IIWA" --partition 0 &
            python /home/guanhuaji/mirage/robot2robot/rendering/generate_target_robot_images.py --robot_dataset "furniture_bench" --target_robot "UR5e" --partition 0
            '''

        os.makedirs(os.path.join(save_paired_images_folder_path, "source_robot_states", f"{target_name}", "end_effector"), exist_ok=True) 
        os.makedirs(os.path.join(save_paired_images_folder_path, "source_robot_states", f"{target_name}", "gripper_distance"), exist_ok=True)
        os.makedirs(os.path.join(save_paired_images_folder_path, "source_robot_states", f"{target_name}", "joint_angles"), exist_ok=True)
        for episode in episodes:
            # if episode in whitelist, then skip
            whitelist_path = Path(os.path.join(save_paired_images_folder_path, "whitelist.json"))
            with locked_json(whitelist_path) as wl:
                robot_list = wl.get(target_name, [])
                if robot_list and episode in robot_list:
                    print(f"\033[92m[WHITELIST] Skipping {target_name} – episode {episode}\033[0m")
                    continue
            print(f"\033[92m[INFO] Processing {target_name} – episode {episode}\033[0m")
            target_env = TargetEnvWrapper(target_name, target_gripper, args.robot_dataset, camera_height, camera_width)
            env = target_env.target_env.env

            target_env.generate_image(
                save_paired_images_folder_path=save_paired_images_folder_path, 
                reference_joint_angles_path=args.reference_joint_angles_path, 
                reference_ee_states_path=args.reference_ee_states_path, 
                robot_dataset=args.robot_dataset, 
                episode=episode,
                camera_height=camera_height,
                camera_width=camera_width
            )
            target_env.target_env.env.close_renderer()