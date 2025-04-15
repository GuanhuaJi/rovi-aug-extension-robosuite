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

from test_server import RobotCameraWrapper, change_brightness
from robot_pose_dict import ROBOT_POSE_DICT
from tqdm import tqdm


from transforms3d.quaternions import quat2mat
def map_panda_to_robot_gripper(panda_gripper, target_robot):
    """
    Map a Panda gripper position (two-finger) to a corresponding gripper configuration 
    for Jaco (3-finger), IIWA (2-finger), or Sawyer (2-finger).
    
    Args:
        panda_gripper (list or tuple of length 2):
            E.g. [0.02, -0.02], representing each Panda finger joint in meters.
        target_robot (str):
            One of {"Jaco", "IIWA", "Sawyer"}.
    
    Returns:
        list of float:
            - For Sawyer or IIWA (both 2-finger), returns 2 values (e.g. [new_left, new_right]).
            - For Jaco (3-finger), returns 3 values (base joint positions).

    Assumptions:
        - Panda max open distance ~ 0.08m
        - Sawyer/IIWA max open distance ~ 0.05m (adjust as needed)
        - Jaco root joint range [0.0, 1.0] (often ~1.0 => fully open, 0.0 => closed)
    """

    # 1) Panda’s max open distance (tune to your actual model)
    panda_max_open = 0.08  # meters

    # Extract the two Panda finger positions
    panda_left, panda_right = panda_gripper

    # Current Panda open distance
    panda_current_open = abs(panda_left - panda_right)

    # 2) Compute open ratio and clamp to [0, 1]
    open_ratio = panda_current_open / panda_max_open
    open_ratio = max(0.0, min(1.0, open_ratio))  # clamp

    # 3) Depending on the target robot, map open_ratio to its range
    if target_robot == "Sawyer" or target_robot == "IIWA":
        # Assume both Sawyer and IIWA have a parallel-jaw gripper with some max open distance
        # (tweak this based on your model’s real maximum)
        target_max_open = 0.05  # meters
        
        # New distance in [0, target_max_open]
        new_dist = open_ratio * target_max_open
        
        # Return two symmetric positions, e.g. +x and -x
        # (If your model uses a different approach—like 0 to new_dist—adapt as needed.)
        left_finger  =  new_dist / 2
        right_finger = -new_dist / 2
        return [left_finger, right_finger]

    elif target_robot == "Jaco":
        # Jaco typically has 3 base joints each in [0, 1] (0 = closed, 1 = open)
        # We treat open_ratio as directly the Jaco joint value
        jaco_val = open_ratio * 1.0  # if max is 1.0
        return [jaco_val, jaco_val, jaco_val]

    else:
        raise ValueError(f"Unknown target_robot: {target_robot}")

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
        print("TARGET_NAME", target_name)
    
    
    def generate_image(self, save_paired_images_folder_path="paired_images", reference_joint_angles_path=None, reference_ee_states_path=None, robot_dataset=None, episode=0):
        npz_path = os.path.join(save_paired_images_folder_path, f"{episode}.npz")
        data = np.load(npz_path, allow_pickle=True)
        target_pose_array = data['pos']
        shift_dist = ROBOT_POSE_DICT[robot_dataset]['offset_dist']
        target_pose_array[:, :3] = offset_in_quaternion_direction_batch(target_pose_array[:, :3], target_pose_array[:, 3:], shift_dist)
        gripper_array = data['grip']

        camera_pose = data['camera']
        camera_pose[:3] -= ROBOT_POSE_DICT[robot_dataset][self.target_name]['displacement']
        # gripper_v = 0.8

        self.target_env.camera_wrapper.set_camera_pose(pos=camera_pose[:3], quat=camera_pose[3:])

        if "fov" in data:
            fov = data["fov"]
            self.target_env.camera_wrapper.set_camera_fov(fov)
        self.target_env.update_camera()

        num_robot_poses = target_pose_array.shape[0]
        target_pose_list = []
        
        for pose_index in tqdm(range(num_robot_poses), desc=f'{self.target_name} Pose Generation'):
            '''
            if pose_index % 30 == 0: # to avoid simulation becoming unstable
                self.target_env.env.reset()
            '''
            target_pose=target_pose_array[pose_index]
            target_pose[:3] -= ROBOT_POSE_DICT[robot_dataset][self.target_name]['displacement']
            '''
            if robot_dataset == "viola" and self.target_name == "Jaco":
                target_pose[:3] -= np.array([0, 0, 0.1])
            elif robot_dataset == "austin_mutex" and self.target_name == "Jaco":
                target_pose[:3] -= np.array([0, 0, 0.1])
            elif robot_dataset == "nyu_franka" and self.target_name == "Jaco":
                target_pose[:3] += np.array([-0.1, 0, 0.1])
            '''
            #print(gripper_array[pose_index], type(gripper_array[pose_index]))
            self.target_env.open_close_gripper(gripper_open=gripper_array[pose_index])
            # if type(gripper_array[pose_index][0]) is bool:
            #     self.target_env.open_close_gripper(gripper_open=gripper_array[pose_index])
            # else:
            #     gripper_values = map_panda_to_robot_gripper(gripper_array[pose_index], self.target_name)
            #     self.target_env.set_gripper_joint_positions(gripper_values, self.target_name)
            target_reached, target_reached_pose = self.target_env.drive_robot_to_target_pose(target_pose=target_pose)
            ppose = self.target_env.compute_eef_pose()[:3] + ROBOT_POSE_DICT[robot_dataset][self.target_name]['displacement']
            target_pose_list.append(ppose)

            #print("TARGET_REACHED_POSE:", ppose)
            
            
            '''
            if robot_dataset == "viola" and self.target_name == "Jaco":
                camera_pose[:3] -= np.array([0, 0, 0.1])
            elif robot_dataset == "austin_mutex" and self.target_name == "Jaco":
                camera_pose[:3] -= np.array([0, 0, 0.1])
            elif robot_dataset == "nyu_franka" and self.target_name == "Jaco":
                camera_pose[:3] += np.array([-0.1, 0, 0.1])
            '''
            
            joint_indices = self.target_env.env.robots[0]._ref_joint_pos_indexes
            current_joint_angles = self.target_env.env.sim.data.qpos[joint_indices]
            print("Current joint angles:", current_joint_angles)

            target_robot_img, target_robot_seg_img = self.target_env.get_observation(white_background=True)
            
            # sample a random integer between -40 and 40
            target_robot_img_brightness_augmented = change_brightness(target_robot_img, value=np.random.randint(-40, 40), mask=target_robot_seg_img)
            target_robot_img_brightness_augmented = cv2.resize(target_robot_img_brightness_augmented, (256, 256), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(os.path.join(save_paired_images_folder_path, f"{target_name}_rgb", f"{episode}/{pose_index}.jpg"), cv2.cvtColor(target_robot_img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(save_paired_images_folder_path, f"{target_name}_rgb_brightness_augmented", f"{episode}/{pose_index}.jpg"), cv2.cvtColor(target_robot_img_brightness_augmented, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(save_paired_images_folder_path, f"{target_name}_mask", f"{episode}/{pose_index}.jpg"), target_robot_seg_img * 255)
        
        target_pose_array = np.vstack(target_pose_list)
        eef_npy_path = os.path.join(save_paired_images_folder_path, f"{self.target_name}_eef_states_{episode}.npy")
        np.save(eef_npy_path, target_pose_array)


        
        

if __name__ == "__main__":

    """
    Registered environments: Lift, Stack, NutAssembly, NutAssemblySingle, NutAssemblySquare, NutAssemblyRound,
                             PickPlace, PickPlaceSingle, PickPlaceMilk, PickPlaceBread, PickPlaceCereal,
                             PickPlaceCan, Door, Wipe, TwoArmLift, TwoArmPegInHole, TwoArmHandover

    Possible robots: Baxter, IIWA, Jaco, Kinova3, Panda, Sawyer, UR5e
    """

    # print welcome info
    print("Welcome to robosuite v{}!".format(suite.__version__))
    print(suite.__logo__)

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="(optional) (optional) set seed")
    parser.add_argument("--target_gripper", type=str, default="Robotiq85Gripper", help="PandaGripper or Robotiq85Gripper")
    parser.add_argument("--num_robot_poses", type=int, default=5, help="(optional) (optional) set seed")
    parser.add_argument("--target_robot", type=str, default="IIWA", help="(optional) (optional) set seed")
    parser.add_argument("--save_paired_images_folder_path", type=str, default="paired_images", help="(optional) folder path to save the paired images")
    parser.add_argument("--robot_dataset", type=str, help="(optional) to match the robot poses from a dataset, provide the dataset name")
    parser.add_argument("--reference_joint_angles_path", type=str, help="(optional) to match the robot poses from a dataset, provide the path to the joint angles file (np.savetxt)")
    parser.add_argument("--reference_ee_states_path", type=str, help="(optional) to match the robot poses from a dataset, provide the path to the ee state file (np.savetxt)")
    parser.add_argument("--episode", type=int, default=0, help="episode number")
    args = parser.parse_args()
    
    
    target_name = args.target_robot

    if target_name == "Sawyer":
        target_gripper = "RethinkGripper"
    elif target_name == "Jaco":
        target_gripper = "JacoThreeFingerGripper"
    elif target_name == "IIWA":
        target_gripper = "Robotiq85Gripper"

    # Save the captured images
    save_paired_images_folder_path = os.path.join("/home/jiguanhua/mirage/robot2robot/rendering/paired_images", args.robot_dataset)
    os.makedirs(os.path.join(save_paired_images_folder_path, "{}_rgb".format(target_name), str(args.episode)), exist_ok=True)
    os.makedirs(os.path.join(save_paired_images_folder_path, "{}_rgb_brightness_augmented".format(target_name), str(args.episode)), exist_ok=True)
    os.makedirs(os.path.join(save_paired_images_folder_path, "{}_mask".format(target_name), str(args.episode)), exist_ok=True)
    
    if args.robot_dataset is not None:
        from dataset_poses_dict import ROBOT_CAMERA_POSES_DICT
        robot_dataset_info = ROBOT_CAMERA_POSES_DICT[args.robot_dataset]
        camera_height = robot_dataset_info["camera_heights"]
        camera_width = robot_dataset_info["camera_widths"]
    else:
        camera_height = 256
        camera_width = 256
    
    target_env = TargetEnvWrapper(target_name, target_gripper, args.robot_dataset, camera_height, camera_width)

    target_env.generate_image(
        save_paired_images_folder_path=save_paired_images_folder_path, 
        reference_joint_angles_path=args.reference_joint_angles_path, 
        reference_ee_states_path=args.reference_ee_states_path, 
        robot_dataset=args.robot_dataset, 
        episode=args.episode
    )

    target_env.target_env.env.close_renderer()