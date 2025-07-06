#from sim.robot_camera_15 import RobotCameraWrapper
from sim.robot_camera import RobotCameraWrapper
from sim.camera import CameraWrapper
import numpy as np
import os
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from sim.dataset_loader import gripper_convert, load_states_from_harsha
from config.dataset_poses_dict import ROBOT_CAMERA_POSES_DICT

class SourceEnvWrapper:
    def __init__(self, source_name, source_gripper, robot_dataset, camera_height=256, camera_width=256, verbose=False):
        self.source_env = RobotCameraWrapper(robotname=source_name, grippername=source_gripper, robot_dataset=robot_dataset, camera_height=camera_height, camera_width=camera_width)
        self.source_name = source_name
        self.robot_dataset = robot_dataset
        self.fixed_cam_positions = None
        self.fixed_cam_quaternions = None
        self.verbose = verbose
    
    def get_source_robot_states(self, gripper_states, joint_angles=None, ee_states=None, episode=0, save_source_robot_states_path="paired_images"):
        if joint_angles is None and ee_states is None:
            raise ValueError("Either joint_angles or ee_states must be provided.")
        if joint_angles is None:
            raise ValueError("joint_angles must be provided")
        info = ROBOT_CAMERA_POSES_DICT[self.robot_dataset]

        target_pose_list = []
        num_frames = joint_angles.shape[0]

        for pose_index in tqdm(range(num_frames), desc=f'{self.source_name} Pose States Calculation'):    
        #for pose_index in range(num_frames):  
            source_reached = False
            attempt_counter = 0
            while source_reached == False:
                attempt_counter += 1
                if attempt_counter > 10:
                    break
                for i in range(5):
                    joint_angle = joint_angles[pose_index]
                    self.source_env.teleport_to_joint_positions(joint_angle)
                    target_pose = self.source_env.compute_eef_pose()

            target_pose_list.append(target_pose)       
        target_pose_array = np.vstack(target_pose_list)
        GREEN = "\033[92m"
        RESET = "\033[0m"
        npz_path = os.path.join(save_source_robot_states_path, f"{episode}.npz")
        if self.source_name == "Panda":
            if np.all(gripper_states <= 0.09):
                gripper_states = np.clip(gripper_states / 0.08, 0, 1)
            else:
                gripper_states = np.clip(gripper_states, 0, 1)
        elif self.source_name == "UR5e":
            if np.all(gripper_states <= 0.06) and np.all(gripper_states >= -0.06):
                gripper_states = np.clip((gripper_states + 0.05) / 0.1, 0, 1)
            else:
                gripper_states = np.clip(gripper_states, 0, 1)

        np.savez(npz_path, pos=target_pose_array, grip=gripper_states)
        print(f"{GREEN}✔ States saved under {npz_path}{RESET}")