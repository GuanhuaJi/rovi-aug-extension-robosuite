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
    
    def get_source_robot_states(self, gripper_states=None, joint_angles=None, ee_states=None, episode=0, save_source_robot_states_path="paired_images"):
        if joint_angles is None and ee_states is None:
            raise ValueError("Either joint_angles or ee_states must be provided.")
        if joint_angle is not None:


        info = ROBOT_CAMERA_POSES_DICT[self.robot_dataset]
        if self.robot_dataset == "ucsd_kitchen_rlds" or self.robot_dataset == "utokyo_pick_and_place":
            joint_angles, gripper_states, translation = load_states_from_harsha(self.robot_dataset, episode, self.source_env.robot_name)
        else:
            joint_angles = np.loadtxt(os.path.join("/home/guanhuaji/mirage/robot2robot/rendering/datasets/states", self.robot_dataset, f"episode_{episode}", "joint_states.txt"))
            gripper_states = np.loadtxt(os.path.join("/home/guanhuaji/mirage/robot2robot/rendering/datasets/states", self.robot_dataset, f"episode_{episode}", "gripper_states.txt"))
        if self.robot_dataset == "toto":
            joint_angles[:, 5] += 3.14159 / 2
            joint_angles[:, 6] += 3.14159 / 4
        elif self.robot_dataset == "autolab_ur5":
            joint_angles[:, 5] += 3.14159 / 2
        elif self.robot_dataset == "asu_table_top_rlds":
            joint_angles[:, 1] -= np.pi / 2
            joint_angles[:, 2] *= -1
            joint_angles[:, 3] -= np.pi / 2
            joint_angles[:, 5] -= np.pi / 2
        elif self.robot_dataset == "viola":
            tol = 1e-8
            for i, row in enumerate(joint_angles):
                if not np.all(np.isclose(row, 0.0, atol=tol)):
                    if i > 0:
                        joint_angles[:i] = row
                        print(f"WARNING: first {i} rows were all zeros; "
                            f"copied row {i} into them.")
                    break
            else:
                print("WARNING: all joint_angles rows are zeros; nothing replaced.")
        elif self.robot_dataset == "austin_buds":
            tol = 1e-8
            N   = len(joint_angles)
            zero_mask = np.all(np.isclose(joint_angles, 0.0, atol=tol), axis=1)
            if zero_mask.all():
                print("WARNING: all joint_angles rows are zeros; nothing replaced.")
            else:
                i = 0
                while i < N:
                    if not zero_mask[i]:
                        i += 1
                        continue
                    start = i
                    while i < N and zero_mask[i]:
                        i += 1
                    end = i

                    left  = start - 1
                    right = end if end < N else None

                    if left < 0 and right is None:
                        print("WARNING: joint_angles entirely zero for austin_buds; left unchanged.")
                        break
                    if left < 0:
                        joint_angles[start:end] = joint_angles[right]
                    elif right is None:
                        joint_angles[start:end] = joint_angles[left]
                    else:
                        gap = right - left
                        for k in range(1, gap):
                            alpha = k / gap
                            joint_angles[left + k] = (1 - alpha) * joint_angles[left] + alpha * joint_angles[right]
                print(f"[INFO] austin_buds: filled {zero_mask.sum()} zero rows via interpolation / copying.")

        for viewpoint in info["viewpoints"]:
            if episode in viewpoint["episodes"]:
                camera_reference_position = viewpoint["camera_position"] + np.array([-0.6, 0.0, 0.912]) 
                roll_deg = viewpoint["roll"]
                pitch_deg = viewpoint["pitch"]
                yaw_deg = viewpoint["yaw"]
                fov = viewpoint["camera_fov"]
                r = R.from_euler('xyz', [roll_deg, pitch_deg, yaw_deg], degrees=True)
                camera_reference_quaternion = r.as_quat()
                camera_reference_pose = np.concatenate((camera_reference_position, camera_reference_quaternion))
                break
        target_pose_list = []
        gripper_list = []
        num_frames = joint_angles.shape[0]

        for pose_index in tqdm(range(num_frames), desc=f'{self.source_name} Pose States Calculation'):    
            source_reached = False
            attempt_counter = 0
            while source_reached == False:
                attempt_counter += 1
                if attempt_counter > 10:
                    break
                if self.robot_dataset == "kaist":
                    gripper_open = False
                elif self.robot_dataset in ["can", "lift", "square", "stack", "three_piece_assembly"]:
                    gripper_open = (gripper_states[pose_index][0] - gripper_states[pose_index][1]) > 0.06
                else:
                    gripper_open = gripper_convert(gripper_states[pose_index], self.robot_dataset)
                for i in range(5):
                    joint_angle = joint_angles[pose_index]
                    self.source_env.teleport_to_joint_positions(joint_angle)
                    target_pose = self.source_env.compute_eef_pose()

            gripper_list.append(gripper_open)
            target_pose_list.append(target_pose)       
        target_pose_array = np.vstack(target_pose_list)
        if self.robot_dataset == "ucsd_kitchen_rlds":
            target_pose_array[:, :3] -= translation
        gripper_array = np.vstack(gripper_list)
        eef_npy_path = os.path.join(
            save_source_robot_states_path, f"{self.source_name}_eef_states_{episode}.npy"
        )
        GREEN = "\033[92m"
        RESET = "\033[0m"
        np.save(eef_npy_path, target_pose_array)
        print(f"{GREEN}✔ End effector saved under {eef_npy_path}{RESET}")
        npz_path = os.path.join(save_source_robot_states_path, f"{episode}.npz")
        
        if self.source_name == "Panda":
            if np.all(gripper_array <= 0.09):
                gripper_states = np.clip(gripper_array / 0.08, 0, 1)
            else:
                gripper_states = np.clip(gripper_array, 0, 1)
        elif self.source_name == "UR5e":
            if np.all(gripper_array <= 0.06) and np.all(gripper_array >= -0.06):
                gripper_states = np.clip((gripper_array + 0.05) / 0.1, 0, 1)
            else:
                gripper_states = np.clip(gripper_array, 0, 1)

        np.savez(npz_path, pos=target_pose_array, grip=gripper_states)
        print(f"{GREEN}✔ States saved under {npz_path}{RESET}")