from sim.robot_camera import RobotCameraWrapper
#from sim.robot_camera_15 import RobotCameraWrapper
from core.signal import reach_further
import numpy as np
import os
from config.dataset_poses_dict import ROBOT_CAMERA_POSES_DICT
from config.robot_pose_dict import ROBOT_POSE_DICT
from core import locked_json, atomic_write_json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import cv2
from pathlib import Path
import imageio.v3 as iio


STEP_MAX = 0.02          # 单帧允许的最大 L1 位移（米），1 cm
ORI_LERP = False

class TargetEnvWrapper:
    def __init__(self, target_name, target_gripper, robot_dataset, camera_height=256, camera_width=256):
        self.target_env = RobotCameraWrapper(robotname=target_name, grippername=target_gripper, robot_dataset=robot_dataset, camera_height=camera_height, camera_width=camera_width)
        self.target_name = target_name
        # self.camera_height = camera_height
        # self.camera_width = camera_width
        self.camera_height = 84
        self.camera_width = 84

    def generate_image(
        self,
        save_paired_images_folder_path="paired_images",
        source_robot_states_path="paired_images",
        robot_dataset=None,
        robot_disp=None,
        unlimited=False,
        episode=0,
        dry_run=False,
    ):
        print(robot_dataset, robot_disp, episode)
        data = np.load(os.path.join(source_robot_states_path, "source_robot_states", f"{episode}.npz"), allow_pickle=True)
        info = ROBOT_CAMERA_POSES_DICT[robot_dataset]
        target_pose_array = data['pos'].copy()


        gripper_array = data['grip']
        if robot_dataset == "can":
            camera_pose = np.array([0.9, 0.1, 1.75, 0.271, 0.271, 0.653, 0.653])
        elif robot_dataset == "lift":
            camera_pose = np.array([0.45, 0, 1.35, 0.271, 0.271, 0.653, 0.653])
        elif robot_dataset == "square":
            camera_pose = np.array([0.45, 0, 1.35, 0.271, 0.271, 0.653, 0.653])
        elif robot_dataset == "stack":
            camera_pose = np.array([0.45, 0, 1.35, 0.271, 0.271, 0.653, 0.653])
        elif robot_dataset == "three_piece_assembly":
            camera_pose = np.array([0.713078462147161, 2.062036796036723e-08, 1.5194726087166726, 0.293668270111084, 0.2936684489250183, 0.6432408690452576, 0.6432409286499023])
        else:
            for viewpoint in info["viewpoints"]:
                if episode in viewpoint["episodes"]:
                    camera_reference_position = viewpoint["camera_position"] + np.array([-0.6, 0.0, 0.912]) 
                    roll_deg = viewpoint["roll"]
                    pitch_deg = viewpoint["pitch"]
                    yaw_deg = viewpoint["yaw"]
                    fov = viewpoint["camera_fov"]
                    r = R.from_euler('xyz', [roll_deg, pitch_deg, yaw_deg], degrees=True)
                    camera_reference_quaternion = r.as_quat()
                    camera_pose = np.concatenate((camera_reference_position, camera_reference_quaternion))
                    break
        if robot_disp is None:
            #robot_disp = ROBOT_POSE_DICT[robot_dataset][self.target_name]
            robot_disp = np.zeros(3, dtype=np.float32)

        camera_pose[:3] -= robot_disp
        
        self.target_env.camera_wrapper.set_camera_pose(pos=camera_pose[:3], quat=camera_pose[3:])
        if "fov" in data:
            fov = data["fov"]
            self.target_env.camera_wrapper.set_camera_fov(fov)
        self.target_env.update_camera()

        # os.makedirs(os.path.join(save_paired_images_folder_path, "{}_rgb".format(self.target_name), str(episode)), exist_ok=True)
        # os.makedirs(os.path.join(save_paired_images_folder_path, "{}_mask".format(self.target_name), str(episode)), exist_ok=True)
        num_robot_poses = target_pose_array.shape[0]
        target_pose_list = []
        joint_angles_list = []
        gripper_width_list = []
        success = True

        mask_dir = Path(save_paired_images_folder_path) / f"{self.target_name}_replay_mask"
        video_dir = Path(save_paired_images_folder_path) / f"{self.target_name}_replay_video"
        mask_dir.mkdir(parents=True, exist_ok=True)
        video_dir.mkdir(parents=True, exist_ok=True)
        mask_path  = mask_dir / f"{episode}.mp4"
        video_path = video_dir / f"{episode}.mp4"
        mask_frames = []
        video_frames = []
        mask_path = video_path = None
        if not dry_run:
            mask_dir = Path(save_paired_images_folder_path) / f"{self.target_name}_replay_mask"
            video_dir = Path(save_paired_images_folder_path) / f"{self.target_name}_replay_video"
            mask_dir.mkdir(parents=True, exist_ok=True)
            video_dir.mkdir(parents=True, exist_ok=True)
            mask_path  = mask_dir / f"{episode}.mp4"
            video_path = video_dir / f"{episode}.mp4"
        suggestion = np.zeros(3)
        for pose_index in range(num_robot_poses):
            target_pose=target_pose_array[pose_index].copy()
            target_pose[:3] -= robot_disp
            #target_pose = reach_further(target_pose, distance=ROBOT_CAMERA_POSES_DICT[robot_dataset]["extend_gripper"])
            _, gripper_dist = self.target_env.get_gripper_width_from_qpos()
            attempt = 0
            while (gripper_dist < gripper_array[pose_index] - 0.1 or gripper_dist > gripper_array[pose_index] + 0.1) and attempt < 10:
                if gripper_dist < gripper_array[pose_index] - 0.1:
                    self.target_env.open_close_gripper(gripper_open=True)
                elif gripper_dist > gripper_array[pose_index] + 0.1:
                    self.target_env.open_close_gripper(gripper_open=False)
                _, gripper_dist = self.target_env.get_gripper_width_from_qpos()
                attempt += 1

                
            target_reached, target_reached_pose, error = (
                self.target_env.drive_robot_to_target_pose(target_pose=target_pose)
            )

            if unlimited == False and not target_reached:
                success = False
                suggestion = target_pose[:3] - target_reached_pose[:3]
                break
            reached_pose = self.target_env.compute_eef_pose()
            reached_pose[:3] += robot_disp
            target_pose_list.append(reached_pose)
            gripper_width_list.append(self.target_env.get_gripper_width_from_qpos())
            
            joint_indices = self.target_env.env.robots[0]._ref_joint_pos_indexes
            joint_angles = self.target_env.env.sim.data.qpos[joint_indices]
            joint_angles_list.append(joint_angles)


            target_robot_img, target_robot_seg_img = self.target_env.get_observation_fast(
                white_background=True,
                width=self.camera_width,
                height=self.camera_height,
            )
            if not dry_run:
                mask_frames.append(target_robot_seg_img)
                video_frames.append(target_robot_img)
        if success:        
            if not dry_run:
                mask_frames_np = np.stack(mask_frames, axis=0).astype(np.uint8) * 255
                video_frames_np = np.stack(video_frames, axis=0)
                iio.imwrite(
                    mask_path,
                    mask_frames_np,          # shape (T, 84, 84) or (T, 84, 84, 3)
                    fps=30,
                    codec="libx264",
                    macro_block_size=1,      # ← disable 16-pixel padding
                    pixelformat="gray"       # or "yuv420p" if your mask is 3-channel
                )

                iio.imwrite(
                    video_path,
                    video_frames_np,         # shape (T, 84, 84, 3)
                    fps=30,
                    codec="libx264",
                    macro_block_size=1,      # ← same here
                    pixelformat="yuv420p"    # keeps the file widely playable
                )
            if unlimited == False:
                print(f"\033[92m[SUCCESS] Generated {self.target_name} – episode {episode}\033[0m")
            else:
                print(f"\033[92m[UNLIMITED] Generated {self.target_name} – episode {episode}\033[0m")

            if not dry_run:
                os.makedirs(
                    os.path.join(
                        save_paired_images_folder_path,
                        "target_robot_states",
                        f"{self.target_name}",
                    ),
                    exist_ok=True,
                )
                target_pose_array = np.vstack(target_pose_list)
                joint_angles_array = np.vstack(joint_angles_list)
                gripper_width_array = np.asarray(gripper_width_list)
                offset_array = robot_disp
                state_npz_path = os.path.join(
                    save_paired_images_folder_path,
                    "target_robot_states",
                    f"{self.target_name}",
                    f"{episode}.npz",
                )
                np.savez(
                    state_npz_path,
                    target_pose=target_pose_array,
                    joint_angles=joint_angles_array,
                    gripper_width=gripper_width_array,
                    offsets=offset_array,
                )
            steps = len(target_pose_list)
            return success, suggestion, steps
        else:
            print(f"\033[91m[FAILURE] Could not reach target pose for {self.target_name} – episode {episode}\033[0m")
            steps = len(target_pose_list)
            return False, suggestion, steps

