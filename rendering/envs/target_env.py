from sim.robot_camera import RobotCameraWrapper
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

STEP_MAX = 0.02          # 单帧允许的最大 L1 位移（米），1 cm
ORI_LERP = False

class TargetEnvWrapper:
    def __init__(self, target_name, target_gripper, robot_dataset, camera_height=256, camera_width=256):
        self.target_env = RobotCameraWrapper(robotname=target_name, grippername=target_gripper, robot_dataset=robot_dataset, camera_height=camera_height, camera_width=camera_width)
        self.target_name = target_name

    def _load_dataset_info(self, dataset_name):
        info = ROBOT_CAMERA_POSES_DICT[dataset_name]
        return info
    
    def generate_image(self, 
                        save_paired_images_folder_path="paired_images", 
                        displacement_csv_path=None,
                        source_robot_states_path="paired_images",
                        reference_joint_angles_path=None, 
                        reference_ee_states_path=None, 
                        robot_dataset=None, 
                        unlimited="False",
                        load_displacement=False,
                        episode=0, 
                        camera_height=256, 
                        camera_width=256):
        data = np.load(os.path.join(source_robot_states_path, "source_robot_states", f"{episode}.npz"), allow_pickle=True)
        info = self._load_dataset_info(robot_dataset)
        # if robot_dataset in ["austin_buds", "austin_sailor"]:
        #     target_pose_array = smooth_xyz_spikes(
        #         data['pos'].copy(),
        #         thresh=0.05)
        # else:
        #     target_pose_array = data['pos'].copy() 
        target_pose_array = data['pos'].copy()

        gripper_array = data['grip']
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
        robot_disp = None

        if load_displacement:
            offset_file = os.path.join(source_robot_states_path, "source_robot_states", self.target_name, "offsets", f"{episode}.npy")
            if os.path.isfile(offset_file):
                robot_disp = np.load(offset_file)
            else:
                robot_disp = np.zeros(3, dtype=np.float32)
                print(f"WARNING: displacement file not found → {offset_file}; "
                    f"using default [0, 0, 0].")
        else:
            robot_disp = ROBOT_POSE_DICT[robot_dataset][self.target_name]

        camera_pose[:3] -= robot_disp
        
        self.target_env.camera_wrapper.set_camera_pose(pos=camera_pose[:3], quat=camera_pose[3:])
        if "fov" in data:
            fov = data["fov"]
            self.target_env.camera_wrapper.set_camera_fov(fov)
        self.target_env.update_camera()

        os.makedirs(os.path.join(save_paired_images_folder_path, "{}_rgb".format(self.target_name), str(episode)), exist_ok=True)
        os.makedirs(os.path.join(save_paired_images_folder_path, "{}_mask".format(self.target_name), str(episode)), exist_ok=True)
        num_robot_poses = target_pose_array.shape[0]
        target_pose_list = []
        joint_angles_list = []
        gripper_width_list = []
        success = True
        
        for pose_index in range(num_robot_poses):
            target_pose=target_pose_array[pose_index].copy()
            target_pose[:3] -= robot_disp
            target_pose = reach_further(target_pose, distance=ROBOT_CAMERA_POSES_DICT[robot_dataset]["extend_gripper"])
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

            if unlimited == "False" and not target_reached:
                print(f"episode {episode} pose {pose_index}")
                blacklist_path = Path(f"{save_paired_images_folder_path}/{self.target_name}/blacklist.json")
                with locked_json(blacklist_path) as blk:
                    robot_list = blk.setdefault(self.target_name, [])
                    if episode not in robot_list:
                        robot_list.append(episode)
                        robot_list.sort()
                        RED   = "\033[91m"
                        RESET = "\033[0m"
                        print(f"{RED}[BLACKLIST] Added {self.target_name} – episode {episode}{RESET}")
                success = False
                try:
                    n = len(target_pose_list)
                    tgt_xy = target_pose_array[:, :2]
                    real_xy = np.array(target_pose_list)[:, :2] - robot_disp[:2]
                    fig, ax = plt.subplots(figsize=(4, 4))
                    ax.plot(tgt_xy[:, 0], tgt_xy[:, 1], "o-", label="target XY")
                    ax.plot(real_xy[:, 0], real_xy[:, 1], "x-", label="reached XY")
                    ax.set_xlabel("X (m)");  ax.set_ylabel("Y (m)")
                    ax.set_title(f"{self.target_name} – episode {episode} - offset {robot_disp.round(3)}")
                    ax.axis("equal");  ax.legend()
                    
                    out_dir = os.path.join(save_paired_images_folder_path,
                                        f"{self.target_name}_traj_plots")
                    os.makedirs(out_dir, exist_ok=True)
                    out_file = os.path.join(out_dir, f"{episode}.png")
                    fig.savefig(out_file,
                                dpi=150, bbox_inches="tight")
                    plt.close(fig)
                    print(f"\033[95m[TRAJ]  Saved XY plot for episode {episode}: {out_file}\033[0m")
                except Exception as e:
                    print(f"[WARN] could not save trajectory plot: {e}")
                break
            reached_pose = self.target_env.compute_eef_pose()
            reached_pose[:3] += robot_disp
            target_pose_list.append(reached_pose)
            gripper_width_list.append(self.target_env.get_gripper_width_from_qpos())
            
            joint_indices = self.target_env.env.robots[0]._ref_joint_pos_indexes
            joint_angles = self.target_env.env.sim.data.qpos[joint_indices]
            joint_angles_list.append(joint_angles)


            target_robot_img, target_robot_seg_img = self.target_env.get_observation_fast(white_background=True, width=camera_width, height=camera_height)
            cv2.imwrite(os.path.join(save_paired_images_folder_path, f"{self.target_name}_rgb", f"{episode}/{pose_index}.png"), cv2.cvtColor(target_robot_img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(save_paired_images_folder_path, f"{self.target_name}_mask", f"{episode}/{pose_index}.png"), target_robot_seg_img * 255)
        
        if success:
            if unlimited == "False":
                blacklist_path = Path(f"{save_paired_images_folder_path}/{self.target_name}/blacklist.json")
                with locked_json(blacklist_path) as blk:
                    robot_list = blk.get(self.target_name, [])
                    if episode in robot_list:
                        robot_list.remove(episode)
                        if robot_list:
                            blk[self.target_name] = robot_list
                        else:
                            blk.pop(self.target_name)
                        print(f"\033[92m[BLACKLIST] Removed {self.target_name} – episode {episode}\033[0m")

                whitelist_path = Path(f"{save_paired_images_folder_path}/{self.target_name}/whitelist.json")
                with locked_json(whitelist_path) as wl:
                    robot_list = wl.get(self.target_name, [])
                    if episode not in robot_list:
                        robot_list.append(episode)
                        robot_list.sort()
                        wl[self.target_name] = robot_list
                        print(f"\033[92m[WHITELIST] Added {self.target_name} – episode {episode}\033[0m")
            else:
                print(f"\033[92m[UNLIMITED] Generated {self.target_name} – episode {episode}\033[0m")

            target_pose_array = np.vstack(target_pose_list)
            joint_angles_array = np.vstack(joint_angles_list)
            gripper_width_array = np.array(gripper_width_list)
            eef_npy_path = os.path.join(save_paired_images_folder_path, "source_robot_states", f"{self.target_name}", "end_effector", f"{episode}.npy")
            np.save(eef_npy_path, target_pose_array)
            gripper_npy_path = os.path.join(save_paired_images_folder_path, "source_robot_states", f"{self.target_name}", "gripper_distance", f"{episode}.npy")
            np.save(gripper_npy_path, gripper_width_array)
            joint_angles_npy_path = os.path.join(save_paired_images_folder_path, "source_robot_states", f"{self.target_name}", "joint_angles", f"{episode}.npy")
            np.save(joint_angles_npy_path, joint_angles_array)
            offset_npy_path = os.path.join(save_paired_images_folder_path, "source_robot_states", f"{self.target_name}", "offsets", f"{episode}.npy")
            np.save(offset_npy_path, robot_disp)