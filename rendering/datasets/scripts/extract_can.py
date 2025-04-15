import os
import h5py
import numpy as np
from PIL import Image

def main():
    hdf5_path = "/home/jiguanhua/mirage/robot2robot/image84/can/image_84.hdf5"

    with h5py.File(hdf5_path, "r") as f:
        data_group = f["data"]
        demo_keys = list(data_group.keys())  # e.g. ["demo_0", "demo_1", ...]

        for i, demo_key in enumerate(demo_keys):
            print(f"[INFO] Processing {demo_key} as episode_{i}")
            demo_group = data_group[demo_key]

            # --- 1) Access observations ---
            obs_group = demo_group["obs"]

            # Joint angles (T, 7)
            robot0_joint_pos = obs_group["robot0_joint_pos"][:]  
            # EEF position (T, 3) and quaternion (T, 4)
            robot0_eef_pos = obs_group["robot0_eef_pos"][:]
            robot0_eef_quat = obs_group["robot0_eef_quat"][:]
            # Gripper state (T, 2)
            robot0_gripper_qpos = obs_group["robot0_gripper_qpos"][:]
            # Camera frames (T, H, W, 3)
            agentview_images = obs_group["agentview_image"][:]  

            # Combine EEF pos+quat to one array (T, 7), if desired
            ee_states = np.concatenate([robot0_eef_pos, robot0_eef_quat], axis=-1)

            # --- 2) Create output folder structure ---
            # ../states/can/episode_{i}/
            episode_path = f"../states/can/episode_{i}"
            os.makedirs(episode_path, exist_ok=True)

            # Create an "images" subfolder
            images_folder = os.path.join(episode_path, "images")
            os.makedirs(images_folder, exist_ok=True)

            # --- 3) Save each frame as .jpeg ---
            for t in range(agentview_images.shape[0]):
                frame = agentview_images[t]  # shape (H, W, 3)
                img = Image.fromarray(frame)
                img.save(os.path.join(images_folder, f"{t}.jpeg"), format="JPEG")

            # --- 4) Save states as .txt ---
            # Joint states
            np.savetxt(os.path.join(episode_path, "joint_states.txt"), robot0_joint_pos)
            # EEF states (pos + quat)
            np.savetxt(os.path.join(episode_path, "ee_states.txt"), ee_states)
            # Gripper states
            np.savetxt(os.path.join(episode_path, "gripper_states.txt"), robot0_gripper_qpos)

            print(f"[INFO] episode_{i} saved with {robot0_joint_pos.shape[0]} timesteps.")

if __name__ == "__main__":
    main()