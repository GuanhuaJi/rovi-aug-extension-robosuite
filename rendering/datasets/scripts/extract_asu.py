import os
import numpy as np
import tensorflow_datasets as tfds
from PIL import Image

# Example GCS/local path for the dataset
DATASET_GCS_PATH = (
    "gs://gresearch/robotics/asu_table_top_converted_externally_to_rlds/0.1.0"
)

def main():
    # 1) Build from the prepared directory (dataset is ready; no download needed)
    builder = tfds.builder_from_directory(builder_dir=DATASET_GCS_PATH)

    # 2) Load the first 20 episodes from 'train' split, without shuffling
    ds = builder.as_dataset(split="train[:20]", shuffle_files=False)

    # 3) Iterate over each episode
    for episode_num, episode in enumerate(ds):
        print(f"[INFO] Processing Episode {episode_num}")

        # Prepare lists for storing step-wise data
        joint_angles_list = []  # shape: (6,) from state[:6]
        gripper_list = []       # shape: (1,) from state[6:7]
        ee_pose_list = []       # shape: (6,) from ground_truth_states['EE']

        # Create an output folder for this episode
        folder_path = f"../states/asu_table_top_rlds/episode_{episode_num}"
        os.makedirs(folder_path, exist_ok=True)

        # Subfolder for images
        images_folder = os.path.join(folder_path, "images")
        os.makedirs(images_folder, exist_ok=True)

        # 4) Go through each step in the episode
        steps_dataset = episode["steps"]
        for step_idx, step in enumerate(steps_dataset):
            # --- Observations ---
            obs_state = step["observation"]["state"].numpy()  # shape (7,)
            # First 6 are joint angles, last 1 is gripper
            joint_angles = obs_state[:6]       # shape (6,)
            gripper_state = obs_state[6:7]     # shape (1,)

            # End-effector from ground_truth_states['EE'] => (6,) [x, y, z, r, p, y]
            ee_pose = step["ground_truth_states"]["EE"].numpy()  # shape (6,)

            # Accumulate
            joint_angles_list.append(joint_angles)
            gripper_list.append(gripper_state)
            ee_pose_list.append(ee_pose)

            # --- Extract and save the image ---
            image_np = step["observation"]["image"].numpy()  # shape (224, 224, 3)
            img = Image.fromarray(image_np)
            image_filename = os.path.join(images_folder, f"{step_idx}.jpeg")
            img.save(image_filename, format="JPEG")

        # 5) Convert data to NumPy arrays
        joint_angles_array = np.array(joint_angles_list)  # shape (T, 6)
        gripper_array = np.array(gripper_list)            # shape (T, 1)
        ee_pose_array = np.array(ee_pose_list)            # shape (T, 6)

        # 6) Save them as .txt
        np.savetxt(os.path.join(folder_path, "joint_states.txt"), joint_angles_array)
        np.savetxt(os.path.join(folder_path, "gripper_states.txt"), gripper_array)
        np.savetxt(os.path.join(folder_path, "ee_pose.txt"), ee_pose_array)

        print(f"[INFO] Episode {episode_num} processed with {joint_angles_array.shape[0]} steps.")

if __name__ == "__main__":
    main()