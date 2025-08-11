import os
import numpy as np
import tensorflow_datasets as tfds
from PIL import Image

# Replace with the path where "austin_sirius_dataset_converted_externally_to_rlds" is stored
DATASET_PATH = "gs://gresearch/robotics/austin_sirius_dataset_converted_externally_to_rlds/0.1.0"
def main():
    # 1) Create builder from specified directory (dataset already prepared; no need to download_and_prepare())
    builder = tfds.builder_from_directory(builder_dir=DATASET_PATH)
    
    # 2) Load the first 20 episodes from the train split without shuffling file order
    ds = builder.as_dataset(split="train", shuffle_files=False)

    # 3) Iterate over each episode
    for episode_num, episode in enumerate(ds):
        print(f"[INFO] Processing Episode {episode_num}")
        
        # Initialize lists to store data for each step
        joint_angles_list = []    # taken from the first 7 elements of observation['state'] or observation['state_joint']
        gripper_state_list = []   # taken from the 8th element of observation['state'] or observation['state_gripper']
        ee_states_list = []       # taken from observation['state_ee']

        # Set current episode storage path
        folder_path = f"../states/austin_sirius/episode_{episode_num}"
        os.makedirs(folder_path, exist_ok=True)
        
        # Create folder for images
        images_folder = os.path.join(folder_path, "images")
        os.makedirs(images_folder, exist_ok=True)
        
        # 4) Iterate over each step in the current episode
        steps_dataset = episode["steps"]
        for step_idx, step in enumerate(steps_dataset):
            # ---- Observation section ----
            # observation['state'] -> (8,) => [0:7] are joint angles, [7] is gripper
            # observation['state_joint'] -> (7,)
            # observation['state_gripper'] -> (1,)
            # observation['state_ee'] -> (16,)
            # observation['image'] -> (84, 84, 3)

            # If you want to mimic the original approach of fetching state at once and slicing:
            state = step["observation"]["state"].numpy()  # (8,)
            joint_angles = state[:7]            # first 7 elements
            gripper_state = state[7:8]          # 8th element (keep (1,) shape)
            
            # End-effector 4x4 homogeneous transform, 16 floats total
            ee_state = step["observation"]["state_ee"].numpy()  # (16,)

            joint_angles_list.append(joint_angles)
            gripper_state_list.append(gripper_state)
            ee_states_list.append(ee_state)
            
            # 5) Extract main camera image (84, 84, 3)
            image_np = step["observation"]["image"].numpy()
            img = Image.fromarray(image_np)
            
            # Save as JPEG, filenames numbered by step: "0.jpeg", "1.jpeg", ...
            image_filename = os.path.join(images_folder, f"{step_idx}.jpeg")
            img.save(image_filename, format="JPEG")
        
        # 6) Convert lists to numpy arrays and save as text files
        joint_angles_array = np.array(joint_angles_list)   # shape: (T, 7)
        gripper_state_array = np.array(gripper_state_list) # shape: (T, 1)
        ee_states_array = np.array(ee_states_list)         # shape: (T, 16)
        
        np.savetxt(os.path.join(folder_path, "joint_states.txt"), joint_angles_array)
        np.savetxt(os.path.join(folder_path, "gripper_states.txt"), gripper_state_array)
        np.savetxt(os.path.join(folder_path, "ee_states.txt"), ee_states_array)
        
        print(f"[INFO] Episode {episode_num} processed with {joint_angles_array.shape[0]} steps.")

if __name__ == "__main__":
    main()
