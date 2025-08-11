import os
import numpy as np
import tensorflow_datasets as tfds
from PIL import Image

# Example GCS path or local directory for the cmu_play_fusion dataset
DATASET_GCS_PATH = "gs://gresearch/robotics/cmu_play_fusion/0.1.0"

def main():
    # 1) Create builder from specified directory (dataset already prepared; no need to download_and_prepare())
    builder = tfds.builder_from_directory(builder_dir=DATASET_GCS_PATH)
    
    # 2) Load first 20 episodes from train split without shuffling file order
    ds = builder.as_dataset(split="train[:20]", shuffle_files=False)
    
    # 3) Iterate over each episode
    for episode_num, episode in enumerate(ds):
        print(f"[INFO] Processing Episode {episode_num}")
        
        # Initialize lists to store data for each step
        joint_angles_list = []    # taken from first 7 elements of observation/state
        gripper_state_list = []   # taken from 8th element of observation/state (stored as array)
        ee_states_list = []       # dataset only has joint + gripper (8 dims); no additional ee state, so use empty placeholder
        
        # Set storage path for current episode
        folder_path = f"../states/cmu_play_fusion/episode_{episode_num}"
        os.makedirs(folder_path, exist_ok=True)
        
        # Create folder for images
        images_folder = os.path.join(folder_path, "images")
        os.makedirs(images_folder, exist_ok=True)
        
        # 4) Iterate over each step in the current episode
        steps_dataset = episode["steps"]
        for step_idx, step in enumerate(steps_dataset):
            # Extract state (8,) from observation
            # Meaning: [0:7] -> 7 joint angles, [7] -> gripper position
            state = step["observation"]["state"].numpy()
            joint_angles = state[:7]          # shape (7,)
            gripper_state = state[7:8]        # shape (1,)
            
            # This dataset has no separate ee state; placeholder here
            ee_state = np.array([], dtype=np.float32)  # shape (0,)
            
            joint_angles_list.append(joint_angles)
            gripper_state_list.append(gripper_state)
            ee_states_list.append(ee_state)
            
            # 5) Extract main camera image (128, 128, 3)
            image_np = step["observation"]["image"].numpy()
            img = Image.fromarray(image_np)
            # Save as JPEG, filenames numbered by step: "0.jpeg", "1.jpeg", ...
            image_filename = os.path.join(images_folder, f"{step_idx}.jpeg")
            img.save(image_filename, format="JPEG")
        
        # 6) Convert lists to numpy arrays and save as text files
        joint_angles_array = np.array(joint_angles_list)   # shape: (T, 7)
        gripper_state_array = np.array(gripper_state_list) # shape: (T, 1)
        ee_states_array = np.array(ee_states_list)         # shape: (T, 0) placeholder only
        
        np.savetxt(os.path.join(folder_path, "joint_states.txt"), joint_angles_array)
        np.savetxt(os.path.join(folder_path, "gripper_states.txt"), gripper_state_array)
        # If you want to keep ee_states.txt here, write out the empty array
        np.savetxt(os.path.join(folder_path, "ee_states.txt"), ee_states_array)
        
        print(f"[INFO] Episode {episode_num} processed with {joint_angles_array.shape[0]} steps.")

if __name__ == "__main__":
    main()
