'''
FeaturesDict({
    'episode_metadata': FeaturesDict({
        'file_path': Text(shape=(), dtype=string),
    }),
    'steps': Dataset({
        'action': Tensor(shape=(7,), dtype=float32, description=Robot action, consists of [6x end effector delta pose, 1x gripper position]),
        'discount': Scalar(shape=(), dtype=float32, description=Discount if provided, default to 1.),
        'is_first': bool,
        'is_last': bool,
        'is_terminal': bool,
        'language_embedding': Tensor(shape=(512,), dtype=float32, description=Kona language embedding. See https://tfhub.dev/google/universal-sentence-encoder-large/5),
        'language_instruction': Text(shape=(), dtype=string),
        'observation': FeaturesDict({
            'image': Image(shape=(128, 128, 3), dtype=uint8, description=Main camera RGB observation.),
            'state': Tensor(shape=(24,), dtype=float32, description=Robot state, consists of [7x robot joint angles, 1x gripper position, 16x robot end-effector homogeneous matrix].),
            'wrist_image': Image(shape=(128, 128, 3), dtype=uint8, description=Wrist camera RGB observation.),
        }),
        'reward': Scalar(shape=(), dtype=float32, description=Reward if provided, 1 on final step for demos.),
    }),
})

'''

import os
import numpy as np
import tensorflow_datasets as tfds
from PIL import Image

# Example GCS path or local directory for the dataset
DATASET_GCS_PATH = "gs://gresearch/robotics/utaustin_mutex/0.1.0"

def main():
    # 1) Create builder from specified directory (dataset ready; no need to download_and_prepare())
    builder = tfds.builder_from_directory(builder_dir=DATASET_GCS_PATH)
    
    # 2) Load first 20 episodes from train split without shuffling file order
    ds = builder.as_dataset(split="train", shuffle_files=False)
    
    # 3) Iterate over each episode
    for episode_num, episode in enumerate(ds):
        print(f"[INFO] Processing Episode {episode_num}")
        
        # Initialize lists to store data for each step
        joint_angles_list = []    # taken from the first 7 elements of observation/state
        gripper_state_list = []   # taken from the 8th element of observation/state (stored as array)
        ee_states_list = []       # taken from the last 16 elements of observation/state
        language_instructions = []  # store language instructions
        
        # Set storage path for current episode
        folder_path = f"../states/utaustin_mutex/episode_{episode_num}"
        os.makedirs(folder_path, exist_ok=True)
        
        # Create folder for images
        images_folder = os.path.join(folder_path, "images")
        os.makedirs(images_folder, exist_ok=True)
        
        # 4) Iterate over each step in the current episode
        steps_dataset = episode["steps"]
        for step_idx, step in enumerate(steps_dataset):
            # Extract state (24,) from observation
            # Meaning: [0:7] -> joint angles, [7] -> gripper state, [8:24] -> end-effector state
            state = step["observation"]["state"].numpy()
            language_instruction = step["language_instruction"].numpy().decode('utf-8')
            joint_angles = state[:7]
            gripper_state = state[7:8]  # keep as (1,) array
            ee_state = state[8:24]
            
            joint_angles_list.append(joint_angles)
            gripper_state_list.append(gripper_state)
            ee_states_list.append(ee_state)
            language_instructions.append(language_instruction)
            
            # 5) Extract main camera image (128, 128, 3)
            image_np = step["observation"]["image"].numpy()
            img = Image.fromarray(image_np)
            # Save as JPEG; filenames numbered by step: "0.jpeg", "1.jpeg", ...
            image_filename = os.path.join(images_folder, f"{step_idx}.jpeg")
            img.save(image_filename, format="JPEG")
        
        # 6) Convert lists to numpy arrays and save as text files
        joint_angles_array = np.array(joint_angles_list)     # shape: (T, 7)
        gripper_state_array = np.array(gripper_state_list)     # shape: (T, 1)
        ee_states_array = np.array(ee_states_list)             # shape: (T, 16)
        
        np.savetxt(os.path.join(folder_path, "joint_states.txt"), joint_angles_array)
        np.savetxt(os.path.join(folder_path, "gripper_states.txt"), gripper_state_array)
        np.savetxt(os.path.join(folder_path, "ee_states.txt"), ee_states_array)
        np.savetxt(os.path.join(folder_path, "language_instructions.txt"), 
                   np.array(language_instructions), fmt="%s")
        
        print(f"[INFO] Episode {episode_num} processed with {joint_angles_array.shape[0]} steps.")

if __name__ == "__main__":
    main()