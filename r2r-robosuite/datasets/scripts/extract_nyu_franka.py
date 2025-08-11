import os
import numpy as np
import tensorflow_datasets as tfds
from PIL import Image

'''
FeaturesDict({
    'episode_metadata': FeaturesDict({
        'file_path': Text(shape=(), dtype=string),
    }),
    'steps': Dataset({
        'action': Tensor(shape=(15,), dtype=float32, description=Robot action, consists of [7x joint velocities, 3x EE delta xyz, 3x EE delta rpy, 1x gripper position, 1x terminate episode].),
        'discount': Scalar(shape=(), dtype=float32, description=Discount if provided, default to 1.),
        'is_first': bool,
        'is_last': bool,
        'is_terminal': bool,
        'language_embedding': Tensor(shape=(512,), dtype=float32, description=Kona language embedding. See https://tfhub.dev/google/universal-sentence-encoder-large/5),
        'language_instruction': Text(shape=(), dtype=string),
        'observation': FeaturesDict({
            'depth': Tensor(shape=(128, 128, 1), dtype=int32, description=Right camera depth observation.),
            'depth_additional_view': Tensor(shape=(128, 128, 1), dtype=int32, description=Left camera depth observation.),
            'image': Image(shape=(128, 128, 3), dtype=uint8, description=Right camera RGB observation.),
            'image_additional_view': Image(shape=(128, 128, 3), dtype=uint8, description=Left camera RGB observation.),
            'state': Tensor(shape=(13,), dtype=float32, description=Robot state, consists of [7x robot joint angles, 3x EE xyz, 3x EE rpy.),
        }),
        'reward': Scalar(shape=(), dtype=float32, description=Reward if provided, 1 on final step for demos.),
    }),
})
'''

# Specify dataset path (GCS or local directory)
DATASET_GCS_PATH = "gs://gresearch/robotics/nyu_franka_play_dataset_converted_externally_to_rlds/0.1.0"

def main():
    # 1) Create builder from specified directory (dataset already prepared; no need to download_and_prepare())
    builder = tfds.builder_from_directory(builder_dir=DATASET_GCS_PATH)
    
    # 2) Load first 20 episodes from train split without shuffling file order
    split_name = "train"
    ds = builder.as_dataset(split=split_name, shuffle_files=False)
    
    # 3) Iterate over each episode
    for episode_num, episode in enumerate(ds):
        print(f"[INFO] Processing Episode {episode_num}")
        
        # Initialize lists to store data for each step
        joint_states_list = []    # 7-dof joint angles (observation/state[0:7])
        ee_states_list = []       # End-effector state (observation/state[7:13], 6-dim)
        gripper_states_list = []  # Gripper state (extracted from action, index 13)
        language_instructions = []  # used to store language instructions
        
        # Set storage path for current episode
        folder_path = f"../states/nyu_franka/{split_name}_episode_{episode_num}"
        os.makedirs(folder_path, exist_ok=True)
        
        # Create folder for images
        images_folder = os.path.join(folder_path, "images")
        os.makedirs(images_folder, exist_ok=True)
        
        # 4) Iterate over each step in the current episode
        steps_dataset = episode["steps"]
        for step_idx, step in enumerate(steps_dataset):
            # Extract state (shape: (13,)) from observation
            state = step["observation"]["state"].numpy()
            joint_state = state[:7]     # first 7 values
            ee_state = state[7:13]      # last 6 values
            
            # Extract gripper state from action (action has shape (15,))
            action = step["action"].numpy()
            gripper_state = action[13:14]  # take 14th element, keep as array
            
            joint_states_list.append(joint_state)
            ee_states_list.append(ee_state)
            gripper_states_list.append(gripper_state)
            
            # 5) Extract main camera image (128x128x3)
            image_np = step["observation"]["image"].numpy()
            img = Image.fromarray(image_np)
            # Save as JPEG, filename format "0.jpeg", "1.jpeg", …
            image_filename = os.path.join(images_folder, f"{step_idx}.jpeg")
            img.save(image_filename, format="JPEG")

            language_instruction = step["language_instruction"].numpy().decode('utf-8')
            language_instructions.append(language_instruction)
        
        # 6) Convert lists to numpy arrays and save as text files
        joint_states_array = np.array(joint_states_list)     # shape: (T, 7)
        ee_states_array = np.array(ee_states_list)             # shape: (T, 6)
        gripper_states_array = np.array(gripper_states_list)   # shape: (T, 1)
        
        np.savetxt(os.path.join(folder_path, "joint_states.txt"), joint_states_array)
        np.savetxt(os.path.join(folder_path, "ee_states.txt"), ee_states_array)
        np.savetxt(os.path.join(folder_path, "gripper_states.txt"), gripper_states_array)
        np.savetxt(os.path.join(folder_path, "language_instruction.txt"), 
                   np.array(language_instructions), fmt="%s")

        
        print(f"[INFO] Episode {episode_num} processed with {joint_states_array.shape[0]} steps.")

if __name__ == "__main__":
    main()
