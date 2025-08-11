import os
import numpy as np
import tensorflow_datasets as tfds
from PIL import Image

'''
FeaturesDict({
    'steps': Dataset({
        'action': FeaturesDict({
            'gripper_closedness_action': float32,
            'rotation_delta': Tensor(shape=(3,), dtype=float32, description=Delta change in roll, pitch, yaw.),
            'terminate_episode': float32,
            'world_vector': Tensor(shape=(3,), dtype=float32, description=Delta change in XYZ.),
        }),
        'is_first': bool,
        'is_last': bool,
        'is_terminal': bool,
        'observation': FeaturesDict({
            'hand_image': Image(shape=(480, 640, 3), dtype=uint8),
            'image': Image(shape=(480, 640, 3), dtype=uint8),
            'image_with_depth': Image(shape=(480, 640, 1), dtype=float32),
            'natural_language_embedding': Tensor(shape=(512,), dtype=float32),
            'natural_language_instruction': string,
            'robot_state': Tensor(shape=(15,), dtype=float32, description=Explanation of the robot state can be found at https://sites.google.com/corp/view/berkeley-ur5),
        }),
        'reward': Scalar(shape=(), dtype=float32),
    }),
})
'''

# Specify dataset path (GCS or local directory)
DATASET_GCS_PATH = "gs://gresearch/robotics/berkeley_autolab_ur5/0.1.0"

def main():
    # 1) Create builder from specified directory (dataset already prepared; no need to download_and_prepare())
    builder = tfds.builder_from_directory(builder_dir=DATASET_GCS_PATH)
    
    # 2) Load first 20 episodes from train split (keep file order)
    ds = builder.as_dataset(split="train", shuffle_files=False)
    
    # 3) Iterate over each episode
    for episode_num, episode in enumerate(ds):
        print(f"[INFO] Processing Episode {episode_num}")
        
        # Initialize lists to store data for each step
        joint_states_list = []    # joint angles: take first 6 elements of robot_state
        gripper_states_list = []  # gripper state: take 7th element of robot_state (keep as (1,) array)
        ee_states_list = []       # end-effector state: take last 8 elements of robot_state
        actions_list = []        # action: take first 7 elements of action
        language_instructions_list = []  # language instruction: from observation's natural_language_instruction
        
        # Set storage path for current episode
        folder_path = f"../states/autolab_ur5/episode_{episode_num}"
        os.makedirs(folder_path, exist_ok=True)
        
        # Create subfolder for images
        images_folder = os.path.join(folder_path, "images")
        os.makedirs(images_folder, exist_ok=True)
        
        # 4) Iterate over each step in the current episode
        steps_dataset = episode["steps"]
        for step_idx, step in enumerate(steps_dataset):
            # Extract robot_state (15,) from observation
            state = step["observation"]["robot_state"].numpy()
            # Assume robot_state structure: [0:6] -> joint angles, [6:7] -> gripper state, [7:15] -> end-effector state
            joint_state = state[:6]
            ee_state = state[6:13]
            gripper_state = state[13:14]

            
            joint_states_list.append(joint_state)
            gripper_states_list.append(gripper_state)
            ee_states_list.append(ee_state)
            # I want word_vector + rotation_delta + gripper_closedness_action
            # Take first 7 elements of action
            action = step["action"]["world_vector"].numpy()
            # Take gripper_closedness_action
            gripper_closedness_action = [step["action"]["gripper_closedness_action"].numpy()]
            # Take rotation_delta
            rotation_delta = step["action"]["rotation_delta"].numpy()
            # Combine action, gripper_closedness_action and rotation_delta into one array
            action_combined = np.concatenate((action, rotation_delta, gripper_closedness_action), axis=0)
            actions_list.append(action_combined)
            language_instructions_list.append(step["observation"]["natural_language_instruction"].numpy().decode("utf-8"))
            
            # 5) Extract main camera image (480, 640, 3)
            image_np = step["observation"]["image"].numpy()
            img = Image.fromarray(image_np)
            # Save as JPEG, filename "0.jpeg", "1.jpeg", ... (no leading zeros)
            image_filename = os.path.join(images_folder, f"{step_idx}.jpeg")
            img.save(image_filename, format="JPEG")
        
        # 6) Convert lists to numpy arrays and save as text files
        joint_states_array = np.vstack(joint_states_list)      # shape: (T, 6)
        gripper_states_array = np.vstack(gripper_states_list)  # shape: (T, 1)
        ee_states_array = np.vstack(ee_states_list)              # shape: (T, 8)
        
        np.savetxt(os.path.join(folder_path, "joint_states.txt"), joint_states_array)
        np.savetxt(os.path.join(folder_path, "gripper_states.txt"), gripper_states_array)
        np.savetxt(os.path.join(folder_path, "ee_states.txt"), ee_states_array)
        np.savetxt(os.path.join(folder_path, "actions.txt"), np.vstack(actions_list))
        np.savetxt(os.path.join(folder_path, "language_instruction.txt"), np.array(language_instructions_list), fmt="%s")
        
        print(f"[INFO] Episode {episode_num} processed with {joint_states_array.shape[0]} steps.")

if __name__ == "__main__":
    main()
