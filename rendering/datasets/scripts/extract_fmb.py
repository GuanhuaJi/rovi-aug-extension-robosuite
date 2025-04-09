import os
import numpy as np
import tensorflow_datasets as tfds
from PIL import Image

# Name of the TFDS dataset
DATASET = "fmb"

try:
    # Load the first 20 episodes from the train split
    ds = tfds.load(DATASET, split='train[:20]')
    print(f"Dataset '{DATASET}' loaded. Processing first 20 episodes...")

    # Enumerate over each "episode"
    for episode_num, episode in enumerate(ds):

        # Prepare lists to store per-step data
        ee_positions_list = []
        joint_angles_list = []
        gripper_states_list = []

        # Create a folder for this episode
        episode_folder = os.path.join('../states', DATASET, f'episode_{episode_num}')
        os.makedirs(episode_folder, exist_ok=True)

        # Subfolder for images
        images_folder = os.path.join(episode_folder, 'images')
        os.makedirs(images_folder, exist_ok=True)

        # Iterate over each step in the episode
        for step_idx, step in enumerate(episode['steps']):
            # --- 1) End effector position ---
            eef_pose = step['observation']['eef_pose'].numpy()  # shape: (7,)
            # Typically, the first 3 elements are x, y, z position
            eef_position = eef_pose[:3]
            ee_positions_list.append(eef_position)

            # --- 2) Joint angles ---
            joint_angle = step['observation']['joint_pos'].numpy()  # shape: (7,)
            joint_angles_list.append(joint_angle)

            # --- 3) Gripper state ---
            gripper_state = step['observation']['state_gripper_pose'].numpy()  # shape: ()
            # Convert to shape (1,) so it stacks easily
            gripper_states_list.append([gripper_state])

            # --- 4) Image side_1 ---
            image_np = step['observation']['image_side_1'].numpy()  # shape: (256, 256, 3)
            img = Image.fromarray(image_np)
            image_filename = os.path.join(images_folder, f"{step_idx}.jpeg")
            img.save(image_filename, format='JPEG')

        # After collecting per-step data, convert to numpy arrays & save
        ee_positions_array = np.vstack(ee_positions_list)     # shape: (T, 3)
        joint_angles_array = np.vstack(joint_angles_list)     # shape: (T, 7)
        gripper_states_array = np.vstack(gripper_states_list) # shape: (T, 1)

        np.savetxt(os.path.join(episode_folder, 'ee_states.txt'), ee_positions_array)
        np.savetxt(os.path.join(episode_folder, 'joint_states.txt'), joint_angles_array)
        np.savetxt(os.path.join(episode_folder, 'gripper_states.txt'), gripper_states_array)

        print(f"Episode {episode_num} processed with {ee_positions_array.shape[0]} steps.")

except Exception as e:
    print(f"Error processing dataset '{DATASET}': {e}")