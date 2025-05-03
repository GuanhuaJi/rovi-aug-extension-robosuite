import os
import numpy as np
import tensorflow_datasets as tfds
from PIL import Image

# The GCS path for "taco_play" might be:
DATASET_PATH = "gs://gresearch/robotics/taco_play/0.1.0"

def main():
    # 1) Create the DatasetBuilder from directory
    builder = tfds.builder_from_directory(builder_dir=DATASET_PATH)
    
    # 2) Load the first 20 episodes from the 'train' split, unshuffled
    ds = builder.as_dataset(split="train[:1000]", shuffle_files=False)

    # 3) Iterate over each episode
    for episode_num, episode in enumerate(ds):
        print(f"[INFO] Processing Episode {episode_num}")
        
        # Prepare lists for storing step-wise data
        joint_states_list = []    # shape => (T, 7)
        ee_states_list = []       # shape => (T, 6)  (3 pos + 3 euler orientation)
        gripper_states_list = []  # shape => (T, 2)  (width, action)

        # Define folder path for the current episode
        folder_path = f"../states/taco_play/episode_{episode_num}"
        os.makedirs(folder_path, exist_ok=True)
        
        # Create a subfolder to store images
        images_folder = os.path.join(folder_path, "images")
        os.makedirs(images_folder, exist_ok=True)
        
        # 4) Iterate over each step in this episode
        steps_dataset = episode["steps"]
        for step_idx, step in enumerate(steps_dataset):
            # ---- Images ----
            # You can also use 'rgb_gripper' if preferred
            # shape (150, 200, 3) for 'rgb_static', shape (84, 84, 3) for 'rgb_gripper'
            image_np = step["observation"]["rgb_static"].numpy()
            img = Image.fromarray(image_np)
            
            # Save image as JPEG: "images/0.jpeg", "images/1.jpeg", ...
            image_filename = os.path.join(images_folder, f"{step_idx}.jpeg")
            img.save(image_filename, format="JPEG")
            
            # ---- Robot Observations (robot_obs) ----
            # 'robot_obs' has length 15:
            #    [0:3] = end-effector position (x, y, z)
            #    [3:6] = end-effector orientation (yaw, pitch, roll)
            #    [6]   = gripper width
            #    [7:14] = 7 joint angles
            #    [14]  = gripper action
            robot_obs = step["observation"]["robot_obs"].numpy()
            
            # Extract pieces
            ee_pos_ori = robot_obs[0:6]       # shape (6,) => 3 pos + 3 euler angles
            gripper_width = robot_obs[6]      # scalar
            joint_angles = robot_obs[7:14]    # shape (7,)
            gripper_action = robot_obs[14]    # scalar
            
            # Append to time-series lists
            ee_states_list.append(ee_pos_ori)
            joint_states_list.append(joint_angles)
            gripper_states_list.append([gripper_action])
        
        # 5) Convert lists to numpy arrays and save as text
        ee_states_array = np.array(ee_states_list)        # shape: (T, 6)
        joint_states_array = np.array(joint_states_list)  # shape: (T, 7)
        gripper_states_array = np.array(gripper_states_list)  # shape: (T, 2)
        
        np.savetxt(os.path.join(folder_path, "ee_states.txt"), ee_states_array)
        np.savetxt(os.path.join(folder_path, "joint_states.txt"), joint_states_array)
        np.savetxt(os.path.join(folder_path, "gripper_states.txt"), gripper_states_array)
        
        print(f"[INFO] Episode {episode_num} processed with {ee_states_array.shape[0]} steps.")

if __name__ == "__main__":
    main()