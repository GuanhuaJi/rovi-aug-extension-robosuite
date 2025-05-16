import os
import numpy as np
import tensorflow_datasets as tfds
from PIL import Image

'''
taco_play
FeaturesDict({
    'steps': Dataset({
        'action': FeaturesDict({
            'actions': Tensor(shape=(7,), dtype=float32, description=absolute desired values for gripper pose (first 6 dimensions are x, y, z, yaw, pitch, roll), last dimension is open_gripper (-1 is open gripper, 1 is close)),
            'rel_actions_gripper': Tensor(shape=(7,), dtype=float32, description=relative actions for gripper pose in the gripper camera frame (first 6 dimensions are x, y, z, yaw, pitch, roll), last dimension is open_gripper (-1 is open gripper, 1 is close)),
            'rel_actions_world': Tensor(shape=(7,), dtype=float32, description=relative actions for gripper pose in the robot base frame (first 6 dimensions are x, y, z, yaw, pitch, roll), last dimension is open_gripper (-1 is open gripper, 1 is close)),
            'terminate_episode': float32,
        }),
        'is_first': bool,
        'is_last': bool,
        'is_terminal': bool,
        'observation': FeaturesDict({
            'depth_gripper': Tensor(shape=(84, 84), dtype=float32),
            'depth_static': Tensor(shape=(150, 200), dtype=float32),
            'natural_language_embedding': Tensor(shape=(512,), dtype=float32),
            'natural_language_instruction': string,
            'rgb_gripper': Image(shape=(84, 84, 3), dtype=uint8),
            'rgb_static': Image(shape=(150, 200, 3), dtype=uint8, description=RGB static image of shape. (150, 200, 3). Subsampled from (200,200, 3) image.),
            'robot_obs': Tensor(shape=(15,), dtype=float32, description=EE position (3), EE orientation in euler angles (3), gripper width (1), joint positions (7), gripper action (1)),
            'structured_language_instruction': string,
        }),
        'reward': Scalar(shape=(), dtype=float32),
    }),
})
'''


# The GCS path for "taco_play" might be:
DATASET_PATH = "gs://gresearch/robotics/taco_play/0.1.0"

def main():
    # 1) Create the DatasetBuilder from directory
    builder = tfds.builder_from_directory(builder_dir=DATASET_PATH)
    
    # 2) Load the first 20 episodes from the 'train' split, unshuffled
    ds = builder.as_dataset(split="train", shuffle_files=False)

    # 3) Iterate over each episode
    for episode_num, episode in enumerate(ds):
        print(f"[INFO] Processing Episode {episode_num}")
        
        # Prepare lists for storing step-wise data
        joint_states_list = []    # shape => (T, 7)
        ee_states_list = []       # shape => (T, 6)  (3 pos + 3 euler orientation)
        gripper_states_list = []  # shape => (T, 2)  (width, action)
        actions_list = []       # shape => (T, 7)
        language_instructions_list = []  # shape => (T, 1)

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
            actions_list.append(step["action"]["actions"].numpy())
            language_instructions_list.append(step["observation"]["structured_language_instruction"].numpy().decode("utf-8"))
        
        # 5) Convert lists to numpy arrays and save as text
        ee_states_array = np.array(ee_states_list)        # shape: (T, 6)
        joint_states_array = np.array(joint_states_list)  # shape: (T, 7)
        gripper_states_array = np.array(gripper_states_list)  # shape: (T, 2)
        actions_array = np.array(actions_list)            # shape: (T, 7)
        language_instructions_array = np.array(language_instructions_list)
        
        np.savetxt(os.path.join(folder_path, "ee_states.txt"), ee_states_array)
        np.savetxt(os.path.join(folder_path, "joint_states.txt"), joint_states_array)
        np.savetxt(os.path.join(folder_path, "gripper_states.txt"), gripper_states_array)
        np.savetxt(os.path.join(folder_path, "actions.txt"), actions_array)
        np.savetxt(os.path.join(folder_path, "language_instruction.txt"), language_instructions_array, fmt="%s")
        
        print(f"[INFO] Episode {episode_num} processed with {ee_states_array.shape[0]} steps.")

if __name__ == "__main__":
    main()