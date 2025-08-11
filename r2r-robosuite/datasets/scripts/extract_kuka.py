import os
import numpy as np
import tensorflow_datasets as tfds

def dataset2path(dataset_name):
    # If stored on GCS or local, adjust this function to return the dataset directory
    return f'gs://gresearch/robotics/{dataset_name}/0.1.0'

DATASET = "kuka"  # or your dataset folder name

try:
    # 1) Create the builder from the directory (local or GCS).
    builder = tfds.builder_from_directory(builder_dir=dataset2path(DATASET))
    print(f"Features for {DATASET}:")
    print(builder.info.features)

    # 2) Check if 'train' split exists
    if 'train' not in builder.info.splits:
        print("No 'train' split found. Exiting.")
    else:
        total_episodes = builder.info.splits['train'].num_examples
        print(f"Dataset {DATASET} has {total_episodes} episodes in 'train' split.")

        # Only load first 20 episodes if more are available
        split = 'train[:20]' if total_episodes >= 20 else 'train'
        ds = builder.as_dataset(split=split)

        # 3) Iterate over each episode
        for episode_num, episode in enumerate(ds):
            eef_states_list = []      # will store [x,y,z,qx,qy,qz,qw] per step
            gripper_list = []         # will store gripper info per step

            # 4) steps is the time-step sequence for this episode
            for step_idx, step in enumerate(episode['steps']):
                # EEF pose from observation
                # shape = (7,) -> (x, y, z, qx, qy, qz, qw)
                eef_pose = step['observation']['clip_function_input/base_pose_tool_reached'].numpy()
                eef_states_list.append(eef_pose)

                # Option A: If you want the *action* that controls the gripper:
                grip_action = step['action']['gripper_closedness_action'].numpy()
                # shape = (1,). You could store as a scalar
                gripper_list.append(grip_action)

                # Option B: If you want the *current observed* state (open vs. closed):
                # grip_closed_obs = step['observation']['gripper_closed'].numpy()
                # gripper_list.append(grip_closed_obs)

            # 5) Convert lists to arrays
            eef_states_array = np.vstack(eef_states_list)   # shape=(T, 7)
            gripper_array = np.vstack(gripper_list)         # shape=(T, 1)

            # 6) Make a local folder for each episode
            folder_path = f'../states/{DATASET}/episode_{episode_num}'
            os.makedirs(folder_path, exist_ok=True)

            # 7) Save arrays to text
            np.savetxt(os.path.join(folder_path, 'ee_states.txt'), eef_states_array)
            np.savetxt(os.path.join(folder_path, 'gripper_states.txt'), gripper_array)

            print(f"Episode {episode_num} extracted: {eef_states_array.shape[0]} steps.")

except Exception as e:
    print(f"Error processing dataset {DATASET}: {e}")