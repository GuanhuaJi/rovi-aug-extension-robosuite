import os
import numpy as np
import tensorflow_datasets as tfds

def dataset2path(dataset_name):
    # Modify this if you have a different local path or GCS bucket
    return f'gs://gresearch/robotics/{dataset_name}/0.1.0'

DATASET = "stanford_kuka_multimodal_dataset_converted_externally_to_rlds"

try:
    # 1) Create the builder from the GCS (or local) directory
    builder = tfds.builder_from_directory(builder_dir=dataset2path(DATASET))
    print(f"Features for {DATASET}:")
    print(builder.info.features)

    # 2) Check if 'train' split exists
    if 'train' not in builder.info.splits:
        print("No 'train' split found. Exiting.")
    else:
        total_episodes = builder.info.splits['train'].num_examples
        print(f"Dataset {DATASET} has {total_episodes} episodes in 'train' split.")

        # Only load first 20 episodes if more are available (just an example)
        split = 'train[:20]' if total_episodes >= 20 else 'train'
        ds = builder.as_dataset(split=split)

        # 3) Iterate over each episode
        for episode_num, episode in enumerate(ds):
            joint_angles_list = []
            gripper_actions_list = []

            # 4) 'steps' is the time-step sequence for this episode
            for step_idx, step in enumerate(episode['steps']):
                # Joint angles from observation (shape = (7,))
                joint_pos = step['observation']['joint_pos'].numpy()
                joint_angles_list.append(joint_pos)

                # Action vector is shape = (4,) => [3x EEF position, 1x gripper open/close]
                action_vec = step['action'].numpy()
                # The last entry (index 3) is the gripper command
                gripper = action_vec[3]
                gripper_actions_list.append(gripper)

            # 5) Convert lists to arrays
            joint_angles_array = np.vstack(joint_angles_list)       # shape=(T, 7)
            gripper_array = np.array(gripper_actions_list).reshape(-1, 1)  # shape=(T,1)

            # 6) Create a folder for each episode
            folder_path = f'../states/{DATASET}/episode_{episode_num}'
            os.makedirs(folder_path, exist_ok=True)

            # 7) Save arrays to text
            np.savetxt(os.path.join(folder_path, 'joint_states.txt'), joint_angles_array)
            np.savetxt(os.path.join(folder_path, 'gripper_actions.txt'), gripper_array)

            print(f"Episode {episode_num} extracted: {joint_angles_array.shape[0]} steps.")

except Exception as e:
    print(f"Error processing dataset {DATASET}: {e}")