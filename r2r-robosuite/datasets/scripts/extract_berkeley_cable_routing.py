import os
import numpy as np
import tensorflow_datasets as tfds

def dataset2path(dataset_name):
    # The GCS path for 'berkeley_cable_routing'
    return f'gs://gresearch/robotics/{dataset_name}/0.1.0'

DATASET = "berkeley_cable_routing"

try:
    # 1) Create the DatasetBuilder from GCS
    builder = tfds.builder_from_directory(builder_dir=dataset2path(DATASET))
    print(f"Features of {DATASET}:")
    print(builder.info.features)

    # 2) Check if there's a 'train' split
    if 'train' not in builder.info.splits:
        print("No 'train' split found. Exiting.")
    else:
        total_episodes = builder.info.splits['train'].num_examples
        print(f"Dataset {DATASET} has {total_episodes} episodes in 'train'.")

        # 3) We'll take the first 20 episodes if available
        split = 'train[:20]' if total_episodes >= 20 else 'train'
        ds = builder.as_dataset(split=split)

        for episode_num, episode in enumerate(ds):
            # We'll collect 'robot_state' for each step
            robot_states_list = []

            # 4) Each 'episode' has a 'steps' dataset
            for step_idx, step in enumerate(episode['steps']):
                # Is 'robot_state' in the observation?
                if 'robot_state' in step['observation']:
                    # shape=(7,)
                    robot_state_7 = step['observation']['robot_state'].numpy()
                    robot_states_list.append(robot_state_7)
                else:
                    # If not found, skip or break
                    pass

            # 5) Save the stacked robot_state array
            if robot_states_list:
                robot_state_array = np.vstack(robot_states_list)
                folder_path = f'../states/{DATASET}/episode_{episode_num}'
                os.makedirs(folder_path, exist_ok=True)

                # Save as text
                # If you confirm these are joint angles, rename e.g. "joint_states.txt"
                # If you confirm these are eef pose, rename e.g. "ee_states.txt"
                np.savetxt(os.path.join(folder_path, 'end_effector_cartesian_pos.txt'), robot_state_array)
                
                print(f"Saved episode {episode_num}, shape={robot_state_array.shape}.")

except Exception as e:
    print(f"Error processing dataset {DATASET}: {e}")