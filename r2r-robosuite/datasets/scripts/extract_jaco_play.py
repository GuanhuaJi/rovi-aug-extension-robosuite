import os
import numpy as np
import tensorflow_datasets as tfds

def dataset2path(dataset_name):
    # Return the GCS path for the jaco_play dataset
    return f'gs://gresearch/robotics/{dataset_name}/0.1.0'

# We'll only extract from jaco_play
DATASET = "jaco_play"

# Build the dataset from GCS
try:
    # 1) Create the DatasetBuilder for jaco_play
    b = tfds.builder_from_directory(builder_dir=dataset2path(DATASET))
    
    # 2) Print out the feature structure (optional debug)
    print(f"Features of {DATASET}:")
    print(b.info.features)
    
    # 3) Check we have a 'train' split
    if 'train' not in b.info.splits:
        print("No 'train' split found. Exiting.")
    else:
        # number of episodes in train
        total_episodes = b.info.splits['train'].num_examples
        print(f"Dataset {DATASET} has {total_episodes} episodes in 'train'.")

        # 4) If there are fewer than 20 episodes, take all; otherwise, take the first 20
        split = 'train[:20]' if total_episodes >= 20 else 'train'
        ds = b.as_dataset(split=split)

        # 5) Iterate over each episode in this split
        for episode_num, episode in enumerate(ds):
            # 'episode' is a dictionary with keys like 'steps'
            # 'steps' is a dataset of time steps

            # We'll collect joint_pos and ee_cartesian_pos for all steps
            joint_states_list = []
            ee_states_list = []

            for step_idx, step in enumerate(episode['steps']):
                # step['observation'] is the dict of observations at this time step

                # Joint states (joint_pos is shape=(8,))
                #   usually 6 for the arm + 2 for the gripper, or as specified
                joint_pos = step['observation']['joint_pos'].numpy()
                joint_states_list.append(joint_pos[:6])

                # End-effector states (end_effector_cartesian_pos is shape=(7,))
                #   3 for position + 4 for quaternion orientation
                ee_cartesian = step['observation']['end_effector_cartesian_pos'].numpy()
                ee_states_list.append(ee_cartesian)

            # Convert lists to NumPy arrays
            joint_states_array = np.vstack(joint_states_list)
            ee_states_array = np.vstack(ee_states_list)

            # 6) Make an output folder for this dataset & episode
            folder_path = f'../states/{DATASET}/episode_{episode_num}'
            os.makedirs(folder_path, exist_ok=True)
            
            # 7) Save the arrays to text files
            #   e.g. joint_states.txt and ee_states.txt
            np.savetxt(os.path.join(folder_path, 'joint_states.txt'), joint_states_array)
            np.savetxt(os.path.join(folder_path, 'ee_states.txt'), ee_states_array)

            print(f"Saved episode {episode_num} with {joint_states_array.shape[0]} steps.")

except Exception as e:
    print(f"Error processing dataset {DATASET}: {e}")