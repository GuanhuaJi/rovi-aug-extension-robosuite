import os
import numpy as np
import tensorflow_datasets as tfds

def dataset2path(dataset_name):
    # GCS path for stanford_hydra_dataset_converted_externally_to_rlds, version 0.1.0
    return f'gs://gresearch/robotics/{dataset_name}/0.1.0'

DATASET = "stanford_hydra_dataset_converted_externally_to_rlds"

try:
    # 1) Create the builder from the GCS directory
    builder = tfds.builder_from_directory(builder_dir=dataset2path(DATASET))
    print(f"Features for {DATASET}:")
    print(builder.info.features)

    # 2) Check if 'train' split exists
    if 'train' not in builder.info.splits:
        print("No 'train' split found. Exiting.")
    else:
        total_episodes = builder.info.splits['train'].num_examples
        print(f"Dataset {DATASET} has {total_episodes} episodes in 'train' split.")

        # 3) Only load first 20 episodes if more are available
        split = 'train[:20]' if total_episodes >= 20 else 'train'
        ds = builder.as_dataset(split=split)

        # 4) Iterate over each episode
        for episode_num, episode in enumerate(ds):
            ee_states_list = []
            joint_states_list = []
            gripper_states_list = []

            # 5) steps is the time-step sequence for this episode
            for step_idx, step in enumerate(episode['steps']):
                # The 'state' is shape=(27,), described as:
                # [0:3]   => EEF position (x,y,z)
                # [3:7]   => EEF orientation in quaternion (qx,qy,qz,qw)
                # [7:10]  => EEF orientation in euler angles (roll,pitch,yaw)
                # [10:17] => 7 robot joint angles
                # [17:24] => 7 robot joint velocities
                # [24:27] => 3 gripper states

                # We only extract the first 7 (EEF pos+quat) and the 7 joint angles
                state_27 = step['observation']['state'].numpy()  # shape=(27,)

                action_7 = step['action'].numpy()
                
                # EEF = [x,y,z,qx,qy,qz,qw]
                ee_pose = state_27[:7]  # first 7
                ee_states_list.append(ee_pose)

                # Joint angles = state[10:17]
                joint_angles = state_27[10:17]
                joint_states_list.append(joint_angles)

                gripper_control = action_7[6]
                gripper_states_list.append(gripper_control)

            # Convert lists to arrays
            ee_states_array = np.vstack(ee_states_list)      # shape=(T,7)
            joint_states_array = np.vstack(joint_states_list)  # shape=(T,7)
            gripper_states_array = np.vstack(gripper_states_list)

            # 6) Make a local folder for each episode
            folder_path = f'../states/{DATASET}/episode_{episode_num}'
            os.makedirs(folder_path, exist_ok=True)

            # 7) Save arrays to text
            np.savetxt(os.path.join(folder_path, 'ee_states.txt'), ee_states_array)
            np.savetxt(os.path.join(folder_path, 'joint_states.txt'), joint_states_array)
            np.savetxt(os.path.join(folder_path, 'gripper_states.txt'), gripper_states_array)

            print(f"Episode {episode_num} extracted: {ee_states_array.shape[0]} steps.")

except Exception as e:
    print(f"Error processing dataset {DATASET}: {e}")