import os
import numpy as np
import tensorflow_datasets as tfds

# GCS location for the prepared dataset
DATASET_GCS_PATH = "gs://gresearch/robotics/stanford_mask_vit_converted_externally_to_rlds/0.1.0"

import numpy as np
import transforms3d.euler as euler

def convert_5dof_to_7dof(end_effector_pose_5dof):
    """
    Convert an end-effector pose of shape (5,)
    [x, y, z, yaw, gripper] -> [x, y, z, qx, qy, qz, qw]
    assuming roll=pitch=0, yaw given.
    """
    # unpack the 5D pose
    x, y, z, yaw, gripper_state = end_effector_pose_5dof

    # Convert yaw -> quaternion (assuming roll=0, pitch=0)
    # transforms3d uses (roll, pitch, yaw) in radians for euler2quat 
    # with default axes='sxyz'
    qx, qy, qz, qw = euler.euler2quat(0.0, 0.0, yaw, axes='sxyz')

    # Combine position + quaternion into a 7D array
    ee_7dof = np.array([x, y, z, qx, qy, qz, qw], dtype=np.float32)
    return ee_7dof


def main():
    """
    1) Use builder_from_directory() with GCS path => builder
    2) Skip download_and_prepare()
    3) Call builder.as_dataset(split='train[:20]', download=False)
    4) Iterate episodes, extracting e.g. a 15D 'state'
    5) Save each episode's data
    """

    # 1) Build from the GCS directory (already prepared)
    builder = tfds.builder_from_directory(
        builder_dir=DATASET_GCS_PATH
    )

    # 2) Do NOT call builder.download_and_prepare(), 
    #    because we assume the data is fully prepared on GCS.

    # 3) Create a dataset from the first 20 episodes (or fewer if the dataset is smaller).
    ds = builder.as_dataset(
        split="train[:20]",
        shuffle_files=False
    )

    # 4) Iterate over each episode
    for episode_num, episode in enumerate(ds):
        print(f"[INFO] Episode {episode_num}")

        # Prepare lists for storing data
        joint_states_list = []
        gripper_states_list = []
        ee_7dof_list = []  # <-- List for storing 7-DOF end-effector poses

        # In the MaskViT example, each episode has a sub-dataset "steps"
        steps_dataset = episode["steps"]
        for step_idx, step in enumerate(steps_dataset):
            # For MaskViT: 'state' is shape (15,):
            #   [0:7] => joint angles
            #   [7:14] => joint velocities
            #   [14]   => gripper
            state_15 = step["observation"]["state"].numpy()
            joint_angles = state_15[0:7]
            gripper_pos  = state_15[14]

            joint_states_list.append(joint_angles)
            gripper_states_list.append([gripper_pos])  # store as 1D for easy stacking

            # Extract 5-DOF end effector pose: [x, y, z, yaw, gripper]
            end_effector_pose_5dof = step["observation"]["end_effector_pose"].numpy()
            # Convert to 7-DOF [x, y, z, qx, qy, qz, qw]
            ee_7dof = convert_5dof_to_7dof(end_effector_pose_5dof)
            ee_7dof_list.append(ee_7dof)

        # Convert to arrays
        joint_states_array   = np.array(joint_states_list)
        gripper_states_array = np.array(gripper_states_list)
        ee_7dof_array        = np.array(ee_7dof_list)

        # Make local folder
        folder_path = f"../states/stanford_mask_vit_converted_externally_to_rlds/episode_{episode_num}"
        os.makedirs(folder_path, exist_ok=True)

        # Save joint angles + gripper states
        np.savetxt(os.path.join(folder_path, "joint_states.txt"), joint_states_array)
        np.savetxt(os.path.join(folder_path, "gripper_states.txt"), gripper_states_array)

        # Save 7DOF end-effector poses
        np.savetxt(os.path.join(folder_path, "end_effector_7dof.txt"), ee_7dof_array)

        print(f"[INFO] Episode {episode_num} done. Steps: {len(joint_states_list)}")

if __name__ == "__main__":
    main()