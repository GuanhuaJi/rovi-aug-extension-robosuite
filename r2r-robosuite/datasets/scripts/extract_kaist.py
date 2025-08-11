'''
FeaturesDict({
    'episode_metadata': FeaturesDict({
        'file_path': Text(shape=(), dtype=string),
    }),
    'steps': Dataset({
        'action': Tensor(shape=(20,), dtype=float32, description=Robot action, consists of [3x end-effector position residual, 3x end-effector axis-angle residual, 7x robot joint k_p gain coefficient, 7x robot joint damping ratio coefficient].The action residuals are global, i.e. multiplied on theleft-hand side of the current end-effector state.),
        'discount': Scalar(shape=(), dtype=float32, description=Discount if provided, default to 1.),
        'is_first': bool,
        'is_last': bool,
        'is_terminal': bool,
        'language_embedding': Tensor(shape=(512,), dtype=float32, description=Kona language embedding. See https://tfhub.dev/google/universal-sentence-encoder-large/5),
        'language_instruction': Text(shape=(), dtype=string),
        'observation': FeaturesDict({
            'image': Image(shape=(480, 640, 3), dtype=uint8, description=Main camera RGB observation.),
            'partial_pointcloud': Tensor(shape=(512, 3), dtype=float32, description=Partial pointcloud observation),
            'state': Tensor(shape=(21,), dtype=float32, description=Robot state, consists of [joint_states, end_effector_pose].Joint states are 14-dimensional, formatted in the order of [q_0, w_0, q_1, w_0, ...].In other words,  joint positions and velocities are interleaved.The end-effector pose is 7-dimensional, formatted in the order of [position, quaternion].The quaternion is formatted in (x,y,z,w) order. The end-effector pose references the tool frame, in the center of the two fingers of the gripper.),
        }),
        'reward': Scalar(shape=(), dtype=float32, description=Reward if provided, 1 on final step for demos.),
    }),
})
'''

import os
import numpy as np
import tensorflow_datasets as tfds
from PIL import Image

# Example GCS/local path for the dataset
DATASET_GCS_PATH = (
    "gs://gresearch/robotics/kaist_nonprehensile_converted_externally_to_rlds/0.1.0"
)

def main():
    # 1) Build from directory (dataset already prepared)
    builder = tfds.builder_from_directory(builder_dir=DATASET_GCS_PATH)

    # 2) Load first 20 episodes from 'train' split, no shuffle
    ds = builder.as_dataset(split="train", shuffle_files=False)

    # 3) Iterate episodes
    for episode_num, episode in enumerate(ds):
        print(f"[INFO] Processing Episode {episode_num}")

        # Prepare lists
        joint_angles_list = []  # from obs_state indices [0,2,4,6,8,10,12]
        gripper_list = []       # no explicit gripper dimension, so we store an empty array
        ee_pose_list = []       # from obs_state[14:21]
        actions_list = []      # not used in this example
        language_instructions_list = []  # not used in this example
        
        # Create output folder
        folder_path = f"/home/guanhuaji/mirage/robot2robot/rendering/datasets/states/kaist_nonprehensile_rlds/episode_{episode_num}"
        os.makedirs(folder_path, exist_ok=True)

        # Images subfolder
        images_folder = os.path.join(folder_path, "images")
        os.makedirs(images_folder, exist_ok=True)

        # 4) Iterate steps
        steps_dataset = episode["steps"]
        for step_idx, step in enumerate(steps_dataset):
            # --- Observation ---
            # The 21-D state is [0..13] => 14 dims (joint pos/vel interleaved), [14..20] => 7-dim EE pose
            obs_state = step["observation"]["state"].numpy()
            
            # Joint angles come from indices [0,2,4,6,8,10,12] of the first 14
            selected_indices = [0, 2, 4, 6, 8, 10, 12]
            joint_angles = obs_state[selected_indices]  # shape (7,)

            # There's no explicit gripper dimension in the dataset
            gripper_state = np.array([], dtype=np.float32)

            # EE pose is the last 7 dims: [14:21]
            ee_pose = obs_state[14:]

            # Accumulate
            joint_angles_list.append(joint_angles)
            gripper_list.append(gripper_state)
            ee_pose_list.append(ee_pose)
            actions_list.append(step["action"].numpy())  # not used in this example
            language_instructions_list.append(step["language_instruction"].numpy().decode("utf-8"))  # not used in this example


            # --- Image extraction ---
            image_np = step["observation"]["image"].numpy()  # (480, 640, 3)
            img = Image.fromarray(image_np)
            image_filename = os.path.join(images_folder, f"{step_idx}.jpeg")
            img.save(image_filename, format="JPEG")

        # 5) Convert to numpy arrays
        joint_angles_array = np.array(joint_angles_list)  # shape (T, 7)
        gripper_array = np.array(gripper_list)            # shape (T, 0)
        ee_pose_array = np.array(ee_pose_list)            # shape (T, 7)
        actions_array = np.array(actions_list)            # shape (T, 20)
        language_instructions_array = np.array(language_instructions_list)

        # 6) Save as .txt
        np.savetxt(os.path.join(folder_path, "joint_states.txt"), joint_angles_array)
        np.savetxt(os.path.join(folder_path, "gripper_states.txt"), gripper_array)
        np.savetxt(os.path.join(folder_path, "ee_pose.txt"), ee_pose_array)
        np.savetxt(os.path.join(folder_path, "actions.txt"), actions_array)
        np.savetxt(os.path.join(folder_path, "language_instruction.txt"), language_instructions_array, fmt="%s")

        print(f"[INFO] Episode {episode_num} processed with {joint_angles_array.shape[0]} steps.")

if __name__ == "__main__":
    main()