import os
import numpy as np
import tensorflow_datasets as tfds
from PIL import Image

# Example GCS path or local directory for the dataset
DATASET_GCS_PATH = (
    "gs://gresearch/robotics/ucsd_kitchen_dataset_converted_externally_to_rlds/0.1.0"
)
'''
FeaturesDict({
    'episode_metadata': FeaturesDict({
        'file_path': Text(shape=(), dtype=string),
    }),
    'steps': Dataset({
        'action': Tensor(shape=(8,), dtype=float32, description=8-dimensional action, consisting of end-effector position and orientation, gripper open/close and a episode termination action.),
        'discount': Scalar(shape=(), dtype=float32, description=Discount if provided, default to 1.),
        'is_first': bool,
        'is_last': bool,
        'is_terminal': bool,
        'language_embedding': Tensor(shape=(512,), dtype=float32, description=Kona language embedding. See https://tfhub.dev/google/universal-sentence-encoder-large/5),
        'language_instruction': Text(shape=(), dtype=string),
        'observation': FeaturesDict({
            'image': Image(shape=(480, 640, 3), dtype=uint8, description=Main camera RGB observation.),
            'state': Tensor(shape=(21,), dtype=float32, description=21-dimensional joint states, consists of robot joint angles, joint velocity and joint torque.),
        }),
        'reward': Scalar(shape=(), dtype=float32, description=Reward if provided, 1 on final step for demos.),
    }),
})

'''
def main():
    # 1) Create the builder from a prepared directory (dataset is ready; no download needed)
    builder = tfds.builder_from_directory(builder_dir=DATASET_GCS_PATH)

    # 2) Load the first 20 episodes from the train split, without shuffling
    ds = builder.as_dataset(split="train", shuffle_files=False)

    # 3) Iterate over each episode
    for episode_num, episode in enumerate(ds):
        print(f"[INFO] Processing Episode {episode_num}")

        # Prepare lists for storing step-wise data
        joint_states_list = []   # from observation['state'] (we’ll take the first 7 dims, i.e. joint angles)
        ee_poses_list = []       # from action[:7] (EE position + orientation)
        gripper_list = []        # from action[7] (gripper open/close)
        language_instructions = []  # to store language instructions

        # Create folders for this episode
        folder_path = f"../states/ucsd_kitchen_rlds/episode_{episode_num}"
        os.makedirs(folder_path, exist_ok=True)

        # Subfolder for images
        images_folder = os.path.join(folder_path, "images")
        os.makedirs(images_folder, exist_ok=True)

        # 4) Iterate over each step in the episode
        steps_dataset = episode["steps"]
        for step_idx, step in enumerate(steps_dataset):
            # --- Extract from observation ---
            # observation['state'] is 21-D: [joint angles (7), joint velocities (7), joint torques (7)]
            # For this example, assume you only need the 7 joint angles:
            obs_state = step["observation"]["state"].numpy()
            language_instruction = step["language_instruction"].numpy().decode('utf-8')
            joint_angles = obs_state[:7]  # shape (7,)

            # --- Extract from action ---
            # action is 8-D: [EE pos & orientation (7), gripper (1), possibly termination]
            # We’ll assume:
            #   action[:7] = End-effector position + orientation
            #   action[7]  = Gripper open/close
            act = step["action"].numpy()
            ee_pose = act[:7]      # shape (7,)
            gripper = act[7:8]     # shape (1,)

            joint_states_list.append(joint_angles)
            ee_poses_list.append(ee_pose)
            gripper_list.append(gripper)

            # --- Extract and save the image ---
            image_np = step["observation"]["image"].numpy()  # shape (480, 640, 3)
            img = Image.fromarray(image_np)
            image_filename = os.path.join(images_folder, f"{step_idx}.jpeg")
            img.save(image_filename, format="JPEG")

        # 5) Convert each list to a numpy array and save to text
        joint_states_array = np.array(joint_states_list)  # shape (T, 7)
        ee_poses_array = np.array(ee_poses_list)          # shape (T, 7)
        gripper_array = np.array(gripper_list)            # shape (T, 1)

        np.savetxt(os.path.join(folder_path, "joint_states.txt"), joint_states_array)
        np.savetxt(os.path.join(folder_path, "ee_states.txt"), ee_poses_array)
        np.savetxt(os.path.join(folder_path, "gripper_states.txt"), gripper_array)
        np.savetxt(os.path.join(folder_path, "language_instruction.txt"), 
                   np.array(language_instructions), fmt="%s")

        print(f"[INFO] Episode {episode_num} processed with {joint_states_array.shape[0]} steps.")

if __name__ == "__main__":
    main()