import os
import numpy as np
import tensorflow_datasets as tfds

# For saving images
from PIL import Image

def dataset2path(dataset_name):
    """
    If you have the dataset locally or elsewhere, change the returned path here.
    For example: return f'/path/to/{dataset_name}/0.1.0'
    or use a Google Cloud Storage path: 'gs://...'
    """
    return f'gs://gresearch/robotics/{dataset_name}/0.1.0'

DATASET = "toto"  # The dataset name

try:
    # 1) Create TFDS builder from directory or GCS path
    builder = tfds.builder_from_directory(builder_dir=dataset2path(DATASET))
    print(f"Features for {DATASET}:")
    print(builder.info.features)

    # 2) Check if there is a 'train' split
    if 'train' not in builder.info.splits:
        print("No 'train' split found. Exiting.")
    else:
        total_episodes = builder.info.splits['train[546:]'].num_examples
        print(f"Dataset {DATASET} has {total_episodes} episodes in 'train' split.")

        # For example, just take the first 20 episodes
        split = 'train'
        ds = builder.as_dataset(split=split)

        # 3) Iterate over episodes
        for episode_num, episode in enumerate(ds):
            joint_angles_list = []
            gripper_status_list = []

            # Create an output folder for this episode
            folder_path = f'../states/{DATASET}/episode_{episode_num + 546}'
            os.makedirs(folder_path, exist_ok=True)
            images_folder = os.path.join(folder_path, "images")
            os.makedirs(images_folder, exist_ok=True)

            # 4) Step through each frame (step) in the episode
            for step_idx, step in enumerate(episode['steps']):
                # --- (A) Extract joint angles (7D)
                joint_pos = step['observation']['state'].numpy()  # shape=(7,)
                joint_angles_list.append(joint_pos)

                # --- (B) Extract gripper status (bool: True => open, False => close)
                gripper_open = step['action']['open_gripper'].numpy()
                gripper_status_list.append(int(gripper_open))

                # --- (C) Extract and save the image
                # step['observation']['image'] is a tf.Tensor of shape (480, 640, 3)
                image_np = step['observation']['image'].numpy()
                image_pil = Image.fromarray(image_np)
                image_filename = os.path.join(folder_path, f'images/{step_idx}.jpeg')
                image_pil.save(image_filename)

            # 5) Convert joint angles and gripper statuses to numpy arrays
            joint_angles_array = np.vstack(joint_angles_list)       # shape=(T, 7)
            gripper_array = np.array(gripper_status_list).reshape(-1, 1)  # shape=(T,1)

            # 6) Save them as text
            np.savetxt(os.path.join(folder_path, 'joint_states.txt'), joint_angles_array)
            np.savetxt(os.path.join(folder_path, 'gripper_states.txt'), gripper_array, fmt='%d')

            print(f"Episode {episode_num + 546} extracted: {joint_angles_array.shape[0]} steps.")

except Exception as e:
    print(f"Error processing dataset {DATASET}: {e}")