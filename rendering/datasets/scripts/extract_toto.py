import os
import numpy as np
import tensorflow_datasets as tfds

'''
FeaturesDict({
    'steps': Dataset({
        'action': FeaturesDict({
            'open_gripper': bool,
            'rotation_delta': Tensor(shape=(3,), dtype=float32),
            'terminate_episode': float32,
            'world_vector': Tensor(shape=(3,), dtype=float32),
        }),
        'is_first': bool,
        'is_last': bool,
        'is_terminal': bool,
        'observation': FeaturesDict({
            'image': Image(shape=(480, 640, 3), dtype=uint8),
            'natural_language_embedding': Tensor(shape=(512,), dtype=float32),
            'natural_language_instruction': string,
            'state': Tensor(shape=(7,), dtype=float32, description=numpy array of shape (7,). Contains the robot joint states (as absolute joint angles) at each timestep),
        }),
        'reward': Scalar(shape=(), dtype=float32),
    }),
})

'''

# For saving images
from PIL import Image

def dataset2path(dataset_name):
    """
    If you have the dataset locally or elsewhere, change the returned path here.
    For example: return f'/path/to/{dataset_name}/0.1.0'
    or use a Google Cloud Storage path: 'gs://...'
    """
    return f'gs://gresearch/robotics/toto/0.1.0'

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
        total_episodes = builder.info.splits['train'].num_examples
        print(f"Dataset {DATASET} has {total_episodes} episodes in 'train' split.")

        # For example, just take the first 20 episodes
        split = 'train'
        ds = builder.as_dataset(split=split)

        # 3) Iterate over episodes
        for episode_num, episode in enumerate(ds):
            joint_angles_list = []
            gripper_status_list = []
            language_instructions = []

            # Create an output folder for this episode
            folder_path = f'/home/guanhuaji/mirage/robot2robot/rendering/datasets/states/{DATASET}/episode_{episode_num}'
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
                # image_np = step['observation']['image'].numpy()
                # language_instruction = step['observation']['natural_language_instruction'].numpy().decode('utf-8')
                # image_pil = Image.fromarray(image_np)
                # image_filename = os.path.join(folder_path, f'images/{step_idx}.jpeg')
                # image_pil.save(image_filename)

            # 5) Convert joint angles and gripper statuses to numpy arrays
            joint_angles_array = np.vstack(joint_angles_list)       # shape=(T, 7)
            gripper_array = np.array(gripper_status_list).reshape(-1, 1)  # shape=(T,1)

            # 6) Save them as text
            np.savetxt(os.path.join(folder_path, 'joint_states.txt'), joint_angles_array)
            np.savetxt(os.path.join(folder_path, 'gripper_states.txt'), gripper_array, fmt='%d')
            np.savetxt(os.path.join(folder_path, 'language_instructions.txt'), 
                       np.array(language_instructions), fmt='%s')

            print(f"Episode {episode_num} extracted: {joint_angles_array.shape[0]} steps.")

except Exception as e:
    print(f"Error processing dataset {DATASET}: {e}")