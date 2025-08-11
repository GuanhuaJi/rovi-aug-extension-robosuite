import os
import numpy as np
import tensorflow_datasets as tfds
from PIL import Image

'''
austin_buds_dataset_converted_externally_to_rlds
FeaturesDict({
    'episode_metadata': FeaturesDict({
        'file_path': Text(shape=(), dtype=string),
    }),
    'steps': Dataset({
        'action': Tensor(shape=(7,), dtype=float32, description=Robot action, consists of [6x end effector delta pose, 1x gripper position].),
        'discount': Scalar(shape=(), dtype=float32, description=Discount if provided, default to 1.),
        'is_first': bool,
        'is_last': bool,
        'is_terminal': bool,
        'language_embedding': Tensor(shape=(512,), dtype=float32, description=Kona language embedding. See https://tfhub.dev/google/universal-sentence-encoder-large/5),
        'language_instruction': Text(shape=(), dtype=string),
        'observation': FeaturesDict({
            'image': Image(shape=(128, 128, 3), dtype=uint8, description=Main camera RGB observation.),
            'state': Tensor(shape=(24,), dtype=float32, description=Robot state, consists of [7x robot joint angles, 1x gripper position, 16x robot end-effector homogeneous matrix].),
            'wrist_image': Image(shape=(128, 128, 3), dtype=uint8, description=Wrist camera RGB observation.),
        }),
        'reward': Scalar(shape=(), dtype=float32, description=Reward if provided, 1 on final step for demos.),
    }),
})
'''

def dataset2path(dataset_name):
    # The GCS path for 'berkeley_cable_routing'
    return f'gs://gresearch/robotics/{dataset_name}/0.1.0'

DATASET = "austin_buds_dataset_converted_externally_to_rlds"

try:
    # Load the first 20 episodes from the train split
    builder = tfds.builder_from_directory(builder_dir=dataset2path(DATASET))

    ds = builder.as_dataset(split='train')

    # Iterate over each episode
    for episode_num, episode in enumerate(ds):
        # Used to store state info for each time step
        ee_states_list = []
        joint_states_list = []
        gripper_states_list = []
        language_instructions_list = []

        # Create storage folder for current episode
        episode_folder = os.path.join('../states', DATASET, f'episode_{episode_num}')
        os.makedirs(episode_folder, exist_ok=True)
        # Create subfolder for images
        images_folder = os.path.join(episode_folder, 'images')
        os.makedirs(images_folder, exist_ok=True)

        # Iterate over each step in the current episode
        for step_idx, step in enumerate(episode['steps']):
            # Extract state (24-dimensional) from observation
            state = step['observation']['state'].numpy()  # shape: (24,)
            # Split as described:
            # joint_states: first 7 values
            joint_state = state[:7]
            # gripper_state: 8th element (as array with shape (1,))
            gripper_state = state[7:8]
            # ee_states: last 16 values
            ee_state = state[8:24]

            joint_states_list.append(joint_state)
            gripper_states_list.append(gripper_state)
            ee_states_list.append(ee_state)

            # Extract main camera image, shape (128, 128, 3)
            image_np = step['observation']['image'].numpy()
            img = Image.fromarray(image_np)
            # Save image as JPEG, named 0.jpeg, 1.jpeg, ...
            image_filename = os.path.join(images_folder, f"{step_idx}.jpeg")
            img.save(image_filename, format='JPEG')

            language_instruction = step['language_instruction'].numpy().decode('utf-8')

        # Convert lists to numpy arrays and save to text files
        ee_states_array = np.vstack(ee_states_list)         # shape: (T, 16)
        joint_states_array = np.vstack(joint_states_list)     # shape: (T, 7)
        gripper_states_array = np.vstack(gripper_states_list) # shape: (T, 1)

        np.savetxt(os.path.join(episode_folder, 'ee_states.txt'), ee_states_array)
        np.savetxt(os.path.join(episode_folder, 'joint_states.txt'), joint_states_array)
        np.savetxt(os.path.join(episode_folder, 'gripper_states.txt'), gripper_states_array)
        np.savetxt(os.path.join(episode_folder, 'language_instruction.txt'), 
                   np.array(language_instructions_list, dtype=object), fmt='%s')

        print(f"Episode {episode_num} processed with {ee_states_array.shape[0]} steps.")

except Exception as e:
    print(f"Error processing dataset '{DATASET}': {e}")