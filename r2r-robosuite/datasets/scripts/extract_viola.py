import os
import numpy as np
import tensorflow_datasets as tfds
from PIL import Image

'''
viola
FeaturesDict({
    'steps': Dataset({
        'action': FeaturesDict({
            'gripper_closedness_action': float32,
            'rotation_delta': Tensor(shape=(3,), dtype=float32),
            'terminate_episode': float32,
            'world_vector': Tensor(shape=(3,), dtype=float32),
        }),
        'is_first': bool,
        'is_last': bool,
        'is_terminal': bool,
        'observation': FeaturesDict({
            'agentview_rgb': Image(shape=(224, 224, 3), dtype=uint8, description=RGB captured by workspace camera),
            'ee_states': Tensor(shape=(16,), dtype=float32, description=Pose of the end effector specified as a homogenous matrix.),
            'eye_in_hand_rgb': Image(shape=(224, 224, 3), dtype=uint8, description=RGB captured by in hand camera),
            'gripper_states': Tensor(shape=(1,), dtype=float32, description=gripper_states = 0 means the gripper is fully closed. The value represents the gripper width of Franka Panda Gripper.),
            'joint_states': Tensor(shape=(7,), dtype=float32, description=joint values),
            'natural_language_embedding': Tensor(shape=(512,), dtype=float32),
            'natural_language_instruction': string,
        }),
        'reward': Scalar(shape=(), dtype=float32),
    }),
})
'''

DATASET_GCS_PATH = "gs://gresearch/robotics/viola/0.1.0"

try:
    # 1. Use tfds.load to download viola dataset and load first 20 episodes of train split
    builder = tfds.builder_from_directory(builder_dir=DATASET_GCS_PATH)
    ds = builder.as_dataset(split="train", shuffle_files=False)

    # 2. Iterate over each episode (each episode contains a 'steps' sequence)
    for episode_num, episode in enumerate(ds):
        # Used to store state info for each step
        ee_states_list = []
        joint_states_list = []
        gripper_states_list = []
        language_instructions_list = []

        # 3. Create local folder for each episode
        folder_path = f'../states/viola/episode_{episode_num}'
        os.makedirs(folder_path, exist_ok=True)
        # Create subfolder for images
        agentview_folder = os.path.join(folder_path, 'images')
        os.makedirs(agentview_folder, exist_ok=True)

        # 4. Iterate over each step
        for step_idx, step in enumerate(episode['steps']):
            # Extract state information
            # ee_states: (16,), joint_states: (7,), gripper_states: (1,)
            ee_state = step['observation']['ee_states'].numpy()
            joint_state = step['observation']['joint_states'].numpy()
            gripper_state = step['observation']['gripper_states'].numpy()

            ee_states_list.append(ee_state)
            joint_states_list.append(joint_state)
            gripper_states_list.append(gripper_state)

            # 5. Read image and save
            # agentview_rgb and eye_in_hand_rgb both have shape=(224,224,3)
            agentview_img_np = step['observation']['agentview_rgb'].numpy()

            # Convert numpy array to PIL Image
            agentview_img = Image.fromarray(agentview_img_np)

            # Construct image filename (numbered by step)
            agentview_filename = os.path.join(agentview_folder, f'{step_idx}.jpeg')

            language_instruction = step['observation']['natural_language_instruction'].numpy().decode('utf-8')

            # Save image
            agentview_img.save(agentview_filename)

        # 6. Convert lists to numpy arrays and save as text files
        ee_states_array = np.vstack(ee_states_list)       # shape (T, 16)
        joint_states_array = np.vstack(joint_states_list)   # shape (T, 7)
        gripper_states_array = np.vstack(gripper_states_list)  # shape (T, 1)

        np.savetxt(os.path.join(folder_path, 'ee_states.txt'), ee_states_array)
        np.savetxt(os.path.join(folder_path, 'joint_states.txt'), joint_states_array)
        np.savetxt(os.path.join(folder_path, 'gripper_states.txt'), gripper_states_array)
        np.savetxt(os.path.join(folder_path, 'language_instruction.txt'), 
                   [language_instruction], fmt='%s')

        print(f"Episode {episode_num} extracted with {ee_states_array.shape[0]} steps.")

except Exception as e:
    print(f"Error processing dataset viola: {e}")
