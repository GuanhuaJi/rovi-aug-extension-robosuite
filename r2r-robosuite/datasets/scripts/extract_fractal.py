'''
fractal20220817_data
FeaturesDict({
    'aspects': FeaturesDict({
        'already_success': bool,
        'feasible': bool,
        'has_aspects': bool,
        'success': bool,
        'undesirable': bool,
    }),
    'attributes': FeaturesDict({
        'collection_mode': int64,
        'collection_mode_name': string,
        'data_type': int64,
        'data_type_name': string,
        'env': int64,
        'env_name': string,
        'location': int64,
        'location_name': string,
        'objects_family': int64,
        'objects_family_name': string,
        'task_family': int64,
        'task_family_name': string,
    }),
    'steps': Dataset({
        'action': FeaturesDict({
            'base_displacement_vector': Tensor(shape=(2,), dtype=float32),
            'base_displacement_vertical_rotation': Tensor(shape=(1,), dtype=float32),
            'gripper_closedness_action': Tensor(shape=(1,), dtype=float32, description=continuous gripper position),
            'rotation_delta': Tensor(shape=(3,), dtype=float32, description=rpy commanded orientation displacement, in base-relative frame),
            'terminate_episode': Tensor(shape=(3,), dtype=int32),
            'world_vector': Tensor(shape=(3,), dtype=float32, description=commanded end-effector displacement, in base-relative frame),
        }),
        'is_first': bool,
        'is_last': bool,
        'is_terminal': bool,
        'observation': FeaturesDict({
            'base_pose_tool_reached': Tensor(shape=(7,), dtype=float32, description=end-effector base-relative position+quaternion pose),
            'gripper_closed': Tensor(shape=(1,), dtype=float32),
            'gripper_closedness_commanded': Tensor(shape=(1,), dtype=float32, description=continuous gripper position),
            'height_to_bottom': Tensor(shape=(1,), dtype=float32, description=height of end-effector from ground),
            'image': Image(shape=(256, 320, 3), dtype=uint8),
            'natural_language_embedding': Tensor(shape=(512,), dtype=float32),
            'natural_language_instruction': string,
            'orientation_box': Tensor(shape=(2, 3), dtype=float32),
            'orientation_start': Tensor(shape=(4,), dtype=float32),
            'robot_orientation_positions_box': Tensor(shape=(3, 3), dtype=float32),
            'rotation_delta_to_go': Tensor(shape=(3,), dtype=float32, description=rotational displacement from current orientation to target),
            'src_rotation': Tensor(shape=(4,), dtype=float32),
            'vector_to_go': Tensor(shape=(3,), dtype=float32, description=displacement from current end-effector position to target),
            'workspace_bounds': Tensor(shape=(3, 3), dtype=float32),
        }),
        'reward': Scalar(shape=(), dtype=float32),
    }),
})
'''

import os
import numpy as np
import tensorflow_datasets as tfds
from PIL import Image

# Example GCS path or local directory for the dataset
DATASET_GCS_PATH = (
    "gs://gresearch/robotics/fractal20220817_data/0.1.0"
)

def main():
    # 1) Create the builder from a prepared directory (dataset is ready; no download needed)
    builder = tfds.builder_from_directory(builder_dir=DATASET_GCS_PATH)

    # 2) Load the first 20 episodes from the train split, without shuffling
    ds = builder.as_dataset(split="train[:200]", shuffle_files=False)

    # 3) Iterate over each episode
    for episode_num, episode in enumerate(ds):
        print(f"[INFO] Processing Episode {episode_num}")

        # Create folders for this episode
        folder_path = f"../states/fractal/episode_{episode_num}"
        os.makedirs(folder_path, exist_ok=True)

        # Subfolder for images
        images_folder = os.path.join(folder_path, "images")
        os.makedirs(images_folder, exist_ok=True)
        instruction_list = []

        # 4) Iterate over each step in the episode
        steps_dataset = episode["steps"]
        for step_idx, step in enumerate(steps_dataset):
            #if episode["episode_metadata"][f"has_image_{i}"].numpy():
            image_np = step["observation"][f"image"].numpy()
            instr = step["observation"]["natural_language_instruction"].numpy().decode("utf-8")
            instruction_list.append(instr)

            img = Image.fromarray(image_np)
            image_filename = os.path.join(images_folder, f"{step_idx}.jpeg")
            img.save(image_filename, format="JPEG")
        
        np.savetxt(os.path.join(folder_path, "language_instruction.txt"), np.array(instruction_list), fmt="%s")


        print(f"[INFO] Episode {episode_num}.")

if __name__ == "__main__":
    main()