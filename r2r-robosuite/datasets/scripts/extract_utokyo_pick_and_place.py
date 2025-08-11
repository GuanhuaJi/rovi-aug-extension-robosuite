'''
FeaturesDict({
    'episode_metadata': FeaturesDict({
        'disclaimer': Text(shape=(), dtype=string),
        'file_path': Text(shape=(), dtype=string),
        'n_transitions': Scalar(shape=(), dtype=int32, description=Number of transitions in the episode.),
        'success': Scalar(shape=(), dtype=bool, description=True if the last state of an episode is a success state, False otherwise.),
        'success_labeled_by': Text(shape=(), dtype=string),
    }),
    'steps': Dataset({
        'action': Tensor(shape=(4,), dtype=float32, description=Robot action, consists of [3x gripper velocities,1x gripper open/close torque].),
        'discount': Scalar(shape=(), dtype=float32, description=Discount if provided, default to 1.),
        'is_first': bool,
        'is_last': bool,
        'is_terminal': bool,
        'language_embedding': Tensor(shape=(512,), dtype=float32, description=Kona language embedding. See https://tfhub.dev/google/universal-sentence-encoder-large/5),
        'language_instruction': Text(shape=(), dtype=string),
        'observation': FeaturesDict({
            'image': Image(shape=(224, 224, 3), dtype=uint8, description=Camera RGB observation.),
            'state': Tensor(shape=(7,), dtype=float32, description=Robot state, consists of [3x gripper position,3x gripper orientation, 1x finger distance].),
        }),
        'reward': Scalar(shape=(), dtype=float32, description=Reward if provided, 1 on final step for demos.),
    }),
})
'''

import os
import numpy as np
import tensorflow_datasets as tfds
from PIL import Image

# Example GCS path or local directory for the dataset
DATASET_GCS_PATH = (
    "gs://gresearch/robotics/utokyo_xarm_pick_and_place_converted_externally_to_rlds/0.1.0"
)

def main():
    # 1) Create the builder from a prepared directory (dataset is ready; no download needed)
    builder = tfds.builder_from_directory(builder_dir=DATASET_GCS_PATH)

    # 2) Load the first 20 episodes from the train split, without shuffling
    ds = builder.as_dataset(split="train", shuffle_files=False)

    # 3) Iterate over each episode
    for episode_num, episode in enumerate(ds):
        print(f"[INFO] Processing Episode {episode_num}")

        # Create folders for this episode
        folder_path = f"../states/utokyo_pick_and_place/episode_{episode_num}"
        os.makedirs(folder_path, exist_ok=True)

        # Subfolder for images
        images_folder = os.path.join(folder_path, "images")
        os.makedirs(images_folder, exist_ok=True)
        language_instructions = []

        # 4) Iterate over each step in the episode
        steps_dataset = episode["steps"]
        for step_idx, step in enumerate(steps_dataset):
            image_np = step["observation"]["image"].numpy()  # shape (480, 640, 3)
            img = Image.fromarray(image_np)
            image_filename = os.path.join(images_folder, f"{step_idx}.jpeg")
            img.save(image_filename, format="JPEG")
            language_instruction = step["language_instruction"].numpy().decode("utf-8")
            language_instructions.append(language_instruction)

        # Save language instructions to a text file
        np.savetxt(os.path.join(folder_path, "language_instruction.txt"), 
                   np.array(language_instructions), fmt="%s")
        print(f"[INFO] Episode {episode_num}.")

if __name__ == "__main__":
    main()