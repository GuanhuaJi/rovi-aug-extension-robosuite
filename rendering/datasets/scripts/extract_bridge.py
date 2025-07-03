# '''
# FeaturesDict({
#     'steps': Dataset({
#         'action': FeaturesDict({
#             'open_gripper': bool,
#             'rotation_delta': Tensor(shape=(3,), dtype=float32),
#             'terminate_episode': float32,
#             'world_vector': Tensor(shape=(3,), dtype=float32),
#         }),
#         'is_first': bool,
#         'is_last': bool,
#         'is_terminal': bool,
#         'observation': FeaturesDict({
#             'image': Image(shape=(480, 640, 3), dtype=uint8),
#             'natural_language_embedding': Tensor(shape=(512,), dtype=float32),
#             'natural_language_instruction': string,
#             'state': Tensor(shape=(7,), dtype=float32),
#         }),
#         'reward': Scalar(shape=(), dtype=float32),
#     }),
# })
# '''

# import os
# import numpy as np
# import tensorflow_datasets as tfds
# from PIL import Image

# # Example GCS path or local directory for the dataset
# DATASET_GCS_PATH = (
#     "gs://gresearch/robotics/bridge/0.1.0"
# )

# def main():
#     # 1) Create the builder from a prepared directory (dataset is ready; no download needed)
#     builder = tfds.builder_from_directory(builder_dir=DATASET_GCS_PATH)

#     # 2) Load the first 20 episodes from the train split, without shuffling
#     ds = builder.as_dataset(split="train[:20]", shuffle_files=False)

#     # 3) Iterate over each episode
#     for episode_num, episode in enumerate(ds):
#         print(f"[INFO] Processing Episode {episode_num}")

#         # Create folders for this episode
#         folder_path = f"../states/bridge/episode_{episode_num}"
#         os.makedirs(folder_path, exist_ok=True)

#         # Subfolder for images
#         images_folder = os.path.join(folder_path, "images")
#         os.makedirs(images_folder, exist_ok=True)

#         # 4) Iterate over each step in the episode
#         steps_dataset = episode["steps"]
#         for step_idx, step in enumerate(steps_dataset):
#             #if episode["episode_metadata"][f"has_image_{i}"].numpy():
#             image_np = step["observation"][f"image"].numpy()
#             img = Image.fromarray(image_np)
#             image_filename = os.path.join(images_folder, f"{step_idx}.jpeg")
#             img.save(image_filename, format="JPEG")


#         print(f"[INFO] Episode {episode_num}.")

# if __name__ == "__main__":
#     main()


import os
import numpy as np
import tensorflow_datasets as tfds
from PIL import Image

# Path to the Bridge dataset (GCS or local)
DATASET_GCS_PATH = "gs://gresearch/robotics/bridge/0.1.0"


def main():
    # 1) Create the builder from the prepared directory
    builder = tfds.builder_from_directory(builder_dir=DATASET_GCS_PATH)

    # 2) Load the first 20 episodes from the 'train' split, without shuffling
    ds = builder.as_dataset(split="train[10000:11000]", shuffle_files=False)

    # 3) Iterate over each episode
    for episode_num, episode in enumerate(ds):
        print(f"[INFO] Processing Episode {episode_num + 10000}")

        # Prepare lists to gather per-step data
        state_list = []
        embedding_list = []
        instruction_list = []
        gripper_state_list = []

        # Setup episode folders
        folder_path = f"../states/bridge/episode_{episode_num + 10000}"
        os.makedirs(folder_path, exist_ok=True)
        images_folder = os.path.join(folder_path, "images")
        os.makedirs(images_folder, exist_ok=True)

        # 4) Iterate over steps within the episode
        for step_idx, step in enumerate(episode["steps"]):
            # Extract and save the main camera image
            img_np = step["observation"]["image"].numpy()
            img = Image.fromarray(img_np)
            img_filename = os.path.join(images_folder, f"{step_idx}.jpeg")
            img.save(img_filename, format="JPEG")

            # Extract state vector
            gripper_state = step["action"]["open_gripper"].numpy()
            state = step["observation"]["state"].numpy()  # shape (7,)
            state_list.append(state)

            # Extract language embedding and instruction
            emb = step["observation"]["natural_language_embedding"].numpy()  # shape (512,)
            embedding_list.append(emb)

            instr = step["observation"]["natural_language_instruction"].numpy().decode("utf-8")
            instruction_list.append(instr)

        # 5) Save collected arrays to disk
        states_arr = np.vstack(state_list)             # shape: (T, 7)
        gripper_states_arr = np.array(gripper_state_list)  # shape: (T,)

        np.savetxt(os.path.join(folder_path, "ee_states.txt"), states_arr)
        np.savetxt(os.path.join(folder_path, "gripper_states.txt"), gripper_states_arr)
        np.savetxt(os.path.join(folder_path, "language_instructions.txt"), np.array(instruction_list), fmt="%s")

        print(f"[INFO] Episode {episode_num + 10000} processed: {states_arr.shape[0]} steps.")


if __name__ == "__main__":
    main()
