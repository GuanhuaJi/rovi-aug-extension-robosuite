import os
import numpy as np
import tensorflow_datasets as tfds
from PIL import Image

'''
FeaturesDict({
    'episode_metadata': FeaturesDict({
        'file_path': Text(shape=(), dtype=string),
    }),
    'steps': Dataset({
        'action': Tensor(shape=(7,), dtype=float32, description=Robot action, consists of [7x joint velocities, 2x gripper velocities, 1x terminate episode].),
        'action_delta': Tensor(shape=(7,), dtype=float32, description=Robot delta action, consists of [7x joint velocities, 2x gripper velocities, 1x terminate episode].),
        'action_inst': Text(shape=(), dtype=string),
        'discount': Scalar(shape=(), dtype=float32, description=Discount if provided, default to 1.),
        'goal_object': Text(shape=(), dtype=string),
        'ground_truth_states': FeaturesDict({
            'EE': Tensor(shape=(6,), dtype=float32, description=xyzrpy),
            'bottle': Tensor(shape=(6,), dtype=float32, description=xyzrpy),
            'bread': Tensor(shape=(6,), dtype=float32, description=xyzrpy),
            'coke': Tensor(shape=(6,), dtype=float32, description=xyzrpy),
            'cube': Tensor(shape=(6,), dtype=float32, description=xyzrpy),
            'milk': Tensor(shape=(6,), dtype=float32, description=xyzrpy),
            'pepsi': Tensor(shape=(6,), dtype=float32, description=xyzrpy),
        }),
        'is_first': bool,
        'is_last': bool,
        'is_terminal': bool,
        'language_embedding': Tensor(shape=(512,), dtype=float32, description=Kona language embedding. See https://tfhub.dev/google/universal-sentence-encoder-large/5),
        'language_instruction': Text(shape=(), dtype=string),
        'observation': FeaturesDict({
            'image': Image(shape=(224, 224, 3), dtype=uint8, description=Main camera RGB observation.),
            'state': Tensor(shape=(7,), dtype=float32, description=Robot state, consists of [6x robot joint angles, 1x gripper position].),
            'state_vel': Tensor(shape=(7,), dtype=float32, description=Robot joint velocity, consists of [6x robot joint angles, 1x gripper position].),
        }),
        'reward': Scalar(shape=(), dtype=float32, description=Reward if provided, 1 on final step for demos.),
    }),
})
'''

DATASET_GCS_PATH = (
    "gs://gresearch/robotics/asu_table_top_converted_externally_to_rlds/0.1.0"
)

def main():
    # 1) Dataset is already materialised at DATASET_GCS_PATH
    builder = tfds.builder_from_directory(builder_dir=DATASET_GCS_PATH)

    # 2) First 20 episodes of the train split
    ds = builder.as_dataset(split="train", shuffle_files=False)

    for episode_num, episode in enumerate(ds):
        print(f"[INFO] Processing Episode {episode_num}")

        # Lists for step-wise data
        joint_angles_list = []
        gripper_list      = []
        ee_pose_list      = []
        lang_list         = []          # <-- NEW

        # Episode output folders
        folder_path  = f"../states/asu_table_top_rlds/episode_{episode_num}"
        images_folder = os.path.join(folder_path, "images")
        os.makedirs(images_folder, exist_ok=True)

        # 4) Iterate through steps
        for step_idx, step in enumerate(episode["steps"]):
            # --- Observations ---
            obs_state = step["observation"]["state"].numpy()      # (7,)
            joint_angles = obs_state[:6]
            gripper_state = obs_state[6:7]

            ee_pose = step["ground_truth_states"]["EE"].numpy()   # (6,)

            # Accumulate data
            joint_angles_list.append(joint_angles)
            gripper_list.append(gripper_state)
            ee_pose_list.append(ee_pose)

            # --- Language instruction ---
            lang_inst = step["language_instruction"].numpy().decode("utf-8")
            lang_list.append(lang_inst)                           # <-- NEW

            # --- Save image ---
            image_np = step["observation"]["image"].numpy()       # (224,224,3)
            Image.fromarray(image_np).save(
                os.path.join(images_folder, f"{step_idx}.jpeg"), format="JPEG"
            )

        # 5) Convert to NumPy
        np.savetxt(os.path.join(folder_path, "joint_states.txt"),
                   np.array(joint_angles_list))
        np.savetxt(os.path.join(folder_path, "gripper_states.txt"),
                   np.array(gripper_list))
        np.savetxt(os.path.join(folder_path, "ee_pose.txt"),
                   np.array(ee_pose_list))

        # 6) Save language instructions  <-- NEW
        with open(os.path.join(folder_path, "language_instruction.txt"), "w") as f:
            f.write("\n".join(lang_list))

        print(f"[INFO] Episode {episode_num} processed with {len(joint_angles_list)} steps.")

if __name__ == "__main__":
    main()
