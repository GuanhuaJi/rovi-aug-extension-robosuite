'''
FeaturesDict({
    'episode_metadata': FeaturesDict({
        'file_path': Text(shape=(), dtype=string),
    }),
    'steps': Dataset({
        'action': Tensor(shape=(8,), dtype=float32, description=Robot action, consists of [3x end-effector position, 4x end-effector quaternion, 1x gripper open/close].),
        'discount': Scalar(shape=(), dtype=float32, description=Discount if provided, default to 1.),
        'is_first': bool,
        'is_last': bool,
        'is_terminal': bool,
        'language_embedding': Tensor(shape=(512,), dtype=float32, description=Kona language embedding. See https://tfhub.dev/google/universal-sentence-encoder-large/5),
        'language_instruction': Text(shape=(), dtype=string),
        'observation': FeaturesDict({
            'image': Image(shape=(360, 640, 3), dtype=uint8, description=Main camera RGB observation.),
            'state': Tensor(shape=(20,), dtype=float32, description=Robot state, consists of [7x robot joint angles, 1x gripper status, 6x joint torques, 6x end-effector force].),
            'wrist_image': Image(shape=(240, 320, 3), dtype=uint8, description=Wrist camera RGB observation.),
        }),
        'reward': Scalar(shape=(), dtype=float32, description=Reward if provided, 1 on final step for demos.),
    }),
})

'''


import os
import numpy as np
import tensorflow_datasets as tfds
from PIL import Image

# The path to the dataset on GCS
DATASET_PATH = "gs://gresearch/robotics/iamlab_cmu_pickup_insert_converted_externally_to_rlds/0.1.0"

def main():
    # 1) Create the DatasetBuilder and load first 20 episodes, unshuffled
    builder = tfds.builder_from_directory(builder_dir=DATASET_PATH)
    ds = builder.as_dataset(split="train", shuffle_files=False)

    # 2) Iterate over episodes
    for ep_idx, episode in enumerate(ds):
        print(f"[INFO] Processing Episode {ep_idx}")
        
        # Create output folder for this episode
        ep_folder = f"../states/iamlab_cmu/episode_{ep_idx}"
        os.makedirs(ep_folder, exist_ok=True)
        
        # Subfolders for main camera and wrist camera
        main_images_folder = os.path.join(ep_folder, "images")
        os.makedirs(main_images_folder, exist_ok=True)
        
        wrist_images_folder = os.path.join(ep_folder, "wrist_images")
        os.makedirs(wrist_images_folder, exist_ok=True)
        
        # Prepare lists to hold time-series arrays for this episode
        ee_pose_list = []       # shape => (T, 7) => [x, y, z, qw, qx, qy, qz]
        joint_angles_list = []  # shape => (T, 7)
        gripper_list = []       # shape => (T, 2) => [commanded_gripper, observed_gripper_status]
        instruction_list = []

        # 3) Iterate over steps within this episode
        for step_idx, step in enumerate(episode["steps"]):
            # --- Extract images ---
            # Main camera: shape = (360, 640, 3)
            # main_img_np = step["observation"]["image"].numpy()
            # main_img = Image.fromarray(main_img_np)
            # main_img.save(os.path.join(main_images_folder, f"{step_idx}.jpeg"), format="JPEG")
            
            # # Wrist camera: shape = (240, 320, 3)
            # wrist_img_np = step["observation"]["wrist_image"].numpy()
            # wrist_img = Image.fromarray(wrist_img_np)
            # wrist_img.save(os.path.join(wrist_images_folder, f"{step_idx}.jpeg"), format="JPEG")
            
            # --- Extract action info (EE command) ---
            # action = [x, y, z, qw, qx, qy, qz, gripper]
            action_arr = step["action"].numpy()
            ee_pos = action_arr[0:3]          # (3,)
            ee_quat = action_arr[3:7]         # (4,)
            gripper_cmd = action_arr[7]       # scalar, e.g. 1=close, 0=open (exact usage depends on dataset spec)
            
            ee_pose = np.concatenate([ee_pos, ee_quat])  # shape (7,)
            
            # --- Extract robot state info ---
            # observation['state'] = [7 joint angles, 1 gripper status, 6 joint torques, 6 ee force] => total 20
            state_arr = step["observation"]["state"].numpy()
            joint_angles = state_arr[0:7]         # shape (7,)
            gripper_status = state_arr[7]         # scalar, e.g. 1=closed, 0=open

            instr = step["language_instruction"].numpy().decode("utf-8")
            instruction_list.append(instr)
            
            # Store them
            ee_pose_list.append(ee_pose)
            joint_angles_list.append(joint_angles)
            gripper_list.append(gripper_status)
        
        # 4) Convert to numpy arrays and save as text
        ee_pose_array = np.array(ee_pose_list)         # (T, 7)
        joint_array = np.array(joint_angles_list)      # (T, 7)
        gripper_array = np.array(gripper_list)         # (T, 2)
        
        np.savetxt(os.path.join(ep_folder, "ee_pose.txt"), ee_pose_array)
        np.savetxt(os.path.join(ep_folder, "joint_states.txt"), joint_array)
        np.savetxt(os.path.join(ep_folder, "gripper_states.txt"), gripper_array)
        np.savetxt(os.path.join(ep_folder, "language_instructions.txt"), np.array(instruction_list), fmt="%s")
        
        print(f"[INFO] Episode {ep_idx} processed. Steps: {len(ee_pose_list)}")

if __name__ == "__main__":
    main()
