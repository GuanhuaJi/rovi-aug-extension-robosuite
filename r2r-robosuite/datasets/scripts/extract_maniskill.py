import os
import numpy as np
import tensorflow_datasets as tfds
from PIL import Image

# The GCS path for "maniskill_dataset_converted_externally_to_rlds"
DATASET_PATH = "gs://gresearch/robotics/maniskill_dataset_converted_externally_to_rlds/0.1.0"

def main():
    # 1) Build the dataset from GCS and load the first 20 episodes (unshuffled)
    builder = tfds.builder_from_directory(builder_dir=DATASET_PATH)
    ds = builder.as_dataset(split="train[:20]", shuffle_files=False)
    
    # 2) Iterate over episodes
    for ep_idx, episode in enumerate(ds):
        print(f"[INFO] Processing Episode {ep_idx}")
        
        # Create a folder for each episode
        ep_folder = f"../states/maniskill/episode_{ep_idx}"
        os.makedirs(ep_folder, exist_ok=True)
        
        # Subfolder for main camera images
        images_folder = os.path.join(ep_folder, "images")
        os.makedirs(images_folder, exist_ok=True)
        
        # Prepare lists for storing time-series data
        joint_angles_list = []  # shape => (T, 7)
        ee_pose_list = []       # shape => (T, 7),  [x, y, z, qw, qx, qy, qz]
        gripper_list = []       # shape => (T,)    , commanded open/close
        
        # 3) Loop over each step within the episode
        steps_dataset = episode["steps"]
        for step_idx, step in enumerate(steps_dataset):
            # --- Extract and save main camera image (256 x 256 x 3) ---
            img_np = step["observation"]["image"].numpy()
            img = Image.fromarray(img_np)
            img_filename = os.path.join(images_folder, f"{step_idx}.jpeg")
            img.save(img_filename, format="JPEG")
            
            # --- Extract joint angles (7) from observation['state'][0:7] ---
            #   'state' shape=18 => [7 joint angles, 2 gripper pos, 7 joint velocities, 2 gripper velocities]
            state_arr = step["observation"]["state"].numpy()
            joint_angles = state_arr[0:7]   # shape (7,)
            
            # --- Extract end-effector (tool-center-point) pose from observation['tcp_pose'] ---
            #   'tcp_pose' shape=7 => [x, y, z, qw, qx, qy, qz]
            tcp_pose = step["observation"]["tcp_pose"].numpy()  # shape (7,)
            
            # --- Extract gripper (open/close) from action[6] ---
            #   action shape=7 => [3 pos deltas, 3 ori deltas, 1 gripper target]
            #   -1 => close, +1 => open
            action_arr = step["action"].numpy()
            gripper_cmd = action_arr[6]  # scalar
            
            # Append to time-series lists
            joint_angles_list.append(joint_angles)
            ee_pose_list.append(tcp_pose)
            gripper_list.append(gripper_cmd)
        
        # 4) Convert to numpy arrays and save
        joint_angles_array = np.array(joint_angles_list)  # (T, 7)
        ee_pose_array = np.array(ee_pose_list)            # (T, 7)
        gripper_array = np.array(gripper_list)            # (T,)

        np.savetxt(os.path.join(ep_folder, "joint_states.txt"), joint_angles_array)
        np.savetxt(os.path.join(ep_folder, "ee_pose.txt"), ee_pose_array)
        np.savetxt(os.path.join(ep_folder, "gripper.txt"), gripper_array)

        print(f"[INFO] Episode {ep_idx} processed, steps: {len(gripper_array)}")

if __name__ == "__main__":
    main()