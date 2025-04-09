import os
import numpy as np
import tensorflow_datasets as tfds
from PIL import Image

# 指定数据集所在路径（GCS 或本地目录）
DATASET_GCS_PATH = "gs://gresearch/robotics/nyu_franka_play_dataset_converted_externally_to_rlds/0.1.0"

def main():
    # 1) 从指定目录创建 builder（数据集已准备好，无需重新 download_and_prepare()）
    builder = tfds.builder_from_directory(builder_dir=DATASET_GCS_PATH)
    
    # 2) 从 train split 中加载前 20 个 episode，不打乱文件顺序
    ds = builder.as_dataset(split="train[:20]", shuffle_files=False)
    
    # 3) 遍历每个 episode
    for episode_num, episode in enumerate(ds):
        print(f"[INFO] Processing Episode {episode_num}")
        
        # 初始化列表，用于存储每个 step 的数据
        joint_states_list = []    # 7-dof joint angles (observation/state[0:7])
        ee_states_list = []       # End-effector state (observation/state[7:13], 6-dim)
        gripper_states_list = []  # Gripper state (extracted from action, index 13)
        
        # 设定当前 episode 的存储路径
        folder_path = f"../states/nyu_franka_play_dataset_converted_externally_to_rlds/episode_{episode_num}"
        os.makedirs(folder_path, exist_ok=True)
        
        # 创建用于存放图像的文件夹
        images_folder = os.path.join(folder_path, "images")
        os.makedirs(images_folder, exist_ok=True)
        
        # 4) 遍历当前 episode 中的每个 step
        steps_dataset = episode["steps"]
        for step_idx, step in enumerate(steps_dataset):
            # 从 observation 中提取 state (shape: (13,))
            state = step["observation"]["state"].numpy()
            joint_state = state[:7]     # 前 7 个数值
            ee_state = state[7:13]      # 后 6 个数值
            
            # 从 action 中提取 gripper state（action 是 shape (15,)）
            action = step["action"].numpy()
            gripper_state = action[13:14]  # 取第14个元素，以数组形式保存
            
            joint_states_list.append(joint_state)
            ee_states_list.append(ee_state)
            gripper_states_list.append(gripper_state)
            
            # 5) 提取主摄像头图像 (128x128x3)
            image_np = step["observation"]["image"].numpy()
            img = Image.fromarray(image_np)
            # 保存为 JPEG 格式，文件名格式为 "0.jpeg", "1.jpeg", …
            image_filename = os.path.join(images_folder, f"{step_idx}.jpeg")
            img.save(image_filename, format="JPEG")
        
        # 6) 将列表转换为 numpy 数组，并保存为文本文件
        joint_states_array = np.array(joint_states_list)     # shape: (T, 7)
        ee_states_array = np.array(ee_states_list)             # shape: (T, 6)
        gripper_states_array = np.array(gripper_states_list)   # shape: (T, 1)
        
        np.savetxt(os.path.join(folder_path, "joint_states.txt"), joint_states_array)
        np.savetxt(os.path.join(folder_path, "ee_states.txt"), ee_states_array)
        np.savetxt(os.path.join(folder_path, "gripper_states.txt"), gripper_states_array)
        
        print(f"[INFO] Episode {episode_num} processed with {joint_states_array.shape[0]} steps.")

if __name__ == "__main__":
    main()