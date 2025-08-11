import os
import numpy as np
import tensorflow_datasets as tfds
from PIL import Image

# Example GCS path or local directory for the cmu_play_fusion dataset
DATASET_GCS_PATH = "gs://gresearch/robotics/cmu_play_fusion/0.1.0"

def main():
    # 1) 从指定目录创建 builder（数据集已准备好，无需重新 download_and_prepare()）
    builder = tfds.builder_from_directory(builder_dir=DATASET_GCS_PATH)
    
    # 2) 从 train split 中加载前 20 个 episode，不打乱文件顺序
    ds = builder.as_dataset(split="train[:20]", shuffle_files=False)
    
    # 3) 遍历每个 episode
    for episode_num, episode in enumerate(ds):
        print(f"[INFO] Processing Episode {episode_num}")
        
        # 初始化列表，用于存储各步数据
        joint_angles_list = []    # 取自 observation/state 前 7 个元素
        gripper_state_list = []   # 取自 observation/state 第 8 个元素（以数组形式保存）
        ee_states_list = []       # 由于该数据集中仅有 joint + gripper，共 8 维，无额外 ee state，故此处仅做空列表占位
        
        # 设定当前 episode 存储路径
        folder_path = f"../states/cmu_play_fusion/episode_{episode_num}"
        os.makedirs(folder_path, exist_ok=True)
        
        # 创建用于存放图像的文件夹
        images_folder = os.path.join(folder_path, "images")
        os.makedirs(images_folder, exist_ok=True)
        
        # 4) 遍历当前 episode 中的每个 step
        steps_dataset = episode["steps"]
        for step_idx, step in enumerate(steps_dataset):
            # 从 observation 中提取 state (8,)
            # 含义：[0:7] -> 7x 关节角, [7] -> gripper position
            state = step["observation"]["state"].numpy()
            joint_angles = state[:7]          # shape (7,)
            gripper_state = state[7:8]        # shape (1,)
            
            # 该数据集无单独的 ee state，这里占位
            ee_state = np.array([], dtype=np.float32)  # shape (0,)
            
            joint_angles_list.append(joint_angles)
            gripper_state_list.append(gripper_state)
            ee_states_list.append(ee_state)
            
            # 5) 提取主摄像头图像 (128, 128, 3)
            image_np = step["observation"]["image"].numpy()
            img = Image.fromarray(image_np)
            # 保存为 JPEG 格式，文件名按步编号命名： "0.jpeg", "1.jpeg", ...
            image_filename = os.path.join(images_folder, f"{step_idx}.jpeg")
            img.save(image_filename, format="JPEG")
        
        # 6) 将各列表转换为 numpy 数组，并保存为文本文件
        joint_angles_array = np.array(joint_angles_list)   # shape: (T, 7)
        gripper_state_array = np.array(gripper_state_list) # shape: (T, 1)
        ee_states_array = np.array(ee_states_list)         # shape: (T, 0) 仅作占位
        
        np.savetxt(os.path.join(folder_path, "joint_states.txt"), joint_angles_array)
        np.savetxt(os.path.join(folder_path, "gripper_states.txt"), gripper_state_array)
        # 若想在此处保留 ee_states.txt，可将空数组写出
        np.savetxt(os.path.join(folder_path, "ee_states.txt"), ee_states_array)
        
        print(f"[INFO] Episode {episode_num} processed with {joint_angles_array.shape[0]} steps.")

if __name__ == "__main__":
    main()