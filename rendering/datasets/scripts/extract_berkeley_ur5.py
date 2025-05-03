import os
import numpy as np
import tensorflow_datasets as tfds
from PIL import Image

# 指定数据集所在路径（GCS 或本地目录）
DATASET_GCS_PATH = "gs://gresearch/robotics/berkeley_autolab_ur5/0.1.0"

def main():
    # 1) 从指定目录创建 builder（数据集已准备好，无需重新 download_and_prepare()）
    builder = tfds.builder_from_directory(builder_dir=DATASET_GCS_PATH)
    
    # 2) 从 train split 中加载前 20 个 episode（不打乱文件顺序）
    ds = builder.as_dataset(split="train", shuffle_files=False)
    
    # 3) 遍历每个 episode
    for episode_num, episode in enumerate(ds):
        print(f"[INFO] Processing Episode {episode_num}")
        
        # 初始化列表用于存储各步数据
        joint_states_list = []    # 关节角（joint state）：取 robot_state 的前 6 个元素
        gripper_states_list = []  # gripper 状态（breaper state）：取 robot_state 的第 7 个元素（保持 (1,) 数组形式）
        ee_states_list = []       # end-effector 状态（EE state）：取 robot_state 的后 8 个元素
        
        # 设定当前 episode 存储路径
        folder_path = f"../states/autolab_ur5/episode_{episode_num}"
        os.makedirs(folder_path, exist_ok=True)
        
        # 创建用于存放图像的子文件夹
        images_folder = os.path.join(folder_path, "images")
        os.makedirs(images_folder, exist_ok=True)
        
        # 4) 遍历当前 episode 中的每个 step
        steps_dataset = episode["steps"]
        for step_idx, step in enumerate(steps_dataset):
            # 从 observation 中提取 robot_state (15,)
            state = step["observation"]["robot_state"].numpy()
            # 假设 robot_state 的结构为：[0:6] -> joint angles, [6:7] -> gripper state, [7:15] -> end-effector state
            joint_state = state[:6]
            ee_state = state[6:13]
            gripper_state = state[13:14]

            
            joint_states_list.append(joint_state)
            gripper_states_list.append(gripper_state)
            ee_states_list.append(ee_state)
            
            # 5) 提取主摄像头图像 (480, 640, 3)
            image_np = step["observation"]["image"].numpy()
            img = Image.fromarray(image_np)
            # 保存为 JPEG 格式，文件名为 "0.jpeg", "1.jpeg", …（不带前导零）
            image_filename = os.path.join(images_folder, f"{step_idx}.jpeg")
            img.save(image_filename, format="JPEG")
        
        # 6) 将列表转换为 numpy 数组，并保存为文本文件
        joint_states_array = np.vstack(joint_states_list)      # shape: (T, 6)
        gripper_states_array = np.vstack(gripper_states_list)  # shape: (T, 1)
        ee_states_array = np.vstack(ee_states_list)              # shape: (T, 8)
        
        np.savetxt(os.path.join(folder_path, "joint_states.txt"), joint_states_array)
        np.savetxt(os.path.join(folder_path, "gripper_states.txt"), gripper_states_array)
        np.savetxt(os.path.join(folder_path, "ee_states.txt"), ee_states_array)
        
        print(f"[INFO] Episode {episode_num} processed with {joint_states_array.shape[0]} steps.")

if __name__ == "__main__":
    main()