import os
import numpy as np
import tensorflow_datasets as tfds
from PIL import Image

# Example GCS path or local directory for the dataset
DATASET_GCS_PATH = "gs://gresearch/robotics/austin_sailor_dataset_converted_externally_to_rlds/0.1.0"

def main():
    # 1) 从指定目录创建 builder（数据集已经准备好，不需要重新 download_and_prepare()）
    builder = tfds.builder_from_directory(builder_dir=DATASET_GCS_PATH)
    
    # 2) 从 train split 中加载前 20 个 episode，不打乱文件顺序
    ds = builder.as_dataset(split="train", shuffle_files=False)

    # 3) 遍历每个 episode
    for episode_num, episode in enumerate(ds):
        print(f"[INFO] Processing Episode {episode_num}")
        
        # 准备列表用于存储各步数据
        ee_states_list = []      # End-effector state (16,)
        joint_states_list = []   # 7-dof joint state (7,)
        gripper_states_list = [] # Gripper state (1,) —— 这里即为“breaper state”
        
        # 设定当前 episode 存储路径
        folder_path = f"../states/austin_sailor_dataset_converted_externally_to_rlds/episode_{episode_num}"
        os.makedirs(folder_path, exist_ok=True)
        
        # 用于存放图像的文件夹
        images_folder = os.path.join(folder_path, "images")
        os.makedirs(images_folder, exist_ok=True)
        
        # 4) 遍历当前 episode 中的每个 step（time step）
        steps_dataset = episode["steps"]
        for step_idx, step in enumerate(steps_dataset):
            # 从 observation 中提取状态信息：
            # EE state: 来自 "state_ee", shape=(16,)
            ee_state = step["observation"]["state_ee"].numpy()
            # Joint state: 来自 "state_joint", shape=(7,)
            joint_state = step["observation"]["state_joint"].numpy()
            # Gripper state: 来自 "state_gripper", shape=(1,)
            gripper_state = step["observation"]["state_gripper"].numpy()
            
            ee_states_list.append(ee_state)
            joint_states_list.append(joint_state)
            gripper_states_list.append(gripper_state)
            
            # 5) 提取主摄像头图像 (128x128x3)
            image_np = step["observation"]["image"].numpy()
            # 转换为 PIL Image 对象
            img = Image.fromarray(image_np)
            # 保存为 JPEG 格式，文件名为 "0.jpeg", "1.jpeg", …（不带前导零）
            image_filename = os.path.join(images_folder, f"{step_idx}.jpeg")
            img.save(image_filename, format="JPEG")
        
        # 6) 将各列表转换为 numpy 数组，并保存为文本文件
        ee_states_array = np.vstack(ee_states_list)         # shape: (T, 16)
        joint_states_array = np.vstack(joint_states_list)     # shape: (T, 7)
        gripper_states_array = np.vstack(gripper_states_list) # shape: (T, 1)
        
        np.savetxt(os.path.join(folder_path, "ee_states.txt"), ee_states_array)
        np.savetxt(os.path.join(folder_path, "joint_states.txt"), joint_states_array)
        np.savetxt(os.path.join(folder_path, "gripper_states.txt"), gripper_states_array)
        
        print(f"[INFO] Episode {episode_num} processed with {ee_states_array.shape[0]} steps.")

if __name__ == "__main__":
    main()