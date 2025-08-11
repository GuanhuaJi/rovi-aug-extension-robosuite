import os
import numpy as np
import tensorflow_datasets as tfds
from PIL import Image

# 替换为实际存放 "austin_sirius_dataset_converted_externally_to_rlds" 的路径
DATASET_PATH = "gs://gresearch/robotics/austin_sirius_dataset_converted_externally_to_rlds/0.1.0"
def main():
    # 1) 从指定目录创建 builder（数据集已准备好，无需重新 download_and_prepare()）
    builder = tfds.builder_from_directory(builder_dir=DATASET_PATH)
    
    # 2) 从 train split 中加载前 20 个 episode，不打乱文件顺序
    ds = builder.as_dataset(split="train", shuffle_files=False)

    # 3) 遍历每个 episode
    for episode_num, episode in enumerate(ds):
        print(f"[INFO] Processing Episode {episode_num}")
        
        # 初始化列表，用于存储各步数据
        joint_angles_list = []    # 取自 observation['state'] 的前 7 个元素 或 observation['state_joint']
        gripper_state_list = []   # 取自 observation['state'] 的第 8 个元素 或 observation['state_gripper']
        ee_states_list = []       # 取自 observation['state_ee']

        # 设定当前 episode 存储路径
        folder_path = f"../states/austin_sirius/episode_{episode_num}"
        os.makedirs(folder_path, exist_ok=True)
        
        # 创建用于存放图像的文件夹
        images_folder = os.path.join(folder_path, "images")
        os.makedirs(images_folder, exist_ok=True)
        
        # 4) 遍历当前 episode 中的每个 step
        steps_dataset = episode["steps"]
        for step_idx, step in enumerate(steps_dataset):
            # ---- 观察部分 ----
            # observation['state'] -> (8,) => [0:7] 是关节角, [7] 是夹爪
            # observation['state_joint'] -> (7,)
            # observation['state_gripper'] -> (1,)
            # observation['state_ee'] -> (16,)
            # observation['image'] -> (84, 84, 3)

            # 如果你想模仿原先“一次性取到 state 并切片”的做法：
            state = step["observation"]["state"].numpy()  # (8,)
            joint_angles = state[:7]            # 前 7 个元素
            gripper_state = state[7:8]          # 第 8 个元素（保留 (1,) 形状）
            
            # 末端执行器 4x4 齐次变换，合计 16 个浮点
            ee_state = step["observation"]["state_ee"].numpy()  # (16,)

            joint_angles_list.append(joint_angles)
            gripper_state_list.append(gripper_state)
            ee_states_list.append(ee_state)
            
            # 5) 提取主摄像头图像 (84, 84, 3)
            image_np = step["observation"]["image"].numpy()
            img = Image.fromarray(image_np)
            
            # 保存为 JPEG 格式，文件名按步编号命名： "0.jpeg", "1.jpeg", ...
            image_filename = os.path.join(images_folder, f"{step_idx}.jpeg")
            img.save(image_filename, format="JPEG")
        
        # 6) 将各列表转换为 numpy 数组，并保存为文本文件
        joint_angles_array = np.array(joint_angles_list)   # shape: (T, 7)
        gripper_state_array = np.array(gripper_state_list) # shape: (T, 1)
        ee_states_array = np.array(ee_states_list)         # shape: (T, 16)
        
        np.savetxt(os.path.join(folder_path, "joint_states.txt"), joint_angles_array)
        np.savetxt(os.path.join(folder_path, "gripper_states.txt"), gripper_state_array)
        np.savetxt(os.path.join(folder_path, "ee_states.txt"), ee_states_array)
        
        print(f"[INFO] Episode {episode_num} processed with {joint_angles_array.shape[0]} steps.")

if __name__ == "__main__":
    main()
