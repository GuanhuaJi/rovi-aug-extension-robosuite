'''
FeaturesDict({
    'episode_metadata': FeaturesDict({
        'file_path': Text(shape=(), dtype=string),
    }),
    'steps': Dataset({
        'action': Tensor(shape=(7,), dtype=float32, description=Robot action, consists of [6x end effector delta pose, 1x gripper position]),
        'discount': Scalar(shape=(), dtype=float32, description=Discount if provided, default to 1.),
        'is_first': bool,
        'is_last': bool,
        'is_terminal': bool,
        'language_embedding': Tensor(shape=(512,), dtype=float32, description=Kona language embedding. See https://tfhub.dev/google/universal-sentence-encoder-large/5),
        'language_instruction': Text(shape=(), dtype=string),
        'observation': FeaturesDict({
            'image': Image(shape=(128, 128, 3), dtype=uint8, description=Main camera RGB observation.),
            'state': Tensor(shape=(24,), dtype=float32, description=Robot state, consists of [7x robot joint angles, 1x gripper position, 16x robot end-effector homogeneous matrix].),
            'wrist_image': Image(shape=(128, 128, 3), dtype=uint8, description=Wrist camera RGB observation.),
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
DATASET_GCS_PATH = "gs://gresearch/robotics/utaustin_mutex/0.1.0"

def main():
    # 1) 从指定目录创建 builder（数据集已准备好，无需重新 download_and_prepare()）
    builder = tfds.builder_from_directory(builder_dir=DATASET_GCS_PATH)
    
    # 2) 从 train split 中加载前 20 个 episode，不打乱文件顺序
    ds = builder.as_dataset(split="train", shuffle_files=False)
    
    # 3) 遍历每个 episode
    for episode_num, episode in enumerate(ds):
        print(f"[INFO] Processing Episode {episode_num}")
        
        # 初始化列表，用于存储各步数据
        joint_angles_list = []    # 取自 observation/state 前 7 个元素
        gripper_state_list = []   # 取自 observation/state 第 8 个元素（以数组形式保存）
        ee_states_list = []       # 取自 observation/state 后 16 个元素
        language_instructions = []  # 存储语言指令
        
        # 设定当前 episode 存储路径
        folder_path = f"../states/utaustin_mutex/episode_{episode_num}"
        os.makedirs(folder_path, exist_ok=True)
        
        # 创建用于存放图像的文件夹
        images_folder = os.path.join(folder_path, "images")
        os.makedirs(images_folder, exist_ok=True)
        
        # 4) 遍历当前 episode 中的每个 step
        steps_dataset = episode["steps"]
        for step_idx, step in enumerate(steps_dataset):
            # 从 observation 中提取 state (24,)
            # 含义：[0:7] -> joint angles, [7] -> gripper state, [8:24] -> end-effector state
            state = step["observation"]["state"].numpy()
            language_instruction = step["language_instruction"].numpy().decode('utf-8')
            joint_angles = state[:7]
            gripper_state = state[7:8]  # 保持 (1,) 数组形式
            ee_state = state[8:24]
            
            joint_angles_list.append(joint_angles)
            gripper_state_list.append(gripper_state)
            ee_states_list.append(ee_state)
            language_instructions.append(language_instruction)
            
            # 5) 提取主摄像头图像 (128, 128, 3)
            image_np = step["observation"]["image"].numpy()
            img = Image.fromarray(image_np)
            # 保存为 JPEG 格式，文件名按步编号命名： "0.jpeg", "1.jpeg", ...
            image_filename = os.path.join(images_folder, f"{step_idx}.jpeg")
            img.save(image_filename, format="JPEG")
        
        # 6) 将各列表转换为 numpy 数组，并保存为文本文件
        joint_angles_array = np.array(joint_angles_list)     # shape: (T, 7)
        gripper_state_array = np.array(gripper_state_list)     # shape: (T, 1)
        ee_states_array = np.array(ee_states_list)             # shape: (T, 16)
        
        np.savetxt(os.path.join(folder_path, "joint_states.txt"), joint_angles_array)
        np.savetxt(os.path.join(folder_path, "gripper_states.txt"), gripper_state_array)
        np.savetxt(os.path.join(folder_path, "ee_states.txt"), ee_states_array)
        np.savetxt(os.path.join(folder_path, "language_instructions.txt"), 
                   np.array(language_instructions), fmt="%s")
        
        print(f"[INFO] Episode {episode_num} processed with {joint_angles_array.shape[0]} steps.")

if __name__ == "__main__":
    main()