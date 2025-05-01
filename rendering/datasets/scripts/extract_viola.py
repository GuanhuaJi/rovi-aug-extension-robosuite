import os
import numpy as np
import tensorflow_datasets as tfds
from PIL import Image

DATASET_GCS_PATH = "gs://gresearch/robotics/viola/0.1.0"

try:
    # 1. 使用 tfds.load 下载 viola 数据集，并只加载 train split 的前20个 episode
    builder = tfds.builder_from_directory(builder_dir=DATASET_GCS_PATH)
    ds = builder.as_dataset(split="train", shuffle_files=False)

    # 2. 遍历每个 episode（每个 episode 包含一个 'steps' 序列）
    for episode_num, episode in enumerate(ds):
        # 用于存储各个步骤的状态信息
        ee_states_list = []
        joint_states_list = []
        gripper_states_list = []

        # 3. 为每个 episode 创建本地文件夹
        folder_path = f'../states/viola/episode_{episode_num}'
        os.makedirs(folder_path, exist_ok=True)
        # 创建用于存放图像的子文件夹
        agentview_folder = os.path.join(folder_path, 'images')
        os.makedirs(agentview_folder, exist_ok=True)

        # 4. 遍历每个 step
        for step_idx, step in enumerate(episode['steps']):
            # 提取状态信息
            # ee_states: (16,), joint_states: (7,), gripper_states: (1,)
            ee_state = step['observation']['ee_states'].numpy()
            joint_state = step['observation']['joint_states'].numpy()
            gripper_state = step['observation']['gripper_states'].numpy()

            ee_states_list.append(ee_state)
            joint_states_list.append(joint_state)
            gripper_states_list.append(gripper_state)

            # 5. 读取图像，并保存
            # agentview_rgb 和 eye_in_hand_rgb 均为 shape=(224,224,3)
            agentview_img_np = step['observation']['agentview_rgb'].numpy()

            # 将 numpy 数组转换为 PIL Image 对象
            agentview_img = Image.fromarray(agentview_img_np)

            # 构造图像保存的文件名（按步编号命名）
            agentview_filename = os.path.join(agentview_folder, f'{step_idx}.jpeg')

            # 保存图像
            agentview_img.save(agentview_filename)

        # 6. 将列表转换为 numpy 数组，并保存为文本文件
        ee_states_array = np.vstack(ee_states_list)       # 形状 (T, 16)
        joint_states_array = np.vstack(joint_states_list)   # 形状 (T, 7)
        gripper_states_array = np.vstack(gripper_states_list)  # 形状 (T, 1)

        np.savetxt(os.path.join(folder_path, 'ee_states.txt'), ee_states_array)
        np.savetxt(os.path.join(folder_path, 'joint_states.txt'), joint_states_array)
        np.savetxt(os.path.join(folder_path, 'gripper_states.txt'), gripper_states_array)

        print(f"Episode {episode_num} extracted with {ee_states_array.shape[0]} steps.")

except Exception as e:
    print(f"Error processing dataset viola: {e}")