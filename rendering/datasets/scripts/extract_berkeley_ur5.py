import os
import numpy as np
import tensorflow_datasets as tfds
from PIL import Image

'''
FeaturesDict({
    'steps': Dataset({
        'action': FeaturesDict({
            'gripper_closedness_action': float32,
            'rotation_delta': Tensor(shape=(3,), dtype=float32, description=Delta change in roll, pitch, yaw.),
            'terminate_episode': float32,
            'world_vector': Tensor(shape=(3,), dtype=float32, description=Delta change in XYZ.),
        }),
        'is_first': bool,
        'is_last': bool,
        'is_terminal': bool,
        'observation': FeaturesDict({
            'hand_image': Image(shape=(480, 640, 3), dtype=uint8),
            'image': Image(shape=(480, 640, 3), dtype=uint8),
            'image_with_depth': Image(shape=(480, 640, 1), dtype=float32),
            'natural_language_embedding': Tensor(shape=(512,), dtype=float32),
            'natural_language_instruction': string,
            'robot_state': Tensor(shape=(15,), dtype=float32, description=Explanation of the robot state can be found at https://sites.google.com/corp/view/berkeley-ur5),
        }),
        'reward': Scalar(shape=(), dtype=float32),
    }),
})
'''

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
        actions_list = []        # 动作（action）：取 action 的前 7 个元素
        language_instructions_list = []  # 语言指令（language instruction）：取 observation 中的 natural_language_instruction
        
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
            # I want word_vector + rotation_delta + gripper_closedness_action
            # 取出动作的前 7 个元素
            action = step["action"]["world_vector"].numpy()
            # 取出 gripper_closedness_action
            gripper_closedness_action = [step["action"]["gripper_closedness_action"].numpy()]
            # 取出 rotation_delta
            rotation_delta = step["action"]["rotation_delta"].numpy()
            # 将动作、gripper_closedness_action 和 rotation_delta 组合成一个数组
            action_combined = np.concatenate((action, rotation_delta, gripper_closedness_action), axis=0)
            actions_list.append(action_combined)
            language_instructions_list.append(step["observation"]["natural_language_instruction"].numpy().decode("utf-8"))
            
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
        np.savetxt(os.path.join(folder_path, "actions.txt"), np.vstack(actions_list))
        np.savetxt(os.path.join(folder_path, "language_instruction.txt"), np.array(language_instructions_list), fmt="%s")
        
        print(f"[INFO] Episode {episode_num} processed with {joint_states_array.shape[0]} steps.")

if __name__ == "__main__":
    main()