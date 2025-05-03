import os
import numpy as np
import tensorflow_datasets as tfds
from PIL import Image  # 用来保存 image

def dataset2path(dataset_name):
    """
    根据实际存储情况，返回对应 GCS 或本地目录。
    以下只是示例，需根据真实路径修改。
    """
    return f'gs://gresearch/robotics/{dataset_name}/0.1.0'

DATASET = "furniture_bench_dataset_converted_externally_to_rlds"

try:
    # 1) 从指定目录创建 builder
    builder = tfds.builder_from_directory(builder_dir=dataset2path(DATASET))
    print(f"Features for {DATASET}:")
    print(builder.info.features)

    # 2) 检查是否存在 'train' split
    if 'train' not in builder.info.splits:
        print("No 'train' split found. Exiting.")
    else:
        total_episodes = builder.info.splits['train'].num_examples
        print(f"Dataset {DATASET} has {total_episodes} episodes in 'train' split.")

        # 3) 只读取前 20 个 episode（如果数据多的话）
        split = 'train[901:]'
        ds = builder.as_dataset(split=split)

        # 4) 遍历每个 episode
        for episode_num, episode in enumerate(ds):
            ee_states_list = []
            joint_states_list = []
            gripper_states_list = []

            # 5) 创建本地文件夹
            folder_path = f'../states/{DATASET}/episode_{episode_num + 901}'
            os.makedirs(folder_path, exist_ok=True)

            # 专门放置图像的子文件夹
            images_folder_path = os.path.join(folder_path, 'images')
            os.makedirs(images_folder_path, exist_ok=True)

            # 6) 逐步提取每个 time step
            for step_idx, step in enumerate(episode['steps']):
                # 'state' 的 shape = (35,) 对应如下：
                # [0:3]   => EEF position (x, y, z)
                # [3:7]   => EEF quaternion (qx, qy, qz, qw)
                # [7:10]  => EEF linear velocity
                # [10:13] => EEF angular velocity
                # [13:20] => 7 robot joint positions
                # [20:27] => 7 robot joint velocities
                # [27:34] => 7 joint torques
                # [34:35] => 1 gripper width

                state_35 = step['observation']['state'].numpy()

                # 前 7 个维度 (0:7) => [EEF position + EEF quaternion]
                ee_pose = state_35[:7]

                # 取关节角在 13:20
                joint_angles = state_35[13:20]

                # gripper 状态（宽度）在最后一个维度 34
                gripper_width = state_35[34]

                ee_states_list.append(ee_pose)
                joint_states_list.append(joint_angles)
                # 保证形状一致，添加到 list 时可以以 [gripper_width] 方式
                gripper_states_list.append([gripper_width])

                # 7) 读取并保存图像
                # observation['image'] shape=(224, 224, 3)
                image_np = step['observation']['image'].numpy()
                img_pil = Image.fromarray(image_np)
                img_filename = os.path.join(images_folder_path, f'{step_idx}.jpeg')
                img_pil.save(img_filename)

            # 8) 将所需的 state 信息保存为 .txt
            ee_states_array = np.vstack(ee_states_list)         # shape=(T, 7)
            joint_states_array = np.vstack(joint_states_list)   # shape=(T, 7)
            gripper_states_array = np.vstack(gripper_states_list)  # shape=(T, 1)

            np.savetxt(os.path.join(folder_path, 'ee_states.txt'), ee_states_array)
            np.savetxt(os.path.join(folder_path, 'joint_states.txt'), joint_states_array)
            np.savetxt(os.path.join(folder_path, 'gripper_states.txt'), gripper_states_array)

            print(f"Episode {episode_num + 901} extracted: {ee_states_array.shape[0]} steps.")

except Exception as e:
    print(f"Error processing dataset {DATASET}: {e}")