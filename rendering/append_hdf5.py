import os
import glob
import h5py
import numpy as np
from PIL import Image

# ---------------------------
# 1. 配置部分：修改为你自己的路径
# ---------------------------
dataset = "lift"


hdf5_path    = f"/home/jiguanhua/mirage/robot2robot/image84/{dataset}/image_84.hdf5"   # 你已有的HDF5文件路径
path_sawyer  = f"/home/jiguanhua/mirage/robot2robot/rendering/cross_inpainting/{dataset}_Sawyer"
path_iiwa     = f"/home/jiguanhua/mirage/robot2robot/rendering/cross_inpainting/{dataset}_IIWA"
path_jaco   = f"/home/jiguanhua/mirage/robot2robot/rendering/cross_inpainting/{dataset}_Jaco"
path_ur5e   = f"/home/jiguanhua/mirage/robot2robot/rendering/cross_inpainting/{dataset}_UR5e"
path_kinova3   = f"/home/jiguanhua/mirage/robot2robot/rendering/cross_inpainting/{dataset}_Kinova3"
path_panda  = f"/home/jiguanhua/mirage/robot2robot/rendering/cross_inpainting/{dataset}_Panda"

num_episodes = 200  # 你有多少个episode，或你想循环到多少

# 是否需要把图像缩放到 84×84
RESIZE_TO_84x84 = True 
path_eef = f"/home/jiguanhua/mirage/robot2robot/rendering/paired_images/{dataset}"  
FILE_PATTERN = "*_eef_states_*.npy"

def numeric_prefix_sort_key(filename):
    """
    Extract the numeric prefix from a filename like '12.jpg' or '003.png' 
    and return it as an integer.
    """
    # 1) Strip out the directory path, leaving just the filename
    basename = os.path.basename(filename)
    # 2) Remove the extension to handle files like '12.jpg' -> '12'
    name_without_ext, _ = os.path.splitext(basename)
    # 3) Convert to integer (assuming the entire name_without_ext is numeric)
    return int(name_without_ext)
# ---------------------------
# 2. 定义一个函数，用于从某个文件夹里顺序读图并拼成四维数组
# ---------------------------
def load_images_as_array(folder_path, sort_key=None):
    """
    从folder_path读取全部图像文件，并拼成 (N, H, W, 3) 的numpy数组。
    可选 sort_key 来进行特殊排序；默认会按文件名升序排序。
    """
    # 找出该文件夹下的所有图片(根据实际情况改后缀，如 *.jpg, *.png)
    image_files = glob.glob(os.path.join(folder_path, "*.jpg"))
    image_files += glob.glob(os.path.join(folder_path, "*.png"))
    # 去重并排序
    image_files = list(set(image_files))
    image_files.sort(key=numeric_prefix_sort_key)
    
    if len(image_files) == 0:
        print(f"警告：文件夹 {folder_path} 下未找到任何图像。")
        return None

    arrays = []
    for img_file in image_files:
        img = Image.open(img_file).convert("RGB")  # 确保是3通道RGB
        if RESIZE_TO_84x84:
            img = img.resize((84, 84), resample=Image.BILINEAR)
        arr = np.array(img, dtype=np.uint8)  # (H, W, 3)
        arrays.append(arr)
    
    data = np.stack(arrays, axis=0)  # (num_frames, H, W, 3)
    return data

def load_eef_states_into_dict(eef_folder, robot_dataset):
    """
    在 eef_folder 中根据 robot_dataset 来匹配文件名，例如:
      如果 robot_dataset = 'Sawyer'，那么匹配 Sawyer_eef_states_*.npy
    然后解析出 episode_id = <数字> 并读入 numpy 数组。
    将其存入一个 dict，键为 episode_id，值为加载的 eef_data。
    
    参数:
      eef_folder: 存放 .npy 文件的目录
      robot_dataset: 机器人名或数据集标识, 例如 'Sawyer'
    
    文件命名假设形如: <robot_dataset>_eef_states_<episode_id>.npy
    """
    # 根据 robot_dataset 动态生成匹配模式, eg: "Sawyer_eef_states_*.npy"
    file_pattern = f"{robot_dataset}_eef_states_*.npy"
    pattern = os.path.join(eef_folder, file_pattern)
    file_list = glob.glob(pattern)
    
    eef_data_dict = {}
    print(file_list)
    for fpath in file_list:
        fname = os.path.basename(fpath)
        # 假设文件名结构: <robot_dataset>_eef_states_<episode_id>.npy
        try:
            parts = fname.split("_")  # 例如: ["Sawyer", "eef", "states", "55.npy"]
            ep_str = parts[-1].replace(".npy", "")  # 得到 "55"
            episode_id = int(ep_str)               # 转成 int
        except Exception as e:
            print(f"文件名 {fname} 无法解析 episode号, 跳过. Error: {e}")
            continue
        
        # 读取 numpy 数组 (arr.shape 视实际数据而定)
        arr = np.load(fpath)
        # 示例：只保留前3列
        eef_data_dict[episode_id] = arr[:, :3]

    return eef_data_dict
# ---------------------------
# 3. 主逻辑：循环所有 episode，把三种视角图像写入 HDF5
# ---------------------------
with h5py.File(hdf5_path, "a") as f:  # 以 'a' 模式打开，允许写入
    demokey_pairing = [0, 1, 10, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 11, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 12, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 13, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 14, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 15, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 16, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 17, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 18, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 19, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 2, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 3, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 4, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 5, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 6, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 7, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 8, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 9, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
    sawyer_eef_dict = load_eef_states_into_dict(path_eef, "Sawyer")
    iiwa_eef_dict   = load_eef_states_into_dict(path_eef,   "IIWA")
    jaco_eef_dict   = load_eef_states_into_dict(path_eef,   "Jaco")
    panda_eef_dict  = load_eef_states_into_dict(path_eef,  "Panda")
    ur5e_eef_dict  = load_eef_states_into_dict(path_eef,  "UR5e")
    kinova3_eef_dict = load_eef_states_into_dict(path_eef, "Kinova3")
    for episode_id in range(0, 200):
        # 3.1 找到 HDF5 中 /data/demo_{episode_id}/obs 这个 group
        group_path = f"data/demo_{demokey_pairing[episode_id]}/obs"
        if group_path not in f:
            print(f"跳过：{group_path} 在HDF5里不存在，可能这个episode不存在。")
            continue
        obs_group = f[group_path]
        
        # 3.2 分别准备三种视角图片数据
        sawyer_folder = os.path.join(path_sawyer, str(episode_id))
        iiwa_folder    = os.path.join(path_iiwa,    str(episode_id))
        jaco_folder  = os.path.join(path_jaco,  str(episode_id))
        ur5e_folder  = os.path.join(path_ur5e,  str(episode_id))
        kinova3_folder = os.path.join(path_kinova3, str(episode_id))
        panda_folder  = os.path.join(path_panda, str(episode_id))

        
        sawyer_data = load_images_as_array(sawyer_folder)
        iiwa_data    = load_images_as_array(iiwa_folder)
        jaco_data  = load_images_as_array(jaco_folder)
        ur5e_data  = load_images_as_array(ur5e_folder)
        kinova3_data = load_images_as_array(kinova3_folder)
        panda_data  = load_images_as_array(panda_folder)

        
        # 如果某个文件夹为空或不存在，你要么跳过，要么给个警告
        if (sawyer_data is None) or (iiwa_data is None) or (jaco_data is None) or (ur5e_data is None) or (kinova3_data is None) or (panda_data is None):
            print(f"警告：Episode {demokey_pairing[episode_id]} 四类视角中有数据加载失败，跳过创建。")
            continue
        
        # 3.3 如果之前已存在同名数据集，先删除
        if "agentview_image_sawyer" in obs_group:
            del obs_group["agentview_image_sawyer"]
        if "agentview_image_iiwa" in obs_group:
            del obs_group["agentview_image_iiwa"]
        if "agentview_image_jaco" in obs_group:
            del obs_group["agentview_image_jaco"]
        if "agentview_image_ur5e" in obs_group:
            del obs_group["agentview_image_ur5e"]
        if "agentview_image_kinova3" in obs_group:
            del obs_group["agentview_image_kinova3"]
        if "agentview_image_panda" in obs_group:
            del obs_group["agentview_image_panda"]
        
        # 3.4 创建新的 dataset
        #    可以考虑 compression、chunks 等参数，示例如下：
        obs_group.create_dataset(
            "agentview_image_sawyer", 
            data=sawyer_data,
            compression="gzip",
            compression_opts=4,
            chunks=True
        )
        obs_group.create_dataset(
            "agentview_image_iiwa", 
            data=iiwa_data,
            compression="gzip",
            compression_opts=4,
            chunks=True
        )
        obs_group.create_dataset(
            "agentview_image_jaco", 
            data=jaco_data,
            compression="gzip",
            compression_opts=4,
            chunks=True
        )
        obs_group.create_dataset(
            "agentview_image_ur5e", 
            data=ur5e_data,
            compression="gzip",
            compression_opts=4,
            chunks=True
        )
        obs_group.create_dataset(
            "agentview_image_kinova3", 
            data=kinova3_data,
            compression="gzip",
            compression_opts=4,
            chunks=True
        )
        obs_group.create_dataset(
            "agentview_image_panda", 
            data=panda_data,
            compression="gzip",
            compression_opts=4,
            chunks=True
        )

        # Sawyer
        if episode_id in sawyer_eef_dict:
            sawyer_eef_arr = sawyer_eef_dict[episode_id]
            if "Sawyer_eef_states" in obs_group:
                del obs_group["Sawyer_eef_states"]
            obs_group.create_dataset(
                "Sawyer_eef_states",
                data=sawyer_eef_arr,
                compression="gzip",
                compression_opts=4,
                chunks=True
            )
        
        # IIWA
        if episode_id in iiwa_eef_dict:
            iiwa_eef_arr = iiwa_eef_dict[episode_id]
            if "IIWA_eef_states" in obs_group:
                del obs_group["IIWA_eef_states"]
            obs_group.create_dataset(
                "IIWA_eef_states",
                data=iiwa_eef_arr,
                compression="gzip",
                compression_opts=4,
                chunks=True
            )
        
        # Jaco
        if episode_id in jaco_eef_dict:
            jaco_eef_arr = jaco_eef_dict[episode_id]
            if "Jaco_eef_states" in obs_group:
                del obs_group["Jaco_eef_states"]
            obs_group.create_dataset(
                "Jaco_eef_states",
                data=jaco_eef_arr,
                compression="gzip",
                compression_opts=4,
                chunks=True
            )
        
        if episode_id in ur5e_eef_dict:
            ur5e_eef_arr = ur5e_eef_dict[episode_id]
            if "UR5e_eef_states" in obs_group:
                del obs_group["UR5e_eef_states"]
            obs_group.create_dataset(
                "UR5e_eef_states",
                data=ur5e_eef_arr,
                compression="gzip",
                compression_opts=4,
                chunks=True
            )
        
        if episode_id in kinova3_eef_dict:
            kinova3_eef_arr = kinova3_eef_dict[episode_id]
            if "Kinova3_eef_states" in obs_group:
                del obs_group["Kinova3_eef_states"]
            obs_group.create_dataset(
                "Kinova3_eef_states",
                data=kinova3_eef_arr,
                compression="gzip",
                compression_opts=4,
                chunks=True
            )

        # Panda
        if episode_id in panda_eef_dict:
            panda_eef_arr = panda_eef_dict[episode_id]
            if "Panda_eef_states" in obs_group:
                del obs_group["Panda_eef_states"]
            obs_group.create_dataset(
                "Panda_eef_states",
                data=panda_eef_arr,
                compression="gzip",
                compression_opts=4,
                chunks=True
            )

        print(f"Episode {episode_id} => 成功写入三组图像数据。")

print("全部处理完毕。")