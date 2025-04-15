import os
import glob
import h5py
import numpy as np
from PIL import Image

# ---------------------------
# 1. 配置部分：修改为你自己的路径
# ---------------------------
dataset = "can"


hdf5_path    = f"/home/jiguanhua/mirage/robot2robot/image84/{dataset}/image_84.hdf5"   # 你已有的HDF5文件路径
path_sawyer  = f"/home/jiguanhua/mirage/robot2robot/rendering/cross_inpainting/{dataset}_IIWA"
path_iiwa     = f"/home/jiguanhua/mirage/robot2robot/rendering/cross_inpainting/{dataset}_IIWA"
path_jaco   = f"/home/jiguanhua/mirage/robot2robot/rendering/cross_inpainting/{dataset}_Jaco"

num_episodes = 200  # 你有多少个episode，或你想循环到多少

# 是否需要把图像缩放到 84×84
RESIZE_TO_84x84 = True 
path_eef = f"/home/jiguanhua/mirage/robot2robot/rendering/paired_images/{dataset}"  
FILE_PATTERN = "*_eef_states_*.npy"

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
    image_files.sort(key=sort_key)  # 默认按文件名排序
    
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

def load_eef_states_into_dict(eef_folder, robot_name):
    """
    在 eef_folder 中找所有 .npy 文件(如 Sawyer_eef_states_55.npy)，
    解析出 episode_id = 55，并读入 numpy 数组。将其存入一个 dict，
    键为 episode_id，值为加载的 eef_data。
    robot_name 只是用于打印或检查，无逻辑需求的话可以去掉。
    """
    pattern = os.path.join(eef_folder, FILE_PATTERN)  
    # eg: "/path/to/sawyer_eef/*_eef_states_*.npy"
    file_list = glob.glob(pattern)
    
    eef_data_dict = {}
    for fpath in file_list:
        # 文件名形如: Sawyer_eef_states_55.npy
        fname = os.path.basename(fpath)
        # 找到下划线最后那个数字(55)
        # 这里假设 fname 的结构固定: <robot>_eef_states_<episode_id>.npy
        # 也有可能有别的命名方式，需要你改 parse 逻辑
        try:
            # 拆分: ["Sawyer", "eef", "states", "55.npy"]
            parts = fname.split("_")
            # 最后一部分 "55.npy"
            ep_str = parts[-1].replace(".npy", "")  # "55"
            episode_id = int(ep_str)               # 55 => int
        except Exception as e:
            print(f"文件名 {fname} 无法解析 episode号, 跳过. Error: {e}")
            continue
        
        # 读 numpy 数组
        arr = np.load(fpath)  # arr.shape 取决于你的数据(eg: (T, dimension)?)
        eef_data_dict[episode_id] = arr

    return eef_data_dict
# ---------------------------
# 3. 主逻辑：循环所有 episode，把三种视角图像写入 HDF5
# ---------------------------
with h5py.File(hdf5_path, "a") as f:  # 以 'a' 模式打开，允许写入
    sawyer_eef_dict = load_eef_states_into_dict(path_eef, "Sawyer")
    iiwa_eef_dict   = load_eef_states_into_dict(path_eef,   "IIWA")
    jaco_eef_dict   = load_eef_states_into_dict(path_eef,   "Jaco")
    panda_eef_dict  = load_eef_states_into_dict(path_eef,  "Panda")
    for episode_id in range(num_episodes):
        # 3.1 找到 HDF5 中 /data/demo_{episode_id}/obs 这个 group
        group_path = f"data/demo_{episode_id}/obs"
        if group_path not in f:
            print(f"跳过：{group_path} 在HDF5里不存在，可能这个episode不存在。")
            continue
        obs_group = f[group_path]
        
        # 3.2 分别准备三种视角图片数据
        sawyer_folder = os.path.join(path_sawyer, str(episode_id))
        eva_folder    = os.path.join(path_iiwa,    str(episode_id))
        jacob_folder  = os.path.join(path_jaco,  str(episode_id))
        
        sawyer_data = load_images_as_array(sawyer_folder)
        eva_data    = load_images_as_array(eva_folder)
        jacob_data  = load_images_as_array(jacob_folder)
        
        # 如果某个文件夹为空或不存在，你要么跳过，要么给个警告
        if (sawyer_data is None) or (eva_data is None) or (jacob_data is None):
            print(f"警告：Episode {episode_id} 三类视角中有数据加载失败，跳过创建。")
            continue
        
        # 3.3 如果之前已存在同名数据集，先删除
        if "agentview_image_sawyer" in obs_group:
            del obs_group["agentview_image_sawyer"]
        if "agentview_image_iiwa" in obs_group:
            del obs_group["agentview_image_iiwa"]
        if "agentview_image_jaco" in obs_group:
            del obs_group["agentview_image_jaco"]
        
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
            data=eva_data,
            compression="gzip",
            compression_opts=4,
            chunks=True
        )
        obs_group.create_dataset(
            "agentview_image_jaco", 
            data=jacob_data,
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