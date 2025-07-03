import os
import glob
import h5py
import numpy as np
from PIL import Image

# ---------------------------
# 1. 配置部分：修改为你自己的路径
# ---------------------------
dataset = "stack"


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
    #demokey_pairing = [0, 1, 10, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 11, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 12, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 13, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 14, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 15, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 16, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 17, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 18, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 19, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 2, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 3, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 4, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 5, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 6, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 7, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 8, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 9, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
    demokey_pairing = [0, 1, 10, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 11, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 12, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 13, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 14, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 15, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 16, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 17, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 18, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 19, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 
                        2, 20, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 21, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 22, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 23, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 24, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 25, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 26, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 27, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 28, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 29, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 
                        3, 30, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 31, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 32, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 33, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 34, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 35, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 36, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 37, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 38, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 39, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 
                        4, 40, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 41, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 42, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 43, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 44, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 45, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 46, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 47, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 48, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 49, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 
                        5, 50, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 51, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 52, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 53, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 54, 540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 55, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 56, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 57, 570, 571, 572, 573, 574, 575, 576, 577, 578, 579, 58, 580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 59, 590, 591, 592, 593, 594, 595, 596, 597, 598, 599, 
                        6, 60, 600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 61, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 62, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 63, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 64, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 65, 650, 651, 652, 653, 654, 655, 656, 657, 658, 659, 66, 660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 67, 670, 671, 672, 673, 674, 675, 676, 677, 678, 679, 68, 680, 681, 682, 683, 684, 685, 686, 687, 688, 689, 69, 690, 691, 692, 693, 694, 695, 696, 697, 698, 699, 
                        7, 70, 700, 701, 702, 703, 704, 705, 706, 707, 708, 709, 71, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719, 72, 720, 721, 722, 723, 724, 725, 726, 727, 728, 729, 73, 730, 731, 732, 733, 734, 735, 736, 737, 738, 739, 74, 740, 741, 742, 743, 744, 745, 746, 747, 748, 749, 75, 750, 751, 752, 753, 754, 755, 756, 757, 758, 759, 76, 760, 761, 762, 763, 764, 765, 766, 767, 768, 769, 77, 770, 771, 772, 773, 774, 775, 776, 777, 778, 779, 78, 780, 781, 782, 783, 784, 785, 786, 787, 788, 789, 79, 790, 791, 792, 793, 794, 795, 796, 797, 798, 799, 
                        8, 80, 800, 801, 802, 803, 804, 805, 806, 807, 808, 809, 81, 810, 811, 812, 813, 814, 815, 816, 817, 818, 819, 82, 820, 821, 822, 823, 824, 825, 826, 827, 828, 829, 83, 830, 831, 832, 833, 834, 835, 836, 837, 838, 839, 84, 840, 841, 842, 843, 844, 845, 846, 847, 848, 849, 85, 850, 851, 852, 853, 854, 855, 856, 857, 858, 859, 86, 860, 861, 862, 863, 864, 865, 866, 867, 868, 869, 87, 870, 871, 872, 873, 874, 875, 876, 877, 878, 879, 88, 880, 881, 882, 883, 884, 885, 886, 887, 888, 889, 89, 890, 891, 892, 893, 894, 895, 896, 897, 898, 899, 
                        9, 90, 900, 901, 902, 903, 904, 905, 906, 907, 908, 909, 91, 910, 911, 912, 913, 914, 915, 916, 917, 918, 919, 92, 920, 921, 922, 923, 924, 925, 926, 927, 928, 929, 93, 930, 931, 932, 933, 934, 935, 936, 937, 938, 939, 94, 940, 941, 942, 943, 944, 945, 946, 947, 948, 949, 95, 950, 951, 952, 953, 954, 955, 956, 957, 958, 959, 96, 960, 961, 962, 963, 964, 965, 966, 967, 968, 969, 97, 970, 971, 972, 973, 974, 975, 976, 977, 978, 979, 98, 980, 981, 982, 983, 984, 985, 986, 987, 988, 989, 99, 990, 991, 992, 993, 994, 995, 996, 997, 998, 999]
    sawyer_eef_dict = load_eef_states_into_dict(path_eef, "Sawyer")
    iiwa_eef_dict   = load_eef_states_into_dict(path_eef,   "IIWA")
    jaco_eef_dict   = load_eef_states_into_dict(path_eef,   "Jaco")
    panda_eef_dict  = load_eef_states_into_dict(path_eef,  "Panda")
    ur5e_eef_dict  = load_eef_states_into_dict(path_eef,  "UR5e")
    kinova3_eef_dict = load_eef_states_into_dict(path_eef, "Kinova3")
    for episode_id in range(0, 1000):
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