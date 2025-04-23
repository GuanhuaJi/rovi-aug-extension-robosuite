import h5py

'''
def print_hdf5_structure(item, indent=0):
    """
    递归打印 HDF5 文件结构的辅助函数。
    
    Parameters:
    - item: 可以是 h5py.File 或者 h5py.Group
    - indent: 缩进层次，用来显示层级关系
    """
    if isinstance(item, h5py.Dataset):
        # 如果是 dataset，则显示其名称、形状、数据类型
        print(' ' * indent + f"Dataset: {item.name}, shape: {item.shape}, dtype: {item.dtype}")
    else:
        # 如果是 group 或者 file（本质也是 group），先打印该 group 的名称
        print(' ' * indent + f"Group: {item.name}")
        # 然后遍历 group 下所有子元素（可再递归调用）
        for key in item.keys():
            subitem = item[key]
            print_hdf5_structure(subitem, indent + 4)  # 缩进增加以便展示层次关系

# 测试：打开一个现有的 HDF5 文件并调用上述函数
hdf5_filename = "/home/jiguanhua/mirage/robot2robot/image84/lift/image_84.hdf5"
with h5py.File(hdf5_filename, 'r') as f:
    print_hdf5_structure(f)
'''


def print_hdf5_structure(item, indent=0):
    """
    递归打印 HDF5 文件结构的辅助函数。
    
    Parameters:
    - item: 可以是 h5py.File 或者 h5py.Group
    - indent: 缩进层次，用来显示层级关系
    """
    if isinstance(item, h5py.Dataset):
        print(' ' * indent + f"Dataset: {item.name}, shape: {item.shape}, dtype: {item.dtype}")
    else:
        print(' ' * indent + f"Group: {item.name}")
        for key in item.keys():
            subitem = item[key]
            print_hdf5_structure(subitem, indent + 4)

# 打开文件后只检查指定子目录（group）
hdf5_filename = "/home/jiguanhua/mirage/robot2robot/image84/lift/image_84.hdf5"
with h5py.File(hdf5_filename, 'r') as f:
    demokey_pairing = [0, 1, 10, 100, 101, 
                       102, 103, 104, 105, 106, 
                       107, 108, 109, 11, 110, 
                       111, 112, 113, 114, 115, 
                       116, 117, 118, 119, 12, 
                       120, 121, 122, 123, 124, 
                       125, 126, 127, 128, 129, 
                       13, 130, 131, 132, 133, 
                       134, 135, 136, 137, 138, 
                       139, 14, 140, 141, 142, 
                       143, 144, 145, 146, 147, 
                       148, 149, 15, 150, 151, 
                       152, 153, 154, 155, 156, 
                       157, 158, 159, 16, 160, 
                       161, 162, 163, 164, 165, 
                       166, 167, 168, 169, 17, 
                       170, 171, 172, 173, 174, 
                       175, 176, 177, 178, 179, 
                       18, 180, 181, 182, 183, 
                       184, 185, 186, 187, 188, 
                       189, 19, 190, 191, 192, 
                       193, 194, 195, 196, 197, 
                       198, 199, 2, 20, 21, 
                       22, 23, 24, 25, 26, 
                       27, 28, 29, 3, 30, 
                       31, 32, 33, 34, 35, 
                       36, 37, 38, 39, 4, 
                       40, 41, 42, 43, 44, 
                       45, 46, 47, 48, 49, 
                       5, 50, 51, 52, 53, 
                       54, 55, 56, 57, 58, 
                       59, 6, 60, 61, 62, 
                       63, 64, 65, 66, 67, 
                       68, 69, 7, 70, 71, 
                       72, 73, 74, 75, 76, 
                       77, 78, 79, 8, 80, 
                       81, 82, 83, 84, 85, 
                       86, 87, 88, 89, 9, 
                       90, 91, 92, 93, 94, 
                       95, 96, 97, 98, 99]
    # Suppose the group you want to check is called 'my_sub_group'
    #subgroup = f[f"data/demo_{demokey_pairing[10]}/obs"]  # replace with your actual path
    subgroup = f[f"data/demo_80/obs"]
    print_hdf5_structure(subgroup)


'''
import numpy as np

for i in range(200):
    file_path = f"/home/jiguanhua/mirage/robot2robot/rendering/paired_images/lift/Panda_eef_states_{i}.npy"
    data = np.load(file_path)
    if data.shape[1] == 3:
        print(i, data.shape)
'''

'''
import numpy as np

i = 0
file_path = f"/home/jiguanhua/mirage/robot2robot/rendering/paired_images/lift/Panda_eef_states_{i}.npy"
data = np.load(file_path)
print("Panda:", data[0, :3])

file_path = f"/home/jiguanhua/mirage/robot2robot/rendering/paired_images/lift/Sawyer_eef_states_{i}.npy"
data = np.load(file_path)
print("Sawyer:", data[0, :3])

file_path = f"/home/jiguanhua/mirage/robot2robot/rendering/paired_images/lift/IIWA_eef_states_{i}.npy"
data = np.load(file_path)
print("IIWA:", data[0, :3])

file_path = f"/home/jiguanhua/mirage/robot2robot/rendering/paired_images/lift/Jaco_eef_states_{i}.npy"
data = np.load(file_path)
print("Jaco:", data[0, :3])
'''