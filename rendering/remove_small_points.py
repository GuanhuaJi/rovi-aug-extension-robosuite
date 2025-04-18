import cv2
import numpy as np
import os
import glob
import argparse

def remove_small_components(mask, min_size=50):
    """
    对二值图像进行连通域分析，移除所有像素面积小于 min_size 的连通区域。

    参数:
      mask: 输入的二值 Mask（0/255，非二值时建议先进行阈值处理）。
      min_size: 连通区域的最小像素面积，小于该值的区域将被清除。

    返回:
      output_mask: 处理后的二值 Mask，保留面积大于等于 min_size 的区域。
    """
    # 使用 8 连通性分析连通域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    # 创建与 mask 相同尺寸的输出图像（初始化为全 0 黑色）
    output_mask = np.zeros_like(mask)
    
    # 从 1 开始，跳过背景（label 0）
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_size:
            output_mask[labels == label] = 255
    return output_mask

def process_folder(input_folder, output_folder, min_size=50):
    """
    批量处理输入文件夹中所有图像，将处理后的结果以相同的文件名保存到输出文件夹。

    参数:
      input_folder: 包含待处理 Mask 图像的文件夹路径
      output_folder: 保存处理后图像的文件夹路径
      min_size: 连通区域最小面积，小于该值的区域将被移除
    """
    # 如果输出文件夹不存在，则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 定义常见图像文件扩展名
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_folder, ext)))
    
    if not image_files:
        print("在输入文件夹中找不到任何图像文件。")
        return

    for file_path in image_files:
        #print(f"处理 {file_path} ...")
        # 以灰度模式读取图像
        mask = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"无法加载图像 {file_path}，跳过。")
            continue
        
        # 如果图像非二值，先进行阈值处理
        _, binary_mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
        
        # 移除小区域
        cleaned_mask = remove_small_components(binary_mask, min_size)
        
        # 生成输出路径，使用同样的文件名
        filename = os.path.basename(file_path)
        output_path = os.path.join(output_folder, filename)
        
        # 保存处理后的图像
        cv2.imwrite(output_path, cleaned_mask)
    
    print(f"Remove small points处理完成：{output_folder}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="批量处理文件夹中的 Mask，移除孤立的小点（面积较小的连通区域），并保存到指定的输出文件夹。"
    )
    parser.add_argument('input_folder', type=str, help='输入 Mask 文件夹路径')
    parser.add_argument('output_folder', type=str, help='输出 Mask 文件夹路径')
    parser.add_argument('--min_size', type=int, default=50, help='连通区域最小面积，小于该值的区域将被移除（默认值 50 像素）')
    args = parser.parse_args()

    process_folder(args.input_folder, args.output_folder, args.min_size)