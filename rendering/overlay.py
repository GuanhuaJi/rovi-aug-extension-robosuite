import os
import re
import cv2
import numpy as np
import argparse


def extract_frame_index(filename):
    """
    假设文件名里会包含一个整数编号，比如 '10.png' 或者 'frame_10.jpg'，
    我们通过正则把它提取出来，如果有多个数字只提取第一个。
    """
    match = re.search(r'(\d+)', filename)
    if match:
        return int(match.group(1))
    else:
        return None

def merge_images_with_mask(
    original_folder: str,
    mask_folder: str,
    overlay_folder: str,
    output_folder: str,
    reverse: bool
):
    """
    从 original_folder、mask_folder、overlay_folder 中，逐帧读取并合成图像，
    最终保存到 output_folder。
    """
    # 如果输出文件夹不存在，则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 读取三个文件夹下的所有文件，并提取它们的编号
    original_files = sorted(os.listdir(original_folder))
    mask_files = sorted(os.listdir(mask_folder))
    overlay_files = sorted(os.listdir(overlay_folder))

    # 把三类文件根据编号分组，方便后面一一对应
    # 用 dict: key = 帧编号, value = 文件名
    original_dict = {}
    for f in original_files:
        idx = extract_frame_index(f)
        if idx is not None:
            original_dict[idx] = f

    mask_dict = {}
    for f in mask_files:
        idx = extract_frame_index(f)
        if idx is not None:
            mask_dict[idx] = f

    overlay_dict = {}
    for f in overlay_files:
        idx = extract_frame_index(f)
        if idx is not None:
            overlay_dict[idx] = f

    # 找出所有同时在三者中存在的帧编号
    valid_indices = sorted(set(original_dict.keys()) & set(mask_dict.keys()) & set(overlay_dict.keys()))

    # 逐帧处理
    for idx in valid_indices:
        # 构造完整路径
        orig_path = os.path.join(original_folder, original_dict[idx])
        mask_path = os.path.join(mask_folder, mask_dict[idx])
        overlay_path = os.path.join(overlay_folder, overlay_dict[idx])

        # 读取原图、mask、overlay
        orig_img = cv2.imread(orig_path, cv2.IMREAD_COLOR)
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # 只读单通道
        overlay_img = cv2.imread(overlay_path, cv2.IMREAD_COLOR)

        if orig_img is None or mask_img is None or overlay_img is None:
            print(f"警告：无法读取编号 {idx} 对应的某些文件，跳过")
            continue

        # 确保三张图尺寸相同，如若不同，可根据需要进行resize或其他处理
        if orig_img.shape[:2] != mask_img.shape[:2] or orig_img.shape[:2] != overlay_img.shape[:2]:
            print(f"警告：编号 {idx} 的图像尺寸不一致，跳过或进行额外处理")
            continue

        # 生成一个掩码： mask==255 的地方替换，mask==0 的地方保留原图
        # 如果你的 Mask 的白色并不是 255，也可以根据阈值来处理
        # 例如： mask_binary = (mask_img > 127).astype(np.uint8)
        # 具体逻辑可自行调整，这里简单假设 mask==255 即表示需要替换
        if reverse:
            mask_binary = (mask_img < 127).astype(np.uint8)
        else:
            mask_binary = (mask_img > 127).astype(np.uint8)

        # 用掩码来做像素替换
        # np.where条件里面是mask_binary>0的地方用overlay，剩余的地方用orig
        # 但是OpenCV里可以用更直接的方式：
        # orig_img[mask_binary == 1] = overlay_img[mask_binary == 1]
        # 这样就可以直接在原图上改动
        orig_img[mask_binary == 1] = overlay_img[mask_binary == 1]

        # 保存
        out_name = f"{idx}.png"  # 你也可以指定其他命名规则
        out_path = os.path.join(output_folder, out_name)
        cv2.imwrite(out_path, orig_img)

        #print(f"合成完成并保存: {out_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Merge images with mask and overlay.")
    parser.add_argument("--original_folder", type=str, required=True, 
                        help="Path to the original images folder.")
    parser.add_argument("--mask_folder", type=str, required=True,
                        help="Path to the mask images folder.")
    parser.add_argument("--overlay_folder", type=str, required=True,
                        help="Path to the overlay images folder.")
    parser.add_argument("--output_folder", type=str, required=True,
                        help="Path to the output folder.")
    parser.add_argument("--reverse", type=bool, required=False, default=False,
                        help="Path to the output folder.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    merge_images_with_mask(
        original_folder=args.original_folder,
        mask_folder=args.mask_folder,
        overlay_folder=args.overlay_folder,
        output_folder=args.output_folder,
        reverse=args.reverse
    )