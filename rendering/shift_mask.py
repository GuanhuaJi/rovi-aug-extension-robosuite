#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(
        description="Shift binary masks upwards and combine them (union)."
    )
    parser.add_argument(
        "--input_folder",
        type=str,
        default="./input",
        help="Folder containing the original binary mask images."
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="./output",
        help="Folder to save the processed mask images."
    )
    parser.add_argument(
        "--shift_pixels",
        type=int,
        default=20,
        help="Number of pixels to shift the mask upwards."
    )
    return parser.parse_args()

def shift_mask_up(mask_binary, shift_pixels):
    """
    将二值掩膜向上平移 shift_pixels 个像素。超出顶部的部分被抛弃，底部新出现的部分填充为0。
    :param mask_binary: 0/255 的二值图 (np.uint8)
    :param shift_pixels: int, 向上平移的像素数
    :return: 平移后的掩膜 (np.uint8, 0/255)
    """
    h, w = mask_binary.shape
    # 创建一个同样大小的空白mask
    shifted = np.zeros((h, w), dtype=np.uint8)

    # 若 shift_pixels >= h，全部移出图像，则结果全为0
    if shift_pixels >= h:
        return shifted  # 全黑

    # 将 mask_binary 的[shift_pixels : h] 区域，复制到 shifted 的[0 : h - shift_pixels]
    shifted[0 : h - shift_pixels, :] = mask_binary[shift_pixels : h, :]

    return shifted

def main():
    args = parse_args()
    input_folder = args.input_folder
    output_folder = args.output_folder
    shift_pixels = args.shift_pixels

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    file_list = os.listdir(input_folder)
    for file_name in file_list:
        in_path = os.path.join(input_folder, file_name)
        out_path = os.path.join(output_folder, file_name)

        # 读取为灰度图
        mask = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"无法读取图像文件: {file_name}, 跳过...")
            continue

        # 二值化（确保是0和255）
        _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # 1) 将掩膜向上平移
        mask_shifted = shift_mask_up(mask_binary, shift_pixels)

        # 2) 取并集：可以直接使用 bitwise_or
        union_mask = cv2.bitwise_or(mask_binary, mask_shifted)

        # 保存结果
        cv2.imwrite(out_path, union_mask)
        print(f"处理完成: {file_name} -> {out_path}")

if __name__ == "__main__":
    main()