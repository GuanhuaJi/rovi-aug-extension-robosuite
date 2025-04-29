#!/usr/bin/env python3
"""
Batch-generate binary masks for every image in a folder.

功能
------
‣ 读取 input_dir 下所有常见格式图片  
‣ 根据用户指定的颜色区间（RGB 或 HSV）用 cv2.inRange 做阈值分割  
‣ 把二值 mask 保存到 output_dir，文件名保持一致，附加 _mask 后缀  

示例
------
# 以 **RGB** 区间分割“偏红”的像素
python gen_color_mask.py ./images ./masks \
        --lower 150,0,0 --upper 255,80,80

# 以 **HSV** 区间分割“纯红” (Hue≈0°) 的像素
python gen_color_mask.py ./images ./masks \
        --use_hsv \
        --lower 0,120,70 --upper 10,255,255
"""

import argparse
import os
import pathlib
import cv2
import numpy as np
from typing import Tuple

# ---------- 工具函数 ----------
def parse_triplet(s: str) -> np.ndarray:
    """把 'r,g,b' or 'h,s,v' 解析成 uint8 三元组数组。"""
    try:
        nums = [int(x) for x in s.split(",")]
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"Bound '{s}' 必须是 3 个逗号分隔整数"
        ) from e
    if len(nums) != 3 or not all(0 <= n <= 255 for n in nums):
        raise argparse.ArgumentTypeError(
            f"Bound '{s}' 必须恰好 3 个 0-255 整数"
        )
    return np.array(nums, dtype=np.uint8)

def valid_image(path: pathlib.Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png",
                                   ".bmp", ".tif", ".tiff"}

# ---------- 主流程 ----------
def generate_masks(
    input_dir: pathlib.Path,
    output_dir: pathlib.Path,
    lower: np.ndarray,
    upper: np.ndarray,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for img_path in input_dir.iterdir():
        if not valid_image(img_path):
            continue

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"[WARN] 读取失败，跳过: {img_path}")
            continue


        img_space = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        mask = cv2.inRange(img_space, lower, upper)
        out_name = f"{img_path.stem}_mask.png"
        out_path = output_dir / out_name
        cv2.imwrite(str(out_path), mask)
        print(f"[OK] 保存: {out_path.relative_to(output_dir.parent)}")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate binary masks for pixels within a given color range."
    )
    parser.add_argument("input_dir", type=pathlib.Path,
                        help="包含图片的文件夹")
    parser.add_argument("output_dir", type=pathlib.Path,
                        help="保存 mask 的文件夹")
    parser.add_argument("--lower", required=True, type=parse_triplet,
                        help="下界 (R,G,B 或 H,S,V)，形如 100,0,0")
    parser.add_argument("--upper", required=True, type=parse_triplet,
                        help="上界 (R,G,B 或 H,S,V)，形如 255,150,150")

    #python generate_color_mask.py /home/jiguanhua/mirage/robot2robot/rendering/datasets/states/stack/episode_1/images /home/jiguanhua/mirage/robot2robot/rendering/datasets/states/stack/episode_1/color_masks --lower 100,0,0 --upper 255,80,80

    args = parser.parse_args()

    generate_masks(
        args.input_dir,
        args.output_dir,
        args.lower,
        args.upper,
    )

if __name__ == "__main__":
    main()
