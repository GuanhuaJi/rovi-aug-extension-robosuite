#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, cv2, argparse, numpy as np

def parse_args():
    p = argparse.ArgumentParser("Shift binary masks and union them")
    p.add_argument("--input_folder",  type=str, default="./input")
    p.add_argument("--output_folder", type=str, default="./output")
    p.add_argument("--shift_x",       type=int, default=0,
                   help=">0 → right, <0 → left")
    p.add_argument("--shift_y",       type=int, default=0,
                   help=">0 → down,  <0 → up")
    return p.parse_args()

def shift_mask(mask, dx, dy):
    """使用 warpAffine 平移，画面外自动填 0"""
    h, w = mask.shape
    M = np.float32([[1, 0, dx],   # 平移矩阵
                    [0, 1, dy]])
    return cv2.warpAffine(
        mask, M, (w, h),
        flags=cv2.INTER_NEAREST,   # 最近邻保持 0/255
        borderValue=0              # 超出区域＝黑＝un-mask
    )

def main():
    args = parse_args()
    os.makedirs(args.output_folder, exist_ok=True)

    for fn in os.listdir(args.input_folder):
        in_path  = os.path.join(args.input_folder,  fn)
        out_path = os.path.join(args.output_folder, fn)

        mask = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Skip {fn}: not an image"); continue

        _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        shifted  = shift_mask(mask_bin, args.shift_x, args.shift_y)
        union    = cv2.bitwise_or(mask_bin, shifted)

        cv2.imwrite(out_path, union)

if __name__ == "__main__":
    main()
