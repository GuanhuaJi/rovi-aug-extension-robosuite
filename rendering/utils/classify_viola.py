#!/usr/bin/env python3
"""
classify_corner.py
------------------
遍历给定目录下的所有一级子目录，读取每个子目录里的 0.gped 图片，
检查右下角 5×5 像素平均灰度值：
    < 128  → 归为「黑」列表
    ≥ 128 → 归为「白」列表

用法：
    python /home/guanhuaji/mirage/robot2robot/rendering/utils/classify_viola.py /home/guanhuaji/mirage/robot2robot/rendering/datasets/states/viola
"""

import sys
import pathlib
from PIL import Image   # pip install pillow

THRESHOLD = 128  # 灰度阈值，可自行调整

def corner_is_dark(img: Image.Image, threshold: int = THRESHOLD) -> bool:
    """返回 True 表示偏黑，False 表示偏白"""
    img = img.convert("L")              # 灰度
    w, h = img.size
    if w < 5 or h < 5:
        raise ValueError("图片尺寸小于 5×5")
    # 取右下角 5×5
    crop = img.crop((w - 5, h - 5, w, h))
    mean = sum(crop.getdata()) / 25
    return mean < threshold

def main(root_dir: pathlib.Path):
    dark_subfolders, white_subfolders = [], []

    for sub in sorted(root_dir.iterdir()):
        if not sub.is_dir():
            continue
        img_path = sub / "images" / "0.jpeg"
        if not img_path.is_file():
            print(f"[WARN] 子目录 {sub.name} 中缺少 0.gped，跳过")
            continue
        try:
            with Image.open(img_path) as im:
                if corner_is_dark(im):
                    dark_subfolders.append(int(sub.name.split("_")[1]))
                else:
                    white_subfolders.append(int(sub.name.split("_")[1]))
        except Exception as e:
            print(f"[ERR] 处理 {img_path} 失败：{e}")

    print("\n=== 偏黑 (右下角较暗) ===")
    print(dark_subfolders)
    # for name in dark_subfolders:
    #     print(name)

    print("\n=== 偏白 (右下角较亮) ===")
    print(white_subfolders)
    # for name in white_subfolders:
    #     print(name)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python classify_corner.py <folder>")
        sys.exit(1)

    root = pathlib.Path(sys.argv[1]).expanduser().resolve()
    if not root.is_dir():
        print(f"路径 {root} 不是目录或不存在")
        sys.exit(1)

    main(root)
