#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import cv2
from PIL import Image
from PIL.Image import Resampling  # Pillow>=9.1.0 里使用 Resampling.LANCZOS 而不是 ANTIALIAS

def composite_with_two_masks(imgA, maskA, imgB, maskB, fill_color=(0, 0, 0, 255)):
    """
    使用两张独立的 mask (maskA, maskB) 对 A/B 进行组合。
    - maskA: 白(255) => 保留 A；黑(0) => 丢弃 A
    - maskB: 白(255) => 从 B 中挖空；黑(0) => 保留 B

    合成顺序：
      1) 先对 B 进行“挖空”（maskB 白 => alpha=0）
      2) 对 A 进行“裁剪”（maskA 白 => alpha=255）
      3) 在同一画布上，先放 B，再贴 A，A 在前景覆盖 B
      4) 对空白区域(alpha=0)用 fill_color 填充

    返回：合成后的 RGBA 图像
    """

    # 1) B_cut: 按 maskB 把 B 挖空。maskB=255 => alpha=0, maskB=0 => alpha=255
    alphaB = Image.eval(maskB, lambda px: 255 - px)
    B_cut = imgB.copy()
    B_cut.putalpha(alphaB)

    # 2) A_cut: 按 maskA 裁剪。maskA=255 => alpha=255, maskA=0 => alpha=0
    A_cut = imgA.copy()
    A_cut.putalpha(maskA)

    # 3) 合成：B_cut 在底，A_cut 在前
    composite_rgba = Image.alpha_composite(B_cut, A_cut)

    # 4) 空白区域用 fill_color 填充
    fill_bg = Image.new("RGBA", composite_rgba.size, fill_color)
    composite_rgba = Image.alpha_composite(fill_bg, composite_rgba)

    return composite_rgba

def process_folders(folder_maskA, folder_A,
                    folder_maskB, folder_B,
                    folder_output,
                    resize_mode="A_to_B",
                    fill_color=(255,255,255,255)):
    """
    批量处理:
    - 对于索引 i (从 0 开始):
      - A: i.jpg
      - maskA: i.jpg
      - B: (i+1).jpeg
      - maskB: i.jpg
    - 若 A/B 大小不一致，根据 resize_mode 做缩放:
        "A_to_B":  A -> B 大小
        "B_to_A":  B -> A 大小
    - 合成后将图像保存到 folder_output/i.png (或 .jpg)。
    """
    if not os.path.exists(folder_output):
        os.makedirs(folder_output)

    maskA_files = sorted(os.listdir(folder_maskA))

    for maskA_file in maskA_files:
        baseA, extA = os.path.splitext(maskA_file)

        # 解析 i
        try:
            i = int(baseA)
        except ValueError:
            print(f"[SKIP] {maskA_file}：文件名前缀不是数字，跳过。")
            continue

        # 构造路径
        path_maskA = os.path.join(folder_maskA, maskA_file)
        path_A     = os.path.join(folder_A, maskA_file)  # 假设 A 跟 maskA 同后缀

        path_B     = os.path.join(folder_B, f"{i}.jpeg")  
        # maskB: i.jpg
        path_maskB = os.path.join(folder_maskB, f"{i}.jpg")

        # 文件是否存在
        if not os.path.isfile(path_maskA):
            print(f"[SKIP] maskA 不存在: {path_maskA}")
            continue
        if not os.path.isfile(path_A):
            print(f"[SKIP] A 不存在: {path_A}")
            continue
        if not os.path.isfile(path_B):
            print(f"[SKIP] B 不存在: {path_B}")
            continue
        if not os.path.isfile(path_maskB):
            print(f"[SKIP] maskB 不存在: {path_maskB}")
            continue

        try:
            # 打开图像
            maskA_img = Image.open(path_maskA).convert("L")
            A_img     = Image.open(path_A).convert("RGBA")

            maskB_img = Image.open(path_maskB).convert("L")
            B_img     = Image.open(path_B).convert("RGBA")

            # 让 maskA 与 A_img 尺寸一致
            if maskA_img.size != A_img.size:
                maskA_img = maskA_img.resize(A_img.size, Resampling.LANCZOS)

            # 让 maskB 与 B_img 尺寸一致
            if maskB_img.size != B_img.size:
                maskB_img = maskB_img.resize(B_img.size, Resampling.LANCZOS)

            # 按需做尺寸匹配
            if resize_mode == "A_to_B":
                # 以 B.size 为准，把 A 和 maskA resize
                if A_img.size != B_img.size:
                    A_img     = A_img.resize(B_img.size, Resampling.LANCZOS)
                    maskA_img = maskA_img.resize(B_img.size, Resampling.LANCZOS)
            elif resize_mode == "B_to_A":
                # 以 A.size 为准，把 B 和 maskB resize
                if B_img.size != A_img.size:
                    B_img     = B_img.resize(A_img.size, Resampling.LANCZOS)
                    maskB_img = maskB_img.resize(A_img.size, Resampling.LANCZOS)
            # 其他不处理

            final_rgba = composite_with_two_masks(A_img, maskA_img, B_img, maskB_img, fill_color=fill_color)

            # 输出文件名
            output_name = f"{i}.png"
            output_path = os.path.join(folder_output, output_name)
            final_rgba.save(output_path, "PNG")

            print(f"[OK] i={i} => {output_name}")
        except Exception as e:
            print(f"[ERROR] i={i} 合成失败: {e}")

def images_to_video(folder_images, output_video_path, fps=25):
    """
    将指定文件夹下的图片按顺序合成为 MP4 视频。
    - folder_images: 存放输出图片的文件夹
    - output_video_path: 生成 MP4 的输出路径，如 "output.mp4"
    - fps: 帧率（可自行调整）
    """
    # 读取该文件夹下所有 PNG/JPG，并按文件名排序
    valid_exts = (".png", ".jpg", ".jpeg")
    image_files = sorted([
        f for f in os.listdir(folder_images)
        if os.path.splitext(f)[1].lower() in valid_exts
    ], key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else x)

    if not image_files:
        print("[WARN] 没有找到任何图片，无法生成视频。")
        return

    # 用第一张图来确定视频的分辨率
    first_img_path = os.path.join(folder_images, image_files[0])
    first_img = cv2.imread(first_img_path)
    if first_img is None:
        print(f"[ERROR] 无法读取第一张图片：{first_img_path}")
        return
    height, width, channels = first_img.shape

    # 定义编码器
    # 注意：某些环境下可用 *'mp4v' 或 *'avc1'，也可以尝试 X264 (需要编译了相关支持)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # 逐帧写入
    for img_name in image_files:
        img_path = os.path.join(folder_images, img_name)
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"[SKIP] 无法读取图像: {img_path}")
            continue
        video_writer.write(frame)

    video_writer.release()
    print(f"[INFO] 视频已生成: {output_video_path}")

def main():
    # 配置路径
    dataset = "austin_buds"
    robot = "IIWA"
    folder_maskA = f"/mnt/newdisk/jiguanhua/mirage/robot2robot/rendering/{dataset}_{robot}_paired_images/{robot.lower()}_mask/0"
    folder_A     = f"/mnt/newdisk/jiguanhua/mirage/robot2robot/rendering/{dataset}_{robot}_paired_images/{robot.lower()}_rgb/0"
    folder_maskB = f"/mnt/newdisk/jiguanhua/mirage/robot2robot/rendering/{dataset}_{robot}_paired_images/panda_mask/0"
    folder_B     = f"/mnt/newdisk/jiguanhua/mirage/robot2robot/rendering/states/austin_buds_dataset_converted_externally_to_rlds/episode_0/images"

    folder_out   = f"{dataset}_overley_{robot}"  # 你想把输出的图像放在这里
    resize_mode  = "A_to_B"
    fill_color   = (255,255,255,255)

    # 第一步：批量生成合成后的图像
    process_folders(folder_maskA, folder_A,
                    folder_maskB, folder_B,
                    folder_out,
                    resize_mode=resize_mode,
                    fill_color=fill_color)

    # 第二步：将生成的图像合成为 MP4 视频
    output_video = f"{dataset}_overley_{robot}.mp4"
    images_to_video(folder_out, output_video_path=output_video, fps=25)

if __name__ == "__main__":
    main()