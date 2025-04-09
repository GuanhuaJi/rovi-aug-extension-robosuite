import os
import cv2
import argparse

def main(folder, output_video, fps):
    # 1. 定义输入图像所在的文件夹和输出视频路径（由命令行参数接收）
    # 2. 获取文件夹中的所有文件，过滤出图像文件，并按“文件名中的数字”排序
    output_dir = os.path.dirname(output_video)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)


    valid_exts = (".png", ".jpg", ".jpeg", ".bmp")  # 可根据需要添加
    all_files = os.listdir(folder)
    
    numeric_files = []
    for f in all_files:
        _, ext = os.path.splitext(f)
        ext = ext.lower()
        
        if ext in valid_exts:
            base = os.path.splitext(f)[0]
            try:
                index = int(base)
                numeric_files.append((index, f))
            except ValueError:
                pass

    if not numeric_files:
        print("错误：在文件夹中没有找到符合命名格式的图像文件。")
        return

    numeric_files.sort(key=lambda x: x[0])
    sorted_filenames = [x[1] for x in numeric_files]
    
    # 3. 读取第一张图像以获取视频尺寸
    first_image_path = os.path.join(folder, sorted_filenames[0])
    frame0 = cv2.imread(first_image_path)
    if frame0 is None:
        print(f"错误：无法读取图像 {first_image_path}")
        return
    height, width, _ = frame0.shape

    # 4. 创建 VideoWriter 对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # 5. 依次读取排序好的图像并写入视频
    for img_name in sorted_filenames:
        img_path = os.path.join(folder, img_name)
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"警告：无法读取图像 {img_path}，跳过。")
            continue
        out.write(frame)

    out.release()
    print(f"视频已生成：{output_video}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert folder of numbered images into a video.")
    parser.add_argument("--folder", type=str, required=True,
                        help="Path to the folder containing the images.")
    parser.add_argument("--output_video", type=str, required=True,
                        help="Path to the output MP4 file.")
    parser.add_argument("--fps", type=int, default=30,
                        help="Frames per second for the output video.")
    
    args = parser.parse_args()
    
    main(args.folder, args.output_video, args.fps)