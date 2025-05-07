#!/usr/bin/env python3
"""
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir /home/jiguanhua/openvla/datasets/bridge_orig \
  --dataset_name bridge_orig \
  --run_root_dir /home/jiguanhua/openvla/log \
  --adapter_tmp_dir /home/jiguanhua/openvla/tmp \
  --lora_rank 32 \
  --batch_size 16 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug False \
  --wandb_project <PROJECT> \
  --wandb_entity <ENTITY> \
  --save_steps <NUMBER OF GRADIENT STEPS PER CHECKPOINT SAVE>


torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py   --vla_path "openvla/openvla-7b"   --data_root_dir /home/jiguanhua/openvla/datasets   --dataset_name bridge_orig   --run_root_dir /home/jiguanhua/openvla/log   --adapter_tmp_dir /home/jiguanhua/openvla/tmp   --lora_rank 32   --batch_size 1   --grad_accumulation_steps 4   --learning_rate 5e-4   --image_aug False


Merge two folders of binary masks (PNG/JPG/TIFF …).
By default outputs the pixel-wise INTERSECTION; switch to UNION via --mode.

用法示例
--------
# 交集（默认）
python union_mask.py /home/jiguanhua/mirage/robot2robot/rendering/datasets/states/stack/episode_0/color_masks /
 /home/jiguanhua/mirage/robot2robot/rendering/paired_images/stack/Panda_mask_expanded/0 /
 /home/jiguanhua/mirage/robot2robot/rendering/paired_images/stack/Panda_mask_expanded_intersect/0

# 并集
python union_mask.py /home/jiguanhua/mirage/robot2robot/rendering/datasets/states/stack/episode_1/color_masks \
 /home/jiguanhua/mirage/robot2robot/rendering/paired_images/stack/Panda_mask_expanded/1 \
 /home/jiguanhua/mirage/robot2robot/rendering/paired_images/stack/Panda_mask_expanded_union/1 --mode union
"""
import argparse
import pathlib
import sys
import cv2
import numpy as np

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def is_img(p: pathlib.Path) -> bool:
    return p.suffix.lower() in IMG_EXTS

def merge_pair(a_path: pathlib.Path, b_path: pathlib.Path, mode: str) -> np.ndarray:
    """Load two masks (grayscale) and return merged binary mask."""
    a = cv2.imread(str(a_path), cv2.IMREAD_GRAYSCALE)
    b = cv2.imread(str(b_path), cv2.IMREAD_GRAYSCALE)

    if a is None or b is None:
        raise ValueError(f"读取失败: '{a_path}' 或 '{b_path}'")

    if a.shape != b.shape:
        raise ValueError(f"尺寸不一致: {a_path.name} vs {b_path.name}")

    # 将非零视为 1
    a_bin = a > 125
    b_bin = b > 125

    if mode == "intersection":
        merged = np.logical_and(a_bin, b_bin)
    else:  # union
        merged = np.logical_or(a_bin, b_bin)

    # 转回 0/255 uint8
    return (merged.astype(np.uint8) * 255)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pixel-wise merge (intersection/union) of two folders of masks."
    )
    parser.add_argument("folder_a", type=pathlib.Path,
                        help="第一组 mask 的文件夹")
    parser.add_argument("folder_b", type=pathlib.Path,
                        help="第二组 mask 的文件夹")
    parser.add_argument("output_dir", type=pathlib.Path,
                        help="输出文件夹 (自动创建)")
    parser.add_argument("--mode", choices=["intersection", "union"],
                        default="intersection",
                        help="合并方式: intersection (默认) 或 union")

    args = parser.parse_args()

    if not args.folder_a.exists() or not args.folder_b.exists():
        sys.exit("❌ 输入文件夹不存在！")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 遍历 A 中的所有图像文件
    for a_path in args.folder_a.iterdir():
        if not is_img(a_path):
            continue


        b_path = args.folder_b / (a_path.name.split("_")[0] + ".jpg")
        if not b_path.exists():
            print(f"[WARN] {b_path.name} 在 B 中缺失，跳过")
            continue

        try:
            merged_mask = merge_pair(a_path, b_path, args.mode)
        except ValueError as e:
            print(f"[WARN] {e}")
            continue

        out_path = args.output_dir / (a_path.name.split("_")[0] + ".jpg")
        cv2.imwrite(str(out_path), merged_mask)
        print(f"[OK] UNION 保存 {out_path.relative_to(args.output_dir.parent)}")

if __name__ == "__main__":
    main()
