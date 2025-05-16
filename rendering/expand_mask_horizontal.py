import os
import cv2
import numpy as np
import argparse


def expand_mask_horizontal(mask_binary: np.ndarray, pad_px: int) -> np.ndarray:
    """
    仅在水平方向扩张 mask，每段分别向左右生长 pad_px 像素，
    保留行内空洞，不把最左–最右之间整条涂满。
    """
    if pad_px <= 0:
        return mask_binary.copy()

    h, w = mask_binary.shape
    expanded = np.zeros_like(mask_binary, dtype=np.uint8)

    for y in range(h):
        cols = np.flatnonzero(mask_binary[y])         # 这一行前景列索引
        if cols.size == 0:
            continue

        # 找到前景段分界：相邻列差值>1 说明中间有 0
        breaks = np.where(np.diff(cols) > 1)[0]
        seg_starts = np.insert(cols[breaks + 1], 0, cols[0])
        seg_ends   = np.append(cols[breaks], cols[-1])

        # 对每一段单独扩张
        for s, e in zip(seg_starts, seg_ends):
            left  = max(s - pad_px, 0)
            right = min(e + pad_px, w - 1)
            expanded[y, left:right + 1] = 255

    return expanded



def process_folder(input_folder: str, output_folder: str, pad_px: int) -> None:
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(input_folder):
        in_path = os.path.join(input_folder, file_name)
        out_path = os.path.join(output_folder, file_name)

        # Read as grayscale
        mask = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"[WARN] Unable to read image: {file_name}. Skipping…")
            continue

        # Binarise to ensure mask contains only 0/255.
        _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        expanded = expand_mask_horizontal(mask_binary, pad_px)
        cv2.imwrite(out_path, expanded)

    print(f"Horizontal expansion complete ➜ {output_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Expand masks only in the left/right directions.")
    parser.add_argument("--input_folder", required=True,
                        help="Folder containing 0/255 mask images.")
    parser.add_argument("--output_folder", required=True,
                        help="Folder to save expanded masks.")
    parser.add_argument("--pad_px", type=int, default=10,
                        help="Number of pixels to extend on both left and right sides (default: 10).")

    args = parser.parse_args()
    process_folder(args.input_folder, args.output_folder, args.pad_px)
