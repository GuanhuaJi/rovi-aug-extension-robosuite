#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(
        description="Shift binary masks in any direction (up, down, left, right, or diagonal) and combine them (union)."
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
        "--shift_x",
        type=int,
        default=0,
        help="Horizontal shift (pixels). Positive = shift right, negative = shift left."
    )
    parser.add_argument(
        "--shift_y",
        type=int,
        default=0,
        help="Vertical shift (pixels). Positive = shift down, negative = shift up."
    )
    return parser.parse_args()

def shift_mask(mask_binary, shift_x, shift_y):
    """
    Shift a binary (0/255) mask in x and y directions.
    :param mask_binary: 0/255 binary image (np.uint8)
    :param shift_x: horizontal shift (int), >0 => right, <0 => left
    :param shift_y: vertical shift (int), >0 => down,  <0 => up
    :return: shifted mask (np.uint8, 0/255)
    """
    h, w = mask_binary.shape
    # Create a new blank mask
    shifted = np.zeros((h, w), dtype=np.uint8)

    # Calculate the valid region for copying
    # Where the new mask can receive pixels:
    new_x_start = max(0, shift_x)
    new_x_end   = min(w, w + shift_x)
    new_y_start = max(0, shift_y)
    new_y_end   = min(h, h + shift_y)

    # Corresponding old region from which we copy:
    old_x_start = max(0, -shift_x)
    old_x_end   = min(w, w - shift_x)
    old_y_start = max(0, -shift_y)
    old_y_end   = min(h, h - shift_y)

    # Copy the valid region
    shifted[new_y_start:new_y_end, new_x_start:new_x_end] = \
        mask_binary[old_y_start:old_y_end, old_x_start:old_x_end]

    return shifted

def main():
    args = parse_args()
    input_folder = args.input_folder
    output_folder = args.output_folder
    shift_x = args.shift_x
    shift_y = args.shift_y

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    file_list = os.listdir(input_folder)
    for file_name in file_list:
        in_path = os.path.join(input_folder, file_name)
        out_path = os.path.join(output_folder, file_name)

        # Read as grayscale
        mask = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Unable to read image file: {file_name}, skipping...")
            continue

        # Binarize (ensure 0 or 255)
        _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # 1) Shift the mask
        mask_shifted = shift_mask(mask_binary, shift_x, shift_y)

        # 2) Take the union: bitwise_or
        union_mask = cv2.bitwise_or(mask_binary, mask_shifted)

        # Save the result
        cv2.imwrite(out_path, union_mask)
        print(f"Processed: {file_name} -> {out_path}")

if __name__ == "__main__":
    main()