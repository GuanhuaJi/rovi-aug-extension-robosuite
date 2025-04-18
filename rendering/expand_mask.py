import os
import cv2
import numpy as np
import argparse
from collections import deque

def get_boundary_points(mask_binary):
    """
    找到掩膜的所有“边界像素”(即在掩膜内，但至少有1个邻域像素在掩膜外)。
    mask_binary: 0/255 的二值掩膜 (uint8)
    返回：列表 [(x1, y1), (x2, y2), ...]
    """
    # 用 findContours 找到轮廓点，这些轮廓点天然就是“边界像素”
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    boundary = []
    for cnt in contours:
        for p in cnt:
            x, y = p[0]
            boundary.append((x, y))
    return boundary

def adaptive_expand_mask_fast(mask_binary, alpha=1.0, use_8_connected=True):
    """
    对单张 0/255 二值掩膜进行“自适应扩展”：
      - 先计算物体内的 distanceTransform，以获得局部厚度 dt_in
      - 以所有边界像素为多源 BFS 起点，边界像素 b 的最大扩展半径 = alpha * dt_in[b]
      - BFS 时，如果到某个外部像素的“步数” <= 该波前的最大扩展半径，就可将其纳入新掩膜
    
    mask_binary: uint8, 0 或 255
    alpha: 扩展系数
    use_8_connected: 是否使用 8 邻域 (否则使用 4 邻域)
    返回：扩展后的二值掩膜
    """
    # 1) 距离变换(只对掩膜内做)
    dt_in = cv2.distanceTransform(mask_binary, cv2.DIST_L2, 3)

    # 2) 找到边界像素(在掩膜内、邻居有背景)
    boundary_pixels = get_boundary_points(mask_binary)
    if not boundary_pixels:
        # 掩膜可能是空的或整幅图全是 255，这种情况直接返回原图
        return mask_binary

    h, w = mask_binary.shape
    dist_map = np.full((h, w), np.inf, dtype=np.float32)
    radius_map = np.zeros((h, w), dtype=np.float32)

    inside_mask = (mask_binary > 0)  # True/False

    # 选择邻域类型
    if use_8_connected:
        neighbors = [(-1, -1), (-1, 0), (-1, 1),
                     (0, -1),           (0, 1),
                     (1, -1),  (1, 0),  (1, 1)]
    else:
        neighbors = [(0, -1), (0, 1), (-1, 0), (1, 0)]

    from collections import deque
    queue = deque()

    # 多源 BFS 初始化
    for (bx, by) in boundary_pixels:
        dist_map[by, bx] = 0
        max_r = alpha * dt_in[by, bx]
        radius_map[by, bx] = max_r
        queue.append((bx, by))

    # 3) BFS
    while queue:
        x, y = queue.popleft()
        current_dist = dist_map[y, x]
        current_radius = radius_map[y, x]

        for dx, dy in neighbors:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h:
                # 只对“原始掩膜外”像素进行扩张
                if not inside_mask[ny, nx]:
                    new_dist = current_dist + 1
                    if new_dist <= current_radius:
                        if (new_dist < dist_map[ny, nx]) or (
                           abs(new_dist - dist_map[ny, nx]) < 1e-6 and current_radius > radius_map[ny, nx]):
                            dist_map[ny, nx] = new_dist
                            radius_map[ny, nx] = current_radius
                            queue.append((nx, ny))

    # 4) 根据 dist_map 判断哪些外部像素被覆盖(dist < inf => 被覆盖)
    expanded_mask = mask_binary.copy()
    outside_covered = (dist_map < np.inf) & (~inside_mask)
    expanded_mask[outside_covered] = 255

    return expanded_mask

def main(input_folder, output_folder, alpha, use_8_connected):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    file_list = os.listdir(input_folder)
    for file_name in file_list:
        in_path = os.path.join(input_folder, file_name)
        out_path = os.path.join(output_folder, file_name)

        # 读取为灰度
        mask = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"无法读取图像文件: {file_name}, 跳过...")
            continue

        # 二值化（确保是 0 和 255）
        _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # 自适应扩展（快速多源 BFS 实现）
        expanded = adaptive_expand_mask_fast(mask_binary, alpha=alpha, use_8_connected=use_8_connected)
        cv2.imwrite(out_path, expanded)
        #print(f"处理完成: {file_name} -> {out_path}")
    print(f"expand_mask处理完成: {output_folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adaptive mask expansion script.")
    parser.add_argument("--input_folder", required=True, help="Input folder containing mask images.")
    parser.add_argument("--output_folder", required=True, help="Output folder for expanded masks.")
    parser.add_argument("--alpha", type=float, default=5.0, help="Adaptive expansion coefficient.")
    parser.add_argument("--use_8_connected", action="store_true",
                        help="Use 8-direction neighbors instead of 4-direction.")
    
    args = parser.parse_args()
    main(
        args.input_folder,
        args.output_folder,
        args.alpha,
        args.use_8_connected
    )