import cv2
import numpy as np
import os
from collections import deque

# ========== 根据需要手动修改以下变量 ==========
input_folder = "/home/jiguanhua/mirage/robot2robot/rendering/autolab_ur5_IIWA_paired_images/ur5e_mask/0"  # 输入掩膜所在文件夹
output_folder = "/home/jiguanhua/mirage/robot2robot/rendering/autolab_ur5_IIWA_paired_images/ur5e_mask_extend/0"  # 处理后结果输出文件夹
alpha = 5.0  # 自适应扩展系数(越大扩越多), 可根据需要调整
use_8_connected = True  # 是否使用 8 邻域 (True=八方向, False=四方向)
# =========================================

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
    # 1) 距离变换(只对掩膜内做), 得到 dt_in[y,x]
    #    这里用 DIST_L2 计算欧几里得距离，但 BFS 会近似为“步数”，稍有偏差
    dt_in = cv2.distanceTransform(mask_binary, cv2.DIST_L2, 3)

    # 2) 找到边界像素(在掩膜内、邻居有背景)
    boundary_pixels = get_boundary_points(mask_binary)
    if not boundary_pixels:
        # 掩膜可能是空的或整幅图全是 255，这种情况直接返回原图
        return mask_binary

    h, w = mask_binary.shape
    # 准备一个 dist 数组记录“从最近的边界像素走过多少步数到达这里” (初始化为很大)
    dist_map = np.full((h, w), np.inf, dtype=np.float32)
    # 准备一个 radius_map 数组记录“当前波前在这里还剩多少可扩展半径”
    # 实际上我们只需要存储“本次波前能扩展的最大步长 = alpha * dt_in[对应该波前起点]”
    # 但需要在 BFS 过程中可能被“更大半径”的波前更新
    radius_map = np.zeros((h, w), dtype=np.float32)

    # 先把原掩膜内部的像素都标记一下 (后面 BFS 只在外部进行)
    inside_mask = (mask_binary > 0)  # True/False

    # BFS 需要的方向(4 或 8邻域)
    if use_8_connected:
        # 8邻域 (dx, dy)
        neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                     (0, 1), (1, -1), (1, 0), (1, 1)]
    else:
        # 4邻域
        neighbors = [(0, -1), (0, 1), (-1, 0), (1, 0)]

    # 3) 多源 BFS 初始化队列
    queue = deque()
    for (bx, by) in boundary_pixels:
        dist_map[by, bx] = 0
        # 该边界像素自带的“最大可扩半径”
        max_r = alpha * dt_in[by, bx]
        radius_map[by, bx] = max_r
        queue.append((bx, by))

    # 4) BFS
    while queue:
        x, y = queue.popleft()
        current_dist = dist_map[y, x]
        current_radius = radius_map[y, x]

        # 向周围邻居扩展
        for dx, dy in neighbors:
            nx, ny = x + dx, y + dy
            # 检查范围
            if 0 <= nx < w and 0 <= ny < h:
                # 只对“原始掩膜外”的像素考虑扩张 (inside_mask[ny, nx] == False)
                if not inside_mask[ny, nx]:
                    new_dist = current_dist + 1  # 每走一步，步数+1
                    # 如果可以继续扩展 (即还有剩余半径)
                    if new_dist <= current_radius:
                        # 若能以更小的 dist 或同样的 dist 但更大的 radius 到达
                        # 就更新 dist_map 和 radius_map
                        if (new_dist < dist_map[ny, nx]) or (
                           abs(new_dist - dist_map[ny, nx]) < 1e-6 and current_radius > radius_map[ny, nx]):
                            dist_map[ny, nx] = new_dist
                            radius_map[ny, nx] = current_radius
                            queue.append((nx, ny))

    # 5) 根据 dist_map 判断哪些“外部像素”被覆盖( dist < inf => 被成功扩张覆盖 )
    expanded_mask = mask_binary.copy()
    # 对于 dist_map[y,x] < inf 的外部像素，说明被至少一个边界波前覆盖
    outside_covered = (dist_map < np.inf) & (~inside_mask)
    expanded_mask[outside_covered] = 255

    return expanded_mask

def main():
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

        # 二值化（确保是0和255）
        _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # 自适应扩展（快速多源 BFS 实现）
        expanded = adaptive_expand_mask_fast(mask_binary, alpha=alpha, use_8_connected=use_8_connected)

        cv2.imwrite(out_path, expanded)
        print(f"处理完成: {file_name} -> {out_path}")

if __name__ == "__main__":
    main()