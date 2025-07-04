import numpy as np

def _find_spike_ranges(xyz: np.ndarray, thresh: float):
    """
    返回 [(start, end_exclusive), …]
    其中 start-1   为最近一次“良好”帧
         start..end-1 为连续异常帧
         end        为下一帧良好，或 ==N 表示落在尾部
    """
    ranges = []
    N = len(xyz)
    prev_good = 0
    i = 1
    while i < N:
        if np.abs(xyz[i] - xyz[prev_good]).sum() <= thresh:
            prev_good = i
            i += 1
            continue
        start = i
        while i < N and np.abs(xyz[i] - xyz[prev_good]).sum() > thresh:
            i += 1
        ranges.append((start, i))   # i==N ⇒ 尾段
        prev_good = i if i < N else prev_good
        i += 1
    return ranges

# ---------- 主函数：无限制修复 ----------
def smooth_xyz_spikes(
        pose_array: np.ndarray,
        thresh: float,
        tail_mode: str = "copy",   # "copy" | "extrap" | "ignore"
        max_passes: int = 3,
        verbose: bool = True
) -> np.ndarray:
    """
    • 任何长度的异常区段都会被尝试修复（已移除 max_gap 限制）
    • 尾段（右端缺参考）行为由 tail_mode 决定：
        "copy"   -> 全部复制最后一帧良好 xyz
        "extrap" -> 线性外推一步的速度
        "ignore" -> 原样保留
    """
    xyz = pose_array[:, :3].copy()
    N   = len(xyz)

    def _interp_block(l_idx: int, r_idx: int):
        """将 (l_idx, r_idx) 之间（不含端点）的 xyz 线性插值"""
        gap = r_idx - l_idx - 1
        for k in range(1, gap + 1):
            t = k / (gap + 1)
            xyz[l_idx + k] = (1 - t) * xyz[l_idx] + t * xyz[r_idx]

    fixed_any = False
    for p in range(1, max_passes + 1):
        spike_ranges = _find_spike_ranges(xyz, thresh)
        if verbose:
            print(f"[SPIKE] pass {p}: {len(spike_ranges)} segment(s) detected")

        fixed_this_pass = False
        for start, end in spike_ranges:
            at_tail = end >= N

            # ---------- (1) 有右端点：直接插值 ----------
            if not at_tail:
                _interp_block(start - 1, end)
                fixed_this_pass = True
                if verbose:
                    print(f"  ↳ fixed frames {start}…{end-1}  (gap={end-start})")
                continue

            # ---------- (2) 尾段 ----------
            if tail_mode == "copy":
                xyz[start:N] = xyz[start - 1]          # 全部复制上一帧
                fixed_this_pass = True
                if verbose:
                    print(f"  ↳ copied last good xyz to tail frames {start}…{N-1}")
            elif tail_mode == "extrap":
                # 使用上一帧速度估计
                vel = xyz[start - 1] - xyz[start - 2] if start >= 2 else 0
                for k in range(start, N):
                    xyz[k] = xyz[start - 1] + (k - start + 1) * vel
                fixed_this_pass = True
                if verbose:
                    print(f"  ↳ extrapolated tail frames {start}…{N-1}")
            # "ignore": 不修复

        fixed_any |= fixed_this_pass
        if not fixed_this_pass:
            if verbose:
                print(f"[SPIKE] pass {p}: no fixable spikes, stopping\n")
            break

    # ---- 写回并打印剩余异常区段 -----------------------------------------
    pose_array[:, :3] = xyz
    remaining = _find_spike_ranges(xyz, thresh)
    if verbose:
        print(f"[SPIKE] cleaning done, remaining segments: {len(remaining)}")
        for s, e in remaining:
            print(f"  • frames {s}…{e-1}  (gap={e-s})  ❗ at tail={e>=N}")
        print()

    return pose_array

def reach_further(eef, distance=0.07):
    eef_pos = eef[:3]
    eef_quat = eef[3:7]  # (x, y, z, w)
    eef_rot = R.from_quat(eef_quat)  # (x, y, z, w)
    rot_mat = eef_rot.as_matrix()
    forward = rot_mat[:, 2]     # 可以改成 [:, 0] or [:, 1] 取决于你定义的方向
    target_pos = eef_pos + distance * forward
    return np.concatenate((target_pos, eef_quat))