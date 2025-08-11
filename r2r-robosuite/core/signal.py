import numpy as np
from scipy.spatial.transform import Rotation as R

def _find_spike_ranges(xyz: np.ndarray, thresh: float):
    """
    Return [(start, end_exclusive), …]
    where start-1   is the most recent "good" frame
          start..end-1 are consecutive abnormal frames
          end        is the next good frame, or ==N meaning at the tail
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
        ranges.append((start, i))   # i==N ⇒ tail segment
        prev_good = i if i < N else prev_good
        i += 1
    return ranges

# ---------- Main function: unlimited repair ----------
def smooth_xyz_spikes(
        pose_array: np.ndarray,
        thresh: float,
        tail_mode: str = "copy",   # "copy" | "extrap" | "ignore"
        max_passes: int = 3,
        verbose: bool = True
) -> np.ndarray:
    """
    • Any length of abnormal segment will be attempted to fix (max_gap limit removed)
    • Tail behavior (missing right reference) determined by tail_mode:
        "copy"   -> copy the last good xyz for all
        "extrap" -> linearly extrapolate one step of velocity
        "ignore" -> leave as is
    """
    xyz = pose_array[:, :3].copy()
    N   = len(xyz)

    def _interp_block(l_idx: int, r_idx: int):
        """Linearly interpolate xyz between (l_idx, r_idx) (excluding endpoints)"""
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

            # ---------- (1) Has a right endpoint: interpolate directly ----------
            if not at_tail:
                _interp_block(start - 1, end)
                fixed_this_pass = True
                if verbose:
                    print(f"  ↳ fixed frames {start}…{end-1}  (gap={end-start})")
                continue

            # ---------- (2) Tail segment ----------
            if tail_mode == "copy":
                xyz[start:N] = xyz[start - 1]          # copy the previous frame for all
                fixed_this_pass = True
                if verbose:
                    print(f"  ↳ copied last good xyz to tail frames {start}…{N-1}")
            elif tail_mode == "extrap":
                # Estimate using previous frame's velocity
                vel = xyz[start - 1] - xyz[start - 2] if start >= 2 else 0
                for k in range(start, N):
                    xyz[k] = xyz[start - 1] + (k - start + 1) * vel
                fixed_this_pass = True
                if verbose:
                    print(f"  ↳ extrapolated tail frames {start}…{N-1}")
            # "ignore": do not fix

        fixed_any |= fixed_this_pass
        if not fixed_this_pass:
            if verbose:
                print(f"[SPIKE] pass {p}: no fixable spikes, stopping\n")
            break

    # ---- Write back and print remaining abnormal segments -----------------------------------------
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
    forward = rot_mat[:, 2]     # can change to [:, 0] or [:, 1] depending on your forward direction definition
    target_pos = eef_pos + distance * forward
    return np.concatenate((target_pos, eef_quat))
