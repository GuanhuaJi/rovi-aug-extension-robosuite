import numpy as np

def quat_dist_rad(q1, q2):
    """
    最小旋转角：两单位四元数内积的 arccos。
    输入 shape=(4,), 顺序 [qw, qx, qy, qz]
    """
    dot = np.abs(np.dot(q1, q2))
    dot = np.clip(dot, -1.0, 1.0)  # 数值安全
    return 2.0 * np.arccos(dot)

def compute_pose_error(current_pose, target_pose,
                       pos_w=1.0, ori_w=0.1):
    """
    current_pose / target_pose: shape=(7,)
        [x, y, z, qw, qx, qy, qz]
    返回一个标量误差，越小越好
    """
    # 位置误差
    p_cur, p_tgt = current_pose[:3], target_pose[:3]
    pos_err = np.linalg.norm(p_cur - p_tgt)

    # 姿态误差（弧度）
    q_cur, q_tgt = current_pose[3:], target_pose[3:]
    ori_err = quat_dist_rad(q_cur, q_tgt)

    return pos_w * pos_err + ori_w * ori_err 