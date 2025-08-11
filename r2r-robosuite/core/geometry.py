import numpy as np


def quat_dist_rad(q1, q2):
    """
    Minimum rotation angle: arccos of the dot product of two unit quaternions.
    Input shape=(4,), order [qw, qx, qy, qz]
    """
    dot = np.abs(np.dot(q1, q2))
    dot = np.clip(dot, -1.0, 1.0)  # numerical safety
    return 2.0 * np.arccos(dot)


def compute_pose_error(current_pose, target_pose,
                       pos_w=1.0, ori_w=0.1):
    """
    current_pose / target_pose: shape=(7,)
        [x, y, z, qw, qx, qy, qz]
    Returns a scalar error; smaller is better
    """
    # position error
    p_cur, p_tgt = current_pose[:3], target_pose[:3]
    pos_err = np.linalg.norm(p_cur - p_tgt)

    # orientation error (radians)
    q_cur, q_tgt = current_pose[3:], target_pose[3:]
    ori_err = quat_dist_rad(q_cur, q_tgt)

    return pos_w * pos_err + ori_w * ori_err
