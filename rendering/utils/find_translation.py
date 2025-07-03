#!/usr/bin/env python3
"""
manual_compare_pose.py
Save PNGs that overlay the original robot EE pose and the
manually-rotated robosuite pose (both translation and quaternion).

Edit: roll_x, pitch_y, yaw_z;  PATHS if needed.
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R  # SciPy Rotation docs :contentReference[oaicite:2]{index=2}

def continuous_quat(q):
    q_cont = q.copy()
    for i in range(1, len(q_cont)):
        if np.dot(q_cont[i-1], q_cont[i]) < 0:   # 180° flip
            q_cont[i] *= -1
    return q_cont
from dataset_pair_location import dataset_path

robot_dataset = 'autolab_ur5'  # e.g., 'autolab_ur5', 'kaist', 'toto', etc.
for episode in range(5):

    # ─── 1. FILE LOCATIONS ────────────────────────────────────────────────────
    MY_FILE   = Path(dataset_path[robot_dataset] +
                    f"/{robot_dataset}/source_robot_states/{episode}.npz")
    REAL_FILE = Path(f"/home/harshapolavaram/rovi-aug-extension/videos/"
                    f"berkeley_autolab_ur5/{episode}/"
                    f"xarm7_replay_info_{episode}.npz")

    my_npz   = np.load(MY_FILE,   allow_pickle=True)
    real_npz = np.load(REAL_FILE, allow_pickle=True)

    pos_my   = my_npz['pos'][:, :3] - np.array([-0.6, 0.0, 0.912])            # (N,3) XYZ
    quat_my  = my_npz['pos'][:, 3:7]            # (N,4) xyzw
    pos_ref  = real_npz['original_eef']    # (N,3)
    quat_ref = real_npz['replay_quats']         # (N,4)
    print(list(real_npz.keys()))
    print(real_npz['rotation'][0])
    # print(real_npz['translation'])

    # ─── 2. MANUAL ROTATION (deg) ─────────────────────────────────────────────
    roll_x, pitch_y, yaw_z = 0, 0, 90        # adjust here
    SEQ = 'xyz'

    R_delta = R.from_euler(SEQ, [roll_x, pitch_y, yaw_z], degrees=True)

    # ─── 3. APPLY ROTATION ────────────────────────────────────────────────────
    quat_map = (R_delta * R.from_quat(quat_my)).as_quat()
    pos_map  = pos_my

    # ─────────── 4.  PLOT & SAVE  ───────────────────────────────────────────────
    out_dir = Path(".")
    out_dir.mkdir(exist_ok=True)

    # 4-A  ── Translation: 3 stacked panels (x, y, z) ───────────────────────────
    fig_tr, axes_tr = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    labels_xyz = ['x', 'y', 'z']
    for i, ax in enumerate(axes_tr):
        ax.plot(pos_ref[:, i], label=f'{labels_xyz[i]} ref')
        ax.plot(pos_map[:, i], linestyle='--', label=f'{labels_xyz[i]} mapped')
        ax.set_ylabel('m')
        ax.legend(loc='upper right')
        ax.grid(True)

    axes_tr[-1].set_xlabel('Frame')
    fig_tr.suptitle(f'EE translation – manual XYZ = ({roll_x},{pitch_y},{yaw_z})°')
    fig_tr.tight_layout(rect=[0, 0, 1, 0.96])
    fig_tr.savefig(out_dir / "ee_translation_overlay.png", dpi=300)
    plt.close(fig_tr)

    # 4-B  ── Rotation: 3 stacked panels (roll, pitch, yaw) ─────────────────────
    # convert both quaternions to Euler with the SAME sequence
    eul_ref  = R.from_quat(quat_ref).as_euler(SEQ, degrees=True)
    eul_map  = R.from_quat(quat_map).as_euler(SEQ, degrees=True)

    fig_rot, axes_rot = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    labels_rpy = ['roll', 'pitch', 'yaw']
    for i, ax in enumerate(axes_rot):
        ax.plot(eul_ref[:, i], label=f'{labels_rpy[i]} ref')
        ax.plot(eul_map[:, i], linestyle='--', label=f'{labels_rpy[i]} mapped')
        ax.set_ylabel('deg')
        ax.legend(loc='upper right')
        ax.grid(True)

    axes_rot[-1].set_xlabel('Frame')
    fig_rot.suptitle(f'EE rotation – manual XYZ = ({roll_x},{pitch_y},{yaw_z})°')
    fig_rot.tight_layout(rect=[0, 0, 1, 0.96])
    fig_rot.savefig(out_dir / "ee_rotation_overlay.png", dpi=300)
    plt.close(fig_rot)

    print(f"Saved  ➜  {out_dir / 'ee_translation_overlay.png'}")
    print(f"Saved  ➜  {out_dir / 'ee_rotation_overlay.png'}")

    q_ref_cont = continuous_quat(quat_ref)
    q_map_cont = continuous_quat(quat_map)

    eul_ref_rad = R.from_quat(q_ref_cont).as_euler(SEQ)      # rad
    eul_map_rad = R.from_quat(q_map_cont).as_euler(SEQ)

    eul_ref = np.rad2deg(np.unwrap(eul_ref_rad, axis=0))
    eul_map = np.rad2deg(np.unwrap(eul_map_rad, axis=0))

    fig_rot, axes_rot = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    labels_rpy = ['roll', 'pitch', 'yaw']
    for i, ax in enumerate(axes_rot):
        ax.plot(eul_ref[:, i], label=f'{labels_rpy[i]} ref')
        ax.plot(eul_map[:, i], linestyle='--', label=f'{labels_rpy[i]} mapped')
        ax.set_ylabel('deg');  ax.legend();  ax.grid(True)

    axes_rot[-1].set_xlabel('Frame')
    fig_rot.suptitle(f'EE rotation – manual XYZ = ({roll_x},{pitch_y},{yaw_z})°')
    fig_rot.tight_layout(rect=[0, 0, 1, 0.96])
    fig_rot.savefig(out_dir / "ee_rotation_overlay.png", dpi=300)
    plt.close(fig_rot)

    trans_err = pos_map - pos_ref                    # (N,3)
    mean_trans_err = np.linalg.norm(trans_err, axis=1).mean()

    # orientation – angle between quaternions
    ang_err_deg = np.rad2deg(
        (R.from_quat(q_map_cont).inv() *
        R.from_quat(q_ref_cont)).magnitude()
    )
    mean_ang_err = ang_err_deg.mean()

    print(f"Mean translation error |Δp| : {mean_trans_err:.4f} m")
    print(f"Mean angular error θ    : {mean_ang_err:.4f} deg")