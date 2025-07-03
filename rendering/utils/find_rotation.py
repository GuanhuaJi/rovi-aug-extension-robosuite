#!/usr/bin/env python3
"""
auto_find_xyz.py
-------------------------------------------------
Automatically discover the single roll-pitch-yaw rotation (and optional
translation) that best maps a robosuite trajectory onto real-robot data
across many episodes.

Requires: NumPy, SciPy, Matplotlib
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from dataset_pair_location import dataset_path         # your helper dict

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
robot_dataset = "kaist"                   # key in dataset_path
EPISODES      = range(5)                  # 0,1,2,3,4  – change as needed
SEQ           = "xyz"                     # Euler sequence for the *print-out*

# subtract this first, then the script will refine a constant ∆t automatically
PREOFF = np.array([-0.6, 0.0, 0.912])     # metres

# ─────────────────────────────────────────────────────────────────────────────
# LOAD EVERYTHING
# ─────────────────────────────────────────────────────────────────────────────
q_sim, q_real, p_sim, p_real = [], [], [], []

for ep in EPISODES:
    my_file   = Path(dataset_path[robot_dataset] +
                     f"/{robot_dataset}/source_robot_states/{ep}.npz")
    real_file = Path(f"/home/harshapolavaram/rovi-aug-extension/videos/"
                     f"kaist_nonprehensile_converted_externally_to_rlds/{ep}/"
                     f"xarm7_replay_info_{ep}.npz")

    sim_npz   = np.load(my_file,  allow_pickle=True)
    real_npz  = np.load(real_file, allow_pickle=True)

    p_sim.append(sim_npz["pos"][:, :3] - PREOFF)      # subtract rough offset
    q_sim.append(sim_npz["pos"][:, 3:7])

    p_real.append(real_npz["original_eef"])           # (N,3)
    q_real.append(real_npz["replay_quats"])           # (N,4)

# stack → shape (N_total, …)
p_sim  = np.vstack(p_sim)
p_real = np.vstack(p_real)
q_sim  = np.vstack(q_sim)
q_real = np.vstack(q_real)

print(f"Loaded {q_sim.shape[0]:,} pose pairs from {len(EPISODES)} episodes")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 – Best rotation (Wahba/Kabsch on quaternions)
# ─────────────────────────────────────────────────────────────────────────────
R_sim  = R.from_quat(q_sim)
R_real = R.from_quat(q_real)

B = sum(Rr.as_matrix() @ Rs.as_matrix().T for Rr, Rs in zip(R_real, R_sim))
U, _, Vt = np.linalg.svd(B)
M        = np.diag([1, 1, np.linalg.det(U @ Vt)])
R_delta  = U @ M @ Vt

# Euler read-out (deg)
euler_deg = R.from_matrix(R_delta).as_euler(SEQ, degrees=True)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 – Optional best translation offset
# (minimise ‖ p_real - (R∆ p_sim + t∆) ‖² )
# closed-form:  t∆ = mean ( p_real - R∆ p_sim )
# ─────────────────────────────────────────────────────────────────────────────
p_sim_rot = (R_delta @ p_sim.T).T
t_delta   = (p_real - p_sim_rot).mean(axis=0)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 – Apply to the whole dataset for metrics
# ─────────────────────────────────────────────────────────────────────────────
q_aligned = (R.from_matrix(R_delta) * R_sim).as_quat()
p_aligned = p_sim_rot + t_delta

# translation error
trans_err = np.linalg.norm(p_aligned - p_real, axis=1)
mean_trans_err = trans_err.mean()

# orientation error (deg) – use continuous quaternion trick
def cont(q):
    q = q.copy()
    for i in range(1, len(q)):
        if (q[i-1] @ q[i]) < 0:           # flip sign for continuity
            q[i] *= -1
    return q

ang_err_deg = np.rad2deg(
    (R.from_quat(cont(q_aligned)).inv() *
     R.from_quat(cont(q_real))).magnitude()
)
mean_ang_err = ang_err_deg.mean()

# ─────────────────────────────────────────────────────────────────────────────
# PRINT RESULTS
# ─────────────────────────────────────────────────────────────────────────────
print("\nOptimal rotation ({}):  roll={:+.5f}°  pitch={:+.5f}°  yaw={:+.5f}°"
      .format(SEQ, *euler_deg))
print("Optimal translation ∆t :  [{:+.4f}  {:+.4f}  {:+.4f}] m".format(*t_delta))
print("Mean translation error |∆p| : {:.4f} m".format(mean_trans_err))
print("Mean angular error    θ     : {:.4f} deg".format(mean_ang_err))

# ─────────────────────────────────────────────────────────────────────────────
# OPTIONAL: overlay plots for the *first* episode to visual-check
# ─────────────────────────────────────────────────────────────────────────────
# grab episode 0 again just for plotting
ep0_sim  = np.load(dataset_path[robot_dataset] +
                   f"/{robot_dataset}/source_robot_states/0.npz", allow_pickle=True)
ep0_real = np.load(f"/home/harshapolavaram/rovi-aug-extension/videos/"
                   f"kaist_nonprehensile_converted_externally_to_rlds/0/"
                   f"xarm7_replay_info_0.npz", allow_pickle=True)

p0_sim  = ep0_sim["pos"][:, :3] - PREOFF
q0_sim  = ep0_sim["pos"][:, 3:7]
p0_real = ep0_real["original_eef"]
q0_real = ep0_real["replay_quats"]

# apply optimal transform
p0_aligned = (R_delta @ p0_sim.T).T + t_delta
q0_aligned = (R.from_matrix(R_delta) * R.from_quat(q0_sim)).as_quat()

# translation plot
fig_tr, ax_tr = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
for i, lbl in enumerate("xyz"):
    ax_tr[i].plot(p0_real[:, i], label=f"{lbl} real")
    ax_tr[i].plot(p0_aligned[:, i], "--", label=f"{lbl} aligned")
    ax_tr[i].set_ylabel("m");  ax_tr[i].legend();  ax_tr[i].grid(True)
ax_tr[-1].set_xlabel("Frame")
fig_tr.suptitle("Episode 0 translation (optimal XYZ)")
fig_tr.tight_layout(rect=[0,0,1,0.96])
fig_tr.savefig("translation_overlay_opt.png", dpi=300)
plt.close(fig_tr)

# rotation plot
eul_real = R.from_quat(cont(q0_real)).as_euler(SEQ, degrees=True)
eul_algn = R.from_quat(cont(q0_aligned)).as_euler(SEQ, degrees=True)

fig_rot, ax_rot = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
for i, lbl in enumerate(["roll", "pitch", "yaw"]):
    ax_rot[i].plot(eul_real[:, i], label=f"{lbl} real")
    ax_rot[i].plot(eul_algn[:, i], "--", label=f"{lbl} aligned")
    ax_rot[i].set_ylabel("deg"); ax_rot[i].legend(); ax_rot[i].grid(True)
ax_rot[-1].set_xlabel("Frame")
fig_rot.suptitle("Episode 0 rotation (optimal XYZ)")
fig_rot.tight_layout(rect=[0,0,1,0.96])
fig_rot.savefig("rotation_overlay_opt.png", dpi=300)
plt.close(fig_rot)

print("Saved  translation_overlay_opt.png and rotation_overlay_opt.png")
