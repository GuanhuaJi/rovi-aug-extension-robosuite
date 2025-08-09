'''
python /home/guanhuaji/mirage/robot2robot/rendering/generate_target_robot_images_old.py --robot_dataset "taco_play" --target_robot "IIWA" --partition 0 --unlimited false --load_displacement False

datasets: 
austin_buds, austin_mutex, austin_sailor, 
autolab_ur5, can, furniture_bench, iamlab_cmu, 
lift, nyu_franka, square, stack, three_piece_assembly, 
taco_play, ucsd_kitchen_rlds, viola
'''




import argparse
import json
import time
import os
import cv2
import socket, pickle, struct
import numpy as np
import matplotlib.pyplot as plt
import robosuite as suite
import robosuite.macros as macros
macros.IMAGE_CONVENTION = "opencv"
from export_source_robot_states import RobotCameraWrapper
from config.robot_pose_dict import ROBOT_POSE_DICT
from tqdm import tqdm
from transforms3d.quaternions import quat2mat
import json
from pathlib import Path
import pynvml
from contextlib import contextmanager
import json, os, tempfile, portalocker, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from robosuite.utils.transform_utils import mat2quat
from config.dataset_poses_dict import ROBOT_CAMERA_POSES_DICT

STEP_MAX = 0.02          # 单帧允许的最大 L1 位移（米），1 cm
ORI_LERP = False


def reach_further(eef, distance=0.07):
    eef_pos = eef[:3]
    eef_quat = eef[3:7]  # (x, y, z, w)
    eef_rot = R.from_quat(eef_quat)  # (x, y, z, w)
    rot_mat = eef_rot.as_matrix()
    forward = rot_mat[:, 2]     # 可以改成 [:, 0] or [:, 1] 取决于你定义的方向
    target_pos = eef_pos + distance * forward
    return np.concatenate((target_pos, eef_quat))


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



@contextmanager
def locked_json(path: pathlib.Path, mode="r+", default=lambda: {}):
    # 1️⃣ acquire an **exclusive lock**
    with portalocker.Lock(str(path), mode, timeout=30) as fp:    # ← blocks here
        try:
            data = json.load(fp)
        except json.JSONDecodeError:
            data = default()
        yield data                       # 🔒  work with the dict while locked
        fp.seek(0), fp.truncate()        # 2️⃣ rewind
        json.dump(data, fp, indent=2)    # 3️⃣ write
        fp.flush(), os.fsync(fp.fileno())  # 4️⃣ durability
    # ➡ lock is released automatically

def atomic_write_json(obj, path: pathlib.Path):
    with tempfile.NamedTemporaryFile(
            dir=path.parent, delete=False, mode="w") as tmp:
        json.dump(obj, tmp, indent=2)
        tmp.flush()
        os.fsync(tmp.fileno())           # make sure it’s on disk
    os.replace(tmp.name, path) 



def pick_best_gpu(policy="free-mem"):
    """
    Return the index of the “least busy” NVIDIA GPU and set CUDA_VISIBLE_DEVICES
    so frameworks (PyTorch, TensorFlow, JAX…) will automatically use it.

    policy
    ------
    "free-mem"   – prefer the card with the most free memory
    "low-util"   – prefer the card with the lowest compute utilisation
    "hybrid"     – most free mem, break ties with lowest utilisation
    """
    pynvml.nvmlInit()
    n = pynvml.nvmlDeviceGetCount()

    best_idx, best_score = None, None
    for i in range(n):
        h = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)          # bytes
        util = pynvml.nvmlDeviceGetUtilizationRates(h)   # %
        if policy == "free-mem":
            score = mem.free
        elif policy == "low-util":
            score = -util.gpu                            # negative ⇒ lower is better
        else:  # hybrid
            score = (mem.free, -util.gpu)                # tuple is fine for max()

        if best_score is None or score > best_score:
            best_idx, best_score = i, score

    os.environ["CUDA_VISIBLE_DEVICES"] = str(best_idx)   # frameworks see *only* this GPU
    print(f"👉  Selected GPU {best_idx}")
    return best_idx

def load_list(path):
    with locked_json(path, "a+") as data:
        return data.copy()

def save_list(path, data):
    atomic_write_json(data, pathlib.Path(path))

def offset_in_quaternion_direction_batch(positions, quaternions, offset_dist=0.05, local_direction=None):
    """
    批量处理:
      - positions: shape = (N, 3) 的 numpy 数组，代表 N 个末端执行器的位置
      - quaternions: shape = (N, 4) 或 (4,) 的 numpy 数组，代表与每个位置对应的四元数 (w, x, y, z)
         * 如果 quaternions 的 shape = (4,) 表示所有位置都用同一个四元数
      - offset_dist: 沿着 local_direction 在世界坐标系中偏移的距离 (默认 0.05)
      - local_direction: 末端执行器在自身坐标系下的“前向”向量 (默认 [0, 0, -1])
    
    返回：
      - new_positions: shape = (N, 3) 的 numpy 数组，偏移后的世界坐标位置
    """
    # 如果用户没有指定末端执行器在自身坐标系中的“指向”，默认用 -Z
    if local_direction is None:
        local_direction = np.array([0, 0, -1], dtype=float)
    else:
        local_direction = np.array(local_direction, dtype=float)

    # 确保 positions 和 quaternions 都是 numpy 数组
    positions = np.array(positions, dtype=float)
    quaternions = np.array(quaternions, dtype=float)

    # 如果只有一个四元数，则对所有 positions 都使用这个四元数
    if quaternions.ndim == 1:  
        # shape = (4,)
        R = quat2mat(quaternions)  # 1 个旋转矩阵
        world_dir = R.dot(local_direction)  # 在世界坐标系中的方向
        new_positions = positions + offset_dist * world_dir
        return new_positions
    else:
        # shape = (N, 4) -> 每个 position 对应一个 quaternion
        new_positions = []
        for pos, quat in zip(positions, quaternions):
            R = quat2mat(quat)
            world_dir = R.dot(local_direction)
            new_pos = pos + offset_dist * world_dir
            new_positions.append(new_pos)
        return np.vstack(new_positions)

class TargetEnvWrapper:
    def __init__(self, target_name, target_gripper, robot_dataset, camera_height=256, camera_width=256):
        self.target_env = RobotCameraWrapper(robotname=target_name, grippername=target_gripper, robot_dataset=robot_dataset, camera_height=camera_height, camera_width=camera_width)
        self.target_name = target_name

    def _load_dataset_info(self, dataset_name):
        info = ROBOT_CAMERA_POSES_DICT[dataset_name]
        return info
    
    def generate_image(self, 
                        save_paired_images_folder_path="paired_images", 
                        displacement_csv_path=None,
                        source_robot_states_path="paired_images",
                        reference_joint_angles_path=None, 
                        reference_ee_states_path=None, 
                        robot_dataset=None, 
                        unlimited="False",
                        load_displacement=False,
                        episode=0, 
                        camera_height=256, 
                        camera_width=256):
        data = np.load(os.path.join(source_robot_states_path, "source_robot_states", f"{episode}.npz"), allow_pickle=True)
        info = self._load_dataset_info(robot_dataset)

        if robot_dataset in ["austin_buds", "austin_sailor"]:
            target_pose_array = smooth_xyz_spikes(
                data['pos'].copy(),
                thresh=0.05)
        else:
            target_pose_array = data['pos'].copy()

        gripper_array = data['grip']
        if robot_dataset == "can":
            camera_pose = np.array([0.9, 0.1, 1.75, 0.271, 0.271, 0.653, 0.653])
            # cam_id = self.source_env.camera_wrapper.env.sim.model.camera_name2id("agentview")
            # fov = self.source_env.camera_wrapper.env.sim.model.cam_fovy[cam_id]
        elif robot_dataset == "lift":
            camera_pose = np.array([0.45, 0, 1.35, 0.271, 0.271, 0.653, 0.653])
            cam_id = self.source_env.camera_wrapper.env.sim.model.camera_name2id("agentview")
            fov = self.source_env.camera_wrapper.env.sim.model.cam_fovy[cam_id]
        elif robot_dataset == "square":
            camera_pose = np.array([0.45, 0, 1.35, 0.271, 0.271, 0.653, 0.653])
            cam_id = self.source_env.camera_wrapper.env.sim.model.camera_name2id("agentview")
            fov = self.source_env.camera_wrapper.env.sim.model.cam_fovy[cam_id]
        elif robot_dataset == "stack":
            camera_pose = np.array([0.45, 0, 1.35, 0.271, 0.271, 0.653, 0.653])
            cam_id = self.source_env.camera_wrapper.env.sim.model.camera_name2id("agentview")
            fov = self.source_env.camera_wrapper.env.sim.model.cam_fovy[cam_id]
        elif robot_dataset == "three_piece_assembly":
            camera_pose = np.array([0.713078462147161, 2.062036796036723e-08, 1.5194726087166726, 0.293668270111084, 0.2936684489250183, 0.6432408690452576, 0.6432409286499023])
            cam_id = self.source_env.camera_wrapper.env.sim.model.camera_name2id("agentview")
            fov = self.source_env.camera_wrapper.env.sim.model.cam_fovy[cam_id]
        else:
            for viewpoint in info["viewpoints"]:
                if episode in viewpoint["episodes"]:
                    camera_reference_position = viewpoint["camera_position"] + np.array([-0.6, 0.0, 0.912]) 
                    roll_deg = viewpoint["roll"]
                    pitch_deg = viewpoint["pitch"]
                    yaw_deg = viewpoint["yaw"]
                    fov = viewpoint["camera_fov"]
                    r = R.from_euler('xyz', [roll_deg, pitch_deg, yaw_deg], degrees=True)
                    camera_reference_quaternion = r.as_quat()
                    camera_pose = np.concatenate((camera_reference_position, camera_reference_quaternion))
                    break
        robot_disp = None

        if load_displacement:
            offset_file = os.path.join(source_robot_states_path, "source_robot_states", self.target_name, "offsets", f"{episode}.npy")
            if os.path.isfile(offset_file):
                robot_disp = np.load(offset_file)
            else:
                robot_disp = np.zeros(3, dtype=np.float32)
                print(f"WARNING: displacement file not found → {offset_file}; "
                    f"using default [0, 0, 0].")
        else:
            #robot_disp = ROBOT_POSE_DICT[robot_dataset][self.target_name]['displacement']
            robot_disp = np.zeros(3, dtype=np.float32)

        camera_pose[:3] -= robot_disp
        if robot_dataset == "iamlab_cmu":
            camera_pose[2] += 0.1
        
        
        self.target_env.camera_wrapper.set_camera_pose(pos=camera_pose[:3], quat=camera_pose[3:])
        if "fov" in data:
            fov = data["fov"]
            self.target_env.camera_wrapper.set_camera_fov(fov)
        self.target_env.update_camera()

        os.makedirs(os.path.join(save_paired_images_folder_path, "{}_rgb".format(target_name), str(episode)), exist_ok=True)
        os.makedirs(os.path.join(save_paired_images_folder_path, "{}_mask".format(target_name), str(episode)), exist_ok=True)
        num_robot_poses = target_pose_array.shape[0]
        target_pose_list = []
        joint_angles_list = []
        gripper_width_list = []
        success = True

        # if ROBOT_POSE_DICT[robot_dataset][self.target_name]['safe_angle'] is not None:
        #     self.target_env.set_robot_joint_positions(ROBOT_POSE_DICT[robot_dataset][self.target_name]['safe_angle'])
        
        for pose_index in range(num_robot_poses):
            target_pose=target_pose_array[pose_index].copy()
            target_pose[:3] -= robot_disp
            if robot_dataset == "viola":
                target_pose = reach_further(target_pose)

            if target_pose_list:
                prev_world = target_pose_list[-1][:3] - robot_disp
            else:
                prev_world = target_pose[:3]

            delta = target_pose[:3] - prev_world
            dist  = np.abs(delta).sum()

            if dist > STEP_MAX:
                n_sub = int(np.ceil(dist / STEP_MAX))
                for s in range(1, n_sub + 1):
                    sub_pose = target_pose.copy()
                    sub_pose[:3] = prev_world + (s / n_sub) * delta
                    if ORI_LERP and n_sub > 1:
                        sub_pose[3:] = (
                            (1 - s / n_sub) * target_pose_array[pose_index - 1][3:] +
                            (s / n_sub)     * target_pose[3:]
                        )
                    self.target_env.drive_robot_to_target_pose(target_pose=sub_pose)

            _, gripper_dist = self.target_env.get_gripper_width_from_qpos()
            attempt = 0
            while (gripper_dist < gripper_array[pose_index] - 0.1 or gripper_dist > gripper_array[pose_index] + 0.1) and attempt < 10:
                if gripper_dist < gripper_array[pose_index] - 0.1:
                    self.target_env.open_close_gripper(gripper_open=True)
                elif gripper_dist > gripper_array[pose_index] + 0.1:
                    self.target_env.open_close_gripper(gripper_open=False)
                _, gripper_dist = self.target_env.get_gripper_width_from_qpos()
                attempt += 1

                
            target_reached, target_reached_pose, error = (
                self.target_env.drive_robot_to_target_pose(target_pose=target_pose)
            )

            if unlimited == "False" and not target_reached:
                print(f"episode {episode} pose {pose_index}")
                blacklist_path = Path(f"{save_paired_images_folder_path}/{target_name}/blacklist.json")
                with locked_json(blacklist_path) as blk:
                    robot_list = blk.setdefault(self.target_name, [])
                    if episode not in robot_list:
                        robot_list.append(episode)
                        robot_list.sort()
                        RED   = "\033[91m"
                        RESET = "\033[0m"
                        print(f"{RED}[BLACKLIST] Added {self.target_name} – episode {episode}{RESET}")
                success = False
                try:
                    n = len(target_pose_list)
                    tgt_xy = target_pose_array[:, :2]                 # desired XY
                    real_xy = np.array(target_pose_list)[:, :2] - robot_disp[:2]        # reached XY
                    fig, ax = plt.subplots(figsize=(4, 4))
                    ax.plot(tgt_xy[:, 0], tgt_xy[:, 1], "o-", label="target XY")
                    ax.plot(real_xy[:, 0], real_xy[:, 1], "x-", label="reached XY")
                    ax.set_xlabel("X (m)");  ax.set_ylabel("Y (m)")
                    ax.set_title(f"{self.target_name} – episode {episode} - offset {robot_disp.round(3)}")
                    ax.axis("equal");  ax.legend()
                    
                    out_dir = os.path.join(save_paired_images_folder_path,
                                        f"{self.target_name}_traj_plots")
                    os.makedirs(out_dir, exist_ok=True)
                    out_file = os.path.join(out_dir, f"{episode}.png")
                    fig.savefig(out_file,
                                dpi=150, bbox_inches="tight")
                    plt.close(fig)
                    print(f"\033[95m[TRAJ]  Saved XY plot for episode {episode}: {out_file}\033[0m")
                except Exception as e:
                    print(f"[WARN] could not save trajectory plot: {e}")
                break
            reached_pose = self.target_env.compute_eef_pose()
            reached_pose[:3] += robot_disp
            target_pose_list.append(reached_pose)
            gripper_width_list.append(self.target_env.get_gripper_width_from_qpos())
            
            joint_indices = self.target_env.env.robots[0]._ref_joint_pos_indexes
            joint_angles = self.target_env.env.sim.data.qpos[joint_indices]
            joint_angles_list.append(joint_angles)


            target_robot_img, target_robot_seg_img = self.target_env.get_observation_fast(white_background=True, width=camera_width, height=camera_height)
            cv2.imwrite(os.path.join(save_paired_images_folder_path, f"{target_name}_rgb", f"{episode}/{pose_index}.png"), cv2.cvtColor(target_robot_img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(save_paired_images_folder_path, f"{target_name}_mask", f"{episode}/{pose_index}.png"), target_robot_seg_img * 255)
        
        if success:
            if unlimited == "False":
                blacklist_path = Path(f"{save_paired_images_folder_path}/{target_name}/blacklist.json")
                with locked_json(blacklist_path) as blk:
                    robot_list = blk.get(self.target_name, [])
                    if episode in robot_list:
                        robot_list.remove(episode)
                        if robot_list:
                            blk[self.target_name] = robot_list
                        else:
                            blk.pop(self.target_name)
                        print(f"\033[92m[BLACKLIST] Removed {self.target_name} – episode {episode}\033[0m")

                whitelist_path = Path(f"{save_paired_images_folder_path}/{target_name}/whitelist.json")
                with locked_json(whitelist_path) as wl:             # 🔒 独占锁
                    robot_list = wl.get(self.target_name, [])
                    if episode not in robot_list:
                        robot_list.append(episode)
                        robot_list.sort()
                        wl[self.target_name] = robot_list
                        print(f"\033[92m[WHITELIST] Added {self.target_name} – episode {episode}\033[0m")
                # 同样在退出 with 时安全写回
            else:
                print(f"\033[92m[UNLIMITED] Generated {self.target_name} – episode {episode}\033[0m")

            target_pose_array = np.vstack(target_pose_list)
            joint_angles_array = np.vstack(joint_angles_list)
            gripper_width_array = np.array(gripper_width_list)
            eef_npy_path = os.path.join(save_paired_images_folder_path, "source_robot_states", f"{self.target_name}", "end_effector", f"{episode}.npy")
            np.save(eef_npy_path, target_pose_array)
            gripper_npy_path = os.path.join(save_paired_images_folder_path, "source_robot_states", f"{self.target_name}", "gripper_distance", f"{episode}.npy")
            np.save(gripper_npy_path, gripper_width_array)
            joint_angles_npy_path = os.path.join(save_paired_images_folder_path, "source_robot_states", f"{self.target_name}", "joint_angles", f"{episode}.npy")
            np.save(joint_angles_npy_path, joint_angles_array)
            offset_npy_path = os.path.join(save_paired_images_folder_path, "source_robot_states", f"{self.target_name}", "offsets", f"{episode}.npy")
            np.save(offset_npy_path, robot_disp)
        
        

if __name__ == "__main__":

    """
    Registered environments: Lift, Stack, NutAssembly, NutAssemblySingle, NutAssemblySquare, NutAssemblyRound,
                             PickPlace, PickPlaceSingle, PickPlaceMilk, PickPlaceBread, PickPlaceCereal,
                             PickPlaceCan, Door, Wipe, TwoArmLift, TwoArmPegInHole, TwoArmHandover

    Possible robots: Baxter, IIWA, Jaco, Kinova3, Panda, Sawyer, UR5e
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="(optional) set seed")
    parser.add_argument("--target_gripper", type=str, default="Robotiq85Gripper", help="PandaGripper or Robotiq85Gripper")
    parser.add_argument("--num_robot_poses", type=int, default=5, help="(optional) (optional) set seed")
    parser.add_argument("--target_robot", nargs="+", default="IIWA", help="(optional) (optional) set seed")
    parser.add_argument("--save_paired_images_folder_path", type=str, default="paired_images", help="(optional) folder path to save the paired images")
    parser.add_argument("--robot_dataset", type=str, help="(optional) to match the robot poses from a dataset, provide the dataset name")
    parser.add_argument("--reference_joint_angles_path", type=str, help="(optional) to match the robot poses from a dataset, provide the path to the joint angles file (np.savetxt)")
    parser.add_argument("--reference_ee_states_path", type=str, help="(optional) to match the robot poses from a dataset, provide the path to the ee state file (np.savetxt)")
    parser.add_argument("--blacklist", type=bool, default=False, help="If set, prints extra debug/warning information")
    parser.add_argument("--partition", type=int, default=0, help="If set, prints extra debug/warning information")
    parser.add_argument("--unlimited", type=str, default="False", help="If set, prints extra debug/warning information")
    parser.add_argument("--load_displacement", type=bool, default=False, help="If set, load the displacement from the source robot states")
    args = parser.parse_args()


    for target_name in args.target_robot:
        if target_name == "Sawyer":
            target_gripper = "RethinkGripper"
        elif target_name == "Jaco":
            target_gripper = "JacoThreeFingerGripper"
        elif target_name == "IIWA":
            target_gripper = "Robotiq85Gripper"
        elif target_name == "UR5e":
            target_gripper = "Robotiq85Gripper"
        elif target_name == "Kinova3":
            target_gripper = "Robotiq85Gripper"
        elif target_name == "Panda":
            target_gripper = "PandaGripper"

        save_paired_images_folder_path = ROBOT_CAMERA_POSES_DICT[args.robot_dataset]["replay_path"]
        source_robot_states_path = save_paired_images_folder_path

        
        if args.robot_dataset is not None:
            robot_dataset_info = ROBOT_CAMERA_POSES_DICT[args.robot_dataset]
            camera_height = robot_dataset_info["camera_height"]
            camera_width = robot_dataset_info["camera_width"]
        else:
            camera_height = 256
            camera_width = 256

        pick_best_gpu()
        NUM_PARTITIONS = 20

        episodes = []
        if args.blacklist:
            blacklist_path = Path(f"{save_paired_images_folder_path}/{target_name}/blacklist.json")
            with locked_json(blacklist_path) as blk:                      # 🔒 进入临界区
                robot_list = blk.get(target_name, [])
                if robot_list:
                    episodes = robot_list
                    print(f"\033[93m[BLACKLIST] Found {target_name} – episode {episodes}\033[0m")
                else:
                    print(f"\033[92m[BLACKLIST] No blacklisted episodes for {target_name}\033[0m")
        else:
            num_episode = ROBOT_CAMERA_POSES_DICT[args.robot_dataset]['num_episodes']
            episodes = range(num_episode * args.partition // NUM_PARTITIONS, num_episode * (args.partition + 1) // NUM_PARTITIONS)


            '''
            python /home/guanhuaji/mirage/robot2robot/rendering/generate_target_robot_images.py --robot_dataset "furniture_bench" --target_robot "Panda" --partition 0 &
            python /home/guanhuaji/mirage/robot2robot/rendering/generate_target_robot_images.py --robot_dataset "furniture_bench" --target_robot "Sawyer" --partition 0 &
            python /home/guanhuaji/mirage/robot2robot/rendering/generate_target_robot_images.py --robot_dataset "furniture_bench" --target_robot "Jaco" --partition 0 &
            python /home/guanhuaji/mirage/robot2robot/rendering/generate_target_robot_images.py --robot_dataset "furniture_bench" --target_robot "Kinova3" --partition 0 &
            python /home/guanhuaji/mirage/robot2robot/rendering/generate_target_robot_images.py --robot_dataset "furniture_bench" --target_robot "IIWA" --partition 0 &
            python /home/guanhuaji/mirage/robot2robot/rendering/generate_target_robot_images.py --robot_dataset "furniture_bench" --target_robot "UR5e" --partition 0

            conda activate mirage
            python /home/guanhuaji/mirage/robot2robot/rendering/generate_target_robot_images.py --robot_dataset "toto" --target_robot "Jaco" --partition 0
            python /home/guanhuaji/mirage/robot2robot/rendering/generate_target_robot_images.py --robot_dataset "autolab_ur5" --target_robot "Sawyer" --partition 0
            python /home/guanhuaji/mirage/robot2robot/rendering/generate_target_robot_images.py --robot_dataset "nyu_franka" --target_robot "Jaco" --partition 0
            python /home/guanhuaji/mirage/robot2robot/rendering/generate_target_robot_images.py --robot_dataset "ucsd_kitchen_rlds" --target_robot "Sawyer" --partition 0
            python /home/guanhuaji/mirage/robot2robot/rendering/generate_target_robot_images.py --robot_dataset "utokyo_pick_and_place" --target_robot "Sawyer" --partition 0

            python /home/guanhuaji/mirage/robot2robot/rendering/generate_target_robot_images.py --robot_dataset "can" --target_robot "Sawyer" --partition 0

            '''

        os.makedirs(os.path.join(save_paired_images_folder_path, "source_robot_states", f"{target_name}", "end_effector"), exist_ok=True) 
        os.makedirs(os.path.join(save_paired_images_folder_path, "source_robot_states", f"{target_name}", "gripper_distance"), exist_ok=True)
        os.makedirs(os.path.join(save_paired_images_folder_path, "source_robot_states", f"{target_name}", "joint_angles"), exist_ok=True)
        os.makedirs(os.path.join(save_paired_images_folder_path, "source_robot_states", f"{target_name}", "offsets"), exist_ok=True)

        # Create displacement.csv if it does not exist
        csv_path = os.path.join(save_paired_images_folder_path, target_name, "displacement.csv")
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        if not os.path.isfile(csv_path):
            with open(csv_path, "w", newline="") as f:
                pass

        whitelist_path = Path(f"{save_paired_images_folder_path}/{target_name}/whitelist.json")
        if not whitelist_path.exists():
            whitelist_path.parent.mkdir(parents=True, exist_ok=True)   # create any missing folders
            whitelist_path.write_text(json.dumps({}))
        blacklist_path = Path(f"{save_paired_images_folder_path}/{target_name}/blacklist.json")
        if not blacklist_path.exists():
            blacklist_path.parent.mkdir(parents=True, exist_ok=True)
            blacklist_path.write_text(json.dumps({}))

        for episode in episodes:
            # if episode in whitelist, then skip
            print("skip", episode)
            whitelist_path = Path(f"{save_paired_images_folder_path}/{target_name}/whitelist.json")
            with locked_json(whitelist_path) as wl:
                robot_list = wl.get(target_name, [])
                if robot_list and episode in robot_list:
                    #print(f"[WHITELIST] Skipping {target_name} – episode {episode}")
                    continue
            print(f"[INFO] Processing {target_name} – episode {episode}")
            target_env = TargetEnvWrapper(target_name, target_gripper, args.robot_dataset, camera_height, camera_width)

            target_env.generate_image(
                save_paired_images_folder_path=save_paired_images_folder_path, 
                displacement_csv_path=csv_path,
                source_robot_states_path=source_robot_states_path,
                reference_joint_angles_path=args.reference_joint_angles_path, 
                reference_ee_states_path=args.reference_ee_states_path, 
                robot_dataset=args.robot_dataset, 
                episode=episode,
                camera_height=camera_height,
                camera_width=camera_width,
                unlimited=args.unlimited,
                load_displacement=args.load_displacement
            )
            target_env.target_env.env.close_renderer()