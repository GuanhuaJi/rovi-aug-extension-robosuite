'''
python /home/guanhuaji/mirage/robot2robot/rendering/export_source_robot_states.py --robot_dataset autolab_ur5


datasets: 
austin_buds, austin_mutex, austin_sailor, 
autolab_ur5, can, furniture_bench, iamlab_cmu, 
lift, nyu_franka, square, stack, three_piece_assembly, 
taco_play, ucsd_kitchen_rlds, viola, kaist
toto, 

'''

import argparse
import json
import os
import cv2
import socket, pickle, struct
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import robosuite as suite
import robosuite.utils.transform_utils as T
import robosuite.utils.camera_utils as camera_utils
from robosuite.utils.camera_utils import CameraMover
import xml.etree.ElementTree as ET
import robosuite.macros as macros
macros.IMAGE_CONVENTION = "opencv"
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from robot_pose_dict import ROBOT_POSE_DICT
from pathlib import Path
from typing import Tuple
from mujoco import mjtObj
import pynvml

import logging
logger = logging.getLogger(__name__) 

np.set_printoptions(suppress=True, precision=6)

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

def load_blacklist(blacklist_path) -> dict:
    if blacklist_path.exists() and blacklist_path.stat().st_size > 0:
        with blacklist_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def quat_dist_rad(q1, q2):
    """
    最小旋转角：两单位四元数内积的 arccos。
    输入 shape=(4,), 顺序 [qw, qx, qy, qz]
    """
    dot = np.abs(np.dot(q1, q2))
    dot = np.clip(dot, -1.0, 1.0)  # 数值安全
    return 2.0 * np.arccos(dot)

def gripper_convert(gripper_state_value, robot_type):
    if robot_type == "autolab_ur5":
        return gripper_state_value == 0
    elif robot_type == "furniture_bench":
        return gripper_state_value > 0.05
    elif robot_type == "viola":
        return gripper_state_value > 0.06
    elif robot_type == "austin_sailor":
        return gripper_state_value > 0.05
    elif robot_type == "austin_mutex":
        return gripper_state_value > 0.05
    elif robot_type == "austin_buds":
        return gripper_state_value > 0.05
    elif robot_type == "nyu_franka":
        return gripper_state_value >= 0
    elif robot_type == "ucsd_kitchen_rlds":
        return gripper_state_value > 0.5
    elif robot_type == "taco_play":
        return gripper_state_value < 0
    elif robot_type == "iamlab_cmu":
        return gripper_state_value > 0.5
    elif robot_type == "toto":
        return gripper_state_value > 0
    elif robot_type == "can":
        return gripper_state_value
    elif robot_type == "lift":
        return gripper_state_value
    elif robot_type == "square":
        return gripper_state_value
    elif robot_type == "stack":
        return gripper_state_value
    elif robot_type == "three_piece_assembly":
        return gripper_state_value
    print("UNKNOWN GRIPPER")
    return None

def image_to_pointcloud(env, depth_map, camera_name, camera_height, camera_width, segmask=None):
    """
    Convert depth image to point cloud
    """
    real_depth_map = camera_utils.get_real_depth_map(env.sim, depth_map)
    # Camera transform matrix to project from camera coordinates to world coordinates.
    extrinsic_matrix = camera_utils.get_camera_extrinsic_matrix(env.sim, camera_name=camera_name)
    intrinsic_matrix = camera_utils.get_camera_intrinsic_matrix(env.sim, camera_name=camera_name, camera_height=camera_height, camera_width=camera_width)

    # Convert depth image to point cloud
    points = [] # 3D points in robot frame of shape […, 3]
    for x in range(camera_width):
        for y in range(camera_height):
            if segmask is not None and segmask[y, x] == 0:
                continue
            coord_cam_frame = np.array([(x-intrinsic_matrix[0, -1])/intrinsic_matrix[0, 0], (y-intrinsic_matrix[1, -1])/intrinsic_matrix[1, 1], 1]) * real_depth_map[y, x]
            coord_world_frame = np.dot(extrinsic_matrix, np.concatenate((coord_cam_frame, [1])))
            points.append(coord_world_frame)

    return points


def sample_half_hemisphere(num_samples):
    radius = np.random.normal(0.85, 0.2, num_samples)
    hemisphere_center = np.array([-0.31558805, -0.04495631,  1.271939112])
    theta = np.random.uniform(np.pi/4, np.pi/2.2, num_samples)  # Angle with respect to the z-axis
    phi = np.random.uniform(-np.pi*3.7/4, np.pi*3.7/4, num_samples)  # Azimuthal angle
    positions = np.zeros((num_samples, 3))
    positions[:, 0] = radius * np.sin(theta) * np.cos(phi)  # x-coordinate
    positions[:, 1] = radius * np.sin(theta) * np.sin(phi)  # y-coordinate
    positions[:, 2] = radius * np.cos(theta)  # z-coordinate

    # Calculate orientations (quaternions)
    backward_directions = positions - hemisphere_center
    backward_directions /= np.linalg.norm(backward_directions, axis=1, keepdims=True)
    right_directions = np.cross(np.tile(np.array([0, 0, 1]), (num_samples, 1)), backward_directions)  # Assuming right direction is along the x-axis
    right_directions /= np.linalg.norm(right_directions, axis=1, keepdims=True)
    up_directions = np.cross(backward_directions, right_directions)
    up_directions /= np.linalg.norm(up_directions, axis=1, keepdims=True)

    rotations = np.array([np.column_stack((right, down, forward)) for right, down, forward in zip(right_directions, up_directions, backward_directions)])
    # rotations = np.array([                                                        [[ 0.   ,       0.70614784, -0.70806442     ],
    #                                                         [ 1.    ,      0.      ,    0.                ],
    #                                                         [ 0.     ,    -0.70806442 ,-0.70614784     ]]])

    # Convert rotation matrices to quaternions
    quaternions = []
    for rotation_matrix in rotations:
        rotation = Rotation.from_matrix(rotation_matrix)
        quaternion = rotation.as_quat()
        quaternions.append(quaternion)

    quaternions = np.array(quaternions)


    return positions, quaternions

def sample_robot_ee_pose():
    # the position should be in the follow range:
    # x: -0.3 ~ 0.3
    # y: -0.3 ~ 0.3
    # z: 0.5 ~ 1.5
    # np.random.seed(0)
    pos = np.random.uniform(-0.25, 0.25, 3)
    pos[2] = np.random.uniform(0.6, 1.3)
    # quat = np.random.uniform(-1, 1, 4)
    # quat /= np.linalg.norm(quat)
    
    def sample_rotation_matrix():
        # Sample theta (zenith angle) from a normal distribution centered around pi
        theta = np.random.normal(loc=np.pi, scale=np.pi/3.5)
        # print("theta: ", theta)
        # Sample phi (azimuthal angle) uniformly between 0 and 2*pi
        phi = np.random.uniform(0, 2*np.pi)
        
        # Convert spherical coordinates to Cartesian coordinates
        z_axis = np.array([np.sin(theta) * np.cos(phi),
                        np.sin(theta) * np.sin(phi),
                        np.cos(theta)])
        
        # Sample a random vector for the rightward direction (perpendicular to z-axis)
        rightward = np.random.uniform(-1, 1, size=3)
        rightward -= np.dot(rightward, z_axis) * z_axis
        rightward /= np.linalg.norm(rightward)
        
        # Compute the inward direction (perpendicular to both z and rightward axes)
        inward = np.cross(rightward, z_axis)
        
        # Construct the rotation matrix
        R = np.column_stack((inward, rightward, z_axis))
    
        return R

    quat = T.mat2quat(sample_rotation_matrix())
    return np.concatenate((pos, quat))

'''
def compute_pose_error(current_pose, target_pose):
    # quarternions are equivalent up to sign
    error = min(np.linalg.norm(current_pose - target_pose), np.linalg.norm(current_pose - np.concatenate((target_pose[:3], -target_pose[3:]))))
    return error 
'''  

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

def change_brightness(img, value=30, mask=None):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    if mask is None:
        mask = np.ones_like(v)
    else:
        mask = mask.squeeze()
    # Apply mask to the brightness channel
    if value > 0:
        lim = 255 - value
        v[(v > lim) & (mask == 1)] = 255
        v[(v <= lim) & (mask == 1)] += value
    else:
        lim = -value
        v[(v < lim) & (mask == 1)] = 0
        v[(v >= lim) & (mask == 1)] -= lim

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

class CameraWrapper:
    def __init__(self, env, camera_name="agentview"):
        self.env = env
        # Create the camera mover
        self.camera_mover = CameraMover(
            env=env,
            camera=camera_name,
        )
        self.cam_tree = ET.Element("camera", attrib={"name": camera_name})
        CAMERA_NAME = self.cam_tree.get("name") # Make sure we're using the camera that we're modifying
        self.camera_id = env.sim.model.camera_name2id(CAMERA_NAME)
        self.env.viewer.set_camera(camera_id=self.camera_id)
        
        
        # Define initial file camera pose
        initial_file_camera_pos, initial_file_camera_quat = self.camera_mover.get_camera_pose()
        initial_file_camera_pose = T.make_pose(initial_file_camera_pos, T.quat2mat(initial_file_camera_quat))

        # remember difference between camera pose in initial tag and absolute camera pose in world
        # usually we just operate in the wolrd frame, so we don't need to worry about the difference
        # but if we ever want to know the camera pose in the file frame, we can use this
        initial_world_camera_pos, initial_world_camera_quat = self.camera_mover.get_camera_pose()
        initial_world_camera_pose = T.make_pose(initial_world_camera_pos, T.quat2mat(initial_world_camera_quat))
        self.world_in_file = initial_file_camera_pose.dot(T.pose_inv(initial_world_camera_pose))
        
    
    def set_camera_fov(self, fov=45.0):
        self.env.sim.model.cam_fovy[self.camera_id] = float(fov)
        # for _ in range(50):
        #     self.env.sim.forward()
        #     self.env.sim.step()
        #     self.env._update_observables()
    
    def set_camera_pose(self, pos, quat, offset=np.array([0, 0, 0])):
        # Robot base world coord: -0.6 0.0 0.912
        self.camera_mover.set_camera_pose(pos=pos + offset, quat=quat)
        target_pose = np.concatenate((pos + offset, quat))
        current_pose = self.get_camera_pose_world_frame()
        error = compute_pose_error(current_pose, target_pose)
        # for _ in range(50):
        #     self.camera_mover.set_camera_pose(pos=pos + offset, quat=quat)
        #     self.env.sim.forward()
        #     self.env.sim.step()
        #     self.env._update_observables()
            
    
    def get_camera_pose_world_frame(self):
        camera_pos, camera_quat = self.camera_mover.get_camera_pose()
        # world_camera_pose = T.make_pose(camera_pos, T.quat2mat(camera_quat))
        # print("Camera pose in the world frame:", camera_pos, camera_quat)
        return np.concatenate((camera_pos, camera_quat))
    
    def get_camera_pose_file_frame(self, world_camera_pose):
        file_camera_pose = self.world_in_file.dot(world_camera_pose)
        camera_pos, camera_quat = T.mat2pose(file_camera_pose)
        camera_quat = T.convert_quat(camera_quat, to="wxyz")

        print("\n\ncurrent camera tag you should copy")
        self.cam_tree.set("pos", "{} {} {}".format(camera_pos[0], camera_pos[1], camera_pos[2]))
        self.cam_tree.set("quat", "{} {} {} {}".format(camera_quat[0], camera_quat[1], camera_quat[2], camera_quat[3]))
        print(ET.tostring(self.cam_tree, encoding="utf8").decode("utf8"))
        
    def perturb_camera(self, seed=None, angle=10, scale=0.15):
        # np.random.seed(seed)
        self.camera_mover.rotate_camera(point=None, axis=np.random.uniform(-1, 1, 3), angle=angle)
        self.camera_mover.move_camera(direction=np.random.uniform(-1, 1, 3), scale=scale)
        # for _ in range(50):
        #     self.env.sim.forward()
        #     self.env.sim.step()
        #     self.env._update_observables()
        

def fast_step(env, action):
    """
    Minimal physics step that **完全绕开渲染与观测**，
    兼容 robosuite 1.0 ~ 1.6 及任何自定义 MujocoEnv。
    返回 (reward, done, info)；不生成 obs。
    """
    # 1. 计算“一个控制周期需要多少 sim 子步”
    substeps = int(env.control_timestep / env.model_timestep)

    # 2. 先把动作写入电机
    policy_step = True
    for _ in range(substeps):
        # 和 robosuite.step() 保持同样的前/后处理
        if hasattr(env, "_pre_action"):
            env._pre_action(action, policy_step=policy_step)
        # 纯物理推进：旧版只有 sim.step()
        if hasattr(env.sim, "step"):
            env.sim.step()
        else:                           # 极早期 robosuite
            env.sim.forward()
        policy_step = False

    # 3. 时间推进
    if hasattr(env, "cur_time"):       # robosuite >=0.4
        env.cur_time += env.control_timestep
    if hasattr(env, "timestep"):       # robosuite <=0.3
        env.timestep += 1

    # 4. 直接返回空奖励等（如果你不在意 reward）
    return 0.0, False, {}

def _robot_geom_ids(env):
    """
    返回当前 robot 在 sim 中的 geom_id 集合。
    兼容所有 robosuite 版本：
        先尝试 .geom_names；
        否则退化为 contact_geoms + visual_geoms。
    """
    robot = env.robots[0]
    model = getattr(robot, "robot_model", robot)     # robosuite <1.4 没有 robot_model
    # 1) 优先使用 geom_names（1.6+ 才有）
    if hasattr(model, "geom_names") and model.geom_names:
        names = model.geom_names
    else:                                            # 2) 通用回退逻辑
        names = []
        if hasattr(model, "contact_geoms"):
            names += list(model.contact_geoms)
        if hasattr(model, "visual_geoms"):
            names += list(model.visual_geoms)
    # 转成 geom_id；过滤掉 xml 中被删除或重命名的几何体
    m = env.sim.model
    return {
        m.geom_name2id(n) for n in names
        if n in m.geom_names
    }
def _robot_geom_ids(env):
    sim   = env.sim
    robot = env.robots[0]

    # ---- 1. arm geoms ----
    model = getattr(robot, "robot_model", robot)
    arm_names = (getattr(model, "geom_names", []) or
                 getattr(model, "visual_geoms", []) +
                 getattr(model, "contact_geoms", []))

    # ---- 2. gripper geoms ----
    grip = getattr(robot, "gripper", None)
    grip_names = []
    if grip is not None:
        grip_names = getattr(grip, "visual_geoms", []) + getattr(grip, "contact_geoms", [])

    # ---- 3. union + id mapping ----
    names = set(arm_names) | set(grip_names)
    ids = {sim.model.geom_name2id(n) for n in names if n in sim.model.geom_names}
    return ids


class RobotCameraWrapper:
    def __init__(self, robotname="Panda", grippername="PandaGripper", robot_dataset=None, camera_height=256, camera_width=256):
        options = {}
        self.env = suite.make(
            **options,
            robots=robotname,
            gripper_types=grippername,
            env_name="Empty",
            has_renderer=True,  # no on-screen renderer
            has_offscreen_renderer=True,  # no off-screen renderer
            ignore_done=True,
            use_camera_obs=True,  # no camera observations
            controller_configs = suite.load_controller_config(default_controller="OSC_POSE"),
            control_freq=20,
            renderer="mujoco",
            camera_names = ["agentview"],  # You can add more camera names if needed
            camera_heights = camera_height,
            camera_widths = camera_width,
            camera_depths = True,
            camera_segmentations = "robot_only",
            hard_reset=False,
        )
        
        self.camera_wrapper = CameraWrapper(self.env)
        from robot_pose_dict import ROBOT_POSE_DICT
        self.robot_name = robotname
        self.robot_base_name = f"robot0_base"
        self.base_body_id = self.env.sim.model.body_name2id(self.robot_base_name)
        self.base_position = self.env.sim.model.body_pos[self.base_body_id].copy()
        
        #print(f"[DEBUG] 机器人基座 '{self.robot_base_name}' 的世界坐标: {self.base_position}")

    def get_gripper_width_from_qpos(self):
        sim   = self.env.sim
        robot = self.env.robots[0]

        # ① robosuite ≥1.4：官方给好了索引
        if hasattr(robot, "_ref_gripper_joint_pos_indexes") and robot._ref_gripper_joint_pos_indexes is not None:
            qpos_idx = robot._ref_gripper_joint_pos_indexes
        else:
            # ② 早期版本：根据关节名自己找
            joint_names = robot.gripper.joints               # e.g. ['gripper0_finger_joint1', 'gripper0_finger_joint2']
            qpos_idx = [sim.model.get_joint_qpos_addr(name) for name in joint_names]

        finger_qpos = sim.data.qpos[qpos_idx]
        # 对两指对称平动夹爪 (Panda / Robotiq 85) —— 单边位移乘 2 就是张开宽度
        return 2.0 * finger_qpos[0]

    def compute_eef_pose(self):
        pos = np.array(self.env.sim.data.site_xpos[self.env.sim.model.site_name2id(self.env.robots[0].controller.eef_name)])
        rot = np.array(T.mat2quat(self.env.sim.data.site_xmat[self.env.sim.model.site_name2id(self.env.robots[0].controller.eef_name)].reshape([3, 3])))
        return np.concatenate((pos, rot))
    
    def teleport_to_joint_positions(self, joint_angles):
        joint_names = self.env.robots[0].robot_joints
        for i, joint_name in enumerate(joint_names):
            qpos_addr = self.env.sim.model.get_joint_qpos_addr(joint_name)
            self.env.sim.data.qpos[qpos_addr] = joint_angles[i]
            self.env.sim.data.qvel[qpos_addr] = 0.0
        self.env.sim.forward()

    def drive_robot_to_target_pose(self, target_pose=None, min_threshold=0.003, max_threshold=0.01, num_iter_max=200):
        # breakpoint()
        # reset robot joint positions so the robot is hopefully not in a weird pose
        # self.set_robot_joint_positions()
        self.env.robots[0].controller.use_delta = False # change to absolute pose for setting the initial state
        
        assert len(target_pose) == 7, "Target pose should be 7DOF"
        current_pose = self.compute_eef_pose()
        error = compute_pose_error(current_pose, target_pose)
        num_iters = 0   

        no_improve_steps = 0
        last_error = error 
        while error > min_threshold and num_iters < num_iter_max:
            # print("Current pose: ", current_pose)
            # print("Target pose: ", target_pose)
            # print("Error: ", error)
            action = np.zeros(7)
            action[:3] = target_pose[:3]
            action[3:6] = T.quat2axisangle(target_pose[3:])

            #obs, _, _, _ = self.env.step(action)
            #self.env.step(action)
            _, _, _ = fast_step(self.env, action)
            current_pose = self.compute_eef_pose()
            current_joints = self.env.sim.data.qpos[self.env.robots[0]._ref_joint_pos_indexes].copy()
            #print("Current joints: ", current_joints)
            self.some_safe_joint_angles = current_joints
            new_error = compute_pose_error(current_pose, target_pose)

            if abs(new_error - error) < 1e-5:
                no_improve_steps += 1
            else:
                no_improve_steps = 0

            error = new_error
            num_iters += 1
        # print("ERROR", error)
        # print("Take {} iterations to drive robot to target pose".format(num_iters))
        current_pose = self.compute_eef_pose()
        self.env.use_camera_obs = True

        if error < max_threshold:
            return True, current_pose
        else:
            print("Robot is not in the target pose")
            print("Current pose: ", current_pose)
            print("Target pose: ", target_pose)
            print("Error: ", error)
            return False, current_pose

    def set_robot_joint_positions(self, joint_angles=None):
        if joint_angles is None:
            joint_angles = self.some_safe_joint_angles
        for _ in range(200):
            self.env.robots[0].set_robot_joint_positions(joint_angles)
            self.env.sim.forward()
            self.env.sim.step()
            self.env._update_observables()

    def set_gripper_joint_positions(self, finger_qpos, robot_name):
        if robot_name == "Panda":
            gripper_joint_names = ["gripper0_finger_joint1", "gripper0_finger_joint2"]
        elif robot_name == "IIWA":
            gripper_joint_names = ["gripper0_finger_joint", "gripper0_right_outer_knuckle_joint"]
        elif robot_name == "Sawyer":
            gripper_joint_names = ['gripper0_l_finger_joint', 'gripper0_r_finger_joint']
        elif robot_name == "Jaco":
            gripper_joint_names = ["gripper0_joint_thumb", "gripper0_joint_index", "gripper0_joint_pinky",]
        
        for i, joint_name in enumerate(gripper_joint_names):
            self.env.sim.data.set_joint_qpos(joint_name, finger_qpos[i])
        for _ in range(10):
            self.env.sim.forward()
            self.env.sim.step()
    
    def open_close_gripper(self, gripper_open=True):
        self.env.robots[0].controller.use_delta = True # change to delta pose
        action = np.zeros(7)
        if not gripper_open:
            action[-1] = 1
        else:
            action[-1] = -1            
        obs, _, _, _ = self.env.step(action)
    
    def hardset_gripper(self, gripper_state_value):
        self.env.robots[0].set_gripper_joint_positions(gripper_state_value)

    
    def update_camera(self):
        for _ in range(50):
            self.env.sim.forward()
            self.env.sim.step()
            self.env._update_observables()
          
    def get_observation_fast(self, camera="agentview",
                            width=640, height=480,
                            white_background=True):
        sim = self.env.sim
        sim.forward()                                           # 同步姿态

        rgb = sim.render(width=width, height=height,
                        camera_name=camera)[::-1]
        seg = sim.render(width=width, height=height,
                        camera_name=camera,
                        segmentation=True)[::-1]               # (H,W,2)

        objtype_img = seg[..., 0]
        objid_img   = seg[..., 1]

        robot_body_ids = _robot_geom_ids(self.env)
        mask = (np.isin(objid_img, list(robot_body_ids))).astype(np.uint8)
        if white_background:
            rgb_out = rgb.copy()
            rgb_out[mask == 0] = 255
        else:
            rgb_out = (rgb * mask[..., None]).astype(np.uint8)

        return rgb_out, mask


    def get_observation(self, white_background=True):
        view = "agentview"
        obs = self.env._get_observations()
        rgb_img_raw = obs[f'{view}_image']
        seg_img = obs[f'{view}_segmentation_robot_only']
        num_robot_pixels = np.sum(seg_img)
        #if num_robot_pixels <= 700:
            #print(num_robot_pixels, " robot pixels in the image")
            #return None, None
        # Create a mask where non-robot pixels are set to 0 and robot pixels are set to 1
        if white_background:
            mask = (np.repeat(seg_img, 3, axis=2)).astype(bool)
            rgb_img = np.where(~mask, [255, 255, 255], rgb_img_raw) # Set non-robot pixels to white
            rgb_img = rgb_img.astype(np.uint8) # Convert the resulting image to uint8 type
        else:
            # only retain the robot part in the rgb_img, set other pixels to black
            rgb_img = (rgb_img_raw * seg_img).astype(np.uint8)
        return rgb_img, seg_img


class SourceEnvWrapper:
    def __init__(self, source_name, source_gripper, robot_dataset, camera_height=256, camera_width=256, verbose=False):
        self.source_env = RobotCameraWrapper(robotname=source_name, grippername=source_gripper, robot_dataset=robot_dataset, camera_height=camera_height, camera_width=camera_width)
        self.source_name = source_name
        self.fixed_cam_positions = None
        self.fixed_cam_quaternions = None
        self.verbose = verbose
    
    def _receive_all_bytes(self, num_bytes: int) -> bytes:
        """
        Receives all the bytes.
        :param num_bytes: The number of bytes.
        :return: The bytes.
        """
        data = bytearray(num_bytes)
        pos = 0
        while pos < num_bytes:
            cr = self.conn.recv_into(memoryview(data)[pos:])
            if cr == 0:
                raise EOFError
            pos += cr
        return data

    def _load_dataset_info(self, dataset_name):
        from dataset_poses_dict import ROBOT_CAMERA_POSES_DICT
        info = ROBOT_CAMERA_POSES_DICT[dataset_name]
        return info
    
    def _load_dataset_files(self, info, dataset_name):
        # 可能包含 joint_angles, ee_states, gripper_states
        joint_angles = None
        ee_states = None
        gripper_states = None
        if "robot_joint_angles_path" in info:
            joint_angles_path = info["robot_joint_angles_path"]
            joint_angles = np.loadtxt(joint_angles_path)
            if dataset_name == "toto":
                joint_angles[:, 5] += 3.14159 / 2
                joint_angles[:, 6] += 3.14159 / 4
            if dataset_name == "autolab_ur5":
                joint_angles[:, 5] += 3.14159 / 2
        if "robot_ee_states_path" in info:
            ee_states_path = info["robot_ee_states_path"]
            ee_states = np.loadtxt(ee_states_path)
        gripper_states_path = info["gripper_states_path"]
        gripper_states = np.loadtxt(gripper_states_path)
        return joint_angles, ee_states, gripper_states
    
    def get_source_robot_states(self, save_source_robot_states_path="paired_images", reference_joint_angles_path=None, reference_ee_states_path=None, reference_gripper_states_path=None, robot_dataset=None, episode=0):
        info = self._load_dataset_info(robot_dataset)
        joint_angles = np.loadtxt(os.path.join("/home/guanhuaji/mirage/robot2robot/rendering/datasets/states", robot_dataset, f"episode_{episode}", "joint_states.txt"))
        gripper_states = np.loadtxt(os.path.join("/home/guanhuaji/mirage/robot2robot/rendering/datasets/states", robot_dataset, f"episode_{episode}", "gripper_states.txt"))
        if robot_dataset == "toto":
            joint_angles[:, 5] += 3.14159 / 2
            joint_angles[:, 6] += 3.14159 / 4
        if robot_dataset == "autolab_ur5":
            joint_angles[:, 5] += 3.14159 / 2
        num_robot_poses = joint_angles.shape[0]

        if robot_dataset == "can":
            camera_reference_pose = np.array([0.9, 0.1, 1.75, 0.271, 0.271, 0.653, 0.653])
            cam_id = self.source_env.camera_wrapper.env.sim.model.camera_name2id("agentview")
            fov = self.source_env.camera_wrapper.env.sim.model.cam_fovy[cam_id]
        elif robot_dataset == "lift":
            camera_reference_pose = np.array([0.45, 0, 1.35, 0.271, 0.271, 0.653, 0.653])
            cam_id = self.source_env.camera_wrapper.env.sim.model.camera_name2id("agentview")
            fov = self.source_env.camera_wrapper.env.sim.model.cam_fovy[cam_id]
        elif robot_dataset == "square":
            camera_reference_pose = np.array([0.45, 0, 1.35, 0.271, 0.271, 0.653, 0.653])
            cam_id = self.source_env.camera_wrapper.env.sim.model.camera_name2id("agentview")
            fov = self.source_env.camera_wrapper.env.sim.model.cam_fovy[cam_id]
        elif robot_dataset == "stack":
            camera_reference_pose = np.array([0.45, 0, 1.35, 0.271, 0.271, 0.653, 0.653])
            cam_id = self.source_env.camera_wrapper.env.sim.model.camera_name2id("agentview")
            fov = self.source_env.camera_wrapper.env.sim.model.cam_fovy[cam_id]
        elif robot_dataset == "three_piece_assembly":
            camera_reference_pose = np.array([0.713078462147161, 2.062036796036723e-08, 1.5194726087166726, 0.293668270111084, 0.2936684489250183, 0.6432408690452576, 0.6432409286499023])
            cam_id = self.source_env.camera_wrapper.env.sim.model.camera_name2id("agentview")
            fov = self.source_env.camera_wrapper.env.sim.model.cam_fovy[cam_id]
        else:
            camera_reference_position = info["camera_position"] + np.array([-0.6, 0.0, 0.912]) 
            roll_deg = info["roll"]
            pitch_deg = info["pitch"]
            yaw_deg = info["yaw"]
            fov = info["camera_fov"]
            r = R.from_euler('xyz', [roll_deg, pitch_deg, yaw_deg], degrees=True)
            camera_reference_quaternion = r.as_quat()
            camera_reference_pose = np.concatenate((camera_reference_position, camera_reference_quaternion))

        target_pose_list = []
        gripper_list = []

        for pose_index in tqdm(range(num_robot_poses), desc=f'{self.source_name} Pose States Calculation'):    
            source_reached = False
            attempt_counter = 0
            while source_reached == False:
                attempt_counter += 1
                if attempt_counter > 10:
                    break
                if robot_dataset == "kaist":
                    gripper_open = False
                elif robot_dataset in ["can", "lift", "square", "stack", "three_piece_assembly"]:
                    gripper_open = (gripper_states[pose_index][0] - gripper_states[pose_index][1]) > 0.06
                else:
                    gripper_open = gripper_convert(gripper_states[pose_index], robot_dataset)
                for i in range(2):
                    joint_angle = joint_angles[pose_index]
                    self.source_env.teleport_to_joint_positions(joint_angle)
                    '''
                    if type(gripper_open) is bool:
                        self.source_env.open_close_gripper(gripper_open=gripper_open)
                    else:
                        self.source_env.set_gripper_joint_positions(gripper_states[pose_index], self.source_name)
                        gripper_open = (gripper_states[pose_index][0] - gripper_states[pose_index][1]) > 0.06
                    '''
                    target_pose = self.source_env.compute_eef_pose()

            gripper_list.append(gripper_open)
            target_pose_list.append(target_pose)       
        target_pose_array = np.vstack(target_pose_list)
        gripper_array = np.vstack(gripper_list)
        eef_npy_path = os.path.join(
            save_source_robot_states_path, f"{self.source_name}_eef_states_{episode}.npy"
        )
        GREEN = "\033[92m"
        RESET = "\033[0m"
        np.save(eef_npy_path, target_pose_array)
        print(f"{GREEN}✔ End effector saved under {eef_npy_path}{RESET}")
        npz_path = os.path.join(save_source_robot_states_path, f"{episode}.npz")
        np.savez(npz_path, pos=target_pose_array, grip=gripper_array, camera=camera_reference_pose, fov=fov)
        print(f"{GREEN}✔ States saved under {npz_path}{RESET}")

        


if __name__ == "__main__":

    """
    Registered environments: Lift, Stack, NutAssembly, NutAssemblySingle, NutAssemblySquare, NutAssemblyRound,
                             PickPlace, PickPlaceSingle, PickPlaceMilk, PickPlaceBread, PickPlaceCereal,
                             PickPlaceCan, Door, Wipe, TwoArmLift, TwoArmPegInHole, TwoArmHandover

    Possible robots: Baxter, IIWA, Jaco, Kinova3, Panda, Sawyer, UR5e
    Possible grippers: 'RethinkGripper', 'PandaGripper', 'JacoThreeFingerGripper', 'JacoThreeFingerDexterousGripper', 
    'WipingGripper', 'Robotiq85Gripper', 'Robotiq140Gripper', 'RobotiqThreeFingerGripper', 'RobotiqThreeFingerDexterousGripper'
    """

    # print welcome info
    #print("Welcome to robosuite v{}!".format(suite.__version__))
    #print(suite.__logo__)

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="(optional) set seed")
    parser.add_argument("--source_gripper", type=str, default="PandaGripper", help="PandaGripper or Robotiq85Gripper")
    parser.add_argument("--save_paired_images_folder_path", type=str, default="paired_images", help="(optional) folder path to save the paired images")
    parser.add_argument("--robot_dataset", type=str, help="(optional) to match the robot poses from a dataset, provide the dataset name")
    parser.add_argument("--reference_joint_angles_path", type=str, help="(optional) to match the robot poses from a dataset, provide the path to the joint angles file (np.savetxt)")
    parser.add_argument("--reference_ee_states_path", type=str, help="(optional) to match the robot poses from a dataset, provide the path to the ee state file (np.savetxt)")
    parser.add_argument("--reference_gripper_states_path", type=str, help="(optional) to match the gripper's open/close status")
    parser.add_argument("--verbose", action='store_true', help="If set, prints extra debug/warning information")
    args = parser.parse_args()

    if args.robot_dataset == "autolab_ur5" or args.robot_dataset == "asu_table_top_rlds":
        source_name = "UR5e"
        source_gripper = "Robotiq85Gripper"
    else:
        source_name = "Panda"
        source_gripper = "PandaGripper"

    save_source_robot_states_path = (Path("/home/guanhuaji/mirage/robot2robot/rendering/paired_images") / args.robot_dataset / "source_robot_states")
    save_source_robot_states_path.mkdir(parents=True, exist_ok=True)
    
    if args.robot_dataset is not None:
        from dataset_poses_dict import ROBOT_CAMERA_POSES_DICT
        robot_dataset_info = ROBOT_CAMERA_POSES_DICT[args.robot_dataset]
        camera_height = robot_dataset_info["camera_heights"]
        camera_width = robot_dataset_info["camera_widths"]
    else:
        camera_height = 256
        camera_width = 256
    
    num_episode = ROBOT_CAMERA_POSES_DICT[args.robot_dataset]['num_episodes']

    
    episodes = range(num_episode // 2)
    episodes = range(num_episode // 2, num_episode)

    '''
    conda activate mirage
    python /home/guanhuaji/mirage/robot2robot/rendering/export_source_robot_states.py --robot_dataset kaist
    '''


    for episode in episodes:
        source_env = SourceEnvWrapper(source_name, source_gripper, args.robot_dataset, camera_height, camera_width, verbose=args.verbose)
        source_env.get_source_robot_states(
            save_source_robot_states_path=save_source_robot_states_path, 
            reference_joint_angles_path=args.reference_joint_angles_path, 
            reference_ee_states_path=args.reference_ee_states_path, 
            reference_gripper_states_path=args.reference_gripper_states_path,
            robot_dataset=args.robot_dataset,
            episode=episode
        )

        source_env.source_env.env.close_renderer()