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
from PIL import Image


np.set_printoptions(suppress=True, precision=6)

CAMERA_HISTORY_FILE = "camera_pose_history.txt"
import math

def load_camera_pose_history(filename=CAMERA_HISTORY_FILE):
    """
    从文本文件里读取所有历史相机姿态 (x, y, z, roll, pitch, yaw, fov)。
    返回一个列表，每个元素都是 [x, y, z, roll, pitch, yaw, fov]。
    如果文件不存在则返回空列表。
    """
    if not os.path.exists(filename):
        return []

    poses = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            arr = line.split(',')
            if len(arr) != 7:
                continue
            float_arr = list(map(float, arr))
            poses.append(float_arr)
    return poses

def append_camera_pose_to_history(x, y, z, roll, pitch, yaw, fov, filename=CAMERA_HISTORY_FILE):
    """
    将当前相机姿态以逗号分隔的形式写到文件末尾。
    """
    with open(filename, 'a') as f:
        line = f"{x},{y},{z},{roll},{pitch},{yaw},{fov}\n"
        f.write(line)


def matrix_to_xyz_quat(T):
    # 1) 提取平移 (x,y,z)
    x = T[0, 3]
    y = T[1, 3]
    z = T[2, 3]

    # 2) 提取旋转矩阵 R_mat
    R_mat = T[:3, :3]

    # 3) 转成四元数
    rot = R.from_matrix(R_mat)
    # SciPy 默认返回 [x, y, z, w] (即最后一个是实部)
    qx, qy, qz, qw = rot.as_quat()

    return np.array([x, y, z]), np.array([qw, qx, qy, qz])


def get_current_joint_positions(env, robot_index=0):
    """
    从仿真数据中读取 env.robots[robot_index] 对应的所有关节位置 (qpos) 并返回为 np.array
    """
    joint_names = env.robots[robot_index].robot_joints  # 拿到该机器人包含的所有关节名称
    joint_positions = []

    for name in joint_names:
        jpos_addr = env.sim.model.get_joint_qpos_addr(name)
        jpos = env.sim.data.qpos[jpos_addr]
        joint_positions.append(jpos)

    return np.array(joint_positions)

def gripper_convert(gripper_state_value, robot_type):
    if robot_type == "hydra":
        return gripper_state_value == 0
    if robot_type == "maskvit":
        return gripper_state_value > 0
    if robot_type == "kuka":
        return gripper_state_value > 0
    if robot_type == "autolab_ur5":
        return gripper_state_value == 0
    if robot_type == "nyu_franka":
        return gripper_state_value >= 0
    if robot_type == "ucsd_kitchen_rlds":
        return gripper_state_value > 0.5
    if robot_type == "kaist":
        return gripper_state_value 
    print("UNKNOWN GRIPPER")

def image_to_pointcloud(env, depth_map, camera_name, camera_height, camera_width, segmask=None):
    """
    Convert depth image to point cloud
    """
    real_depth_map = camera_utils.get_real_depth_map(env.sim, depth_map)
    extrinsic_matrix = camera_utils.get_camera_extrinsic_matrix(env.sim, camera_name=camera_name)
    intrinsic_matrix = camera_utils.get_camera_intrinsic_matrix(env.sim, camera_name=camera_name, camera_height=camera_height, camera_width=camera_width)

    points = []
    for x in range(camera_width):
        for y in range(camera_height):
            if segmask is not None and segmask[y, x] == 0:
                continue
            coord_cam_frame = np.array([(x-intrinsic_matrix[0, -1])/intrinsic_matrix[0, 0],
                                        (y-intrinsic_matrix[1, -1])/intrinsic_matrix[1, 1], 1]) * real_depth_map[y, x]
            coord_world_frame = np.dot(extrinsic_matrix, np.concatenate((coord_cam_frame, [1])))
            points.append(coord_world_frame)

    return points


def sample_half_hemisphere(num_samples):
    radius = np.random.normal(0.85, 0.2, num_samples)
    hemisphere_center = np.array([-0.31558805, -0.04495631,  1.271939112])
    theta = np.random.uniform(np.pi/4, np.pi/2.2, num_samples)
    phi = np.random.uniform(-np.pi*3.7/4, np.pi*3.7/4, num_samples)
    positions = np.zeros((num_samples, 3))
    positions[:, 0] = radius * np.sin(theta) * np.cos(phi)
    positions[:, 1] = radius * np.sin(theta) * np.sin(phi)
    positions[:, 2] = radius * np.cos(theta)

    backward_directions = positions - hemisphere_center
    backward_directions /= np.linalg.norm(backward_directions, axis=1, keepdims=True)
    right_directions = np.cross(np.tile(np.array([0, 0, 1]), (num_samples, 1)), backward_directions)
    right_directions /= np.linalg.norm(right_directions, axis=1, keepdims=True)
    up_directions = np.cross(backward_directions, right_directions)
    up_directions /= np.linalg.norm(up_directions, axis=1, keepdims=True)

    rotations = np.array([np.column_stack((right, down, forward)) 
                          for right, down, forward in zip(right_directions, up_directions, backward_directions)])
    quaternions = []
    for rotation_matrix in rotations:
        rotation = Rotation.from_matrix(rotation_matrix)
        quaternion = rotation.as_quat()
        quaternions.append(quaternion)
    quaternions = np.array(quaternions)
    return positions, quaternions

def sample_robot_ee_pose():
    pos = np.random.uniform(-0.25, 0.25, 3)
    pos[2] = np.random.uniform(0.6, 1.3)
    
    def sample_rotation_matrix():
        theta = np.random.normal(loc=np.pi, scale=np.pi/3.5)
        phi = np.random.uniform(0, 2*np.pi)
        z_axis = np.array([np.sin(theta) * np.cos(phi),
                           np.sin(theta) * np.sin(phi),
                           np.cos(theta)])
        rightward = np.random.uniform(-1, 1, size=3)
        rightward -= np.dot(rightward, z_axis) * z_axis
        rightward /= np.linalg.norm(rightward)
        inward = np.cross(rightward, z_axis)
        R_mat = np.column_stack((inward, rightward, z_axis))
        return R_mat

    quat = T.mat2quat(sample_rotation_matrix())
    return np.concatenate((pos, quat))


def compute_pose_error(current_pose, target_pose):
    error = min(np.linalg.norm(current_pose - target_pose),
                np.linalg.norm(current_pose - np.concatenate((target_pose[:3], -target_pose[3:]))))
    return error
            
def change_brightness(img, value=30, mask=None):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    if mask is None:
        mask = np.ones_like(v)
    else:
        mask = mask.squeeze()
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

class Data:
    obs = {}
    robot_pose = np.zeros(7)
    camera_pose = np.zeros(7)
    fov = 0
    done = False
    success = False
    message = ""        
    gripper_open = True
    pose_index = 0


class CameraWrapper:
    def __init__(self, env, camera_name="agentview"):
        self.env = env
        self.camera_mover = CameraMover(env=env, camera=camera_name)
        self.cam_tree = ET.Element("camera", attrib={"name": camera_name})
        CAMERA_NAME = self.cam_tree.get("name")
        self.camera_id = env.sim.model.camera_name2id(CAMERA_NAME)
        self.env.viewer.set_camera(camera_id=self.camera_id)
        
        initial_file_camera_pos, initial_file_camera_quat = self.camera_mover.get_camera_pose()
        initial_file_camera_pose = T.make_pose(initial_file_camera_pos, T.quat2mat(initial_file_camera_quat))
        initial_world_camera_pos, initial_world_camera_quat = self.camera_mover.get_camera_pose()
        initial_world_camera_pose = T.make_pose(initial_world_camera_pos, T.quat2mat(initial_world_camera_quat))
        self.world_in_file = initial_file_camera_pose.dot(T.pose_inv(initial_world_camera_pose))
        
    def get_camera_extrinsic(self):
        pos = self.env.sim.data.cam_xpos[self.camera_id]
        xmat = self.env.sim.data.cam_xmat[self.camera_id]
        R_cam2world = xmat.reshape(3,3).T
        E_cam2world = np.eye(4)
        E_cam2world[:3,:3] = R_cam2world
        E_cam2world[:3, 3] = pos
        return E_cam2world

    def set_camera_fov(self, fov=45.0):
        self.env.sim.model.cam_fovy[self.camera_id] = float(fov)
    
    def set_camera_pose(self, pos, quat, offset=np.array([0, 0, 0])):
        self.camera_mover.set_camera_pose(pos=pos + offset, quat=quat)
        target_pose = np.concatenate((pos + offset, quat))
        current_pose = self.get_camera_pose_world_frame()
        error = compute_pose_error(current_pose, target_pose)

    def set_camera_ball_params(self, lookat, distance, azimuth, elevation):
        pos, quat = ball_to_cam_pose(lookat, distance, azimuth, elevation)
        self.set_camera_pose(pos, quat)
            
    
    def get_camera_pose_world_frame(self):
        camera_pos, camera_quat = self.camera_mover.get_camera_pose()
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
        self.camera_mover.rotate_camera(point=None, axis=np.random.uniform(-1, 1, 3), angle=angle)
        self.camera_mover.move_camera(direction=np.random.uniform(-1, 1, 3), scale=scale)
        

class RobotCameraWrapper:
    def __init__(self, robotname="Panda", grippername="PandaGripper", camera_height=256, camera_width=256):
        options = {}
        self.env = suite.make(
            **options,
            robots=robotname,
            gripper_types=grippername,
            env_name="Empty",
            has_renderer=True,
            has_offscreen_renderer=True,
            ignore_done=True,
            use_camera_obs=True,
            controller_configs = suite.load_controller_config(default_controller="OSC_POSE"),
            control_freq=20,
            renderer="mujoco",
            camera_names = ["agentview"],
            camera_heights = camera_height,
            camera_widths = camera_width,
            camera_depths = True,
            camera_segmentations = "robot_only",
            hard_reset=False,
        )
        self.env.reset()
        self.camera_wrapper = CameraWrapper(self.env)
        
        print(robotname)
        
        if robotname == "Panda":
            self.some_safe_joint_angles = [-6.102706193923950195e-01, -1.455744981765747070e+00, 1.501405358314514160e+00, -2.240571022033691406e+00, -2.229462265968322754e-01, 2.963621616363525391e+00, -5.898305177688598633e-01]
            #self.some_safe_joint_angles = [0.0, -0.785398, 0.0, -2.35619, 0.0, 1.5708, 0.785398]
            #self.some_safe_joint_angles = [0.0, -1.785398, 0.7, 2.35619, 2.0, -1.5708, -0.785398]
        elif robotname == "UR5e":
            self.some_safe_joint_angles = [-3.3, -1.207356,  2.514808, -2.433074, -1.849945,  4.024987]
            #self.some_safe_joint_angles = [-1.595451831817626953e-01, 4.841250777244567871e-01, -2.058936595916748047e+00, 6.007540225982666016e-01, 1.412800908088684082e+00, -3.485666529741138220e-04]
        elif robotname == "Sawyer":
            self.some_safe_joint_angles = [0.2553876936435699463, -0.03010351583361625671, -0.9372422099113464355, 1.432788133621215820, 1.421311497688293457, 0.9351797103881835938, -3.228875875473022461]
        elif robotname == "IIWA":
            self.some_safe_joint_angles = [0.0, -0.6, 0.0, 1.2, 0.0, 1.0, 0.0]
        elif robotname == "Jaco":
            self.some_safe_joint_angles = [0.0, -1.2, 1.5, 0.0, 0.0, 0.0, 0.5]

        self.robot_base_name = "robot0_base"
        self.base_body_id = self.env.sim.model.body_name2id(self.robot_base_name)
        self.base_position = self.env.sim.model.body_pos[self.base_body_id].copy()
        self.robot_name = robotname
        
        print(f"[DEBUG] 机器人基座 '{self.robot_base_name}' 的世界坐标: {self.base_position}")
        '''
        intrinsic_matrix = camera_utils.get_camera_intrinsic_matrix(
            self.env.sim,
            camera_name="agentview",
            camera_height=240,
            camera_width=320
        )
        print("INTRINSIC MATRIX:", intrinsic_matrix)
        '''
    
    
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

    def drive_robot_to_target_pose(self, target_pose=None, tracking_error_threshold=0.003, num_iter_max=100):
        self.set_robot_joint_positions()
        self.env.robots[0].controller.use_delta = False
        assert len(target_pose) == 7, "Target pose should be 7DOF"
        current_pose = self.compute_eef_pose()
        error = compute_pose_error(current_pose, target_pose)
        num_iters = 0   
        no_improve_steps = 0
        while error > tracking_error_threshold and num_iters < num_iter_max:
            action = np.zeros(7)
            action[:3] = target_pose[:3]
            action[3:6] = T.quat2axisangle(target_pose[3:])
            if no_improve_steps > 10:
                if self.robot_name == 'UR5e':
                    self.some_safe_joint_angles += np.random.normal(0, 0.1, (6))
                else:
                    self.some_safe_joint_angles += np.random.normal(0, 0.1, (7))
                self.set_robot_joint_positions()
                for _ in range(200):
                    self.env.sim.forward()
                    self.env.sim.step()
                no_improve_steps = 0
            obs, _, _, _ = self.env.step(action)
            current_pose = self.compute_eef_pose()
            current_joints = self.env.sim.data.qpos[self.env.robots[0]._ref_joint_pos_indexes].copy()
            self.some_safe_joint_angles = current_joints
            new_error = compute_pose_error(current_pose, target_pose)
            if abs(new_error - error) < 1e-5:
                no_improve_steps += 1
            else:
                no_improve_steps = 0
            error = new_error
            num_iters += 1

        print("ERROR", error)
        print("NUM_ITERS", num_iters)
        current_pose = self.compute_eef_pose()
        try:
            assert error < tracking_error_threshold, "Starting states are not the same\n"
            return True, current_pose
        except:
            return False, current_pose


    def set_robot_joint_positions(self, joint_angles=None):
        if joint_angles is None:
            joint_angles = self.some_safe_joint_angles
        for _ in range(200):
            self.env.robots[0].set_robot_joint_positions(joint_angles)
            self.env.sim.forward()
            self.env.sim.step()
            self.env._update_observables()
    
    def open_close_gripper(self, gripper_open=True):
        self.env.robots[0].controller.use_delta = True
        action = np.zeros(7)
        if not gripper_open:
            action[-1] = 1
        else:
            action[-1] = -1
        for _ in range(10):            
            obs, _, _, _ = self.env.step(action)
    
    def update_camera(self):
        for _ in range(50):
            self.env.sim.forward()
            self.env.sim.step()
            self.env._update_observables()
          
    def get_observation(self, white_background=True):
        view = "agentview"
        obs = self.env._get_observations()
        rgb_img_raw = obs[f'{view}_image']
        seg_img = obs[f'{view}_segmentation_robot_only']
        num_robot_pixels = np.sum(seg_img)
        if num_robot_pixels <= 700:
            print(num_robot_pixels, " robot pixels in the image")
        if white_background:
            mask = (np.repeat(seg_img, 3, axis=2)).astype(bool)
            rgb_img = np.where(~mask, [255, 255, 255], rgb_img_raw)
            rgb_img = rgb_img.astype(np.uint8)
        else:
            rgb_img = (rgb_img_raw * seg_img).astype(np.uint8)
        return rgb_img, seg_img


class SourceEnvWrapper:
    def __init__(self, source_name, source_gripper, camera_height=256, camera_width=256, connection=None, port=50007, verbose=False):
        self.source_env = RobotCameraWrapper(robotname=source_name, grippername=source_gripper, camera_height=camera_height, camera_width=camera_width)
        self.source_name = source_name
        self.fixed_cam_positions = None
        self.fixed_cam_quaternions = None
        self.verbose = verbose
        # 用于记录标记的状态
        self.marked_pose = None

        self.camera_pose_history = load_camera_pose_history()
        if len(self.camera_pose_history) == 0:
            print("[INFO] 没有历史文件或内容为空，使用初始默认相机姿态。")
            default_pose = [1.25, -0.05, 0.34, 70.0, 0.0, 88.0, 22.0]
            self.camera_pose_history.append(default_pose)
            append_camera_pose_to_history(*default_pose)
        else:
            print(f"[INFO] 已加载历史相机姿态，共 {len(self.camera_pose_history)} 条。当前使用最后一条为初始。")
        
        self.current_camera_pose = self.camera_pose_history[-1]
        self.apply_camera_pose()
    
    def _receive_all_bytes(self, num_bytes: int) -> bytes:
        data = bytearray(num_bytes)
        pos = 0
        while pos < num_bytes:
            cr = self.conn.recv_into(memoryview(data)[pos:])
            if cr == 0:
                raise EOFError
            pos += cr
        return data

    def apply_camera_pose(self):
        x, y, z, roll, pitch, yaw, fov = self.current_camera_pose
        from scipy.spatial.transform import Rotation as R
        r = R.from_euler('xyz', [roll, pitch, yaw], degrees=True)
        quat = r.as_quat()
        pos = np.array([x, y, z]) + np.array([-0.6, 0.0, 0.912])
        self.source_env.camera_wrapper.set_camera_pose(pos, quat)
        self.source_env.camera_wrapper.set_camera_fov(fov)
        for _ in range(10):
            self.source_env.env.sim.forward()
            self.source_env.env.sim.step()
            self.source_env.env._update_observables()

    def _load_dataset_info(self, dataset_name):
        from dataset_poses_dict import ROBOT_CAMERA_POSES_DICT
        info = ROBOT_CAMERA_POSES_DICT[dataset_name]
        return info
    
    def _load_dataset_files(self, info, dataset_name):
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
            if dataset_name == "asu_table_top_rlds":
                joint_angles[:, 1] -= np.pi / 2
                joint_angles[:, 2] *= -1
                joint_angles[:, 3] -= np.pi / 2
        if "robot_ee_states_path" in info:
            ee_states_path = info["robot_ee_states_path"]
            ee_states = np.loadtxt(ee_states_path)
        gripper_states_path = info["gripper_states_path"]
        gripper_states = np.loadtxt(gripper_states_path)
        return joint_angles, ee_states, gripper_states
    
    def _parse_user_command(self, x, y, z, roll, pitch, yaw, fov):
        user_in = input("Modification: ")
        stripped = user_in.strip()
        # 如果输入 m 则标记当前状态
        if stripped == 'm':
            self.marked_pose = [x, y, z, roll, pitch, yaw, fov]
            print(f"[INFO] 标记当前状态: x={x}, y={y}, z={z}, roll={roll}, pitch={pitch}, yaw={yaw}, fov={fov}")
            return x, y, z, roll, pitch, yaw, fov
        # 如果输入 b 则回退到上一个标记的状态
        if stripped == 'b':
            if self.marked_pose is not None:
                x, y, z, roll, pitch, yaw, fov = self.marked_pose
                print(f"[INFO] 回退到标记状态: x={x}, y={y}, z={z}, roll={roll}, pitch={pitch}, yaw={yaw}, fov={fov}")
                append_camera_pose_to_history(x, y, z, roll, pitch, yaw, fov)
                self.camera_pose_history.append([x, y, z, roll, pitch, yaw, fov])
                self.current_camera_pose = [x, y, z, roll, pitch, yaw, fov]
                return x, y, z, roll, pitch, yaw, fov
            else:
                print("[WARN] 未标记状态，无法回退。")
                return x, y, z, roll, pitch, yaw, fov
        # 如果输入纯数字则回退 n 步
        if user_in.isdigit():
            n = int(user_in)
            if n <= 0:
                print("[WARN] 回退步数应当 > 0")
                return x, y, z, roll, pitch, yaw, fov
            current_len = len(self.camera_pose_history)
            if n >= current_len:
                print(f"[WARN] 历史只有 {current_len} 条，无法回退 {n} 步，自动回到最早一条")
                chosen = self.camera_pose_history[0]
            else:
                chosen = self.camera_pose_history[-(n+1)]
            x, y, z, roll, pitch, yaw, fov = chosen
            print(f"[INFO] 回退到 {n} 步前 => x={x}, y={y}, z={z}, roll={roll}, pitch={pitch}, yaw={yaw}, fov={fov}")
            append_camera_pose_to_history(x, y, z, roll, pitch, yaw, fov)
            self.camera_pose_history.append([x, y, z, roll, pitch, yaw, fov])
            self.current_camera_pose = [x, y, z, roll, pitch, yaw, fov]
            return x, y, z, roll, pitch, yaw, fov

        if stripped == 'e':
            print(f"当前: x={x}, y={y}, z={z}, roll={roll}, pitch={pitch}, yaw={yaw}, fov={fov}")
            return x, y, z, roll, pitch, yaw, fov

        try:
            command, number = user_in.split()
            number = float(number)
            if command == 'x':
                x = number
            elif command == 'y':
                y = number
            elif command == 'z':
                z = number
            elif command == 'roll':
                roll = number
            elif command == 'pitch':
                pitch = number
            elif command == 'yaw':
                yaw = number
            elif command == 'fov':
                fov = number
            else:
                print("[WARNING] Unrecognized input, ignoring.")
        except:
            print("[WARNING] 输入格式错误，请重新输入。")

        print(f"Now => x={x}, y={y}, z={z}, roll={roll}, pitch={pitch}, yaw={yaw}, fov={fov}")
        append_camera_pose_to_history(x, y, z, roll, pitch, yaw, fov)
        self.camera_pose_history.append([x, y, z, roll, pitch, yaw, fov])
        self.current_camera_pose = [x, y, z, roll, pitch, yaw, fov]
        return x, y, z, roll, pitch, yaw, fov
    

    def _drive_robot_to_pose_if_needed(self, pose_index,
                                       reference_joint_angles, reference_ee_states, gripper_states,
                                       robot_dataset):
        if reference_ee_states is not None and reference_joint_angles is None:
            target_pose = reference_ee_states[pose_index]
            reached, actual_pose = self.source_env.drive_robot_to_target_pose(target_pose=target_pose)
        elif reference_joint_angles is not None:
            joint_angle = reference_joint_angles[pose_index]
            self.source_env.teleport_to_joint_positions(joint_angle)
            reached, actual_pose = self.source_env.drive_robot_to_target_pose(self.source_env.compute_eef_pose())
        else:
            random_pose = sample_robot_ee_pose()
            reached, actual_pose = self.source_env.drive_robot_to_target_pose(target_pose=random_pose)
        
        if robot_dataset == "kaist":
            gripper_open = False
        else:
            gripper_open = gripper_convert(gripper_states[pose_index], robot_dataset)
        self.source_env.open_close_gripper(gripper_open=gripper_open)
        return reached, actual_pose

    def generate_image(self, num_robot_poses=5, num_cam_poses_per_robot_pose=10, save_paired_images_folder_path="paired_images", reference_joint_angles_path=None, reference_ee_states_path=None, reference_gripper_states_path=None, robot_dataset=None, start_id=0):
        x, y, z, roll, pitch, yaw, fov = self.current_camera_pose
        info = self._load_dataset_info(robot_dataset)
        joint_angles, ee_states, gripper_states = self._load_dataset_files(info, robot_dataset)
        num_frames = joint_angles.shape[0]
        idxs = [
            0,
            num_frames * 1 // 5,
            num_frames * 2 // 5,
            num_frames * 3 // 5,
            num_frames * 4 // 5,
            num_frames - 1
        ]
        print("选取的4个帧下标:", idxs)

        while True:
            x, y, z, roll, pitch, yaw, fov = self._parse_user_command(x, y, z, roll, pitch, yaw, fov)
            camera_reference_position = np.array([x, y, z]) + np.array([-0.6, 0.0, 0.912])
            roll_deg = roll
            pitch_deg = pitch
            yaw_deg = yaw
            r = R.from_euler('xyz', [roll_deg, pitch_deg, yaw_deg], degrees=True)
            camera_reference_quaternion = r.as_quat()
            camera_reference_pose = np.concatenate((camera_reference_position, camera_reference_quaternion))   
            print("camera_reference_pose:", camera_reference_pose)         
            counter = 0
            blended_images = []
            for idx in idxs:
                os.makedirs(os.path.join(save_paired_images_folder_path, f"{self.source_name.lower()}_rgb", str(idx)), exist_ok=True)
                os.makedirs(os.path.join(save_paired_images_folder_path, f"{self.source_name.lower()}_rgb_brightness_augmented", str(idx)), exist_ok=True)
                os.makedirs(os.path.join(save_paired_images_folder_path, f"{self.source_name.lower()}_mask", str(idx)), exist_ok=True)
                reached = False
                while not reached:
                    reached, actual_pose = self._drive_robot_to_pose_if_needed(
                        idx,
                        joint_angles, ee_states, gripper_states,
                        robot_dataset
                    )
                    if not reached:
                        print("Failed to reach pose", idx)
                    else:
                        print("Reached pose:", actual_pose)

                if camera_reference_pose is not None:
                    ref_cam_position, ref_cam_quaternion = camera_reference_pose[:3], camera_reference_pose[3:]
                    cam_positions = [ref_cam_position] * num_cam_poses_per_robot_pose
                    cam_quaternions = [ref_cam_quaternion] * num_cam_poses_per_robot_pose
                for i, (pos, quat) in enumerate(zip(cam_positions, cam_quaternions)):
                    self.source_env.camera_wrapper.set_camera_pose(pos=pos, quat=quat)
                    camera_pose = self.source_env.camera_wrapper.get_camera_pose_world_frame()
                    print("camera_pose", camera_pose)
                    self.source_env.camera_wrapper.set_camera_fov(fov=fov)
                    self.source_env.update_camera()
                    
                    source_robot_img, source_robot_seg_img = self.source_env.get_observation(white_background=True)
                    source_robot_img_brightness_augmented = change_brightness(source_robot_img, value=np.random.randint(-40, 40), mask=source_robot_seg_img)
                    print(f"\033[32m[NOTICE] 已完成第{idx}个pose的生成\033[0m")
                    cv2.imwrite(os.path.join(save_paired_images_folder_path, f"{self.source_name.lower()}_rgb", f"{idx}/{counter}.jpg"), cv2.cvtColor(source_robot_img, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(os.path.join(save_paired_images_folder_path, f"{self.source_name.lower()}_rgb_brightness_augmented", f"{idx}/{counter}.jpg"), cv2.cvtColor(source_robot_img_brightness_augmented, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(os.path.join(save_paired_images_folder_path, f"{self.source_name.lower()}_mask", f"{idx}/{counter}.jpg"), source_robot_seg_img * 255)
                    
                    img2 = Image.open(os.path.join(save_paired_images_folder_path, f"{self.source_name.lower()}_rgb", f"{idx}/{counter}.jpg")).convert("RGBA")
                    img1 = Image.open(f"./datasets/states/{robot_dataset}/episode_0/images/{idx}.jpeg").convert("RGBA")

                    if img1.size != img2.size:
                        img2 = img2.resize(img1.size)
                    blended = Image.blend(img1, img2, alpha=0.5)
                    blended_images.append(blended.convert("RGB"))
                    output_path = os.path.join(save_paired_images_folder_path, f"{idx}.jpg")
                    blended.convert("RGB").save(output_path, "JPEG")

            fig, axes = plt.subplots(2, 3, figsize=(8, 8))
            fig.suptitle("4帧合成对比", fontsize=16)

            for i, ax in enumerate(axes.flat):
                if i < len(blended_images) and blended_images[i] is not None:
                    ax.imshow(blended_images[i])
                    ax.set_title(f"Frame idx={idxs[i]}")
                else:
                    ax.text(0.5, 0.5, "No image", ha='center', va='center')
                ax.axis("off")

            save_path = os.path.join(save_paired_images_folder_path, f"{self.source_name.lower()}_4frames_blend.jpg")
            plt.savefig(save_path, bbox_inches='tight')
            plt.show()

            print(f"4帧合成结果已保存到 {save_path}")
            print(f"[INFO] x={x}, y={y}, z={z}, roll={roll}, pitch={pitch}, yaw={yaw}, fov={fov}")
            

if __name__ == "__main__":

    """
    Registered environments: Lift, Stack, NutAssembly, NutAssemblySingle, NutAssemblySquare, NutAssemblyRound,
                             PickPlace, PickPlaceSingle, PickPlaceMilk, PickPlaceBread, PickPlaceCereal,
                             PickPlaceCan, Door, Wipe, TwoArmLift, TwoArmPegInHole, TwoArmHandover

    Possible robots: Baxter, IIWA, Jaco, Kinova3, Panda, Sawyer, UR5e
    Possible grippers: 'RethinkGripper', 'PandaGripper', 'JacoThreeFingerGripper', 'JacoThreeFingerDexterousGripper', 
    'WipingGripper', 'Robotiq85Gripper', 'Robotiq140Gripper', 'RobotiqThreeFingerGripper', 'RobotiqThreeFingerDexterousGripper'
    """

    print("Welcome to robosuite v{}!".format(suite.__version__))
    print(suite.__logo__)

    parser = argparse.ArgumentParser()
    parser.add_argument("--connection", action='store_true', help="if True, the source robot will wait for the target robot to connect to it")
    parser.add_argument("--port", type=int, default=50007, help="(optional) port for socket connection")
    parser.add_argument("--seed", type=int, default=0, help="(optional) set seed")
    parser.add_argument("--source_gripper", type=str, default="PandaGripper", help="PandaGripper or Robotiq85Gripper")
    parser.add_argument("--num_robot_poses", type=int, default=5, help="(optional) number of robot poses to sample")
    parser.add_argument("--num_cam_poses_per_robot_pose", type=int, default=5, help="(optional) number of camera poses per robot pose to sample")
    parser.add_argument("--save_paired_images_folder_path", type=str, default="paired_images", help="(optional) folder path to save the paired images")
    parser.add_argument("--robot_dataset", type=str, help="(optional) to match the robot poses from a dataset, provide the dataset name")
    parser.add_argument("--reference_joint_angles_path", type=str, help="(optional) to match the robot poses from a dataset, provide the path to the joint angles file (np.savetxt)")
    parser.add_argument("--reference_ee_states_path", type=str, help="(optional) to match the robot poses from a dataset, provide the path to the ee state file (np.savetxt)")
    parser.add_argument("--reference_gripper_states_path", type=str, help="(optional) to match the gripper's open/close status")
    parser.add_argument("--start_id", type=int, default=0, help="(optional) starting index of the robot poses")
    parser.add_argument("--verbose", action='store_true', help="If set, prints extra debug/warning information")
    args = parser.parse_args()

    if args.robot_dataset == "autolab_ur5" or args.robot_dataset == "asu_table_top_rlds":
        source_name = "UR5e"
        source_gripper = "Robotiq85Gripper"
    else:
        source_name = "Panda"
        source_gripper = "PandaGripper"
    
    save_paired_images_folder_path = args.save_paired_images_folder_path
    os.makedirs(os.path.join(save_paired_images_folder_path, f"{source_name.lower()}_rgb"), exist_ok=True)
    os.makedirs(os.path.join(save_paired_images_folder_path, f"{source_name.lower()}_rgb_brightness_augmented"), exist_ok=True)
    os.makedirs(os.path.join(save_paired_images_folder_path, f"{source_name.lower()}_mask"), exist_ok=True)
    
    if args.robot_dataset is not None:
        from dataset_poses_dict import ROBOT_CAMERA_POSES_DICT
        robot_dataset_info = ROBOT_CAMERA_POSES_DICT[args.robot_dataset]
        camera_height = robot_dataset_info["camera_heights"]
        camera_width = robot_dataset_info["camera_widths"]
    else:
        camera_height = 256
        camera_width = 256
    
    source_env = SourceEnvWrapper(source_name, source_gripper, camera_height, camera_width, connection=args.connection, port=args.port, verbose=args.verbose)
    source_env.generate_image(num_robot_poses=args.num_robot_poses, num_cam_poses_per_robot_pose=args.num_cam_poses_per_robot_pose, save_paired_images_folder_path=save_paired_images_folder_path, reference_joint_angles_path=args.reference_joint_angles_path, reference_ee_states_path=args.reference_ee_states_path, reference_gripper_states_path=args.reference_gripper_states_path, robot_dataset=args.robot_dataset, start_id=args.start_id)

    source_env.source_env.env.close_renderer()