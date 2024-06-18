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
    hemisphere_center = np.array([0, 0, 0])
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


def compute_pose_error(current_pose, target_pose):
    # quarternions are equivalent up to sign
    error = min(np.linalg.norm(current_pose - target_pose), np.linalg.norm(current_pose - np.concatenate((target_pose[:3], -target_pose[3:]))))
    return error
            

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

class Data:
    obs = {}
    robot_pose = np.zeros(7)
    camera_pose = np.zeros(7)
    fov = 0
    done = False
    success = False
    message = ""        
    gripper_open = True


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
        

class RobotCameraWrapper:
    def __init__(self, robotname="Panda", grippername="PandaGripper", camera_height=256, camera_width=256):
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
        self.env.reset()
        
        self.camera_wrapper = CameraWrapper(self.env)
        
        if robotname == "Panda":
            self.default_joint_angles = [9.44962915e-03,  2.04028892e-01,  2.27688289e-02, -2.64987059e+00, 1.89505922e-03,  2.91240765e+00,  7.87470020e-01]
        elif robotname == "UR5e":
            self.default_joint_angles = [-0.46205679, -1.76595556, 2.47328085, -2.24068841, -1.59642081, -1.99376873]
        elif robotname == "Sawyer":
            self.default_joint_angles = [-0.02366884, -1.19710107, -0.02385198, 2.18294157, 0.01508528, 0.55936629, -1.58903075]
        elif robotname == "Jaco":
            self.default_joint_angles = [3.18165494, 3.64849233, 0.02417776, 1.16953827, 0.05959387, 3.74331165, 3.13904329]
    
    
    def compute_eef_pose(self):
        pos = np.array(self.env.sim.data.site_xpos[self.env.sim.model.site_name2id(self.env.robots[0].controller.eef_name)])
        rot = np.array(T.mat2quat(self.env.sim.data.site_xmat[self.env.sim.model.site_name2id(self.env.robots[0].controller.eef_name)].reshape([3, 3])))
        return np.concatenate((pos, rot))

    def drive_robot_to_target_pose(self, target_pose=None, tracking_error_threshold=0.02, num_iter_max=100):
        # breakpoint()
        # reset robot joint positions so the robot is hopefully not in a weird pose
        self.set_robot_joint_positions()
        self.env.robots[0].controller.use_delta = False # change to absolute pose for setting the initial state
        
        assert len(target_pose) == 7, "Target pose should be 7DOF"
        current_pose = self.compute_eef_pose()
        error = compute_pose_error(current_pose, target_pose)
        num_iters = 0    
        while error > tracking_error_threshold and num_iters < num_iter_max:
            action = np.zeros(7)
            action[:3] = target_pose[:3]
            action[3:6] = T.quat2axisangle(target_pose[3:])
            
            obs, _, _, _ = self.env.step(action)

            current_pose = self.compute_eef_pose()
            error = compute_pose_error(current_pose, target_pose)
            num_iters += 1

        # print("Take {} iterations to drive robot to target pose".format(num_iters))
        try:
            assert error < tracking_error_threshold, "Starting states are not the same\n"
            return True, current_pose
        except:
            # print("Starting states are not the same\n"
            #         "Source: ", current_pose,
            #         "Target: ", target_pose
            #         )
            return False, current_pose

    def set_robot_joint_positions(self, joint_angles=None):
        if joint_angles is None:
            joint_angles = self.default_joint_angles
        for _ in range(200):
            self.env.robots[0].set_robot_joint_positions(joint_angles)
            self.env.sim.forward()
            self.env.sim.step()
            self.env._update_observables()
    
    def open_close_gripper(self, gripper_open=True):
        self.env.robots[0].controller.use_delta = True # change to delta pose
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
        # count the number of robot pixels
        num_robot_pixels = np.sum(seg_img)
        if num_robot_pixels <= 700:
            print(num_robot_pixels, " robot pixels in the image")
            return None, None
        
        # Create a mask where non-robot pixels are set to 0 and robot pixels are set to 1
        if white_background:
            mask = (np.repeat(seg_img, 3, axis=2)).astype(bool)
            rgb_img = np.where(~mask, [255, 255, 255], rgb_img_raw) # Set non-robot pixels to white
            rgb_img = rgb_img.astype(np.uint8) # Convert the resulting image to uint8 type
        else:
            # only retain the robot part in the rgb_img, set other pixels to black
            rgb_img = (rgb_img * seg_img).astype(np.uint8)
        return rgb_img, seg_img


class SourceEnvWrapper:
    def __init__(self, source_name, source_gripper, camera_height=256, camera_width=256, connection=None, connection_num=1, port=50007):
        self.source_env = RobotCameraWrapper(robotname=source_name, grippername=source_gripper, camera_height=camera_height, camera_width=camera_width)
        self.source_name = source_name
        self.connection_num = connection_num
        if connection:
            HOST = 'localhost'
            PORT = port
            self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.s.bind((HOST, PORT))
            self.s.listen(self.connection_num)
            self.conns = []
            for _ in range(self.connection_num):
                conn, addr = self.s.accept()
                self.conns.append(conn)
            print('Connected by', addr)
        else:
            self.s = None
            self.conns = []
    
    def _receive_all_bytes(self, conn, num_bytes: int) -> bytes:
        """
        Receives all the bytes.
        :param num_bytes: The number of bytes.
        :return: The bytes.
        """
        data = bytearray(num_bytes)
        pos = 0
        while pos < num_bytes:
            cr = conn.recv_into(memoryview(data)[pos:])
            if cr == 0:
                raise EOFError
            pos += cr
        return data

    def generate_image(self, num_robot_poses=5, num_cam_poses_per_robot_pose=10, save_paired_images_folder_path="paired_images", reference_joint_angles_path=None, reference_ee_states_path=None, robot_dataset=None, use_cam_pose_only=False, start_id=0):
        # read desired joint angles
        if reference_ee_states_path is not None:
            ee_states = np.loadtxt(reference_ee_states_path)
            num_robot_poses = ee_states.shape[0]
            # viola
            camera_reference_pose = np.array([0.5 , 0.04  , 1.37, 0.27104094, 0.27104094, 0.65309786, 0.65309786])
            fov_range = (45, 60)
            
        if reference_joint_angles_path is not None:
            joint_angles = np.loadtxt(reference_joint_angles_path)
            num_robot_poses = joint_angles.shape[0]
            # mirage
            camera_pos_mirage = np.array([0.68,0.37,0.47]) + np.array([-0.6, 0.0, 0.912])
            camera_rot_mirage = np.array([[-0.87844054,  0.32722496, -0.34823273],
                                        [ 0.47077211,  0.46765169, -0.74811464],
                                        [-0.08195015, -0.82111249, -0.56485259]])
            # robosuite camera is not right, down, forward but right, up, backward
            camera_rot_mirage[:, 1] = -camera_rot_mirage[:, 1]
            camera_rot_mirage[:, 2] = -camera_rot_mirage[:, 2]
            camera_quat_mirage = T.mat2quat(camera_rot_mirage)
            camera_reference_pose = np.concatenate((camera_pos_mirage, camera_quat_mirage))
            fov_range = (55, 85)
            
        if robot_dataset is not None:
            from dataset_poses_dict import ROBOT_CAMERA_POSES_DICT
            robot_dataset_info = ROBOT_CAMERA_POSES_DICT[robot_dataset]
            reference_joint_angles_path = robot_dataset_info["robot_joint_angles_path"]
            reference_ee_states_path = robot_dataset_info["robot_ee_states_path"]
            joint_angles = np.loadtxt(reference_joint_angles_path)
            ee_states = np.loadtxt(reference_ee_states_path)
            num_robot_poses = min(joint_angles.shape[0], 10000)
            
            camera_reference_position = robot_dataset_info["camera_position"] + np.array([-0.6, 0.0, 0.912])
            camera_reference_orientation = robot_dataset_info["camera_orientation"]
            # robosuite camera is not right, down, forward but right, up, backward
            camera_reference_orientation[:, 1] = -camera_reference_orientation[:, 1]
            camera_reference_orientation[:, 2] = -camera_reference_orientation[:, 2]
            camera_reference_quaternion = T.mat2quat(camera_reference_orientation)
            camera_reference_pose = np.concatenate((camera_reference_position, camera_reference_quaternion))            
            fov_range = (robot_dataset_info["camera_fov"] - 15, robot_dataset_info["camera_fov"] + 15)
        else:
            camera_reference_pose = None    
        
        for pose_index in range(start_id, min(start_id+3000, num_robot_poses)):
            if pose_index % 30 == 0: # to avoid simulation becoming unstable
                self.source_env.env.reset()
            
            print(pose_index)
            counter = 0
            os.makedirs(os.path.join(save_paired_images_folder_path, f"{self.source_name.lower()}_rgb", str(pose_index)), exist_ok=True)
            os.makedirs(os.path.join(save_paired_images_folder_path, f"{self.source_name.lower()}_rgb_brightness_augmented", str(pose_index)), exist_ok=True)
            os.makedirs(os.path.join(save_paired_images_folder_path, f"{self.source_name.lower()}_mask", str(pose_index)), exist_ok=True)
            
            # sample robot eef pose
            both_reached = False
            num_trial = 0
            while not both_reached:
                if num_trial >= 10 and (reference_joint_angles_path is not None or reference_ee_states_path is not None):
                    break
                target_env_robot_state = Data()
                # sample gripper opening/closing with 35% probability of closing
                gripper_open = np.random.choice([True, False], p=[0.7, 0.3])
                if reference_ee_states_path is not None and reference_joint_angles_path is None and not use_cam_pose_only:
                    ee_state = ee_states[pose_index]
                    target_pos, target_quat = T.mat2pose(ee_state.reshape((4, 4)))
                    target_quat = T.quat_multiply(target_quat, np.array([ 0, 0, -0.7071068, 0.7071068 ]))
                    target_pose = np.concatenate((target_pos, target_quat))
                    source_reached, source_reached_pose = self.source_env.drive_robot_to_target_pose(target_pose=target_pose)
                    target_pose = source_reached_pose # to avoid source not reaching its target pose
                elif reference_joint_angles_path is not None and not use_cam_pose_only:
                    joint_angle = joint_angles[pose_index]
                    # add noise to joint angles
                    joint_angle += np.random.normal(0, 0.05, 7)
                    joint_angle[-1] += np.random.normal(0, 0.3)
                    self.source_env.set_robot_joint_positions(joint_angle)
                    source_reached_pose = self.source_env.compute_eef_pose()
                    source_reached, source_reached_pose = self.source_env.drive_robot_to_target_pose(target_pose=source_reached_pose)
                    target_pose = source_reached_pose
                else: # both are None
                    target_pose = sample_robot_ee_pose()
                    source_reached, source_reached_pose = self.source_env.drive_robot_to_target_pose(target_pose=target_pose, tracking_error_threshold=0.04) # no need to track so accurately for the source robot
                    target_pose = source_reached_pose
                if not source_reached:
                    if reference_joint_angles_path is not None:
                        print("Source robot failed to reach the desired pose")
                        # ideal: jump out of the while loop and directly go to the next pose
                        # the issue is the index on the target robot side will be messed up.
                    else:
                        num_trial += 1
                        continue
                # gripper action
                self.source_env.open_close_gripper(gripper_open=gripper_open)
                
                ########### Send message to target robot ############
                variable = Data()
                variable.robot_pose = target_pose
                variable.gripper_open = gripper_open
                variable.message = "Source reached a target pose. Send target pose"
                # Pickle the object and send it to the server
                data_string = pickle.dumps(variable)
                message_length = struct.pack("!I", len(data_string))
                for conn in self.conns:
                    conn.send(message_length)
                    conn.send(data_string)
                
                ########### Receive message from target robot ############
                target_env_robot_state_all = []
                for j, conn in enumerate(self.conns):
                    pickled_message_size = self._receive_all_bytes(conn, 4)
                    message_size = struct.unpack("!I", pickled_message_size)[0]
                    data = self._receive_all_bytes(conn, message_size)
                    target_env_robot_state = pickle.loads(data)
                    target_env_robot_state_all.append(target_env_robot_state)
                    assert target_env_robot_state.message == "Target robot tries the target pose", "Wrong synchronization"
                
                # check if all robots are successful
                all_success = all([target_env_robot_state.success for target_env_robot_state in target_env_robot_state_all])
                if not all_success:
                    if reference_joint_angles_path is not None:
                        print(robot_dataset, "Target robot failed to reach the desired pose")
                    
                    ########### Send message to target robot ############
                    variable = Data()
                    variable.success = False
                    variable.message = "Communicating results on whether all robots reach the same pose"
                    # Pickle the object and send it to the server
                    data_string = pickle.dumps(variable)
                    message_length = struct.pack("!I", len(data_string))
                    for conn in self.conns:
                        conn.send(message_length)
                        conn.send(data_string)
                    
                    num_trial += 1
                    continue          
                else:
                    both_reached = True
                    
                    ########### Send message to target robot ############
                    variable = Data()
                    variable.success = True
                    variable.message = "Communicating results on whether all robots reach the same pose"
                    # Pickle the object and send it to the server
                    data_string = pickle.dumps(variable)
                    message_length = struct.pack("!I", len(data_string))
                    for conn in self.conns:
                        conn.send(message_length)
                        conn.send(data_string)
                    # print("Source robot pose: ", source_reached_pose)
            
            # if the target robot fails to reach the source robot pose in 1 trial, skip this pose
            if num_trial >= 1 and (reference_joint_angles_path is not None or reference_ee_states_path is not None):
                continue
            
            if camera_reference_pose is not None:
                ref_cam_position, ref_cam_quaternion = camera_reference_pose[:3], camera_reference_pose[3:]
                cam_positions, cam_quaternions = [ref_cam_position] * num_cam_poses_per_robot_pose, [ref_cam_quaternion] * num_cam_poses_per_robot_pose
            else:
                cam_positions, cam_quaternions = sample_half_hemisphere(num_cam_poses_per_robot_pose) # Generate random camera poses
            # Capture images from each camera pose
            for i, (pos, quat) in enumerate(zip(cam_positions, cam_quaternions)):
                if camera_reference_pose is not None:
                    # just set the camera pose to the reference pose with slight perturbation
                    self.source_env.camera_wrapper.set_camera_pose(pos=pos, quat=quat)
                    self.source_env.camera_wrapper.perturb_camera(angle=8, scale=0.1)
                    camera_pose = self.source_env.camera_wrapper.get_camera_pose_world_frame()
                    fov = np.random.uniform(fov_range[0], fov_range[1])
                    # print("Actual camera pose: ", camera_pose)
                else:
                    self.source_env.camera_wrapper.set_camera_pose(pos=pos, quat=quat, offset=target_pose[:3])
                    self.source_env.camera_wrapper.perturb_camera()
                    camera_pose = self.source_env.camera_wrapper.get_camera_pose_world_frame()
                    # print("Desired camera pose: ", pos+target_pose[:3], quat)
                    # print("Actual camera pose: ", camera_pose)
                    # sample an fov
                    fov = np.random.uniform(40, 70)
                self.source_env.camera_wrapper.set_camera_fov(fov=fov)
                self.source_env.update_camera()
                
                source_robot_img, source_robot_seg_img = self.source_env.get_observation(white_background=True)
                
                
                ########### Send message to target robot ############
                variable = Data()
                variable.camera_pose = camera_pose
                variable.fov = fov
                variable.success = (source_robot_img is not None)
                variable.message = "Source robot has captured an image"
                # Pickle the object and send it to the server
                data_string = pickle.dumps(variable)
                message_length = struct.pack("!I", len(data_string))
                for conn in self.conns:
                    conn.send(message_length)
                    conn.send(data_string)
                
                if source_robot_img is None:
                    print("No robot pixels in the image")
                    continue
                
                ########### Receive message from target robot ############
                for j, conn in enumerate(self.conns):
                    pickled_message_size = self._receive_all_bytes(conn, 4)
                    message_size = struct.unpack("!I", pickled_message_size)[0]
                    data = self._receive_all_bytes(conn, message_size)
                    target_env_robot_state = pickle.loads(data)
                    assert target_env_robot_state.message == "Target robot has captured an image", "Wrong synchronization"
                    success = target_env_robot_state.success
                    if not success:
                        continue
                
                # sample a random integer between -40 and 40
                source_robot_img_brightness_augmented = change_brightness(source_robot_img, value=np.random.randint(-40, 40), mask=source_robot_seg_img)
                source_robot_img = cv2.resize(source_robot_img, (256, 256), interpolation=cv2.INTER_LINEAR)
                source_robot_img_brightness_augmented = cv2.resize(source_robot_img_brightness_augmented, (256, 256), interpolation=cv2.INTER_LINEAR)
                source_robot_seg_img = cv2.resize(source_robot_seg_img, (256, 256), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(os.path.join(save_paired_images_folder_path, f"{self.source_name.lower()}_rgb", f"{pose_index}/{counter}.jpg"), cv2.cvtColor(source_robot_img, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(save_paired_images_folder_path, f"{self.source_name.lower()}_rgb_brightness_augmented", f"{pose_index}/{counter}.jpg"), cv2.cvtColor(source_robot_img_brightness_augmented, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(save_paired_images_folder_path, f"{self.source_name.lower()}_mask", f"{pose_index}/{counter}.jpg"), source_robot_seg_img * 255)
                counter += 1

        


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
    print("Welcome to robosuite v{}!".format(suite.__version__))
    print(suite.__logo__)

    parser = argparse.ArgumentParser()
    parser.add_argument("--connection", action='store_true', help="if True, the source robot will wait for the target robot to connect to it")
    parser.add_argument("--connection_num", type=int, default=1, help="(optional) number of target robots")
    parser.add_argument("--port", type=int, default=50007, help="(optional) port for socket connection")
    parser.add_argument("--seed", type=int, default=0, help="(optional) set seed")
    parser.add_argument("--source_robot", type=str, default="Panda", help="Panda or UR5e or Jaco or Sawyer")
    parser.add_argument("--source_gripper", type=str, default="PandaGripper", help="PandaGripper or Robotiq85Gripper")
    parser.add_argument("--num_robot_poses", type=int, default=5, help="(optional) number of robot poses to sample")
    parser.add_argument("--num_cam_poses_per_robot_pose", type=int, default=5, help="(optional) number of camera poses per robot pose to sample")
    parser.add_argument("--save_paired_images_folder_path", type=str, default="paired_images", help="(optional) folder path to save the paired images")
    parser.add_argument("--robot_dataset", type=str, help="(optional) to match the robot poses from a dataset, provide the dataset name")
    parser.add_argument("--use_cam_pose_only", action='store_true', help="if True, only use the camera poses from the reference dataset and not the robot poses")
    parser.add_argument("--reference_joint_angles_path", type=str, help="(optional) to match the robot poses from a dataset, provide the path to the joint angles file (np.savetxt)")
    parser.add_argument("--reference_ee_states_path", type=str, help="(optional) to match the robot poses from a dataset, provide the path to the ee state file (np.savetxt)")
    parser.add_argument("--start_id", type=int, default=0, help="(optional) starting index of the robot poses")
    args = parser.parse_args()
    
    
    source_name = args.source_robot
    source_gripper = args.source_gripper

    # Save the captured images
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
    
    source_env = SourceEnvWrapper(source_name, source_gripper, camera_height, camera_width, connection=args.connection, connection_num=args.connection_num, port=args.port)
    source_env.generate_image(num_robot_poses=args.num_robot_poses, num_cam_poses_per_robot_pose=args.num_cam_poses_per_robot_pose, save_paired_images_folder_path=save_paired_images_folder_path, reference_joint_angles_path=args.reference_joint_angles_path, reference_ee_states_path=args.reference_ee_states_path, robot_dataset=args.robot_dataset, use_cam_pose_only=args.use_cam_pose_only, start_id=args.start_id)

    source_env.source_env.env.close_renderer()
    
        
    # source_env = RobotCameraWrapper(robotname=source_name)
    # target_env = RobotCameraWrapper(robotname=target_name)
    
    # positions, quaternions = sample_half_hemisphere(num_cam_poses_per_robot_pose) # Generate random camera poses
    # counter = 0
    # # for each robot pose
    # for k in range(num_robot_poses):
    #     # sample robot eef pose
    #     both_reached = False
    #     while not both_reached:
    #         target_pose = sample_robot_ee_pose()
    #         source_reached, source_reached_pose = source_env.drive_robot_to_target_pose(target_pose=target_pose)
    #         if not source_reached:
    #             continue
    #         target_reached, target_reached_pose = target_env.drive_robot_to_target_pose(target_pose=target_pose)
    #         if not target_reached:
    #             continue
    #         both_reached = True
    #     print("Source robot pose: ", source_reached_pose)
    #     print("Target robot pose: ", target_reached_pose)
        
        
        
    #     # Capture images from each camera pose
    #     for i, (pos, quat) in enumerate(zip(positions, quaternions)):
    #         source_env.camera_wrapper.set_camera_pose(pos=positions[i], quat=quaternions[i], offset=target_pose[:3])
    #         source_env.camera_wrapper.perturb_camera()
    #         camera_pos, camera_quat = source_env.camera_wrapper.get_camera_pose_world_frame()
    #         target_env.camera_wrapper.set_camera_pose(pos=positions[i], quat=quaternions[i])
            
    #         # sample an fov
    #         fov = np.random.uniform(30, 60)
    #         source_env.camera_wrapper.set_camera_fov(fov=fov)
    #         target_env.camera_wrapper.set_camera_fov(fov=fov)
            
            
    #         source_robot_img = source_env.get_observation(white_background=True)
    #         target_robot_img = target_env.get_observation(white_background=True)
            
    #         if source_robot_img is None or target_robot_img is None:
    #             print("No robot pixels in the image")
    #             continue
    #         # breakpoint()
    #         cv2.imwrite(f"output/{counter}_{source_name}.png", cv2.cvtColor(source_robot_img, cv2.COLOR_RGB2BGR))
    #         cv2.imwrite(f"output/{counter}_{target_name}.png", cv2.cvtColor(target_robot_img, cv2.COLOR_RGB2BGR))
    #         counter += 1

    # source_env.env.close_renderer()
    # target_env.env.close_renderer()
    # print("Done.")
