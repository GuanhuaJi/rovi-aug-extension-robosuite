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

from test_server import Data, RobotCameraWrapper, change_brightness
from robot_pose_dict import ROBOT_POSE_DICT


class TargetEnvWrapper:
    def __init__(self, target_name, target_gripper, robot_dataset, camera_height=256, camera_width=256, connection=None, port=50007):
        self.target_env = RobotCameraWrapper(robotname=target_name, grippername=target_gripper, robot_dataset=robot_dataset, camera_height=camera_height, camera_width=camera_width)
        self.target_name = target_name
        print("TARGET_NAME", target_name)
        if connection:
            HOST = 'localhost'
            PORT = port
            self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.s.connect((HOST, PORT))
        else:
            self.s = None
    
    def _receive_all_bytes(self, num_bytes: int) -> bytes:
        """
        Receives all the bytes.
        :param num_bytes: The number of bytes.
        :return: The bytes.
        """
        data = bytearray(num_bytes)
        pos = 0
        while pos < num_bytes:
            cr = self.s.recv_into(memoryview(data)[pos:])
            if cr == 0:
                raise EOFError
            pos += cr
        return data
    
    def generate_image(self, num_robot_poses=5, num_cam_poses_per_robot_pose=10, save_paired_images_folder_path="paired_images", reference_joint_angles_path=None, reference_ee_states_path=None, robot_dataset=None, start_id=0):
        # read desired joint angles
        if reference_ee_states_path is not None:
            ee_states = np.loadtxt(reference_ee_states_path)
            num_robot_poses = ee_states.shape[0]
        
        if reference_joint_angles_path is not None:
            joint_angles = np.loadtxt(reference_joint_angles_path)
            num_robot_poses = joint_angles.shape[0]
        
        if robot_dataset is not None:
            from dataset_poses_dict import ROBOT_CAMERA_POSES_DICT
            robot_dataset_info = ROBOT_CAMERA_POSES_DICT[robot_dataset]
            reference_joint_angles_path = None
            reference_ee_states_path = None
            joint_angles = None
            ee_states = None
            if "robot_joint_angles_path" in robot_dataset_info:
                reference_joint_angles_path = robot_dataset_info["robot_joint_angles_path"]
                joint_angles = np.loadtxt(reference_joint_angles_path)
            if "robot_ee_states_path" in robot_dataset_info:
                reference_ee_states_path = robot_dataset_info["robot_ee_states_path"]
                ee_states = np.loadtxt(reference_ee_states_path)
            num_robot_poses = min(joint_angles.shape[0], 10000)
        
        for pose_index in range(num_robot_poses):
            if pose_index % 30 == 0: # to avoid simulation becoming unstable
                self.target_env.env.reset()
            counter = 0
            os.makedirs(os.path.join(save_paired_images_folder_path, f"{target_name.lower()}_rgb", str(counter)), exist_ok=True)
            os.makedirs(os.path.join(save_paired_images_folder_path, f"{target_name.lower()}_rgb_brightness_augmented", str(counter)), exist_ok=True)
            os.makedirs(os.path.join(save_paired_images_folder_path, f"{target_name.lower()}_mask", str(counter)), exist_ok=True)
            
            # sample robot eef pose
            both_reached = False
            while not both_reached:
                ########### Receive message from source robot ############
                pickled_message_size = self._receive_all_bytes(4)
                message_size = struct.unpack("!I", pickled_message_size)[0]
                data = self._receive_all_bytes(message_size)
                source_env_robot_state = pickle.loads(data)
                assert source_env_robot_state.message == "Source reached a target pose. Send target pose", "Wrong synchronization"
                target_pose=source_env_robot_state.robot_pose

                target_pose[:3] -= ROBOT_POSE_DICT[robot_dataset][self.target_name]['displacement']
                '''
                if robot_dataset == "viola" and self.target_name == "Jaco":
                    target_pose[:3] -= np.array([0, 0, 0.1])
                elif robot_dataset == "austin_mutex" and self.target_name == "Jaco":
                    target_pose[:3] -= np.array([0, 0, 0.1])
                elif robot_dataset == "nyu_franka" and self.target_name == "Jaco":
                    target_pose[:3] += np.array([-0.1, 0, 0.1])
                '''
                
                self.target_env.open_close_gripper(gripper_open=source_env_robot_state.gripper_open)
                target_reached, target_reached_pose = self.target_env.drive_robot_to_target_pose(target_pose=target_pose)
                source_index = source_env_robot_state.pose_index
                ppose = self.target_env.compute_eef_pose()[:3] + ROBOT_POSE_DICT[robot_dataset][self.target_name]['displacement']
                print("TARGET_REACHED_POSE:", ppose)

                
                ########### Send message to source robot ############
                variable = Data()
                variable.success = target_reached
                variable.message = "Target robot tries the target pose"
                # Pickle the object and send it to the server
                data_string = pickle.dumps(variable)
                message_length = struct.pack("!I", len(data_string))
                self.s.send(message_length)
                self.s.send(data_string)
                    
                if not target_reached:
                    continue
                else:
                    both_reached = True
                    #print("Target robot pose: ", targt_reached_pose)
            
            
            # Capture images from each camera pose
            for i in range(num_cam_poses_per_robot_pose):
                ########### Receive message from source robot ############
                pickled_message_size = self._receive_all_bytes(4)
                message_size = struct.unpack("!I", pickled_message_size)[0]
                data = self._receive_all_bytes(message_size)
                source_env_robot_state = pickle.loads(data)
                assert source_env_robot_state.message == "Source robot has captured an image", "Wrong synchronization"
                fov = source_env_robot_state.fov
                camera_pose = source_env_robot_state.camera_pose

                camera_pose[:3] -= ROBOT_POSE_DICT[robot_dataset][self.target_name]['displacement']
                '''
                if robot_dataset == "viola" and self.target_name == "Jaco":
                    camera_pose[:3] -= np.array([0, 0, 0.1])
                elif robot_dataset == "austin_mutex" and self.target_name == "Jaco":
                    camera_pose[:3] -= np.array([0, 0, 0.1])
                elif robot_dataset == "nyu_franka" and self.target_name == "Jaco":
                    camera_pose[:3] += np.array([-0.1, 0, 0.1])
                '''
                

                success = source_env_robot_state.success
                if not success:
                    continue
                
                # self.target_env.camera_wrapper.set_camera_fov(fov=fov)
                joint_indices = self.target_env.env.robots[0]._ref_joint_pos_indexes
                # Retrieve the joint angles from the simulation’s qpos vector
                current_joint_angles = self.target_env.env.sim.data.qpos[joint_indices]
                print("Current joint angles:", current_joint_angles)
                self.target_env.camera_wrapper.set_camera_pose(pos=camera_pose[:3], quat=camera_pose[3:])
                self.target_env.camera_wrapper.set_camera_fov(fov=fov)
                self.target_env.update_camera()
                    
                # print("Desired camera pose: ", camera_pose)
                # print("Actual camera pose: ", self.target_env.camera_wrapper.get_camera_pose_world_frame())
                target_robot_img, target_robot_seg_img = self.target_env.get_observation(white_background=True)
                
                
                ########### Send message to source robot ############
                variable = Data()
                variable.success = (target_robot_img is not None)
                variable.message = "Target robot has captured an image"
                # Pickle the object and send it to the server
                data_string = pickle.dumps(variable)
                message_length = struct.pack("!I", len(data_string))
                self.s.send(message_length)
                self.s.send(data_string)
                
                if target_robot_img is None:
                    print("No robot pixels in the image")
                    # breakpoint()
                    continue
                
                # sample a random integer between -40 and 40
                target_robot_img_brightness_augmented = change_brightness(target_robot_img, value=np.random.randint(-40, 40), mask=target_robot_seg_img)
                #target_robot_img = cv2.resize(target_robot_img, (256, 256), interpolation=cv2.INTER_LINEAR)
                target_robot_img_brightness_augmented = cv2.resize(target_robot_img_brightness_augmented, (256, 256), interpolation=cv2.INTER_LINEAR)
                #target_robot_img = cv2.resize(target_robot_img, (256, 256), interpolation=cv2.INTER_LINEAR)
                #target_robot_seg_img = cv2.resize(target_robot_seg_img, (256, 256), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(os.path.join(save_paired_images_folder_path, f"{target_name.lower()}_rgb", f"{counter}/{source_index}.jpg"), cv2.cvtColor(target_robot_img, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(save_paired_images_folder_path, f"{target_name.lower()}_rgb_brightness_augmented", f"{counter}/{source_index}.jpg"), cv2.cvtColor(target_robot_img_brightness_augmented, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(save_paired_images_folder_path, f"{target_name.lower()}_mask", f"{counter}/{source_index}.jpg"), target_robot_seg_img * 255)
                counter += 1

        
        

if __name__ == "__main__":

    """
    Registered environments: Lift, Stack, NutAssembly, NutAssemblySingle, NutAssemblySquare, NutAssemblyRound,
                             PickPlace, PickPlaceSingle, PickPlaceMilk, PickPlaceBread, PickPlaceCereal,
                             PickPlaceCan, Door, Wipe, TwoArmLift, TwoArmPegInHole, TwoArmHandover

    Possible robots: Baxter, IIWA, Jaco, Kinova3, Panda, Sawyer, UR5e
    """
    time.sleep(3)

    # print welcome info
    print("Welcome to robosuite v{}!".format(suite.__version__))
    print(suite.__logo__)

    parser = argparse.ArgumentParser()
    parser.add_argument("--connection", action='store_true', help="if True, the source robot will wait for the target robot to connect to it")
    parser.add_argument("--port", type=int, default=50007, help="(optional) port for socket connection")
    parser.add_argument("--seed", type=int, default=0, help="(optional) (optional) set seed")
    parser.add_argument("--target_gripper", type=str, default="Robotiq85Gripper", help="PandaGripper or Robotiq85Gripper")
    parser.add_argument("--num_robot_poses", type=int, default=5, help="(optional) (optional) set seed")
    parser.add_argument("--target_robot", type=str, default="IIWA", help="(optional) (optional) set seed")
    parser.add_argument("--num_cam_poses_per_robot_pose", type=int, default=5, help="(optional) (optional) set seed")
    parser.add_argument("--save_paired_images_folder_path", type=str, default="paired_images", help="(optional) folder path to save the paired images")
    parser.add_argument("--robot_dataset", type=str, help="(optional) to match the robot poses from a dataset, provide the dataset name")
    parser.add_argument("--reference_joint_angles_path", type=str, help="(optional) to match the robot poses from a dataset, provide the path to the joint angles file (np.savetxt)")
    parser.add_argument("--reference_ee_states_path", type=str, help="(optional) to match the robot poses from a dataset, provide the path to the ee state file (np.savetxt)")
    parser.add_argument("--start_id", type=int, default=0, help="(optional) starting index of the robot poses")
    args = parser.parse_args()
    
    
    target_name = args.target_robot

    if target_name == "Sawyer":
        target_gripper = "RethinkGripper"
    elif target_name == "Jaco":
        target_gripper = "JacoThreeFingerGripper"
    elif target_name == "IIWA":
        target_gripper = "Robotiq85Gripper"

    # Save the captured images
    save_paired_images_folder_path = "paired_images/" + args.robot_dataset + "_" + args.target_robot + "_" + args.save_paired_images_folder_path
    os.makedirs(os.path.join(save_paired_images_folder_path, "{}_rgb".format(target_name.lower())), exist_ok=True)
    os.makedirs(os.path.join(save_paired_images_folder_path, "{}_rgb_brightness_augmented".format(target_name.lower())), exist_ok=True)
    os.makedirs(os.path.join(save_paired_images_folder_path, "{}_mask".format(target_name.lower())), exist_ok=True)
    
    if args.robot_dataset is not None:
        from dataset_poses_dict import ROBOT_CAMERA_POSES_DICT
        robot_dataset_info = ROBOT_CAMERA_POSES_DICT[args.robot_dataset]
        camera_height = robot_dataset_info["camera_heights"]
        camera_width = robot_dataset_info["camera_widths"]
    else:
        camera_height = 256
        camera_width = 256
    
    target_env = TargetEnvWrapper(target_name, target_gripper, args.robot_dataset, camera_height, camera_width, connection=args.connection, port=args.port)

    target_env.generate_image(num_robot_poses=args.num_robot_poses, num_cam_poses_per_robot_pose=args.num_cam_poses_per_robot_pose, save_paired_images_folder_path=save_paired_images_folder_path, reference_joint_angles_path=args.reference_joint_angles_path, reference_ee_states_path=args.reference_ee_states_path, robot_dataset=args.robot_dataset, start_id=args.start_id)

    target_env.target_env.env.close_renderer()