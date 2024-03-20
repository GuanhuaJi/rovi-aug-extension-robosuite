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

from paired_images_data_gen_server import Data, RobotCameraWrapper


class TargetEnvWrapper:
    def __init__(self, target_name, connection=None, port=50007):
        self.target_env = RobotCameraWrapper(robotname=target_name)
        self.target_name = target_name
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
    
    def generate_image(self, num_robot_poses=5, num_cam_poses_per_robot_pose=10):
        
        
        for pose_index in range(num_robot_poses):
            print(pose_index)
            counter = 0
            os.makedirs(os.path.join(save_paired_images_folder_path, f"{target_name.lower()}_rgb", str(pose_index)), exist_ok=True)
            os.makedirs(os.path.join(save_paired_images_folder_path, f"{target_name.lower()}_mask", str(pose_index)), exist_ok=True)
            
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
                target_reached, targt_reached_pose = self.target_env.drive_robot_to_target_pose(target_pose=target_pose)
                
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
                    print("Target robot pose: ", targt_reached_pose)
            
            
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
                success = source_env_robot_state.success
                if not success:
                    continue
                
                # self.target_env.camera_wrapper.set_camera_fov(fov=fov)
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
                
                cv2.imwrite(os.path.join(save_paired_images_folder_path, f"{target_name.lower()}_rgb", f"{pose_index}/{counter}.jpg"), cv2.cvtColor(target_robot_img, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(save_paired_images_folder_path, f"{target_name.lower()}_mask", f"{pose_index}/{counter}.jpg"), target_robot_seg_img * 255)
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
    parser.add_argument("--num_robot_poses", type=int, default=5, help="(optional) (optional) set seed")
    parser.add_argument("--num_cam_poses_per_robot_pose", type=int, default=5, help="(optional) (optional) set seed")
    parser.add_argument("--save_paired_images_folder_path", type=str, default="paired_images", help="(optional) folder path to save the paired images")
    args = parser.parse_args()
    
    
    target_name = "UR5e"

    # Save the captured images
    save_paired_images_folder_path = args.save_paired_images_folder_path
    os.makedirs(os.path.join(save_paired_images_folder_path, "{}_rgb".format(target_name.lower())), exist_ok=True)
    os.makedirs(os.path.join(save_paired_images_folder_path, "{}_mask".format(target_name.lower())), exist_ok=True)
    
    
    
    target_env = TargetEnvWrapper(target_name, connection=args.connection, port=args.port)
    target_env.generate_image(num_robot_poses=args.num_robot_poses, num_cam_poses_per_robot_pose=args.num_cam_poses_per_robot_pose)

    target_env.target_env.env.close_renderer()
        
    