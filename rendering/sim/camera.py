import numpy as np
import robosuite.utils.transform_utils as T
import robosuite.utils.camera_utils as camera_utils
from robosuite.utils.camera_utils import CameraMover
import xml.etree.ElementTree as ET
from core.geometry import compute_pose_error

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

        cid = self.env.sim.model.camera_name2id("agentview")
        R   = self.env.sim.data.cam_xmat[cid].reshape(3, 3)

        forward_world = -R[:, 2]   # –Z column
        up_world      =  R[:, 1]   # +Y column

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