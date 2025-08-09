import robosuite as suite
from sim.camera import CameraWrapper
import numpy as np
import robosuite.utils.transform_utils as T
from core.physics import fast_step
from core.geometry import compute_pose_error
from sim.geom_utils import _robot_geom_ids

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
        self.robot_name = robotname
        self.robot_base_name = f"robot0_base"
        self.base_body_id = self.env.sim.model.body_name2id(self.robot_base_name)
        self.base_position = self.env.sim.model.body_pos[self.base_body_id].copy()

    def get_gripper_width_from_qpos(self):
        sim   = self.env.sim
        robot = self.env.robots[0]
        if hasattr(robot, "_ref_gripper_joint_pos_indexes") and robot._ref_gripper_joint_pos_indexes is not None:
            qpos_idx = robot._ref_gripper_joint_pos_indexes
        else:
            joint_names = robot.gripper.joints
            qpos_idx = [sim.model.get_joint_qpos_addr(name) for name in joint_names]

        finger_qpos = sim.data.qpos[qpos_idx]
        if self.robot_name == "Panda":
            return 2.0 * finger_qpos[0], np.clip(2.0 * finger_qpos[0] / 0.08, 0, 1) # close 0 -> open 0.08
        elif self.robot_name == "UR5e" or self.robot_name == "Kinova3" or self.robot_name == "IIWA":
            return 2.0 * finger_qpos[0], (1 - np.clip(2.0 * finger_qpos[0], 0, 1)) # close 1 -> open 0
        elif self.robot_name == "Sawyer":
            return 2.0 * finger_qpos[0], 1 - np.clip(2.0 * finger_qpos[0] / -0.024, 0, 1) # close -0.024 -> open 0
        elif self.robot_name == "Jaco":
            return 2.0 * finger_qpos[0], np.clip(2.0 * finger_qpos[0] / 2.2, 0, 1) # close 0 -> open 2.2
        

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

    def drive_robot_to_target_pose(self, target_pose=None, min_threshold=0.003, max_threshold=0.02, num_iter_max=100):
        self.env.robots[0].controller.use_delta = False # change to absolute pose for setting the initial state
        assert len(target_pose) == 7, "Target pose should be 7DOF"
        current_pose = self.compute_eef_pose()
        error = compute_pose_error(current_pose, target_pose)
        num_iters = 0   

        no_improve_steps = 0
        last_error = error 
        while error > min_threshold and num_iters < num_iter_max:
            action = np.zeros(7)
            action[:3] = target_pose[:3]
            action[3:6] = T.quat2axisangle(target_pose[3:])
            _, _, _ = fast_step(self.env, action)
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
        # print("ERROR", error)
        # print("Take {} iterations to drive robot to target pose".format(num_iters))
        current_pose = self.compute_eef_pose()
        self.env.use_camera_obs = True

        if error < max_threshold:
            return True, current_pose, error
        else:
            # print("Failed to drive robot to target pose")
            # print("SUGGESTION: ", target_pose - current_pose)
            return False, current_pose, error

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
        fast_step(self.env, action)
    
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