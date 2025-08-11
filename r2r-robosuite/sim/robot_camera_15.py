# import robosuite as suite
# from sim.camera import CameraWrapper
# import numpy as np
# import robosuite.utils.transform_utils as T
# from core.physics import fast_step
# from core.geometry import compute_pose_error
# from sim.geom_utils import _robot_geom_ids

# from robosuite.controllers import load_part_controller_config
# from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
# from robosuite.models.arenas import EmptyArena
# from robosuite.models.tasks import ManipulationTask
# import robosuite.controllers
# from robosuite.controllers import load_composite_controller_config

# ctrl_cfg = load_composite_controller_config(controller="BASIC")

# class Empty(ManipulationEnv):
#     def _load_model(self):
#         # 1️⃣ first create robot wrappers → self.robots
#         self._load_robots()                    # essential

#         # 2️⃣ Arena
#         self.arena = EmptyArena()

#         # 3️⃣ extract **robot_model** list
#         robot_xmls = [r.robot_model for r in self.robots]

#         # 4️⃣ create Task (note new parameter names)
#         self.model = ManipulationTask(
#             mujoco_arena=self.arena,
#             mujoco_robots=robot_xmls,
#             mujoco_objects=[],                # pass objects here if any
#         )

# from robosuite.environments.base import register_env
# register_env(Empty)

# class RobotCameraWrapper:
#     def __init__(self, robotname="Panda", grippername="PandaGripper", robot_dataset=None, camera_height=256, camera_width=256):
#         options = {}
#         self.env = suite.make(
#             **options,
#             robots=robotname,
#             gripper_types=grippername,
#             env_name="Empty",
#             #env_name="Lift",
#             has_renderer=True,  # no on-screen renderer
#             has_offscreen_renderer=True,  # no off-screen renderer
#             ignore_done=True,
#             use_camera_obs=True,  # no camera observations
#             #controller_configs = suite.load_controller_config(default_controller="OSC_POSE"),
#             controller_configs = ctrl_cfg,
#             control_freq=20,
#             renderer="mujoco",
#             camera_names = ["agentview"],  # You can add more camera names if needed
#             camera_heights = camera_height,
#             camera_widths = camera_width,
#             camera_depths = True,
#             camera_segmentations = "robot_only",
#             hard_reset=False,
#         )
        
#         self.camera_wrapper = CameraWrapper(self.env)
#         self.robot_name = robotname
#         self.robot_base_name = f"robot0_base"
#         self.base_body_id = self.env.sim.model.body_name2id(self.robot_base_name)
#         self.base_position = self.env.sim.model.body_pos[self.base_body_id].copy()

#     def get_gripper_width_from_qpos(self):
#         sim   = self.env.sim
#         robot = self.env.robots[0]
#         if hasattr(robot, "_ref_gripper_joint_pos_indexes") and robot._ref_gripper_joint_pos_indexes is not None:
#             qpos_idx = robot._ref_gripper_joint_pos_indexes
#         else:
#             joint_names = robot.gripper.joints
#             qpos_idx = [sim.model.get_joint_qpos_addr(name) for name in joint_names]

#         finger_qpos = sim.data.qpos[qpos_idx]
#         if self.robot_name == "Panda":
#             return 2.0 * finger_qpos[0], np.clip(2.0 * finger_qpos[0] / 0.08, 0, 1) # close 0 -> open 0.08
#         elif self.robot_name == "UR5e" or self.robot_name == "Kinova3" or self.robot_name == "IIWA":
#             return 2.0 * finger_qpos[0], (1 - np.clip(2.0 * finger_qpos[0], 0, 1)) # close 1 -> open 0
#         elif self.robot_name == "Sawyer":
#             return 2.0 * finger_qpos[0], 1 - np.clip(2.0 * finger_qpos[0] / -0.024, 0, 1) # close -0.024 -> open 0
#         elif self.robot_name == "Jaco":
#             return 2.0 * finger_qpos[0], np.clip(2.0 * finger_qpos[0] / 2.2, 0, 1) # close 0 -> open 2.2
        
#     def get_gripper_width(self):
#     # directly read current environment observation
#         return self.env._observables["robot0_gripper_qpos"].obs[0]
        

#     def compute_eef_pose(self):
#         arm_ctrl = self.env.robots[0].part_controllers["right"]
#         sid = self.env.sim.model.site_name2id(arm_ctrl.ref_name)      # eef_name → ref_name :contentReference[oaicite:3]{index=3}
#         pos = self.env.sim.data.site_xpos[sid]
#         rot = T.mat2quat(self.env.sim.data.site_xmat[sid].reshape(3,3))
#         return np.concatenate([pos, rot])
        
#     def teleport_to_joint_positions(self, joint_angles):
#         joint_names = self.env.robots[0].robot_joints
#         for i, joint_name in enumerate(joint_names):
#             qpos_addr = self.env.sim.model.get_joint_qpos_addr(joint_name)
#             self.env.sim.data.qpos[qpos_addr] = joint_angles[i]
#             self.env.sim.data.qvel[qpos_addr] = 0.0
#         self.env.sim.forward()

#     def drive_robot_to_target_pose(
#             self,
#             target_pose,                   # 7-DoF (x,y,z, qw,qx,qy,qz)
#             pos_thr  = 3e-3,               # m         | success when both
#             rot_thr  = 1e-2,               # rad (~0.6°)| errors below
#             step_max = 300):

#         assert len(target_pose) == 7, "Target pose should be 7-DoF <xyz,quat>"

#         # ----  controller handle -------------------------------------------------
#         arm_ctrl = self.env.robots[0].part_controllers["right"]     # single arm  :contentReference[oaicite:2]{index=2}
#         arm_ctrl.control_delta = True                               # send Δ pose
#         Δ_scale   = arm_ctrl.action_scale                           # default 0.05 m  :contentReference[oaicite:3]{index=3}

#         def pose_error(pose_a, pose_b):
#             """returns (pos_err, rot_err)"""
#             dp   = np.linalg.norm(pose_a[:3] - pose_b[:3])
#             dq   = T.quat_distance(pose_a[3:], pose_b[3:])          # scalar rad
#             return dp, dq

#         for step in range(step_max):
#             cur = self.compute_eef_pose()
#             pos_err, rot_err = pose_error(cur, target_pose)
#             if pos_err < pos_thr and rot_err < rot_thr:
#                 return True, cur, (pos_err, rot_err)

#             # ---------- build δ-action (normalised to ±1) ------------------------
#             d_xyz = target_pose[:3] - cur[:3]
#             d_rot = T.quat2axisangle(T.quat_multiply(target_pose[3:], T.quat_conjugate(cur[3:])))
#             δ     = np.concatenate([d_xyz, d_rot])

#             # clip each component to one Δ-step
#             δ_clipped = np.clip(δ / Δ_scale, -1.0, 1.0)

#             act = np.zeros(self.env.action_dim)      # length 7  :contentReference[oaicite:4]{index=4}
#             act[:6] = δ_clipped                      # arm 6-DoF
#             fast_step(self.env, act)                # apply action

#         # -------------------------------------------------------------------------
#         # if we exit the loop we failed to converge
#         print(f"Failed: residual (pos,rot) = {pos_err:.4f}, {rot_err:.4f} rad")
#         return False, cur, (pos_err, rot_err)


#     def set_robot_joint_positions(self, joint_angles=None):
#         if joint_angles is None:
#             joint_angles = self.some_safe_joint_angles
#         for _ in range(200):
#             self.env.robots[0].set_robot_joint_positions(joint_angles)
#             self.env.sim.forward()
#             self.env.sim.step()
#             self.env._update_observables()

#     def set_gripper_joint_positions(self, finger_qpos, robot_name):
#         if robot_name == "Panda":
#             gripper_joint_names = ["gripper0_finger_joint1", "gripper0_finger_joint2"]
#         elif robot_name == "IIWA":
#             gripper_joint_names = ["gripper0_finger_joint", "gripper0_right_outer_knuckle_joint"]
#         elif robot_name == "Sawyer":
#             gripper_joint_names = ['gripper0_l_finger_joint', 'gripper0_r_finger_joint']
#         elif robot_name == "Jaco":
#             gripper_joint_names = ["gripper0_joint_thumb", "gripper0_joint_index", "gripper0_joint_pinky",]
        
#         for i, joint_name in enumerate(gripper_joint_names):
#             self.env.sim.data.set_joint_qpos(joint_name, finger_qpos[i])
#         for _ in range(10):
#             self.env.sim.forward()
#             self.env.sim.step()
    
#     def open_close_gripper(self, gripper_open=True):
#         arm_ctrl = self.env.robots[0].part_controllers["right"]
#         action = np.zeros(7)
#         action[-1] = -1.0 if gripper_open else 1.0
#         fast_step(self.env, action)
    
#     def update_camera(self):
#         for _ in range(50):
#             self.env.sim.forward()
#             self.env.sim.step()
#             self.env._update_observables()
          
#     def get_observation_fast(self, camera="agentview",
#                             width=640, height=480,
#                             white_background=True):
#         sim = self.env.sim
#         sim.forward()                                           # synchronize pose

#         rgb = sim.render(width=width, height=height,
#                         camera_name=camera)[::-1]
#         seg = sim.render(width=width, height=height,
#                         camera_name=camera,
#                         segmentation=True)[::-1]               # (H,W,2)

#         objtype_img = seg[..., 0]
#         objid_img   = seg[..., 1]

#         robot_body_ids = _robot_geom_ids(self.env)
#         mask = (np.isin(objid_img, list(robot_body_ids))).astype(np.uint8)
#         if white_background:
#             rgb_out = rgb.copy()
#             rgb_out[mask == 0] = 255
#         else:
#             rgb_out = (rgb * mask[..., None]).astype(np.uint8)

#         return rgb_out, mask