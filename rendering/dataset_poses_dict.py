import numpy as np

ROBOT_CAMERA_POSES_DICT = {
    "austin_buds": {
        "robot_joint_angles_path": "./datasets/states/austin_buds/episode_0/joint_states.txt",
        "gripper_states_path": "./datasets/states/austin_buds/episode_0/gripper_states.txt",
        "camera_position": np.array([0.97,0.065,0.78]),
        "roll": 29, # large up, small down
        "pitch": 0, # large cw, small ccw
        "yaw": 89.0, # large left, small right
        "camera_fov": 50,
        "camera_heights": 128,
        "camera_widths": 128,
        "num_episodes": 50
    },
    "austin_sailor": {
        "robot_joint_angles_path": "./datasets/states/austin_sailor/episode_1/joint_states.txt",
        "gripper_states_path": "./datasets/states/austin_sailor/episode_1/gripper_states.txt",
        "camera_position": np.array([0.72,-0.49,0.55]),
        "roll": 35.0, # large up, small down
        "pitch": 0.0, # large cw, small ccw
        "yaw": 22.0, # large left, small right
        "camera_fov": 46.0,
        "camera_heights": 128,
        "camera_widths": 128,
        "num_episodes": 240
    },
    "autolab_ur5": {
        "robot_joint_angles_path": "./datasets/states/autolab_ur5/episode_0/joint_states.txt",
        "robot_ee_states_path": "./datasets/states/autolab_ur5/episode_0/eef_pose.txt",
        "gripper_states_path": "./datasets/states/autolab_ur5/episode_0/gripper_states.txt",
        "camera_position": np.array([-0.03, -0.45, 0.46]),
        #"roll": 53,
        #"pitch": -4,
        #"yaw": 49,
        "roll": 51.37949173756049,
        "pitch": -4.0,
        "yaw": 49.2377567979282,
        "camera_fov": 57.82240163683314,
        "camera_heights": 480,
        "camera_widths": 640,
        "num_episodes": 896,
    },
    "austin_mutex": {
        "robot_joint_angles_path": "./datasets/states/austin_mutex/episode_0/joint_states.txt",
        "gripper_states_path": "./datasets/states/austin_mutex/episode_0/gripper_states.txt",
        "camera_position": np.array([-0.05,0.47,0.49]),
        "roll": 66, # large up, small down
        "pitch": -5.0, # large cw, small ccw
        "yaw": -130, # large left, small right
        "camera_fov": 60.0,
        "camera_heights": 128,
        "camera_widths": 128,
        "num_episodes": 1500,
    },   
    "nyu_franka": {
        "robot_joint_angles_path": "./datasets/states/nyu_franka/episode_0/joint_states.txt",
        "gripper_states_path": "./datasets/states/nyu_franka/episode_0/gripper_states.txt",
        "camera_position": np.array([-0.47,-0.5,0.75]),
        "roll": 78, # large up, small down
        "pitch": 4.0, # large cw, small ccw
        "yaw": -57.0, # large left, small right
        "camera_fov": 37.0,
        "camera_heights": 128,
        "camera_widths": 128,
        "num_episodes": 365,
    },
    "kaist": {
        "robot_joint_angles_path": "./datasets/states/kaist/episode_4/joint_states.txt",
        "gripper_states_path": "./datasets/states/kaist/episode_4/gripper_states.txt",
        "camera_position": np.array([0.95,-0.33,0.345]), # np.array([0.93,-0.34,0.28])
        "roll": 51.3, # 54.0
        "pitch": 0.0, # 0.0
        "yaw": 52.0, # 47.0
        "camera_fov": 43.0, # 52.0
        "camera_heights": 480,
        "camera_widths": 640,
        "num_episodes": 201,
    },
    "toto": {
        "robot_joint_angles_path": "./datasets/states/toto/episode_0/joint_states.txt",
        "gripper_states_path": "./datasets/states/toto/episode_0/gripper_states.txt",
        "camera_position": np.array([0.82,-0.82,0.55]),
        "roll": 67, # large up, small down
        "pitch": -3, # large cw, small ccw
        "yaw": 28, # large left, small right
        "camera_fov": 42,
        "camera_heights": 480,
        "camera_widths": 640,
        "num_episodes": 901,
    },
    "asu_table_top_rlds": {
        "robot_joint_angles_path": "./datasets/states/asu_table_top_rlds/episode_0/joint_states.txt",
        "gripper_states_path": "./datasets/states/asu_table_top_rlds/episode_0/gripper_states.txt",
        "camera_position": np.array([1.7,0,1.5]),
        "roll": 45, # large up, small down
        "pitch": 0, # large cw, small ccw
        "yaw": 90, # large left, small right
        "camera_fov": 45,
        "camera_heights": 1024,
        "camera_widths": 1024,
    },
    "taco_play": {
        "robot_joint_angles_path": "./datasets/states/taco_play/episode_1/joint_states.txt",
        "gripper_states_path": "./datasets/states/taco_play/episode_1/gripper_states.txt",
        "camera_position": np.array([-0.4,-0.83,1.3]), # np.array([-0.52,-0.92,1.4])
        "roll": 50.0, # 50.0
        "pitch": 0.0, # 0.0
        "yaw": -45.0, # -46.0
        "camera_fov": 40.0, # 37.0
        "camera_heights": 150,
        "camera_widths": 200,
        "num_episodes": 1000,
    },
    "furniture_bench": {
        "robot_joint_angles_path": "./datasets/states/furniture_bench/episode_0/joint_states.txt",
        "gripper_states_path": "./datasets/states/furniture_bench/episode_0/gripper_states.txt",
        #v1
        # "camera_position": np.array([1.35,-0.05,0.35]),
        # "roll": 72.0, # large up, small down
        # "pitch": 0.0, # large cw, small ccw
        # "yaw": 88.0, # large left, small right
        # "camera_fov": 24.0,
        "camera_position": np.array([1.12,-0.03,0.28]),
        "roll": 72.0, # large up, small down
        "pitch": 0.0, # large cw, small ccw
        "yaw": 88.0, # large left, small right
        "camera_fov": 29.0,
        "camera_heights": 224,
        "camera_widths": 224,
        "num_episodes": 5100,
    },
    "viola": {
        "robot_joint_angles_path": "./datasets/states/viola/episode_0/joint_states.txt",
        "gripper_states_path": "./datasets/states/viola/episode_0/gripper_states.txt",
        #v1
        # "camera_position": np.array([1.01, -0.55, 0.64]),
        # "roll": 56.0, # large up, small down
        # "pitch": 0.0, # large cw, small ccw
        # "yaw": 35.0, # large left, small right
        # "camera_fov": 36.0,
        "camera_position": np.array([0.98, -0.52, 0.62]),
        "roll": 56.0, # large up, small down
        "pitch": 0.0, # large cw, small ccw
        "yaw": 35.0, # large left, small right
        "camera_fov": 36.0,
        "camera_heights": 224,
        "camera_widths": 224,
        "num_episodes": 135,
    },
    "iamlab_cmu": {
        "robot_joint_angles_path": "./datasets/states/iamlab_cmu/episode_0/joint_states.txt",
        "gripper_states_path": "./datasets/states/iamlab_cmu/episode_0/gripper_states.txt",
        "camera_position": np.array([0.57, 0.52, 0.45]),
        "roll": 45.0, # large up, small down
        "pitch": 0.0, # large cw, small ccw
        "yaw": 178.0, # large left, small right
        "camera_fov": 50.0,
        "camera_heights": 360,
        "camera_widths": 640,
        "num_episodes": 520,
    },
    "can": {
        "camera_heights": 84,
        "camera_widths": 84,
    },
    "lift": {
        "camera_heights": 84,
        "camera_widths": 84,
    },
    "square": {
        "camera_heights": 84,
        "camera_widths": 84,
    },
    "stack": {
        "camera_heights": 84,
        "camera_widths": 84,
    },
    "three_piece_assembly": {
        "camera_heights": 84,
        "camera_widths": 84,
        "num_episodes": 1000,
    }
}