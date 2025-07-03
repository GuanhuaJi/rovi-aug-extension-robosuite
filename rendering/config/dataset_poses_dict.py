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
        "camera_position": np.array([-0.33533, -0.32647,  0.698]),
        "roll": 78.1, # large up, small down
        "pitch": 0, # large cw, small ccw
        "yaw": 60.55, # large left, small right
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
        "camera_position": np.array([2.2, 0.0, 1.52]),
        "roll": 52.0, # large up, small down
        "pitch": 0.0, # large cw, small ccw
        "yaw": 90.0, # large left, small right
        "camera_fov": 45,
        "camera_heights": 224,
        "camera_widths": 224,
        "num_episodes": 110,
    },
    "ucsd_kitchen_rlds": {
        "camera_position": np.array([ 0.38741, -0.82647,  0.33115]),
        "roll": 86.5, # large up, small down
        "pitch": 0.0, # large cw, small ccw
        "yaw": 1.7, # large left, small right
        "camera_fov": 45,
        "camera_heights": 480,
        "camera_widths": 640,
        "num_episodes": 150,
    },
    "utokyo_pick_and_place": {
        "camera_position": np.array([ 0.95476, -0.01469,  0.68365]),
        "roll": 55.52, # large up, small down
        "pitch": 0.0, # large cw, small ccw
        "yaw": 84.6, # large left, small right
        "camera_fov": 45,
        "camera_heights": 480,
        "camera_widths": 640,
        "num_episodes": 92,
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
        #"num_episodes": 3242,
        "num_episodes": 200,
    },
    "furniture_bench": {
        "robot_joint_angles_path": "./datasets/states/furniture_bench/episode_0/joint_states.txt",
        "gripper_states_path": "./datasets/states/furniture_bench/episode_0/gripper_states.txt",
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
        "robot_joint_angles_path": "./datasets/states/viola/episode_5/joint_states.txt",
        "gripper_states_path": "./datasets/states/viola/episode_5/gripper_states.txt",
        # front view
        "view_1": {
            "camera_position": np.array([1.27, 0.05, 0.71]),
            "roll": 38.0,
            "pitch": 0.0,
            "yaw": 90.0,
            "camera_fov": 40.0,
            "episodes": [0, 1, 101, 103, 104, 105, 106, 107, 108, 109, 11, 112, 113, 114, 115, 118, 119, 12, 121, 122, 123, 124, 125, 127, 128, 13, 131, 133, 134, 14, 15, 16, 17, 19, 2, 22, 24, 26, 27, 31, 32, 33, 35, 36, 37, 40, 41, 42, 44, 46, 47, 48, 49, 5, 50, 51, 53, 57, 58, 59, 6, 60, 61, 62, 65, 66, 69, 7, 70, 71, 72, 73, 75, 76, 78, 8, 81, 82, 83, 85, 87, 88, 89, 90, 91, 92, 93, 94, 95, 99]
        },
        # side view
        "view_2": {
            "camera_position": np.array([0.98, -0.52, 0.62]),
            "roll": 56.0,
            "pitch": 0.0,
            "yaw": 35.0,
            "camera_fov": 36.0,
            "episodes": [10, 100, 102, 110, 111, 116, 117, 120, 126, 129, 130, 132, 18, 20, 21, 23, 25, 28, 29, 3, 30, 34, 38, 39, 4, 43, 45, 52, 54, 55, 56, 63, 64, 67, 68, 74, 77, 79, 80, 84, 86, 9, 96, 97, 98]
        },

        "camera_heights": 224,
        "camera_widths": 224,
        "num_episodes": 135,
    },
    # "iamlab_cmu": {
    #     "robot_joint_angles_path": "./datasets/states/iamlab_cmu/episode_0/joint_states.txt",
    #     "gripper_states_path": "./datasets/states/iamlab_cmu/episode_0/gripper_states.txt",
    #     "camera_position": np.array([0.57, 0.52, 0.45]),
    #     "roll": 45.0, # large up, small down
    #     "pitch": 0.0, # large cw, small ccw
    #     "yaw": 178.0, # large left, small right
    #     "camera_fov": 50.0,
    #     "camera_heights": 360,
    #     "camera_widths": 640,
    #     "num_episodes": 520,
    # },
    "iamlab_cmu": {
        "robot_joint_angles_path": "./datasets/states/iamlab_cmu/episode_0/joint_states.txt",
        "gripper_states_path": "./datasets/states/iamlab_cmu/episode_0/gripper_states.txt",
        "camera_position": np.array([0.56004, 0.54349, 0.41225]),
        "roll": 52.25, # large up, small down
        "pitch": 0.0, # large cw, small ccw
        "yaw": 179.5, # large left, small right
        "camera_fov": 45.0,
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