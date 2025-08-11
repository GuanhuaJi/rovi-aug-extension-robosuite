import numpy as np
import config.processing_episode as processing_episode

ROBOT_CAMERA_POSES_DICT = {
    "austin_buds": { #full name
        "viewpoints": [
            {
                "camera_position": np.array([0.97,0.065,0.78]),
                "roll": 29, # large up, small down
                "pitch": 0, # large cw, small ccw
                "yaw": 89.0, # large left, small right
                "camera_fov": 50,
                "episodes": list(range(0, 50))
            }
        ],
        "inpaint_path": "/shared/projects/mirage2/final_inpainted_vids/austin_buds_dataset_converted_externally_to_rlds",
        "replay_path": "/home/abrashid/paired_images/austin_buds",
        "save_path": "./replay_videos/austin_buds",
        "robot": "Panda",
        "gripper": "PandaGripper",
        "extend_gripper": 0.0,
        "binarized_gripper": False,
        "GCS_path": "gs://gresearch/robotics/austin_buds_dataset_converted_externally_to_rlds/0.1.0", #remove
        "processing_function": processing_episode.process_episode_austin_buds
    },
    "austin_sailor": {
        "viewpoints": [
            {
                "camera_position": np.array([0.72,-0.49,0.55]),
                "roll": 35.0, # large up, small down
                "pitch": 0.0, # large cw, small ccw
                "yaw": 22.0, # large left, small right
                "camera_fov": 46.0,
                "episodes": list(range(0, 240))
            }
        ],
        "inpaint_path": "/shared/projects/mirage2/final_inpainted_vids/austin_sailor_dataset_converted_externally_to_rlds",
        "replay_path": "./replay_videos/austin_sailor",
        "camera_height": 128,
        "camera_width": 128,
        "num_episodes": 240,
        "robot": "Panda",
        "gripper": "PandaGripper",
        "extend_gripper": 0.0,
        "binarized_gripper": False,
        "GCS_path": "gs://gresearch/robotics/austin_sailor_dataset_converted_externally_to_rlds/0.1.0",
        "processing_function": processing_episode.process_episode_austin_sailor
    },
    "autolab_ur5": {
        "viewpoints": [
            {
                "camera_position": np.array([-0.03, -0.45, 0.46]),
                "roll": 51.37949173756049,
                "pitch": -4.0,
                "yaw": 49.2377567979282,
                "camera_fov": 57.82240163683314,
                "episodes": list(range(0, 896))
            }
        ],
        "inpainting_path": "/shared/projects/mirage2/final_inpainted_vids/berkeley_autolab_ur5",
        "replay_path": "./replay_videos/autolab_ur5",
        "overlay_path": "./replay_videos/autolab_ur5_overlay",
        "camera_height": 128,
        "camera_width": 128,
        "num_episodes": 896,
        "robot": "UR5e",
        "gripper": "Robotiq85Gripper",
        "extend_gripper": 0.0,
        "binarized_gripper": True,
        "GCS_path": "gs://gresearch/robotics/berkeley_autolab_ur5/0.1.0",
        "processing_function": processing_episode.process_episode_autolab_ur5
    },
    "austin_mutex": {
        "viewpoints": [
            {
                "camera_position": np.array([-0.05, 0.47, 0.49]),
                "roll": 66,
                "pitch": -5.0,
                "yaw": -130,
                "camera_fov": 60.0,
                "episodes": list(range(0, 1500))
            }
        ],
        "inpaint_path": "/shared/projects/mirage2/final_inpainted_vids/utaustin_mutex",
        "replay_path": "./replay_videos/austin_mutex",
        "camera_height": 128,
        "camera_width": 128,
        "num_episodes": 1500,
        "robot": "Panda",
        "gripper": "PandaGripper",
        "extend_gripper": 0.0,
        "binarized_gripper": False,
        "GCS_path": "gs://gresearch/robotics/utaustin_mutex/0.1.0",
        "processing_function": processing_episode.process_episode_austin_mutex
    },   
    "nyu_franka": {
        "viewpoints": [
            {
                "camera_position": np.array([-0.33533, -0.32647,  0.698]),
                "roll": 78.1,
                "pitch": 0,
                "yaw": -60.55,
                "camera_fov": 37.0,
                "episodes": list(range(0, 365))
            }
        ],
        "inpaint_path": "/shared/projects/mirage2/final_inpainted_vids/nyu_franka_play_dataset_converted_externally_to_rlds",
        "replay_path": "./replay_videos/nyu_franka",
        "camera_heights": 128,
        "camera_widths": 128,
        "num_episodes": 365,
        "robot": "Panda",
        "gripper": "PandaGripper",
        "extend_gripper": 0.0,
        "binarized_gripper": True,
        "GCS_path": "gs://gresearch/robotics/nyu_franka_play_dataset_converted_externally_to_rlds/0.1.0",
        "processing_function": processing_episode.process_episode_nyu_franka
    },
    "kaist": {
        "viewpoints": [
            {
                "camera_position": np.array([0.95,-0.33,0.345]), # np.array([0.93,-0.34,0.28])
                "roll": 51.3, # 54.0
                "pitch": 0.0, # 0.0
                "yaw": 52.0, # 47.0
                "camera_fov": 43.0, # 52.0
                "episodes": list(range(0, 201))
            }
        ],
        "inpaint_path": "/shared/projects/mirage2/final_inpainted_vids/kaist_nonprehensile_converted_externally_to_rlds",
        "replay_path": "./replay_videos/kaist",
        "camera_height": 480,
        "camera_width": 640,
        "num_episodes": 201,
        "robot": "Panda",
        "gripper": "PandaGripper",
        "extend_gripper": 0.0,
        "binarized_gripper": False,
        "GCS_path": "gs://gresearch/robotics/kaist_nonprehensile_converted_externally_to_rlds/0.1.0",
        "processing_function": processing_episode.process_episode_kaist
    },
    "toto": {
        "viewpoints": [
            {
                "camera_position": np.array([0.82,-0.82,0.55]),
                "roll": 67, # large up, small down
                "pitch": -3, # large cw, small ccw
                "yaw": 28, # large left, small right
                "camera_fov": 42,
                "episodes": list(range(0, 901))
            }
        ],
        "inpaint_path": "/shared/projects/mirage2/final_inpainted_vids/toto",
        "replay_path": "./replay_videos/toto",
        "camera_height": 480,
        "camera_width": 640,
        "num_episodes": 901,
        "robot": "Panda",
        "gripper": "PandaGripper",
        "extend_gripper": 0.0,
        "binarized_gripper": True,
        "GCS_path": "gs://gresearch/robotics/toto/0.1.0",
        "processing_function": processing_episode.process_episode_toto
    },
    "asu_table_top_rlds": {
        "viewpoints": [
            {
                "camera_position": np.array([2.2, 0.0, 1.52]),
                "roll": 52.0, # large up, small down
                "pitch": 0.0, # large cw, small ccw
                "yaw": 90.0, # large left, small right
                "camera_fov": 45,
                "episodes": list(range(0, 110))
            }
        ],
        "inpaint_path": "/shared/projects/mirage2/final_inpainted_vids/asu_table_top_converted_externally_to_rlds",
        "replay_path": "./replay_videos/asu_table_top_rlds",
        "camera_height": 224,
        "camera_width": 224,
        "num_episodes": 110,
        "robot": "Panda",
        "gripper": "PandaGripper",
        "extend_gripper": 0.0,
        "GCS_path": "gs://gresearch/robotics/asu_table_top_converted_externally_to_rlds/0.1.0",
        "processing_function": processing_episode.process_episode_asu_table_top_rlds,
    },
    "ucsd_kitchen_rlds": {
        "viewpoints": [
            {
                "camera_position": np.array([ 0.38741, -0.82647,  0.33115]),
                "roll": 86.5, # large up, small down
                "pitch": 0.0, # large cw, small ccw
                "yaw": 1.7, # large left, small right
                "camera_fov": 45,
                "episodes": list(range(0, 150))
            }
        ],
        "inpaint_path": "/shared/projects/mirage2/final_inpainted_vids/ucsd_kitchen_dataset_converted_externally_to_rlds",
        "replay_path": "./replay_videos/ucsd_kitchen_rlds",
        "camera_heights": 480,
        "camera_widths": 640,
        "num_episodes": 150,
        "robot": "Panda",
        "gripper": "PandaGripper",
        "extend_gripper": 0.0,
        "binarized_gripper": False,
        "GCS_path": "gs://gresearch/robotics/ucsd_kitchen_dataset_converted_externally_to_rlds/0.1.0",
        "processing_function": processing_episode.process_episode_ucsd_kitchen_rlds
    },
    "utokyo_pick_and_place": {
        "viewpoints": [
            {
                "camera_position": np.array([ 0.95476, -0.01469,  0.68365]),
                "roll": 55.52, # large up, small down
                "pitch": 0.0, # large cw, small ccw
                "yaw": 84.6, # large left, small right
                "camera_fov": 45,
                "episodes": list(range(0, 92))
            }
        ],
        "inpaint_path": "/shared/projects/mirage2/final_inpainted_vids/utokyo_xarm_pick_and_place_converted_externally_to_rlds",
        "replay_path": "./replay_videos/utokyo_pick_and_place",
        "camera_height": 480,
        "camera_width": 640,
        "num_episodes": 92,
        "robot": "Panda",
        "gripper": "PandaGripper",
        "extend_gripper": 0.0,
        "binarized_gripper": False,
        "GCS_path": "gs://gresearch/robotics/utokyo_xarm_pick_and_place_converted_externally_to_rlds/0.1.0",
        "processing_function": processing_episode.process_episode_utokyo_pick_and_place
    },
    "taco_play": {
        "viewpoints": [
            {
                "camera_position": np.array([-0.4,-0.83,1.3]), # np.array([-0.52,-0.92,1.4])
                "roll": 50.0, # 50.0
                "pitch": 0.0, # 0.0
                "yaw": -45.0, # -46.0
                "camera_fov": 40.0, # 37.0
                "episodes": list(range(0, 3242))
            }
        ],
        "inpaint_path": "/home/harshapolavaram/rovi-aug-extension/ProPainter/taco_play",
        "replay_path": "./replay_videos/taco_play",
        "camera_height": 150,
        "camera_width": 200,
        "num_episodes": 3242,
        "robot": "Panda",
        "gripper": "PandaGripper",
        "extend_gripper": 0.0,
        "binarized_gripper": True,
        "GCS_path": "gs://gresearch/robotics/taco_play/0.1.0",
        "processing_function": processing_episode.process_episode_taco_play
    },
    "furniture_bench": {
        "viewpoints": [
            {
                "camera_position": np.array([1.12,-0.03,0.28]),
                "roll": 72.0, # large up, small down
                "pitch": 0.0, # large cw, small ccw
                "yaw": 88.0, # large left, small right
                "camera_fov": 29.0,
                "episodes": list(range(0, 5100))
            }
        ],
        "inpaint_path": "/home/abrashid/video_inpainting/furniture_bench",
        "replay_path": "./replay_videos/furniture_bench",
        "camera_height": 224,
        "camera_width": 224,
        "num_episodes": 5100,
        "robot": "Panda",
        "gripper": "PandaGripper",
        "extend_gripper": 0.0,
        "binarized_gripper": False,
        "GCS_path": "gs://gresearch/robotics/furniture_bench_dataset_converted_externally_to_rlds/0.1.0",
        "processing_function": processing_episode.process_episode_furniture_bench
    },
    "viola": {
        "viewpoints": [
            {
                "camera_position": np.array([1.27, 0.05, 0.71]),
                "roll": 38.0,
                "pitch": 0.0,
                "yaw": 90.0,
                "camera_fov": 40.0,
                "episodes": [0, 1, 101, 103, 104, 105, 106, 107, 108, 109, 11, 112, 113, 114, 115, 118, 119, 12, 121, 122, 123, 124, 125, 127, 128, 13, 131, 133, 134, 14, 15, 16, 17, 19, 2, 22, 24, 26, 27, 31, 32, 33, 35, 36, 37, 40, 41, 42, 44, 46, 47, 48, 49, 5, 50, 51, 53, 57, 58, 59, 6, 60, 61, 62, 65, 66, 69, 7, 70, 71, 72, 73, 75, 76, 78, 8, 81, 82, 83, 85, 87, 88, 89, 90, 91, 92, 93, 94, 95, 99]
            },
            {
                "camera_position": np.array([0.98, -0.52, 0.62]),
                "roll": 56.0,
                "pitch": 0.0,
                "yaw": 35.0,
                "camera_fov": 36.0,
                "episodes": [10, 100, 102, 110, 111, 116, 117, 120, 126, 129, 130, 132, 18, 20, 21, 23, 25, 28, 29, 3, 30, 34, 38, 39, 4, 43, 45, 52, 54, 55, 56, 63, 64, 67, 68, 74, 77, 79, 80, 84, 86, 9, 96, 97, 98]
            },
        ],
        "inpaint_path": "/home/abrashid/video_inpainting/viola",
        "replay_path": "./replay_videos/viola",
        "camera_height": 224,
        "camera_width": 224,
        "num_episodes": 135,
        "robot": "Panda",
        "gripper": "PandaGripper",
        "extend_gripper": 0.08,
        "binarized_gripper": False,
        "GCS_path": "gs://gresearch/robotics/viola/0.1.0",
        "processing_function": processing_episode.process_episode_viola
    },
    "iamlab_cmu": {
        "viewpoints": [
            {
                "camera_position": np.array([0.56004, 0.54349, 0.41225]),
                "roll": 52.25, # large up, small down
                "pitch": 0.0, # large cw, small ccw
                "yaw": 179.5, # large left, small right
                "camera_fov": 45.0,
                "episodes": list(range(0, 520))
            },
            {
                "camera_position": np.array([0.6792,  0.00989, 0.78965]),
                "roll": 4.25, # large up, small down
                "pitch": 0.0, # large cw, small ccw
                "yaw": -268.25, # large left, small right
                "camera_fov": 48.0,
                "episodes": list(range(0, 520))
            }
        ],
        "inpaint_path": "/shared/projects/mirage2/final_inpainted_vids/iamlab_cmu_pickup_insert_converted_externally_to_rlds",
        "replay_path": "./replay_videos/iamlab_cmu",
        "camera_height": 360,
        "camera_width": 640,
        "num_episodes": 520,
        "robot": "Panda",
        "gripper": "PandaGripper",
        "extend_gripper": 0.08,
        "binarized_gripper": True,
        "GCS_path": "gs://gresearch/robotics/iamlab_cmu_pickup_insert_converted_externally_to_rlds/0.1.0",
        "processing_function": processing_episode.process_episode_iamlab_cmu
    },
    "can": {
        "replay_path": "./replay_videos/can",
        "camera_height": 84,
        "camera_width": 84,
        "robot": "Panda",
        "gripper": "PandaGripper",
        "extend_gripper": 0.0,
        "num_episodes": 200,
    },
    "lift": {
        "replay_path": "./replay_videos/lift",
        "camera_height": 84,
        "camera_width": 84,
        "robot": "Panda",
        "gripper": "PandaGripper",
        "extend_gripper": 0.0
    },
    "square": {
        "replay_path": "./replay_videos/square",
        "camera_height": 84,
        "camera_width": 84,
        "robot": "Panda",
        "gripper": "PandaGripper",
        "extend_gripper": 0.0
    },
    "stack": {
        "replay_path": "./replay_videos/stack",
        "camera_height": 84,
        "camera_width": 84,
        "robot": "Panda",
        "gripper": "PandaGripper",
        "extend_gripper": 0.0
    },
    "two_piece": {
        "replay_path": "./replay_videos/two_piece",
        "camera_height": 84,
        "camera_width": 84,
        "robot": "Panda",
        "gripper": "PandaGripper",
        "extend_gripper": 0.0
    }
}