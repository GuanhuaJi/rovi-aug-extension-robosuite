"""
['austin_buds', 'austin_sailor', 'autolab_ur5', 'furniture_bench', 'hydra', 'jaco_play', 'mirage', 'mutex', 'nyu_franka', 'roboturk', 'taco_play', 'toto', 'viola']
CUDA_VISIBLE_DEVICES=0 python paired_images_data_gen_server.py --connection --port 50016 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_mirage_franka_gripper --robot_dataset austin_buds --start_id 0 &
CUDA_VISIBLE_DEVICES=0 python paired_images_data_gen_client.py --connection --port 50016  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_mirage_franka_gripper --robot_dataset austin_buds --start_id 0 &
"""

devices_ids = [0, 1, 2, 3, 4] * 1000
port_id = 42013
datasets = ['austin_buds', 'austin_sailor', 'autolab_ur5', 'furniture_bench', 'hydra', 'jaco_play', 'mirage', 'mutex', 'nyu_franka', 'roboturk', 'taco_play', 'toto', 'viola']
dataset_length = [10000, 5000, 2000, 4000, 5000, 3000, 10000, 10000, 6000, 4000, 10000, 3000, 10000]

for start_id in range(0, 10000, 1000):
    for k, data in enumerate(datasets[:-1]):
        if start_id >= dataset_length[k]:
            continue
        device_id = devices_ids.pop()
        print(f"CUDA_VISIBLE_DEVICES={device_id} python paired_images_data_gen_server.py --connection --port {port_id} --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_pairs_all/paired_images_{data} --robot_dataset {data} --start_id {start_id} &")
        print(f"CUDA_VISIBLE_DEVICES={device_id} python paired_images_data_gen_client.py --connection --port {port_id}  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_pairs_all/paired_images_{data} --robot_dataset {data} --start_id {start_id} &")
        port_id += 3
        print()