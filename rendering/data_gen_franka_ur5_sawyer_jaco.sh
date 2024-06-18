

CUDA_VISIBLE_DEVICES=4 python paired_images_data_gen_multiserver.py --connection --connection_num 3 --port 41390 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_sawyer_jaco_open_gripper/paired_images_austin_sailor --robot_dataset austin_sailor --start_id 0 &
CUDA_VISIBLE_DEVICES=4 python paired_images_data_gen_multiclient.py --connection --port 41390  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_sawyer_jaco_open_gripper/paired_images_austin_sailor --robot_dataset austin_sailor --target_robot UR5e --target_gripper Robotiq85Gripper --start_id 0 &
CUDA_VISIBLE_DEVICES=4 python paired_images_data_gen_multiclient.py --connection --port 41390  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_sawyer_jaco_open_gripper/paired_images_austin_sailor --robot_dataset austin_sailor --target_robot Sawyer  --target_gripper RethinkGripper --start_id 0 &
CUDA_VISIBLE_DEVICES=4 python paired_images_data_gen_multiclient.py --connection --port 41390  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_sawyer_jaco_open_gripper/paired_images_austin_sailor --robot_dataset austin_sailor --target_robot Jaco --target_gripper JacoThreeFingerGripper --start_id 0 &

sleep 10

CUDA_VISIBLE_DEVICES=4 python paired_images_data_gen_multiserver.py --connection --connection_num 3 --port 41391 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_sawyer_jaco_open_gripper/paired_images_autolab_ur5 --robot_dataset autolab_ur5 --start_id 0 &
CUDA_VISIBLE_DEVICES=4 python paired_images_data_gen_multiclient.py --connection --port 41391  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_sawyer_jaco_open_gripper/paired_images_autolab_ur5 --robot_dataset autolab_ur5 --target_robot UR5e --target_gripper Robotiq85Gripper --start_id 0 &
CUDA_VISIBLE_DEVICES=4 python paired_images_data_gen_multiclient.py --connection --port 41391  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_sawyer_jaco_open_gripper/paired_images_autolab_ur5 --robot_dataset autolab_ur5 --target_robot Sawyer  --target_gripper RethinkGripper --start_id 0 &
CUDA_VISIBLE_DEVICES=4 python paired_images_data_gen_multiclient.py --connection --port 41391  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_sawyer_jaco_open_gripper/paired_images_autolab_ur5 --robot_dataset autolab_ur5 --target_robot Jaco --target_gripper JacoThreeFingerGripper --start_id 0 &

sleep 10

CUDA_VISIBLE_DEVICES=3 python paired_images_data_gen_multiserver.py --connection --connection_num 3 --port 41392 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_sawyer_jaco_open_gripper/paired_images_hydra --robot_dataset hydra --start_id 0 &
CUDA_VISIBLE_DEVICES=3 python paired_images_data_gen_multiclient.py --connection --port 41392  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_sawyer_jaco_open_gripper/paired_images_hydra --robot_dataset hydra --target_robot UR5e --target_gripper Robotiq85Gripper --start_id 0 &
CUDA_VISIBLE_DEVICES=3 python paired_images_data_gen_multiclient.py --connection --port 41392  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_sawyer_jaco_open_gripper/paired_images_hydra --robot_dataset hydra --target_robot Sawyer  --target_gripper RethinkGripper --start_id 0 &
CUDA_VISIBLE_DEVICES=3 python paired_images_data_gen_multiclient.py --connection --port 41392  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_sawyer_jaco_open_gripper/paired_images_hydra --robot_dataset hydra --target_robot Jaco --target_gripper JacoThreeFingerGripper --start_id 0 &

sleep 10

CUDA_VISIBLE_DEVICES=2 python paired_images_data_gen_multiserver.py --connection --connection_num 3 --port 41393 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_sawyer_jaco_open_gripper/paired_images_mirage --robot_dataset mirage --start_id 0 &
CUDA_VISIBLE_DEVICES=2 python paired_images_data_gen_multiclient.py --connection --port 41393  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_sawyer_jaco_open_gripper/paired_images_mirage --robot_dataset mirage --target_robot UR5e --target_gripper Robotiq85Gripper --start_id 0 &
CUDA_VISIBLE_DEVICES=2 python paired_images_data_gen_multiclient.py --connection --port 41393  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_sawyer_jaco_open_gripper/paired_images_mirage --robot_dataset mirage --target_robot Sawyer  --target_gripper RethinkGripper --start_id 0 &
CUDA_VISIBLE_DEVICES=2 python paired_images_data_gen_multiclient.py --connection --port 41393  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_sawyer_jaco_open_gripper/paired_images_mirage --robot_dataset mirage --target_robot Jaco --target_gripper JacoThreeFingerGripper --start_id 0 &

sleep 10

CUDA_VISIBLE_DEVICES=1 python paired_images_data_gen_multiserver.py --connection --connection_num 3 --port 41394 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_sawyer_jaco_open_gripper/paired_images_mutex --robot_dataset mutex --start_id 0 &
CUDA_VISIBLE_DEVICES=1 python paired_images_data_gen_multiclient.py --connection --port 41394  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_sawyer_jaco_open_gripper/paired_images_mutex --robot_dataset mutex --target_robot UR5e --target_gripper Robotiq85Gripper --start_id 0 &
CUDA_VISIBLE_DEVICES=1 python paired_images_data_gen_multiclient.py --connection --port 41394  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_sawyer_jaco_open_gripper/paired_images_mutex --robot_dataset mutex --target_robot Sawyer  --target_gripper RethinkGripper --start_id 0 &
CUDA_VISIBLE_DEVICES=1 python paired_images_data_gen_multiclient.py --connection --port 41394  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_sawyer_jaco_open_gripper/paired_images_mutex --robot_dataset mutex --target_robot Jaco --target_gripper JacoThreeFingerGripper --start_id 0 &

sleep 10

CUDA_VISIBLE_DEVICES=0 python paired_images_data_gen_multiserver.py --connection --connection_num 3 --port 41395 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_sawyer_jaco_open_gripper/paired_images_nyu_franka --robot_dataset nyu_franka --start_id 0 &
CUDA_VISIBLE_DEVICES=0 python paired_images_data_gen_multiclient.py --connection --port 41395  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_sawyer_jaco_open_gripper/paired_images_nyu_franka --robot_dataset nyu_franka --target_robot UR5e --target_gripper Robotiq85Gripper --start_id 0 &
CUDA_VISIBLE_DEVICES=0 python paired_images_data_gen_multiclient.py --connection --port 41395  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_sawyer_jaco_open_gripper/paired_images_nyu_franka --robot_dataset nyu_franka --target_robot Sawyer  --target_gripper RethinkGripper --start_id 0 &
CUDA_VISIBLE_DEVICES=0 python paired_images_data_gen_multiclient.py --connection --port 41395  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_sawyer_jaco_open_gripper/paired_images_nyu_franka --robot_dataset nyu_franka --target_robot Jaco --target_gripper JacoThreeFingerGripper --start_id 0 &

sleep 10

CUDA_VISIBLE_DEVICES=4 python paired_images_data_gen_multiserver.py --connection --connection_num 3 --port 42711 --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_sawyer_jaco_open_gripper/paired_images_random_poses --start_id 0 &
CUDA_VISIBLE_DEVICES=4 python paired_images_data_gen_multiclient.py --connection --port 42711  --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_sawyer_jaco_open_gripper/paired_images_random_poses --target_robot Sawyer --target_gripper RethinkGripper --start_id 0 &
CUDA_VISIBLE_DEVICES=4 python paired_images_data_gen_multiclient.py --connection --port 42711  --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_sawyer_jaco_open_gripper/paired_images_random_poses --target_robot Jaco --target_gripper JacoThreeFingerGripper --start_id 0 &
CUDA_VISIBLE_DEVICES=4 python paired_images_data_gen_multiclient.py --connection --port 42711  --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_sawyer_jaco_open_gripper/paired_images_random_poses --target_gripper Robotiq85Gripper --start_id 0 &

sleep 10


CUDA_VISIBLE_DEVICES=4 python paired_images_data_gen_multiserver.py --connection --connection_num 3 --port 42712 --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_sawyer_jaco_open_gripper/paired_images_random_poses --start_id 3000 &
CUDA_VISIBLE_DEVICES=4 python paired_images_data_gen_multiclient.py --connection --port 42712  --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_sawyer_jaco_open_gripper/paired_images_random_poses --target_robot Sawyer --target_gripper RethinkGripper --start_id 3000 &
CUDA_VISIBLE_DEVICES=4 python paired_images_data_gen_multiclient.py --connection --port 42712  --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_sawyer_jaco_open_gripper/paired_images_random_poses --target_robot Jaco --target_gripper JacoThreeFingerGripper --start_id 3000 &
CUDA_VISIBLE_DEVICES=4 python paired_images_data_gen_multiclient.py --connection --port 42712  --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_sawyer_jaco_open_gripper/paired_images_random_poses --target_gripper Robotiq85Gripper --start_id 3000 &

sleep 10


CUDA_VISIBLE_DEVICES=3 python paired_images_data_gen_multiserver.py --connection --connection_num 3 --port 41398 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_sawyer_jaco_open_gripper/paired_images_austin_sailor --robot_dataset austin_sailor --start_id 3000 &
CUDA_VISIBLE_DEVICES=3 python paired_images_data_gen_multiclient.py --connection --port 41398  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_sawyer_jaco_open_gripper/paired_images_austin_sailor --robot_dataset austin_sailor --target_robot UR5e --target_gripper Robotiq85Gripper --start_id 3000 &
CUDA_VISIBLE_DEVICES=3 python paired_images_data_gen_multiclient.py --connection --port 41398  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_sawyer_jaco_open_gripper/paired_images_austin_sailor --robot_dataset austin_sailor --target_robot Sawyer  --target_gripper RethinkGripper --start_id 3000 &
CUDA_VISIBLE_DEVICES=3 python paired_images_data_gen_multiclient.py --connection --port 41398  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_sawyer_jaco_open_gripper/paired_images_austin_sailor --robot_dataset austin_sailor --target_robot Jaco --target_gripper JacoThreeFingerGripper --start_id 3000 &

sleep 10

CUDA_VISIBLE_DEVICES=2 python paired_images_data_gen_multiserver.py --connection --connection_num 3 --port 41399 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_sawyer_jaco_open_gripper/paired_images_hydra --robot_dataset hydra --start_id 3000 &
CUDA_VISIBLE_DEVICES=2 python paired_images_data_gen_multiclient.py --connection --port 41399  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_sawyer_jaco_open_gripper/paired_images_hydra --robot_dataset hydra --target_robot UR5e --target_gripper Robotiq85Gripper --start_id 3000 &
CUDA_VISIBLE_DEVICES=2 python paired_images_data_gen_multiclient.py --connection --port 41399  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_sawyer_jaco_open_gripper/paired_images_hydra --robot_dataset hydra --target_robot Sawyer  --target_gripper RethinkGripper --start_id 3000 &
CUDA_VISIBLE_DEVICES=2 python paired_images_data_gen_multiclient.py --connection --port 41399  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_sawyer_jaco_open_gripper/paired_images_hydra --robot_dataset hydra --target_robot Jaco --target_gripper JacoThreeFingerGripper --start_id 3000 &

sleep 10

CUDA_VISIBLE_DEVICES=1 python paired_images_data_gen_multiserver.py --connection --connection_num 3 --port 41400 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_sawyer_jaco_open_gripper/paired_images_mirage --robot_dataset mirage --start_id 3000 &
CUDA_VISIBLE_DEVICES=1 python paired_images_data_gen_multiclient.py --connection --port 41400  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_sawyer_jaco_open_gripper/paired_images_mirage --robot_dataset mirage --target_robot UR5e --target_gripper Robotiq85Gripper --start_id 3000 &
CUDA_VISIBLE_DEVICES=1 python paired_images_data_gen_multiclient.py --connection --port 41400  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_sawyer_jaco_open_gripper/paired_images_mirage --robot_dataset mirage --target_robot Sawyer  --target_gripper RethinkGripper --start_id 3000 &
CUDA_VISIBLE_DEVICES=1 python paired_images_data_gen_multiclient.py --connection --port 41400  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_sawyer_jaco_open_gripper/paired_images_mirage --robot_dataset mirage --target_robot Jaco --target_gripper JacoThreeFingerGripper --start_id 3000 &

sleep 10

CUDA_VISIBLE_DEVICES=0 python paired_images_data_gen_multiserver.py --connection --connection_num 3 --port 41401 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_sawyer_jaco_open_gripper/paired_images_mutex --robot_dataset mutex --start_id 3000 &
CUDA_VISIBLE_DEVICES=0 python paired_images_data_gen_multiclient.py --connection --port 41401  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_sawyer_jaco_open_gripper/paired_images_mutex --robot_dataset mutex --target_robot UR5e --target_gripper Robotiq85Gripper --start_id 3000 &
CUDA_VISIBLE_DEVICES=0 python paired_images_data_gen_multiclient.py --connection --port 41401  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_sawyer_jaco_open_gripper/paired_images_mutex --robot_dataset mutex --target_robot Sawyer  --target_gripper RethinkGripper --start_id 3000 &
CUDA_VISIBLE_DEVICES=0 python paired_images_data_gen_multiclient.py --connection --port 41401  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_sawyer_jaco_open_gripper/paired_images_mutex --robot_dataset mutex --target_robot Jaco --target_gripper JacoThreeFingerGripper --start_id 3000 

sleep 10




CUDA_VISIBLE_DEVICES=4 python paired_images_data_gen_multiserver.py --connection --connection_num 3 --port 41402 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_sawyer_jaco_open_gripper/paired_images_nyu_franka --robot_dataset nyu_franka --start_id 3000 &
CUDA_VISIBLE_DEVICES=4 python paired_images_data_gen_multiclient.py --connection --port 41402  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_sawyer_jaco_open_gripper/paired_images_nyu_franka --robot_dataset nyu_franka --target_robot UR5e --target_gripper Robotiq85Gripper --start_id 3000 &
CUDA_VISIBLE_DEVICES=4 python paired_images_data_gen_multiclient.py --connection --port 41402  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_sawyer_jaco_open_gripper/paired_images_nyu_franka --robot_dataset nyu_franka --target_robot Sawyer  --target_gripper RethinkGripper --start_id 3000 &
CUDA_VISIBLE_DEVICES=4 python paired_images_data_gen_multiclient.py --connection --port 41402  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_sawyer_jaco_open_gripper/paired_images_nyu_franka --robot_dataset nyu_franka --target_robot Jaco --target_gripper JacoThreeFingerGripper --start_id 3000 &

sleep 10

CUDA_VISIBLE_DEVICES=4 python paired_images_data_gen_multiserver.py --connection --connection_num 3 --port 41403 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_sawyer_jaco_open_gripper/paired_images_taco_play --robot_dataset taco_play --start_id 3000 &
CUDA_VISIBLE_DEVICES=4 python paired_images_data_gen_multiclient.py --connection --port 41403  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_sawyer_jaco_open_gripper/paired_images_taco_play --robot_dataset taco_play --target_robot UR5e --target_gripper Robotiq85Gripper --start_id 3000 &
CUDA_VISIBLE_DEVICES=4 python paired_images_data_gen_multiclient.py --connection --port 41403  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_sawyer_jaco_open_gripper/paired_images_taco_play --robot_dataset taco_play --target_robot Sawyer  --target_gripper RethinkGripper --start_id 3000 &
CUDA_VISIBLE_DEVICES=4 python paired_images_data_gen_multiclient.py --connection --port 41403  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_sawyer_jaco_open_gripper/paired_images_taco_play --robot_dataset taco_play --target_robot Jaco --target_gripper JacoThreeFingerGripper --start_id 3000 &

sleep 10

CUDA_VISIBLE_DEVICES=3 python paired_images_data_gen_multiserver.py --connection --connection_num 3 --port 41404 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_sawyer_jaco_open_gripper/paired_images_viola --robot_dataset viola --start_id 3000 &
CUDA_VISIBLE_DEVICES=3 python paired_images_data_gen_multiclient.py --connection --port 41404  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_sawyer_jaco_open_gripper/paired_images_viola --robot_dataset viola --target_robot UR5e --target_gripper Robotiq85Gripper --start_id 3000 &
CUDA_VISIBLE_DEVICES=3 python paired_images_data_gen_multiclient.py --connection --port 41404  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_sawyer_jaco_open_gripper/paired_images_viola --robot_dataset viola --target_robot Sawyer  --target_gripper RethinkGripper --start_id 3000 &
CUDA_VISIBLE_DEVICES=3 python paired_images_data_gen_multiclient.py --connection --port 41404  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_sawyer_jaco_open_gripper/paired_images_viola --robot_dataset viola --target_robot Jaco --target_gripper JacoThreeFingerGripper --start_id 3000 &

sleep 10


CUDA_VISIBLE_DEVICES=2 python paired_images_data_gen_multiserver.py --connection --connection_num 3 --port 42713 --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_sawyer_jaco_open_gripper/paired_images_random_poses --start_id 6000 &
CUDA_VISIBLE_DEVICES=2 python paired_images_data_gen_multiclient.py --connection --port 42713  --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_sawyer_jaco_open_gripper/paired_images_random_poses --target_robot Sawyer --target_gripper RethinkGripper --start_id 6000 &
CUDA_VISIBLE_DEVICES=2 python paired_images_data_gen_multiclient.py --connection --port 42713  --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_sawyer_jaco_open_gripper/paired_images_random_poses --target_robot Jaco --target_gripper JacoThreeFingerGripper --start_id 6000 &
CUDA_VISIBLE_DEVICES=2 python paired_images_data_gen_multiclient.py --connection --port 42713  --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_sawyer_jaco_open_gripper/paired_images_random_poses --target_gripper Robotiq85Gripper --start_id 6000 &

sleep 10

CUDA_VISIBLE_DEVICES=1 python paired_images_data_gen_multiserver.py --connection --connection_num 3 --port 41396 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_sawyer_jaco_open_gripper/paired_images_taco_play --robot_dataset taco_play --start_id 0 &
CUDA_VISIBLE_DEVICES=1 python paired_images_data_gen_multiclient.py --connection --port 41396  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_sawyer_jaco_open_gripper/paired_images_taco_play --robot_dataset taco_play --target_robot UR5e --target_gripper Robotiq85Gripper --start_id 0 &
CUDA_VISIBLE_DEVICES=1 python paired_images_data_gen_multiclient.py --connection --port 41396  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_sawyer_jaco_open_gripper/paired_images_taco_play --robot_dataset taco_play --target_robot Sawyer  --target_gripper RethinkGripper --start_id 0 &
CUDA_VISIBLE_DEVICES=1 python paired_images_data_gen_multiclient.py --connection --port 41396  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_sawyer_jaco_open_gripper/paired_images_taco_play --robot_dataset taco_play --target_robot Jaco --target_gripper JacoThreeFingerGripper --start_id 0 &

sleep 10

CUDA_VISIBLE_DEVICES=0 python paired_images_data_gen_multiserver.py --connection --connection_num 3 --port 41397 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_sawyer_jaco_open_gripper/paired_images_viola --robot_dataset viola --start_id 0 &
CUDA_VISIBLE_DEVICES=0 python paired_images_data_gen_multiclient.py --connection --port 41397  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_sawyer_jaco_open_gripper/paired_images_viola --robot_dataset viola --target_robot UR5e --target_gripper Robotiq85Gripper --start_id 0 &
CUDA_VISIBLE_DEVICES=0 python paired_images_data_gen_multiclient.py --connection --port 41397  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_sawyer_jaco_open_gripper/paired_images_viola --robot_dataset viola --target_robot Sawyer  --target_gripper RethinkGripper --start_id 0 &
CUDA_VISIBLE_DEVICES=0 python paired_images_data_gen_multiclient.py --connection --port 41397  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_sawyer_jaco_open_gripper/paired_images_viola --robot_dataset viola --target_robot Jaco --target_gripper JacoThreeFingerGripper --start_id 0 &

sleep 10

