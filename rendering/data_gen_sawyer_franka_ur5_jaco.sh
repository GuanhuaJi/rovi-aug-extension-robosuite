CUDA_VISIBLE_DEVICES=4 python paired_images_data_gen_multiserver.py --connection --connection_num 3 --port 51001 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_austin_buds --robot_dataset austin_buds --source_robot Sawyer  --source_gripper RethinkGripper --use_cam_pose_only --start_id 0 &
CUDA_VISIBLE_DEVICES=4 python paired_images_data_gen_multiclient.py --connection --port 51001  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_austin_buds --robot_dataset austin_buds --target_robot Panda --target_gripper PandaGripper --use_cam_pose_only --start_id 0 &
CUDA_VISIBLE_DEVICES=4 python paired_images_data_gen_multiclient.py --connection --port 51001  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_austin_buds --robot_dataset austin_buds --target_robot UR5e --target_gripper Robotiq85Gripper --use_cam_pose_only --start_id 0 &
CUDA_VISIBLE_DEVICES=4 python paired_images_data_gen_multiclient.py --connection --port 51001  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_austin_buds --robot_dataset austin_buds --target_robot Jaco --target_gripper JacoThreeFingerGripper --use_cam_pose_only --start_id 0 &

sleep 10

CUDA_VISIBLE_DEVICES=3 python paired_images_data_gen_multiserver.py --connection --connection_num 3 --port 51002 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_austin_sailor --robot_dataset austin_sailor --source_robot Sawyer  --source_gripper RethinkGripper --use_cam_pose_only --start_id 0 &
CUDA_VISIBLE_DEVICES=3 python paired_images_data_gen_multiclient.py --connection --port 51002  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_austin_sailor --robot_dataset austin_sailor --target_robot Panda --target_gripper PandaGripper --use_cam_pose_only --start_id 0 &
CUDA_VISIBLE_DEVICES=3 python paired_images_data_gen_multiclient.py --connection --port 51002  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_austin_sailor --robot_dataset austin_sailor --target_robot UR5e --target_gripper Robotiq85Gripper --use_cam_pose_only --start_id 0 &
CUDA_VISIBLE_DEVICES=3 python paired_images_data_gen_multiclient.py --connection --port 51002  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_austin_sailor --robot_dataset austin_sailor --target_robot Jaco --target_gripper JacoThreeFingerGripper --use_cam_pose_only --start_id 0 &

sleep 10

CUDA_VISIBLE_DEVICES=2 python paired_images_data_gen_multiserver.py --connection --connection_num 3 --port 51003 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_autolab_ur5 --robot_dataset autolab_ur5 --source_robot Sawyer  --source_gripper RethinkGripper --use_cam_pose_only --start_id 0 &
CUDA_VISIBLE_DEVICES=2 python paired_images_data_gen_multiclient.py --connection --port 51003  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_autolab_ur5 --robot_dataset autolab_ur5 --target_robot Panda --target_gripper PandaGripper --use_cam_pose_only --start_id 0 &
CUDA_VISIBLE_DEVICES=2 python paired_images_data_gen_multiclient.py --connection --port 51003  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_autolab_ur5 --robot_dataset autolab_ur5 --target_robot UR5e --target_gripper Robotiq85Gripper --use_cam_pose_only --start_id 0 &
CUDA_VISIBLE_DEVICES=2 python paired_images_data_gen_multiclient.py --connection --port 51003  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_autolab_ur5 --robot_dataset autolab_ur5 --target_robot Jaco --target_gripper JacoThreeFingerGripper --use_cam_pose_only --start_id 0 &

sleep 10

CUDA_VISIBLE_DEVICES=1 python paired_images_data_gen_multiserver.py --connection --connection_num 3 --port 51004 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_furniture_bench --robot_dataset furniture_bench --source_robot Sawyer  --source_gripper RethinkGripper --use_cam_pose_only --start_id 0 &
CUDA_VISIBLE_DEVICES=1 python paired_images_data_gen_multiclient.py --connection --port 51004  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_furniture_bench --robot_dataset furniture_bench --target_robot Panda --target_gripper PandaGripper --use_cam_pose_only --start_id 0 &
CUDA_VISIBLE_DEVICES=1 python paired_images_data_gen_multiclient.py --connection --port 51004  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_furniture_bench --robot_dataset furniture_bench --target_robot UR5e --target_gripper Robotiq85Gripper --use_cam_pose_only --start_id 0 &
CUDA_VISIBLE_DEVICES=1 python paired_images_data_gen_multiclient.py --connection --port 51004  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_furniture_bench --robot_dataset furniture_bench --target_robot Jaco --target_gripper JacoThreeFingerGripper --use_cam_pose_only --start_id 0 &

sleep 10

CUDA_VISIBLE_DEVICES=0 python paired_images_data_gen_multiserver.py --connection --connection_num 3 --port 51005 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_hydra --robot_dataset hydra --source_robot Sawyer  --source_gripper RethinkGripper --use_cam_pose_only --start_id 0 &
CUDA_VISIBLE_DEVICES=0 python paired_images_data_gen_multiclient.py --connection --port 51005  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_hydra --robot_dataset hydra --target_robot Panda --target_gripper PandaGripper --use_cam_pose_only --start_id 0 &
CUDA_VISIBLE_DEVICES=0 python paired_images_data_gen_multiclient.py --connection --port 51005  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_hydra --robot_dataset hydra --target_robot UR5e --target_gripper Robotiq85Gripper --use_cam_pose_only --start_id 0 &
CUDA_VISIBLE_DEVICES=0 python paired_images_data_gen_multiclient.py --connection --port 51005  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_hydra --robot_dataset hydra --target_robot Jaco --target_gripper JacoThreeFingerGripper --use_cam_pose_only --start_id 0 &

sleep 10

CUDA_VISIBLE_DEVICES=4 python paired_images_data_gen_multiserver.py --connection --connection_num 3 --port 51006 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_jaco_play --robot_dataset jaco_play --source_robot Sawyer  --source_gripper RethinkGripper --use_cam_pose_only --start_id 0 &
CUDA_VISIBLE_DEVICES=4 python paired_images_data_gen_multiclient.py --connection --port 51006  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_jaco_play --robot_dataset jaco_play --target_robot Panda --target_gripper PandaGripper --use_cam_pose_only --start_id 0 &
CUDA_VISIBLE_DEVICES=4 python paired_images_data_gen_multiclient.py --connection --port 51006  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_jaco_play --robot_dataset jaco_play --target_robot UR5e --target_gripper Robotiq85Gripper --use_cam_pose_only --start_id 0 &
CUDA_VISIBLE_DEVICES=4 python paired_images_data_gen_multiclient.py --connection --port 51006  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_jaco_play --robot_dataset jaco_play --target_robot Jaco --target_gripper JacoThreeFingerGripper --use_cam_pose_only --start_id 0 &

sleep 10

CUDA_VISIBLE_DEVICES=3 python paired_images_data_gen_multiserver.py --connection --connection_num 3 --port 51007 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_mirage --robot_dataset mirage --source_robot Sawyer  --source_gripper RethinkGripper --use_cam_pose_only --start_id 0 &
CUDA_VISIBLE_DEVICES=3 python paired_images_data_gen_multiclient.py --connection --port 51007  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_mirage --robot_dataset mirage --target_robot Panda --target_gripper PandaGripper --use_cam_pose_only --start_id 0 &
CUDA_VISIBLE_DEVICES=3 python paired_images_data_gen_multiclient.py --connection --port 51007  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_mirage --robot_dataset mirage --target_robot UR5e --target_gripper Robotiq85Gripper --use_cam_pose_only --start_id 0 &
CUDA_VISIBLE_DEVICES=3 python paired_images_data_gen_multiclient.py --connection --port 51007  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_mirage --robot_dataset mirage --target_robot Jaco --target_gripper JacoThreeFingerGripper --use_cam_pose_only --start_id 0 &

sleep 10

CUDA_VISIBLE_DEVICES=2 python paired_images_data_gen_multiserver.py --connection --connection_num 3 --port 51008 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_mutex --robot_dataset mutex --source_robot Sawyer  --source_gripper RethinkGripper --use_cam_pose_only --start_id 0 &
CUDA_VISIBLE_DEVICES=2 python paired_images_data_gen_multiclient.py --connection --port 51008  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_mutex --robot_dataset mutex --target_robot Panda --target_gripper PandaGripper --use_cam_pose_only --start_id 0 &
CUDA_VISIBLE_DEVICES=2 python paired_images_data_gen_multiclient.py --connection --port 51008  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_mutex --robot_dataset mutex --target_robot UR5e --target_gripper Robotiq85Gripper --use_cam_pose_only --start_id 0 &
CUDA_VISIBLE_DEVICES=2 python paired_images_data_gen_multiclient.py --connection --port 51008  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_mutex --robot_dataset mutex --target_robot Jaco --target_gripper JacoThreeFingerGripper --use_cam_pose_only --start_id 0 &

sleep 10

CUDA_VISIBLE_DEVICES=1 python paired_images_data_gen_multiserver.py --connection --connection_num 3 --port 51009 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_nyu_franka --robot_dataset nyu_franka --source_robot Sawyer  --source_gripper RethinkGripper --use_cam_pose_only --start_id 0 &
CUDA_VISIBLE_DEVICES=1 python paired_images_data_gen_multiclient.py --connection --port 51009  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_nyu_franka --robot_dataset nyu_franka --target_robot Panda --target_gripper PandaGripper --use_cam_pose_only --start_id 0 &
CUDA_VISIBLE_DEVICES=1 python paired_images_data_gen_multiclient.py --connection --port 51009  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_nyu_franka --robot_dataset nyu_franka --target_robot UR5e --target_gripper Robotiq85Gripper --use_cam_pose_only --start_id 0 &
CUDA_VISIBLE_DEVICES=1 python paired_images_data_gen_multiclient.py --connection --port 51009  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_nyu_franka --robot_dataset nyu_franka --target_robot Jaco --target_gripper JacoThreeFingerGripper --use_cam_pose_only --start_id 0 &

sleep 10

CUDA_VISIBLE_DEVICES=0 python paired_images_data_gen_multiserver.py --connection --connection_num 3 --port 51010 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_taco_play --robot_dataset taco_play --source_robot Sawyer  --source_gripper RethinkGripper --use_cam_pose_only --start_id 0 &
CUDA_VISIBLE_DEVICES=0 python paired_images_data_gen_multiclient.py --connection --port 51010  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_taco_play --robot_dataset taco_play --target_robot Panda --target_gripper PandaGripper --use_cam_pose_only --start_id 0 &
CUDA_VISIBLE_DEVICES=0 python paired_images_data_gen_multiclient.py --connection --port 51010  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_taco_play --robot_dataset taco_play --target_robot UR5e --target_gripper Robotiq85Gripper --use_cam_pose_only --start_id 0 &
CUDA_VISIBLE_DEVICES=0 python paired_images_data_gen_multiclient.py --connection --port 51010  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_taco_play --robot_dataset taco_play --target_robot Jaco --target_gripper JacoThreeFingerGripper --use_cam_pose_only --start_id 0 &

sleep 10

CUDA_VISIBLE_DEVICES=4 python paired_images_data_gen_multiserver.py --connection --connection_num 3 --port 51011 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_viola --robot_dataset viola --source_robot Sawyer  --source_gripper RethinkGripper --use_cam_pose_only --start_id 0 &
CUDA_VISIBLE_DEVICES=4 python paired_images_data_gen_multiclient.py --connection --port 51011  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_viola --robot_dataset viola --target_robot Panda --target_gripper PandaGripper --use_cam_pose_only --start_id 0 &
CUDA_VISIBLE_DEVICES=4 python paired_images_data_gen_multiclient.py --connection --port 51011  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_viola --robot_dataset viola --target_robot UR5e --target_gripper Robotiq85Gripper --use_cam_pose_only --start_id 0 &
CUDA_VISIBLE_DEVICES=4 python paired_images_data_gen_multiclient.py --connection --port 51011  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_viola --robot_dataset viola --target_robot Jaco --target_gripper JacoThreeFingerGripper --use_cam_pose_only --start_id 0 &

sleep 10

CUDA_VISIBLE_DEVICES=3 python paired_images_data_gen_multiserver.py --connection --connection_num 3 --port 51012 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_toto --robot_dataset toto --source_robot Sawyer  --source_gripper RethinkGripper --use_cam_pose_only --start_id 0 &
CUDA_VISIBLE_DEVICES=3 python paired_images_data_gen_multiclient.py --connection --port 51012  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_toto --robot_dataset toto --target_robot Panda --target_gripper PandaGripper --use_cam_pose_only --start_id 0 &
CUDA_VISIBLE_DEVICES=3 python paired_images_data_gen_multiclient.py --connection --port 51012  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_toto --robot_dataset toto --target_robot UR5e --target_gripper Robotiq85Gripper --use_cam_pose_only --start_id 0 &
CUDA_VISIBLE_DEVICES=3 python paired_images_data_gen_multiclient.py --connection --port 51012  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_toto --robot_dataset toto --target_robot Jaco --target_gripper JacoThreeFingerGripper --use_cam_pose_only --start_id 0 &

sleep 10

CUDA_VISIBLE_DEVICES=2 python paired_images_data_gen_multiserver.py --connection --connection_num 3 --port 42711 --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_random_poses --source_robot Sawyer  --source_gripper RethinkGripper --start_id 0 &
CUDA_VISIBLE_DEVICES=2 python paired_images_data_gen_multiclient.py --connection --port 42711  --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_random_poses --target_robot Panda --target_gripper PandaGripper --start_id 0 &
CUDA_VISIBLE_DEVICES=2 python paired_images_data_gen_multiclient.py --connection --port 42711  --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_random_poses --target_robot Jaco --target_gripper JacoThreeFingerGripper --start_id 0 &
CUDA_VISIBLE_DEVICES=2 python paired_images_data_gen_multiclient.py --connection --port 42711  --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_random_poses --target_gripper Robotiq85Gripper --start_id 0 &

sleep 10


CUDA_VISIBLE_DEVICES=1 python paired_images_data_gen_multiserver.py --connection --connection_num 3 --port 42712 --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_random_poses --source_robot Sawyer  --source_gripper RethinkGripper --start_id 3000 &
CUDA_VISIBLE_DEVICES=1 python paired_images_data_gen_multiclient.py --connection --port 42712  --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_random_poses --target_robot Panda --target_gripper PandaGripper --start_id 3000 &
CUDA_VISIBLE_DEVICES=1 python paired_images_data_gen_multiclient.py --connection --port 42712  --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_random_poses --target_robot Jaco --target_gripper JacoThreeFingerGripper --start_id 3000 &
CUDA_VISIBLE_DEVICES=1 python paired_images_data_gen_multiclient.py --connection --port 42712  --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_random_poses --target_gripper Robotiq85Gripper --start_id 3000 &

sleep 10

CUDA_VISIBLE_DEVICES=0 python paired_images_data_gen_multiserver.py --connection --connection_num 3 --port 51015 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_furniture_bench --robot_dataset furniture_bench --source_robot Sawyer  --source_gripper RethinkGripper --use_cam_pose_only --start_id 3000 &
CUDA_VISIBLE_DEVICES=0 python paired_images_data_gen_multiclient.py --connection --port 51015  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_furniture_bench --robot_dataset furniture_bench --target_robot Panda --target_gripper PandaGripper --use_cam_pose_only --start_id 3000 &
CUDA_VISIBLE_DEVICES=0 python paired_images_data_gen_multiclient.py --connection --port 51015  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_furniture_bench --robot_dataset furniture_bench --target_robot UR5e --target_gripper Robotiq85Gripper --use_cam_pose_only --start_id 3000 &
CUDA_VISIBLE_DEVICES=0 python paired_images_data_gen_multiclient.py --connection --port 51015  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_furniture_bench --robot_dataset furniture_bench --target_robot Jaco --target_gripper JacoThreeFingerGripper --use_cam_pose_only --start_id 3000 

sleep 10




CUDA_VISIBLE_DEVICES=4 python paired_images_data_gen_multiserver.py --connection --connection_num 3 --port 51016 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_hydra --robot_dataset hydra --source_robot Sawyer  --source_gripper RethinkGripper --use_cam_pose_only --start_id 3000 &
CUDA_VISIBLE_DEVICES=4 python paired_images_data_gen_multiclient.py --connection --port 51016  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_hydra --robot_dataset hydra --target_robot Panda --target_gripper PandaGripper --use_cam_pose_only --start_id 3000 &
CUDA_VISIBLE_DEVICES=4 python paired_images_data_gen_multiclient.py --connection --port 51016  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_hydra --robot_dataset hydra --target_robot UR5e --target_gripper Robotiq85Gripper --use_cam_pose_only --start_id 3000 &
CUDA_VISIBLE_DEVICES=4 python paired_images_data_gen_multiclient.py --connection --port 51016  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_hydra --robot_dataset hydra --target_robot Jaco --target_gripper JacoThreeFingerGripper --use_cam_pose_only --start_id 3000 &

sleep 10

CUDA_VISIBLE_DEVICES=3 python paired_images_data_gen_multiserver.py --connection --connection_num 3 --port 51017 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_mirage --robot_dataset mirage --source_robot Sawyer  --source_gripper RethinkGripper --use_cam_pose_only --start_id 3000 &
CUDA_VISIBLE_DEVICES=3 python paired_images_data_gen_multiclient.py --connection --port 51017  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_mirage --robot_dataset mirage --target_robot Panda --target_gripper PandaGripper --use_cam_pose_only --start_id 3000 &
CUDA_VISIBLE_DEVICES=3 python paired_images_data_gen_multiclient.py --connection --port 51017  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_mirage --robot_dataset mirage --target_robot UR5e --target_gripper Robotiq85Gripper --use_cam_pose_only --start_id 3000 &
CUDA_VISIBLE_DEVICES=3 python paired_images_data_gen_multiclient.py --connection --port 51017  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_mirage --robot_dataset mirage --target_robot Jaco --target_gripper JacoThreeFingerGripper --use_cam_pose_only --start_id 3000 &

sleep 10

CUDA_VISIBLE_DEVICES=2 python paired_images_data_gen_multiserver.py --connection --connection_num 3 --port 51018 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_mutex --robot_dataset mutex --source_robot Sawyer  --source_gripper RethinkGripper --use_cam_pose_only --start_id 3000 &
CUDA_VISIBLE_DEVICES=2 python paired_images_data_gen_multiclient.py --connection --port 51018  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_mutex --robot_dataset mutex --target_robot Panda --target_gripper PandaGripper --use_cam_pose_only --start_id 3000 &
CUDA_VISIBLE_DEVICES=2 python paired_images_data_gen_multiclient.py --connection --port 51018  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_mutex --robot_dataset mutex --target_robot UR5e --target_gripper Robotiq85Gripper --use_cam_pose_only --start_id 3000 &
CUDA_VISIBLE_DEVICES=2 python paired_images_data_gen_multiclient.py --connection --port 51018  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_mutex --robot_dataset mutex --target_robot Jaco --target_gripper JacoThreeFingerGripper --use_cam_pose_only --start_id 3000 &

sleep 10

CUDA_VISIBLE_DEVICES=1 python paired_images_data_gen_multiserver.py --connection --connection_num 3 --port 51019 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_nyu_franka --robot_dataset nyu_franka --source_robot Sawyer  --source_gripper RethinkGripper --use_cam_pose_only --start_id 3000 &
CUDA_VISIBLE_DEVICES=1 python paired_images_data_gen_multiclient.py --connection --port 51019  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_nyu_franka --robot_dataset nyu_franka --target_robot Panda --target_gripper PandaGripper --use_cam_pose_only --start_id 3000 &
CUDA_VISIBLE_DEVICES=1 python paired_images_data_gen_multiclient.py --connection --port 51019  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_nyu_franka --robot_dataset nyu_franka --target_robot UR5e --target_gripper Robotiq85Gripper --use_cam_pose_only --start_id 3000 &
CUDA_VISIBLE_DEVICES=1 python paired_images_data_gen_multiclient.py --connection --port 51019  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_nyu_franka --robot_dataset nyu_franka --target_robot Jaco --target_gripper JacoThreeFingerGripper --use_cam_pose_only --start_id 3000 &

sleep 10

CUDA_VISIBLE_DEVICES=0 python paired_images_data_gen_multiserver.py --connection --connection_num 3 --port 51020 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_taco_play --robot_dataset taco_play --source_robot Sawyer  --source_gripper RethinkGripper --use_cam_pose_only --start_id 3000 &
CUDA_VISIBLE_DEVICES=0 python paired_images_data_gen_multiclient.py --connection --port 51020  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_taco_play --robot_dataset taco_play --target_robot Panda --target_gripper PandaGripper --use_cam_pose_only --start_id 3000 &
CUDA_VISIBLE_DEVICES=0 python paired_images_data_gen_multiclient.py --connection --port 51020  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_taco_play --robot_dataset taco_play --target_robot UR5e --target_gripper Robotiq85Gripper --use_cam_pose_only --start_id 3000 &
CUDA_VISIBLE_DEVICES=0 python paired_images_data_gen_multiclient.py --connection --port 51020  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_taco_play --robot_dataset taco_play --target_robot Jaco --target_gripper JacoThreeFingerGripper --use_cam_pose_only --start_id 3000 &

sleep 10

CUDA_VISIBLE_DEVICES=4 python paired_images_data_gen_multiserver.py --connection --connection_num 3 --port 51021 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_viola --robot_dataset viola --source_robot Sawyer  --source_gripper RethinkGripper --use_cam_pose_only --start_id 3000 &
CUDA_VISIBLE_DEVICES=4 python paired_images_data_gen_multiclient.py --connection --port 51021  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_viola --robot_dataset viola --target_robot Panda --target_gripper PandaGripper --use_cam_pose_only --start_id 3000 &
CUDA_VISIBLE_DEVICES=4 python paired_images_data_gen_multiclient.py --connection --port 51021  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_viola --robot_dataset viola --target_robot UR5e --target_gripper Robotiq85Gripper --use_cam_pose_only --start_id 3000 &
CUDA_VISIBLE_DEVICES=4 python paired_images_data_gen_multiclient.py --connection --port 51021  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_viola --robot_dataset viola --target_robot Jaco --target_gripper JacoThreeFingerGripper --use_cam_pose_only --start_id 3000 &

sleep 10

CUDA_VISIBLE_DEVICES=3 python paired_images_data_gen_multiserver.py --connection --connection_num 3 --port 42713 --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_random_poses --source_robot Sawyer  --source_gripper RethinkGripper --start_id 6000 &
CUDA_VISIBLE_DEVICES=3 python paired_images_data_gen_multiclient.py --connection --port 42713  --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_random_poses --target_robot Panda --target_gripper PandaGripper --start_id 6000 &
CUDA_VISIBLE_DEVICES=3 python paired_images_data_gen_multiclient.py --connection --port 42713  --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_random_poses --target_robot Jaco --target_gripper JacoThreeFingerGripper --start_id 6000 &
CUDA_VISIBLE_DEVICES=3 python paired_images_data_gen_multiclient.py --connection --port 42713  --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_random_poses --target_gripper Robotiq85Gripper --start_id 6000 &

sleep 10

CUDA_VISIBLE_DEVICES=2 python paired_images_data_gen_multiserver.py --connection --connection_num 3 --port 42714 --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_random_poses --source_robot Sawyer  --source_gripper RethinkGripper --start_id 9000 &
CUDA_VISIBLE_DEVICES=2 python paired_images_data_gen_multiclient.py --connection --port 42714  --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_random_poses --target_robot Panda --target_gripper PandaGripper --start_id 9000 &
CUDA_VISIBLE_DEVICES=2 python paired_images_data_gen_multiclient.py --connection --port 42714  --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_random_poses --target_robot Jaco --target_gripper JacoThreeFingerGripper --start_id 9000 &
CUDA_VISIBLE_DEVICES=2 python paired_images_data_gen_multiclient.py --connection --port 42714  --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_random_poses --target_gripper Robotiq85Gripper --start_id 9000 &

sleep 10


CUDA_VISIBLE_DEVICES=1 python paired_images_data_gen_multiserver.py --connection --connection_num 3 --port 42715 --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_random_poses --source_robot Sawyer  --source_gripper RethinkGripper --start_id 12000 &
CUDA_VISIBLE_DEVICES=1 python paired_images_data_gen_multiclient.py --connection --port 42715  --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_random_poses --target_robot Panda --target_gripper PandaGripper --start_id 12000 &
CUDA_VISIBLE_DEVICES=1 python paired_images_data_gen_multiclient.py --connection --port 42715  --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_random_poses --target_robot Jaco --target_gripper JacoThreeFingerGripper --start_id 12000 &
CUDA_VISIBLE_DEVICES=1 python paired_images_data_gen_multiclient.py --connection --port 42715  --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_random_poses --target_gripper Robotiq85Gripper --start_id 12000 &

sleep 10

CUDA_VISIBLE_DEVICES=0 python paired_images_data_gen_multiserver.py --connection --connection_num 3 --port 42716 --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_random_poses --source_robot Sawyer  --source_gripper RethinkGripper --start_id 15000 &
CUDA_VISIBLE_DEVICES=0 python paired_images_data_gen_multiclient.py --connection --port 42716  --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_random_poses --target_robot Panda --target_gripper PandaGripper --start_id 15000 &
CUDA_VISIBLE_DEVICES=0 python paired_images_data_gen_multiclient.py --connection --port 42716  --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_random_poses --target_robot Jaco --target_gripper JacoThreeFingerGripper --start_id 15000 &
CUDA_VISIBLE_DEVICES=0 python paired_images_data_gen_multiclient.py --connection --port 42716  --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_random_poses --target_gripper Robotiq85Gripper --start_id 15000 &

sleep 10

CUDA_VISIBLE_DEVICES=4 python paired_images_data_gen_multiserver.py --connection --connection_num 3 --port 42717 --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_random_poses --source_robot Sawyer  --source_gripper RethinkGripper --start_id 18000 &
CUDA_VISIBLE_DEVICES=4 python paired_images_data_gen_multiclient.py --connection --port 42717  --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_random_poses --target_robot Panda --target_gripper PandaGripper --start_id 18000 &
CUDA_VISIBLE_DEVICES=4 python paired_images_data_gen_multiclient.py --connection --port 42717  --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_random_poses --target_robot Jaco --target_gripper JacoThreeFingerGripper --start_id 18000 &
CUDA_VISIBLE_DEVICES=4 python paired_images_data_gen_multiclient.py --connection --port 42717  --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_random_poses --target_gripper Robotiq85Gripper --start_id 18000 &

sleep 10


CUDA_VISIBLE_DEVICES=3 python paired_images_data_gen_multiserver.py --connection --connection_num 3 --port 42718 --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_random_poses --source_robot Sawyer  --source_gripper RethinkGripper --start_id 21000 &
CUDA_VISIBLE_DEVICES=3 python paired_images_data_gen_multiclient.py --connection --port 42718  --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_random_poses --target_robot Panda --target_gripper PandaGripper --start_id 21000 &
CUDA_VISIBLE_DEVICES=3 python paired_images_data_gen_multiclient.py --connection --port 42718  --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_random_poses --target_robot Jaco --target_gripper JacoThreeFingerGripper --start_id 21000 &
CUDA_VISIBLE_DEVICES=3 python paired_images_data_gen_multiclient.py --connection --port 42718  --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_random_poses --target_gripper Robotiq85Gripper --start_id 21000 &

sleep 10

CUDA_VISIBLE_DEVICES=2 python paired_images_data_gen_multiserver.py --connection --connection_num 3 --port 51013 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_austin_buds --robot_dataset austin_buds --source_robot Sawyer  --source_gripper RethinkGripper --use_cam_pose_only --start_id 3000 &
CUDA_VISIBLE_DEVICES=2 python paired_images_data_gen_multiclient.py --connection --port 51013  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_austin_buds --robot_dataset austin_buds --target_robot Panda --target_gripper PandaGripper --use_cam_pose_only --start_id 3000 &
CUDA_VISIBLE_DEVICES=2 python paired_images_data_gen_multiclient.py --connection --port 51013  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_austin_buds --robot_dataset austin_buds --target_robot UR5e --target_gripper Robotiq85Gripper --use_cam_pose_only --start_id 3000 &
CUDA_VISIBLE_DEVICES=2 python paired_images_data_gen_multiclient.py --connection --port 51013  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_austin_buds --robot_dataset austin_buds --target_robot Jaco --target_gripper JacoThreeFingerGripper --use_cam_pose_only --start_id 3000 &

sleep 10

CUDA_VISIBLE_DEVICES=1 python paired_images_data_gen_multiserver.py --connection --connection_num 3 --port 51014 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_austin_sailor --robot_dataset austin_sailor --source_robot Sawyer  --source_gripper RethinkGripper --use_cam_pose_only --start_id 3000 &
CUDA_VISIBLE_DEVICES=1 python paired_images_data_gen_multiclient.py --connection --port 51014  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_austin_sailor --robot_dataset austin_sailor --target_robot Panda --target_gripper PandaGripper --use_cam_pose_only --start_id 3000 &
CUDA_VISIBLE_DEVICES=1 python paired_images_data_gen_multiclient.py --connection --port 51014  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_austin_sailor --robot_dataset austin_sailor --target_robot UR5e --target_gripper Robotiq85Gripper --use_cam_pose_only --start_id 3000 &
CUDA_VISIBLE_DEVICES=1 python paired_images_data_gen_multiclient.py --connection --port 51014  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/sawyer_franka_ur5_jaco/paired_images_austin_sailor --robot_dataset austin_sailor --target_robot Jaco --target_gripper JacoThreeFingerGripper --use_cam_pose_only --start_id 3000 &
