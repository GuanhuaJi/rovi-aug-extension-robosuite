CUDA_VISIBLE_DEVICES=0 python paired_images_data_gen_server.py --connection --port 50006 --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_pairs_all/paired_images_random_poses --start_id 0 &
CUDA_VISIBLE_DEVICES=0 python paired_images_data_gen_client.py --connection --port 50006 --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_pairs_all/paired_images_random_poses --start_id 0 &

CUDA_VISIBLE_DEVICES=1 python paired_images_data_gen_server.py --connection --port 50007 --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_pairs_all/paired_images_random_poses --start_id 1000 &
CUDA_VISIBLE_DEVICES=1 python paired_images_data_gen_client.py --connection --port 50007 --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_pairs_all/paired_images_random_poses --start_id 1000 &

CUDA_VISIBLE_DEVICES=2 python paired_images_data_gen_server.py --connection --port 50008 --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_pairs_all/paired_images_random_poses --start_id 2000 &
CUDA_VISIBLE_DEVICES=2 python paired_images_data_gen_client.py --connection --port 50008 --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_pairs_all/paired_images_random_poses --start_id 2000 &

CUDA_VISIBLE_DEVICES=3 python paired_images_data_gen_server.py --connection --port 50009 --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_pairs_all/paired_images_random_poses --start_id 3000 &
CUDA_VISIBLE_DEVICES=3 python paired_images_data_gen_client.py --connection --port 50009 --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_pairs_all/paired_images_random_poses --start_id 3000 &

CUDA_VISIBLE_DEVICES=4 python paired_images_data_gen_server.py --connection --port 50010 --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_pairs_all/paired_images_random_poses --start_id 4000 &
CUDA_VISIBLE_DEVICES=4 python paired_images_data_gen_client.py --connection --port 50010 --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_pairs_all/paired_images_random_poses --start_id 4000 &



CUDA_VISIBLE_DEVICES=0 python paired_images_data_gen_server.py --connection --port 50011 --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_pairs_all/paired_images_random_poses --start_id 5000 &
CUDA_VISIBLE_DEVICES=0 python paired_images_data_gen_client.py --connection --port 50011 --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_pairs_all/paired_images_random_poses --start_id 5000 &

CUDA_VISIBLE_DEVICES=1 python paired_images_data_gen_server.py --connection --port 50012 --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_pairs_all/paired_images_random_poses --start_id 6000 &
CUDA_VISIBLE_DEVICES=1 python paired_images_data_gen_client.py --connection --port 50012 --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_pairs_all/paired_images_random_poses --start_id 6000 &

CUDA_VISIBLE_DEVICES=2 python paired_images_data_gen_server.py --connection --port 50013 --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_pairs_all/paired_images_random_poses --start_id 7000 &
CUDA_VISIBLE_DEVICES=2 python paired_images_data_gen_client.py --connection --port 50013 --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_pairs_all/paired_images_random_poses --start_id 7000 &

CUDA_VISIBLE_DEVICES=3 python paired_images_data_gen_server.py --connection --port 50014 --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_pairs_all/paired_images_random_poses --start_id 8000 &
CUDA_VISIBLE_DEVICES=3 python paired_images_data_gen_client.py --connection --port 50014 --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_pairs_all/paired_images_random_poses --start_id 8000 &

CUDA_VISIBLE_DEVICES=4 python paired_images_data_gen_server.py --connection --port 50015 --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_pairs_all/paired_images_random_poses --start_id 9000 &
CUDA_VISIBLE_DEVICES=4 python paired_images_data_gen_client.py --connection --port 50015 --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_pairs_all/paired_images_random_poses --start_id 9000 

wait

CUDA_VISIBLE_DEVICES=0 python paired_images_data_gen_server.py --connection --port 50016 --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_pairs_all/paired_images_random_poses --start_id 10000 &
CUDA_VISIBLE_DEVICES=0 python paired_images_data_gen_client.py --connection --port 50016 --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_pairs_all/paired_images_random_poses --start_id 10000 &

CUDA_VISIBLE_DEVICES=1 python paired_images_data_gen_server.py --connection --port 50017 --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_pairs_all/paired_images_random_poses --start_id 11000 &
CUDA_VISIBLE_DEVICES=1 python paired_images_data_gen_client.py --connection --port 50017 --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_pairs_all/paired_images_random_poses --start_id 11000 &

CUDA_VISIBLE_DEVICES=2 python paired_images_data_gen_server.py --connection --port 50018 --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_pairs_all/paired_images_random_poses --start_id 12000 &
CUDA_VISIBLE_DEVICES=2 python paired_images_data_gen_client.py --connection --port 50018 --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_pairs_all/paired_images_random_poses --start_id 12000 &

CUDA_VISIBLE_DEVICES=3 python paired_images_data_gen_server.py --connection --port 50019 --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_pairs_all/paired_images_random_poses --start_id 13000 &
CUDA_VISIBLE_DEVICES=3 python paired_images_data_gen_client.py --connection --port 50019 --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_pairs_all/paired_images_random_poses --start_id 13000 &

CUDA_VISIBLE_DEVICES=4 python paired_images_data_gen_server.py --connection --port 50020 --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_pairs_all/paired_images_random_poses --start_id 14000 &
CUDA_VISIBLE_DEVICES=4 python paired_images_data_gen_client.py --connection --port 50020 --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_pairs_all/paired_images_random_poses --start_id 14000 &



CUDA_VISIBLE_DEVICES=0 python paired_images_data_gen_server.py --connection --port 50021 --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_pairs_all/paired_images_random_poses --start_id 15000 &
CUDA_VISIBLE_DEVICES=0 python paired_images_data_gen_client.py --connection --port 50021 --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_pairs_all/paired_images_random_poses --start_id 15000 &

CUDA_VISIBLE_DEVICES=1 python paired_images_data_gen_server.py --connection --port 50022 --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_pairs_all/paired_images_random_poses --start_id 16000 &
CUDA_VISIBLE_DEVICES=1 python paired_images_data_gen_client.py --connection --port 50022 --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_pairs_all/paired_images_random_poses --start_id 16000 &

CUDA_VISIBLE_DEVICES=2 python paired_images_data_gen_server.py --connection --port 50023 --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_pairs_all/paired_images_random_poses --start_id 17000 &
CUDA_VISIBLE_DEVICES=2 python paired_images_data_gen_client.py --connection --port 50023 --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_pairs_all/paired_images_random_poses --start_id 17000 &

CUDA_VISIBLE_DEVICES=3 python paired_images_data_gen_server.py --connection --port 50024 --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_pairs_all/paired_images_random_poses --start_id 18000 &
CUDA_VISIBLE_DEVICES=3 python paired_images_data_gen_client.py --connection --port 50024 --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_pairs_all/paired_images_random_poses --start_id 18000 &

CUDA_VISIBLE_DEVICES=4 python paired_images_data_gen_server.py --connection --port 50025 --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_pairs_all/paired_images_random_poses --start_id 19000 &
CUDA_VISIBLE_DEVICES=4 python paired_images_data_gen_client.py --connection --port 50025 --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_pairs_all/paired_images_random_poses --start_id 19000 &



CUDA_VISIBLE_DEVICES=0 python paired_images_data_gen_server.py --connection --port 50026 --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_pairs_all/paired_images_random_poses --start_id 20000 &
CUDA_VISIBLE_DEVICES=0 python paired_images_data_gen_client.py --connection --port 50026 --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_pairs_all/paired_images_random_poses --start_id 20000 &

CUDA_VISIBLE_DEVICES=1 python paired_images_data_gen_server.py --connection --port 50027 --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_pairs_all/paired_images_random_poses --start_id 21000 &
CUDA_VISIBLE_DEVICES=1 python paired_images_data_gen_client.py --connection --port 50027 --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_pairs_all/paired_images_random_poses --start_id 21000 &

CUDA_VISIBLE_DEVICES=2 python paired_images_data_gen_server.py --connection --port 50028 --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_pairs_all/paired_images_random_poses --start_id 22000 &
CUDA_VISIBLE_DEVICES=2 python paired_images_data_gen_client.py --connection --port 50028 --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_pairs_all/paired_images_random_poses --start_id 22000 &

CUDA_VISIBLE_DEVICES=3 python paired_images_data_gen_server.py --connection --port 50029 --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_pairs_all/paired_images_random_poses --start_id 23000 &
CUDA_VISIBLE_DEVICES=3 python paired_images_data_gen_client.py --connection --port 50029 --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_pairs_all/paired_images_random_poses --start_id 23000 &

CUDA_VISIBLE_DEVICES=4 python paired_images_data_gen_server.py --connection --port 50030 --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_pairs_all/paired_images_random_poses --start_id 24000 &
CUDA_VISIBLE_DEVICES=4 python paired_images_data_gen_client.py --connection --port 50030 --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_pairs_all/paired_images_random_poses --start_id 24000 

wait

CUDA_VISIBLE_DEVICES=0 python paired_images_data_gen_server.py --connection --port 50031 --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_pairs_all/paired_images_random_poses --start_id 25000 &
CUDA_VISIBLE_DEVICES=0 python paired_images_data_gen_client.py --connection --port 50031 --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_pairs_all/paired_images_random_poses --start_id 25000 &

CUDA_VISIBLE_DEVICES=1 python paired_images_data_gen_server.py --connection --port 50032 --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_pairs_all/paired_images_random_poses --start_id 26000 &
CUDA_VISIBLE_DEVICES=1 python paired_images_data_gen_client.py --connection --port 50032 --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_pairs_all/paired_images_random_poses --start_id 26000 &

CUDA_VISIBLE_DEVICES=2 python paired_images_data_gen_server.py --connection --port 50033 --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_pairs_all/paired_images_random_poses --start_id 27000 &
CUDA_VISIBLE_DEVICES=2 python paired_images_data_gen_client.py --connection --port 50033 --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_pairs_all/paired_images_random_poses --start_id 27000 &

CUDA_VISIBLE_DEVICES=3 python paired_images_data_gen_server.py --connection --port 50034 --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_pairs_all/paired_images_random_poses --start_id 28000 &
CUDA_VISIBLE_DEVICES=3 python paired_images_data_gen_client.py --connection --port 50034 --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_pairs_all/paired_images_random_poses --start_id 28000 &

CUDA_VISIBLE_DEVICES=4 python paired_images_data_gen_server.py --connection --port 50035 --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_pairs_all/paired_images_random_poses --start_id 29000 &
CUDA_VISIBLE_DEVICES=4 python paired_images_data_gen_client.py --connection --port 50035 --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_pairs_all/paired_images_random_poses --start_id 29000 &




CUDA_VISIBLE_DEVICES=0 python paired_images_data_gen_server.py --connection --port 50036 --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_pairs_all/paired_images_random_poses --start_id 30000 &
CUDA_VISIBLE_DEVICES=0 python paired_images_data_gen_client.py --connection --port 50036 --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_pairs_all/paired_images_random_poses --start_id 30000 &

CUDA_VISIBLE_DEVICES=1 python paired_images_data_gen_server.py --connection --port 50037 --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_pairs_all/paired_images_random_poses --start_id 31000 &
CUDA_VISIBLE_DEVICES=1 python paired_images_data_gen_client.py --connection --port 50037 --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_pairs_all/paired_images_random_poses --start_id 31000 &

CUDA_VISIBLE_DEVICES=2 python paired_images_data_gen_server.py --connection --port 50038 --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_pairs_all/paired_images_random_poses --start_id 32000 &
CUDA_VISIBLE_DEVICES=2 python paired_images_data_gen_client.py --connection --port 50038 --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_pairs_all/paired_images_random_poses --start_id 32000 &

CUDA_VISIBLE_DEVICES=3 python paired_images_data_gen_server.py --connection --port 50039 --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_pairs_all/paired_images_random_poses --start_id 33000 &
CUDA_VISIBLE_DEVICES=3 python paired_images_data_gen_client.py --connection --port 50039 --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_pairs_all/paired_images_random_poses --start_id 33000 &

CUDA_VISIBLE_DEVICES=4 python paired_images_data_gen_server.py --connection --port 50040 --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_pairs_all/paired_images_random_poses --start_id 34000 &
CUDA_VISIBLE_DEVICES=4 python paired_images_data_gen_client.py --connection --port 50040 --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_pairs_all/paired_images_random_poses --start_id 34000 &



CUDA_VISIBLE_DEVICES=0 python paired_images_data_gen_server.py --connection --port 50041 --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_pairs_all/paired_images_random_poses --start_id 35000 &
CUDA_VISIBLE_DEVICES=0 python paired_images_data_gen_client.py --connection --port 50041 --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_pairs_all/paired_images_random_poses --start_id 35000 &

CUDA_VISIBLE_DEVICES=1 python paired_images_data_gen_server.py --connection --port 50042 --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_pairs_all/paired_images_random_poses --start_id 36000 &
CUDA_VISIBLE_DEVICES=1 python paired_images_data_gen_client.py --connection --port 50042 --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_pairs_all/paired_images_random_poses --start_id 36000 &

CUDA_VISIBLE_DEVICES=2 python paired_images_data_gen_server.py --connection --port 50043 --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_pairs_all/paired_images_random_poses --start_id 37000 &
CUDA_VISIBLE_DEVICES=2 python paired_images_data_gen_client.py --connection --port 50043 --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_pairs_all/paired_images_random_poses --start_id 37000 &

CUDA_VISIBLE_DEVICES=3 python paired_images_data_gen_server.py --connection --port 50044 --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_pairs_all/paired_images_random_poses --start_id 38000 &
CUDA_VISIBLE_DEVICES=3 python paired_images_data_gen_client.py --connection --port 50044 --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_pairs_all/paired_images_random_poses --start_id 38000 &

CUDA_VISIBLE_DEVICES=4 python paired_images_data_gen_server.py --connection --port 50045 --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_pairs_all/paired_images_random_poses --start_id 39000 &
CUDA_VISIBLE_DEVICES=4 python paired_images_data_gen_client.py --connection --port 50045 --num_robot_poses 50000 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path data/franka_ur5_pairs_all/paired_images_random_poses --start_id 39000 &


# CUDA_VISIBLE_DEVICES=1 python paired_images_data_gen_server.py --connection --port 50017 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_viola_bowlplatefork --reference_ee_states_path /home/lawrence/xembody_followup/viola_dataset/ee_states.txt --start_id 420 &
# CUDA_VISIBLE_DEVICES=1 python paired_images_data_gen_client.py --connection --port 50017  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_viola_bowlplatefork --reference_ee_states_path /home/lawrence/xembody_followup/viola_dataset/ee_states.txt --start_id 420 &

# CUDA_VISIBLE_DEVICES=1 python paired_images_data_gen_server.py --connection --port 50018 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_viola_bowlplatefork --reference_ee_states_path /home/lawrence/xembody_followup/viola_dataset/ee_states.txt --start_id 1420 &
# CUDA_VISIBLE_DEVICES=1 python paired_images_data_gen_client.py --connection --port 50018  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_viola_bowlplatefork --reference_ee_states_path /home/lawrence/xembody_followup/viola_dataset/ee_states.txt --start_id 1420 &

# CUDA_VISIBLE_DEVICES=1 python paired_images_data_gen_server.py --connection --port 50019 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_viola_bowlplatefork --reference_ee_states_path /home/lawrence/xembody_followup/viola_dataset/ee_states.txt --start_id 2420 &
# CUDA_VISIBLE_DEVICES=1 python paired_images_data_gen_client.py --connection --port 50019  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_viola_bowlplatefork --reference_ee_states_path /home/lawrence/xembody_followup/viola_dataset/ee_states.txt --start_id 2420 &

# CUDA_VISIBLE_DEVICES=1 python paired_images_data_gen_server.py --connection --port 50020 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_viola_bowlplatefork --reference_ee_states_path /home/lawrence/xembody_followup/viola_dataset/ee_states.txt --start_id 3420 &
# CUDA_VISIBLE_DEVICES=1 python paired_images_data_gen_client.py --connection --port 50020  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_viola_bowlplatefork --reference_ee_states_path /home/lawrence/xembody_followup/viola_dataset/ee_states.txt --start_id 3420 &

# CUDA_VISIBLE_DEVICES=2 python paired_images_data_gen_server.py --connection --port 50021 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_viola_bowlplatefork --reference_ee_states_path /home/lawrence/xembody_followup/viola_dataset/ee_states.txt --start_id 4420 &
# CUDA_VISIBLE_DEVICES=2 python paired_images_data_gen_client.py --connection --port 50021  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_viola_bowlplatefork --reference_ee_states_path /home/lawrence/xembody_followup/viola_dataset/ee_states.txt --start_id 4420 &

# CUDA_VISIBLE_DEVICES=2 python paired_images_data_gen_server.py --connection --port 50022 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_viola_bowlplatefork --reference_ee_states_path /home/lawrence/xembody_followup/viola_dataset/ee_states.txt --start_id 5420 &
# CUDA_VISIBLE_DEVICES=2 python paired_images_data_gen_client.py --connection --port 50022  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_viola_bowlplatefork --reference_ee_states_path /home/lawrence/xembody_followup/viola_dataset/ee_states.txt --start_id 5420 &

# CUDA_VISIBLE_DEVICES=2 python paired_images_data_gen_server.py --connection --port 50023 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_viola_bowlplatefork --reference_ee_states_path /home/lawrence/xembody_followup/viola_dataset/ee_states.txt --start_id 6420 &
# CUDA_VISIBLE_DEVICES=2 python paired_images_data_gen_client.py --connection --port 50023  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_viola_bowlplatefork --reference_ee_states_path /home/lawrence/xembody_followup/viola_dataset/ee_states.txt --start_id 6420 &

# CUDA_VISIBLE_DEVICES=2 python paired_images_data_gen_server.py --connection --port 50024 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_viola_bowlplatefork --reference_ee_states_path /home/lawrence/xembody_followup/viola_dataset/ee_states.txt --start_id 7420 &
# CUDA_VISIBLE_DEVICES=2 python paired_images_data_gen_client.py --connection --port 50024  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_viola_bowlplatefork --reference_ee_states_path /home/lawrence/xembody_followup/viola_dataset/ee_states.txt --start_id 7420 &

# CUDA_VISIBLE_DEVICES=3 python paired_images_data_gen_server.py --connection --port 50025 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_viola_bowlplatefork --reference_ee_states_path /home/lawrence/xembody_followup/viola_dataset/ee_states.txt --start_id 8420 &
# CUDA_VISIBLE_DEVICES=3 python paired_images_data_gen_client.py --connection --port 50025  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_viola_bowlplatefork --reference_ee_states_path /home/lawrence/xembody_followup/viola_dataset/ee_states.txt --start_id 8420 &

# CUDA_VISIBLE_DEVICES=3 python paired_images_data_gen_server.py --connection --port 50026 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_viola_bowlplatefork --reference_ee_states_path /home/lawrence/xembody_followup/viola_dataset/ee_states.txt --start_id 9420 &
# CUDA_VISIBLE_DEVICES=3 python paired_images_data_gen_client.py --connection --port 50026  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_viola_bowlplatefork --reference_ee_states_path /home/lawrence/xembody_followup/viola_dataset/ee_states.txt --start_id 9420 &

# CUDA_VISIBLE_DEVICES=3 python paired_images_data_gen_server.py --connection --port 50027 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_viola_bowlplatefork --reference_ee_states_path /home/lawrence/xembody_followup/viola_dataset/ee_states.txt --start_id 10420 &
# CUDA_VISIBLE_DEVICES=3 python paired_images_data_gen_client.py --connection --port 50027  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_viola_bowlplatefork --reference_ee_states_path /home/lawrence/xembody_followup/viola_dataset/ee_states.txt --start_id 10420 &

# CUDA_VISIBLE_DEVICES=3 python paired_images_data_gen_server.py --connection --port 50028 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_viola_bowlplatefork --reference_ee_states_path /home/lawrence/xembody_followup/viola_dataset/ee_states.txt --start_id 11420 &
# CUDA_VISIBLE_DEVICES=3 python paired_images_data_gen_client.py --connection --port 50028  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_viola_bowlplatefork --reference_ee_states_path /home/lawrence/xembody_followup/viola_dataset/ee_states.txt --start_id 11420 &

# CUDA_VISIBLE_DEVICES=4 python paired_images_data_gen_server.py --connection --port 50029 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_viola_bowlplatefork --reference_ee_states_path /home/lawrence/xembody_followup/viola_dataset/ee_states.txt --start_id 12420 &
# CUDA_VISIBLE_DEVICES=4 python paired_images_data_gen_client.py --connection --port 50029  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_viola_bowlplatefork --reference_ee_states_path /home/lawrence/xembody_followup/viola_dataset/ee_states.txt --start_id 12420 &

# CUDA_VISIBLE_DEVICES=4 python paired_images_data_gen_server.py --connection --port 50030 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_viola_bowlplatefork --reference_ee_states_path /home/lawrence/xembody_followup/viola_dataset/ee_states.txt --start_id 13420 &
# CUDA_VISIBLE_DEVICES=4 python paired_images_data_gen_client.py --connection --port 50030  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_viola_bowlplatefork --reference_ee_states_path /home/lawrence/xembody_followup/viola_dataset/ee_states.txt --start_id 13420 &

# CUDA_VISIBLE_DEVICES=1 python paired_images_data_gen_server.py --connection --port 50016 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_mirage_franka_gripper --reference_joint_angles_path /home/lawrence/xembody_followup/mirage_data/joint_states.txt --start_id 0 --source_gripper PandaGripper &
# CUDA_VISIBLE_DEVICES=1 python paired_images_data_gen_client.py --connection --port 50016  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_mirage_franka_gripper --reference_joint_angles_path /home/lawrence/xembody_followup/mirage_data/joint_states.txt --start_id 0 --target_gripper Robotiq85Gripper &

# CUDA_VISIBLE_DEVICES=1 python paired_images_data_gen_server.py --connection --port 50018 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_mirage_franka_gripper --reference_joint_angles_path /home/lawrence/xembody_followup/mirage_data/joint_states.txt --start_id 1000 --source_gripper PandaGripper &
# CUDA_VISIBLE_DEVICES=1 python paired_images_data_gen_client.py --connection --port 50018  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_mirage_franka_gripper --reference_joint_angles_path /home/lawrence/xembody_followup/mirage_data/joint_states.txt --start_id 1000 --target_gripper Robotiq85Gripper &

# CUDA_VISIBLE_DEVICES=1 python paired_images_data_gen_server.py --connection --port 50019 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_mirage_franka_gripper --reference_joint_angles_path /home/lawrence/xembody_followup/mirage_data/joint_states.txt --start_id 2000 --source_gripper PandaGripper &
# CUDA_VISIBLE_DEVICES=1 python paired_images_data_gen_client.py --connection --port 50019  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_mirage_franka_gripper --reference_joint_angles_path /home/lawrence/xembody_followup/mirage_data/joint_states.txt --start_id 2000 --target_gripper Robotiq85Gripper &

# CUDA_VISIBLE_DEVICES=1 python paired_images_data_gen_server.py --connection --port 50020 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_mirage_franka_gripper --reference_joint_angles_path /home/lawrence/xembody_followup/mirage_data/joint_states.txt --start_id 3000 --source_gripper PandaGripper &
# CUDA_VISIBLE_DEVICES=1 python paired_images_data_gen_client.py --connection --port 50020  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_mirage_franka_gripper --reference_joint_angles_path /home/lawrence/xembody_followup/mirage_data/joint_states.txt --start_id 3000 --target_gripper Robotiq85Gripper &

# CUDA_VISIBLE_DEVICES=2 python paired_images_data_gen_server.py --connection --port 50021 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_mirage_franka_gripper --reference_joint_angles_path /home/lawrence/xembody_followup/mirage_data/joint_states.txt --start_id 4000 --source_gripper PandaGripper &
# CUDA_VISIBLE_DEVICES=2 python paired_images_data_gen_client.py --connection --port 50021  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_mirage_franka_gripper --reference_joint_angles_path /home/lawrence/xembody_followup/mirage_data/joint_states.txt --start_id 4000 --target_gripper Robotiq85Gripper &

# CUDA_VISIBLE_DEVICES=2 python paired_images_data_gen_server.py --connection --port 50022 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_mirage_franka_gripper --reference_joint_angles_path /home/lawrence/xembody_followup/mirage_data/joint_states.txt --start_id 5000 --source_gripper PandaGripper &
# CUDA_VISIBLE_DEVICES=2 python paired_images_data_gen_client.py --connection --port 50022  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_mirage_franka_gripper --reference_joint_angles_path /home/lawrence/xembody_followup/mirage_data/joint_states.txt --start_id 5000 --target_gripper Robotiq85Gripper &

# CUDA_VISIBLE_DEVICES=2 python paired_images_data_gen_server.py --connection --port 50023 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_mirage_franka_gripper --reference_joint_angles_path /home/lawrence/xembody_followup/mirage_data/joint_states.txt --start_id 6000 --source_gripper PandaGripper &
# CUDA_VISIBLE_DEVICES=2 python paired_images_data_gen_client.py --connection --port 50023  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_mirage_franka_gripper --reference_joint_angles_path /home/lawrence/xembody_followup/mirage_data/joint_states.txt --start_id 6000 --target_gripper Robotiq85Gripper &

# CUDA_VISIBLE_DEVICES=2 python paired_images_data_gen_server.py --connection --port 50024 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_mirage_franka_gripper --reference_joint_angles_path /home/lawrence/xembody_followup/mirage_data/joint_states.txt --start_id 7000 --source_gripper PandaGripper &
# CUDA_VISIBLE_DEVICES=2 python paired_images_data_gen_client.py --connection --port 50024  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_mirage_franka_gripper --reference_joint_angles_path /home/lawrence/xembody_followup/mirage_data/joint_states.txt --start_id 7000 --target_gripper Robotiq85Gripper &

# CUDA_VISIBLE_DEVICES=3 python paired_images_data_gen_server.py --connection --port 50025 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_mirage_franka_gripper --reference_joint_angles_path /home/lawrence/xembody_followup/mirage_data/joint_states.txt --start_id 8000 --source_gripper PandaGripper &
# CUDA_VISIBLE_DEVICES=3 python paired_images_data_gen_client.py --connection --port 50025  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_mirage_franka_gripper --reference_joint_angles_path /home/lawrence/xembody_followup/mirage_data/joint_states.txt --start_id 8000 --target_gripper Robotiq85Gripper &

# CUDA_VISIBLE_DEVICES=0 python paired_images_data_gen_server.py --connection --port 50026 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_mirage_franka_gripper --reference_joint_angles_path /home/lawrence/xembody_followup/mirage_data/joint_states.txt --start_id 9000 --source_gripper PandaGripper &
# CUDA_VISIBLE_DEVICES=0 python paired_images_data_gen_client.py --connection --port 50026  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_mirage_franka_gripper --reference_joint_angles_path /home/lawrence/xembody_followup/mirage_data/joint_states.txt --start_id 9000 --target_gripper Robotiq85Gripper &

# CUDA_VISIBLE_DEVICES=3 python paired_images_data_gen_server.py --connection --port 50027 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_mirage_franka_gripper --reference_joint_angles_path /home/lawrence/xembody_followup/mirage_data/joint_states.txt --start_id 10000 --source_gripper PandaGripper &
# CUDA_VISIBLE_DEVICES=3 python paired_images_data_gen_client.py --connection --port 50027  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_mirage_franka_gripper --reference_joint_angles_path /home/lawrence/xembody_followup/mirage_data/joint_states.txt --start_id 10000 --target_gripper Robotiq85Gripper &

# CUDA_VISIBLE_DEVICES=3 python paired_images_data_gen_server.py --connection --port 50028 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_mirage_franka_gripper --reference_joint_angles_path /home/lawrence/xembody_followup/mirage_data/joint_states.txt --start_id 11000 --source_gripper PandaGripper &
# CUDA_VISIBLE_DEVICES=3 python paired_images_data_gen_client.py --connection --port 50028  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_mirage_franka_gripper --reference_joint_angles_path /home/lawrence/xembody_followup/mirage_data/joint_states.txt --start_id 11000 --target_gripper Robotiq85Gripper &

# CUDA_VISIBLE_DEVICES=4 python paired_images_data_gen_server.py --connection --port 50029 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_mirage_franka_gripper --reference_joint_angles_path /home/lawrence/xembody_followup/mirage_data/joint_states.txt --start_id 12000 --source_gripper PandaGripper &
# CUDA_VISIBLE_DEVICES=4 python paired_images_data_gen_client.py --connection --port 50029  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_mirage_franka_gripper --reference_joint_angles_path /home/lawrence/xembody_followup/mirage_data/joint_states.txt --start_id 12000 --target_gripper Robotiq85Gripper &

# CUDA_VISIBLE_DEVICES=0 python paired_images_data_gen_server.py --connection --port 50030 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_mirage_franka_gripper --reference_joint_angles_path /home/lawrence/xembody_followup/mirage_data/joint_states.txt --start_id 13000 --source_gripper PandaGripper &
# CUDA_VISIBLE_DEVICES=0 python paired_images_data_gen_client.py --connection --port 50030  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_mirage_franka_gripper --reference_joint_angles_path /home/lawrence/xembody_followup/mirage_data/joint_states.txt --start_id 13000 --target_gripper Robotiq85Gripper &

# CUDA_VISIBLE_DEVICES=4 python paired_images_data_gen_server.py --connection --port 50031 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_mirage_franka_gripper --reference_joint_angles_path /home/lawrence/xembody_followup/mirage_data/joint_states.txt --start_id 14000 --source_gripper PandaGripper &
# CUDA_VISIBLE_DEVICES=4 python paired_images_data_gen_client.py --connection --port 50031  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_mirage_franka_gripper --reference_joint_angles_path /home/lawrence/xembody_followup/mirage_data/joint_states.txt --start_id 14000 --target_gripper Robotiq85Gripper &

# CUDA_VISIBLE_DEVICES=4 python paired_images_data_gen_server.py --connection --port 50032 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_mirage_franka_gripper --reference_joint_angles_path /home/lawrence/xembody_followup/mirage_data/joint_states.txt --start_id 15000 --source_gripper PandaGripper &
# CUDA_VISIBLE_DEVICES=4 python paired_images_data_gen_client.py --connection --port 50032  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_mirage_franka_gripper --reference_joint_angles_path /home/lawrence/xembody_followup/mirage_data/joint_states.txt --start_id 15000 --target_gripper Robotiq85Gripper &

# CUDA_VISIBLE_DEVICES=4 python paired_images_data_gen_server.py --connection --port 50033 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_mirage_franka_gripper --reference_joint_angles_path /home/lawrence/xembody_followup/mirage_data/joint_states.txt --start_id 16000 --source_gripper PandaGripper &
# CUDA_VISIBLE_DEVICES=4 python paired_images_data_gen_client.py --connection --port 50033  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_mirage_franka_gripper --reference_joint_angles_path /home/lawrence/xembody_followup/mirage_data/joint_states.txt --start_id 16000 --target_gripper Robotiq85Gripper &

# CUDA_VISIBLE_DEVICES=4 python paired_images_data_gen_server.py --connection --port 50034 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_mirage_robotiq_gripper --reference_joint_angles_path /home/lawrence/xembody_followup/mirage_data/joint_states.txt --start_id 17000 --source_gripper Robotiq85Gripper &
# CUDA_VISIBLE_DEVICES=4 python paired_images_data_gen_client.py --connection --port 50034  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_mirage_robotiq_gripper --reference_joint_angles_path /home/lawrence/xembody_followup/mirage_data/joint_states.txt --start_id 17000 --target_gripper Robotiq85Gripper &

# CUDA_VISIBLE_DEVICES=1 python paired_images_data_gen_server.py --connection --port 50035 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_mirage_robotiq_gripper --reference_joint_angles_path /home/lawrence/xembody_followup/mirage_data/joint_states.txt --start_id 18000 --source_gripper Robotiq85Gripper &
# CUDA_VISIBLE_DEVICES=1 python paired_images_data_gen_client.py --connection --port 50035  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_mirage_robotiq_gripper --reference_joint_angles_path /home/lawrence/xembody_followup/mirage_data/joint_states.txt --start_id 18000 --target_gripper Robotiq85Gripper &

# CUDA_VISIBLE_DEVICES=2 python paired_images_data_gen_server.py --connection --port 50036 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_mirage_robotiq_gripper --reference_joint_angles_path /home/lawrence/xembody_followup/mirage_data/joint_states.txt --start_id 19000 --source_gripper Robotiq85Gripper &
# CUDA_VISIBLE_DEVICES=2 python paired_images_data_gen_client.py --connection --port 50036  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_mirage_robotiq_gripper --reference_joint_angles_path /home/lawrence/xembody_followup/mirage_data/joint_states.txt --start_id 19000 --target_gripper Robotiq85Gripper &

# CUDA_VISIBLE_DEVICES=3 python paired_images_data_gen_server.py --connection --port 50037 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_mirage_robotiq_gripper --reference_joint_angles_path /home/lawrence/xembody_followup/mirage_data/joint_states.txt --start_id 20000 --source_gripper Robotiq85Gripper &
# CUDA_VISIBLE_DEVICES=3 python paired_images_data_gen_client.py --connection --port 50037  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_mirage_robotiq_gripper --reference_joint_angles_path /home/lawrence/xembody_followup/mirage_data/joint_states.txt --start_id 20000 --target_gripper Robotiq85Gripper &

# CUDA_VISIBLE_DEVICES=0 python paired_images_data_gen_server.py --connection --port 50038 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_mirage_robotiq_gripper --reference_joint_angles_path /home/lawrence/xembody_followup/mirage_data/joint_states.txt --start_id 21000 --source_gripper Robotiq85Gripper &
# CUDA_VISIBLE_DEVICES=0 python paired_images_data_gen_client.py --connection --port 50038  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_mirage_robotiq_gripper --reference_joint_angles_path /home/lawrence/xembody_followup/mirage_data/joint_states.txt --start_id 21000 --target_gripper Robotiq85Gripper &

# CUDA_VISIBLE_DEVICES=0 python paired_images_data_gen_server.py --connection --port 50039 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_mirage_robotiq_gripper --reference_joint_angles_path /home/lawrence/xembody_followup/mirage_data/joint_states.txt --start_id 22000 --source_gripper Robotiq85Gripper &
# CUDA_VISIBLE_DEVICES=0 python paired_images_data_gen_client.py --connection --port 50039  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_mirage_robotiq_gripper --reference_joint_angles_path /home/lawrence/xembody_followup/mirage_data/joint_states.txt --start_id 22000 --target_gripper Robotiq85Gripper &

# CUDA_VISIBLE_DEVICES=0 python paired_images_data_gen_server.py --connection --port 50040 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_mirage_robotiq_gripper --reference_joint_angles_path /home/lawrence/xembody_followup/mirage_data/joint_states.txt --start_id 23000 --source_gripper Robotiq85Gripper &
# CUDA_VISIBLE_DEVICES=0 python paired_images_data_gen_client.py --connection --port 50040  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_mirage_robotiq_gripper --reference_joint_angles_path /home/lawrence/xembody_followup/mirage_data/joint_states.txt --start_id 23000 --target_gripper Robotiq85Gripper &

# CUDA_VISIBLE_DEVICES=1 python paired_images_data_gen_server.py --connection --port 50041 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_mirage_robotiq_gripper --reference_joint_angles_path /home/lawrence/xembody_followup/mirage_data/joint_states.txt --start_id 24000 --source_gripper Robotiq85Gripper &
# CUDA_VISIBLE_DEVICES=1 python paired_images_data_gen_client.py --connection --port 50041  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_mirage_robotiq_gripper --reference_joint_angles_path /home/lawrence/xembody_followup/mirage_data/joint_states.txt --start_id 24000 --target_gripper Robotiq85Gripper &

# CUDA_VISIBLE_DEVICES=2 python paired_images_data_gen_server.py --connection --port 50042 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_mirage_robotiq_gripper --reference_joint_angles_path /home/lawrence/xembody_followup/mirage_data/joint_states.txt --start_id 25000 --source_gripper Robotiq85Gripper &
# CUDA_VISIBLE_DEVICES=2 python paired_images_data_gen_client.py --connection --port 50042  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_mirage_robotiq_gripper --reference_joint_angles_path /home/lawrence/xembody_followup/mirage_data/joint_states.txt --start_id 25000 --target_gripper Robotiq85Gripper &

# CUDA_VISIBLE_DEVICES=3 python paired_images_data_gen_server.py --connection --port 50043 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_mirage_robotiq_gripper --reference_joint_angles_path /home/lawrence/xembody_followup/mirage_data/joint_states.txt --start_id 26000 --source_gripper Robotiq85Gripper &
# CUDA_VISIBLE_DEVICES=3 python paired_images_data_gen_client.py --connection --port 50043  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_mirage_robotiq_gripper --reference_joint_angles_path /home/lawrence/xembody_followup/mirage_data/joint_states.txt --start_id 26000 --target_gripper Robotiq85Gripper &

# CUDA_VISIBLE_DEVICES=4 python paired_images_data_gen_server.py --connection --port 50044 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_mirage_robotiq_gripper --reference_joint_angles_path /home/lawrence/xembody_followup/mirage_data/joint_states.txt --start_id 27000 --source_gripper Robotiq85Gripper &
# CUDA_VISIBLE_DEVICES=4 python paired_images_data_gen_client.py --connection --port 50044  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_mirage_robotiq_gripper --reference_joint_angles_path /home/lawrence/xembody_followup/mirage_data/joint_states.txt --start_id 27000 --target_gripper Robotiq85Gripper &

# CUDA_VISIBLE_DEVICES=3 python paired_images_data_gen_server.py --connection --port 50045 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_mirage_robotiq_gripper --reference_joint_angles_path /home/lawrence/xembody_followup/mirage_data/joint_states.txt --start_id 28000 --source_gripper Robotiq85Gripper &
# CUDA_VISIBLE_DEVICES=3 python paired_images_data_gen_client.py --connection --port 50045  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_mirage_robotiq_gripper --reference_joint_angles_path /home/lawrence/xembody_followup/mirage_data/joint_states.txt --start_id 28000 --target_gripper Robotiq85Gripper &

# CUDA_VISIBLE_DEVICES=4 python paired_images_data_gen_server.py --connection --port 50046 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_mirage_robotiq_gripper --reference_joint_angles_path /home/lawrence/xembody_followup/mirage_data/joint_states.txt --start_id 29000 --source_gripper Robotiq85Gripper &
# CUDA_VISIBLE_DEVICES=4 python paired_images_data_gen_client.py --connection --port 50046  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_mirage_robotiq_gripper --reference_joint_angles_path /home/lawrence/xembody_followup/mirage_data/joint_states.txt --start_id 29000 --target_gripper Robotiq85Gripper &

# CUDA_VISIBLE_DEVICES=1 python paired_images_data_gen_server.py --connection --port 50047 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_mirage_robotiq_gripper --reference_joint_angles_path /home/lawrence/xembody_followup/mirage_data/joint_states.txt --start_id 30000 --source_gripper Robotiq85Gripper &
# CUDA_VISIBLE_DEVICES=1 python paired_images_data_gen_client.py --connection --port 50047  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_mirage_robotiq_gripper --reference_joint_angles_path /home/lawrence/xembody_followup/mirage_data/joint_states.txt --start_id 30000 --target_gripper Robotiq85Gripper &

# CUDA_VISIBLE_DEVICES=1 python paired_images_data_gen_server.py --connection --port 50048 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_mirage_robotiq_gripper --reference_joint_angles_path /home/lawrence/xembody_followup/mirage_data/joint_states.txt --start_id 31000 --source_gripper Robotiq85Gripper &
# CUDA_VISIBLE_DEVICES=1 python paired_images_data_gen_client.py --connection --port 50048  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_mirage_robotiq_gripper --reference_joint_angles_path /home/lawrence/xembody_followup/mirage_data/joint_states.txt --start_id 31000 --target_gripper Robotiq85Gripper &

