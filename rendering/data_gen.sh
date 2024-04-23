# CUDA_VISIBLE_DEVICES=0 python paired_images_data_gen_server.py --connection --port 50006 --num_robot_poses 5000 --num_cam_poses_per_robot_pose 50 --save_paired_images_folder_path paired_images_1 &
# CUDA_VISIBLE_DEVICES=0 python paired_images_data_gen_client.py --connection --port 50006 --num_robot_poses 5000 --num_cam_poses_per_robot_pose 50 --save_paired_images_folder_path paired_images_1 &

# CUDA_VISIBLE_DEVICES=0 python paired_images_data_gen_server.py --connection --port 50007 --num_robot_poses 5000 --num_cam_poses_per_robot_pose 50 --save_paired_images_folder_path paired_images_2 &
# CUDA_VISIBLE_DEVICES=0 python paired_images_data_gen_client.py --connection --port 50007 --num_robot_poses 5000 --num_cam_poses_per_robot_pose 50 --save_paired_images_folder_path paired_images_2 &

# CUDA_VISIBLE_DEVICES=1 python paired_images_data_gen_server.py --connection --port 50008 --num_robot_poses 5000 --num_cam_poses_per_robot_pose 50 --save_paired_images_folder_path paired_images_3 &
# CUDA_VISIBLE_DEVICES=1 python paired_images_data_gen_client.py --connection --port 50008 --num_robot_poses 5000 --num_cam_poses_per_robot_pose 50 --save_paired_images_folder_path paired_images_3 &

# CUDA_VISIBLE_DEVICES=1 python paired_images_data_gen_server.py --connection --port 50009 --num_robot_poses 5000 --num_cam_poses_per_robot_pose 50 --save_paired_images_folder_path paired_images_4 &
# CUDA_VISIBLE_DEVICES=1 python paired_images_data_gen_client.py --connection --port 50009 --num_robot_poses 5000 --num_cam_poses_per_robot_pose 50 --save_paired_images_folder_path paired_images_4 &

# CUDA_VISIBLE_DEVICES=2 python paired_images_data_gen_server.py --connection --port 50010 --num_robot_poses 5000 --num_cam_poses_per_robot_pose 50 --save_paired_images_folder_path paired_images_5 &
# CUDA_VISIBLE_DEVICES=2 python paired_images_data_gen_client.py --connection --port 50010 --num_robot_poses 5000 --num_cam_poses_per_robot_pose 50 --save_paired_images_folder_path paired_images_5 &

# CUDA_VISIBLE_DEVICES=2 python paired_images_data_gen_server.py --connection --port 50011 --num_robot_poses 5000 --num_cam_poses_per_robot_pose 50 --save_paired_images_folder_path paired_images_6 &
# CUDA_VISIBLE_DEVICES=2 python paired_images_data_gen_client.py --connection --port 50011 --num_robot_poses 5000 --num_cam_poses_per_robot_pose 50 --save_paired_images_folder_path paired_images_6 &

# CUDA_VISIBLE_DEVICES=3 python paired_images_data_gen_server.py --connection --port 50012 --num_robot_poses 5000 --num_cam_poses_per_robot_pose 50 --save_paired_images_folder_path paired_images_7 &
# CUDA_VISIBLE_DEVICES=3 python paired_images_data_gen_client.py --connection --port 50012 --num_robot_poses 5000 --num_cam_poses_per_robot_pose 50 --save_paired_images_folder_path paired_images_7 &

# CUDA_VISIBLE_DEVICES=4 python paired_images_data_gen_server.py --connection --port 50013 --num_robot_poses 5000 --num_cam_poses_per_robot_pose 50 --save_paired_images_folder_path paired_images_8 &
# CUDA_VISIBLE_DEVICES=4 python paired_images_data_gen_client.py --connection --port 50013 --num_robot_poses 5000 --num_cam_poses_per_robot_pose 50 --save_paired_images_folder_path paired_images_8 &


CUDA_VISIBLE_DEVICES=1 python paired_images_data_gen_server.py --connection --port 50017 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_viola_bowlplatefork --reference_joint_angles_path /home/lawrence/xembody_followup/viola_dataset/ee_states.txt --start_id 420 &
CUDA_VISIBLE_DEVICES=1 python paired_images_data_gen_client.py --connection --port 50017  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_viola_bowlplatefork --reference_joint_angles_path /home/lawrence/xembody_followup/viola_dataset/ee_states.txt --start_id 420 &

CUDA_VISIBLE_DEVICES=1 python paired_images_data_gen_server.py --connection --port 50018 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_viola_bowlplatefork --reference_joint_angles_path /home/lawrence/xembody_followup/viola_dataset/ee_states.txt --start_id 1420 &
CUDA_VISIBLE_DEVICES=1 python paired_images_data_gen_client.py --connection --port 50018  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_viola_bowlplatefork --reference_joint_angles_path /home/lawrence/xembody_followup/viola_dataset/ee_states.txt --start_id 1420 &

CUDA_VISIBLE_DEVICES=1 python paired_images_data_gen_server.py --connection --port 50019 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_viola_bowlplatefork --reference_joint_angles_path /home/lawrence/xembody_followup/viola_dataset/ee_states.txt --start_id 2420 &
CUDA_VISIBLE_DEVICES=1 python paired_images_data_gen_client.py --connection --port 50019  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_viola_bowlplatefork --reference_joint_angles_path /home/lawrence/xembody_followup/viola_dataset/ee_states.txt --start_id 2420 &

CUDA_VISIBLE_DEVICES=1 python paired_images_data_gen_server.py --connection --port 50020 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_viola_bowlplatefork --reference_joint_angles_path /home/lawrence/xembody_followup/viola_dataset/ee_states.txt --start_id 3420 &
CUDA_VISIBLE_DEVICES=1 python paired_images_data_gen_client.py --connection --port 50020  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_viola_bowlplatefork --reference_joint_angles_path /home/lawrence/xembody_followup/viola_dataset/ee_states.txt --start_id 3420 &

CUDA_VISIBLE_DEVICES=2 python paired_images_data_gen_server.py --connection --port 50021 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_viola_bowlplatefork --reference_joint_angles_path /home/lawrence/xembody_followup/viola_dataset/ee_states.txt --start_id 4420 &
CUDA_VISIBLE_DEVICES=2 python paired_images_data_gen_client.py --connection --port 50021  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_viola_bowlplatefork --reference_joint_angles_path /home/lawrence/xembody_followup/viola_dataset/ee_states.txt --start_id 4420 &

CUDA_VISIBLE_DEVICES=2 python paired_images_data_gen_server.py --connection --port 50022 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_viola_bowlplatefork --reference_joint_angles_path /home/lawrence/xembody_followup/viola_dataset/ee_states.txt --start_id 5420 &
CUDA_VISIBLE_DEVICES=2 python paired_images_data_gen_client.py --connection --port 50022  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_viola_bowlplatefork --reference_joint_angles_path /home/lawrence/xembody_followup/viola_dataset/ee_states.txt --start_id 5420 &

CUDA_VISIBLE_DEVICES=2 python paired_images_data_gen_server.py --connection --port 50023 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_viola_bowlplatefork --reference_joint_angles_path /home/lawrence/xembody_followup/viola_dataset/ee_states.txt --start_id 6420 &
CUDA_VISIBLE_DEVICES=2 python paired_images_data_gen_client.py --connection --port 50023  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_viola_bowlplatefork --reference_joint_angles_path /home/lawrence/xembody_followup/viola_dataset/ee_states.txt --start_id 6420 &

CUDA_VISIBLE_DEVICES=2 python paired_images_data_gen_server.py --connection --port 50024 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_viola_bowlplatefork --reference_joint_angles_path /home/lawrence/xembody_followup/viola_dataset/ee_states.txt --start_id 7420 &
CUDA_VISIBLE_DEVICES=2 python paired_images_data_gen_client.py --connection --port 50024  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_viola_bowlplatefork --reference_joint_angles_path /home/lawrence/xembody_followup/viola_dataset/ee_states.txt --start_id 7420 &

CUDA_VISIBLE_DEVICES=3 python paired_images_data_gen_server.py --connection --port 50025 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_viola_bowlplatefork --reference_joint_angles_path /home/lawrence/xembody_followup/viola_dataset/ee_states.txt --start_id 8420 &
CUDA_VISIBLE_DEVICES=3 python paired_images_data_gen_client.py --connection --port 50025  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_viola_bowlplatefork --reference_joint_angles_path /home/lawrence/xembody_followup/viola_dataset/ee_states.txt --start_id 8420 &

CUDA_VISIBLE_DEVICES=3 python paired_images_data_gen_server.py --connection --port 50026 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_viola_bowlplatefork --reference_joint_angles_path /home/lawrence/xembody_followup/viola_dataset/ee_states.txt --start_id 9420 &
CUDA_VISIBLE_DEVICES=3 python paired_images_data_gen_client.py --connection --port 50026  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_viola_bowlplatefork --reference_joint_angles_path /home/lawrence/xembody_followup/viola_dataset/ee_states.txt --start_id 9420 &

CUDA_VISIBLE_DEVICES=3 python paired_images_data_gen_server.py --connection --port 50027 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_viola_bowlplatefork --reference_joint_angles_path /home/lawrence/xembody_followup/viola_dataset/ee_states.txt --start_id 10420 &
CUDA_VISIBLE_DEVICES=3 python paired_images_data_gen_client.py --connection --port 50027  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_viola_bowlplatefork --reference_joint_angles_path /home/lawrence/xembody_followup/viola_dataset/ee_states.txt --start_id 10420 &

CUDA_VISIBLE_DEVICES=3 python paired_images_data_gen_server.py --connection --port 50028 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_viola_bowlplatefork --reference_joint_angles_path /home/lawrence/xembody_followup/viola_dataset/ee_states.txt --start_id 11420 &
CUDA_VISIBLE_DEVICES=3 python paired_images_data_gen_client.py --connection --port 50028  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_viola_bowlplatefork --reference_joint_angles_path /home/lawrence/xembody_followup/viola_dataset/ee_states.txt --start_id 11420 &

CUDA_VISIBLE_DEVICES=4 python paired_images_data_gen_server.py --connection --port 50029 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_viola_bowlplatefork --reference_joint_angles_path /home/lawrence/xembody_followup/viola_dataset/ee_states.txt --start_id 12420 &
CUDA_VISIBLE_DEVICES=4 python paired_images_data_gen_client.py --connection --port 50029  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_viola_bowlplatefork --reference_joint_angles_path /home/lawrence/xembody_followup/viola_dataset/ee_states.txt --start_id 12420 &

CUDA_VISIBLE_DEVICES=4 python paired_images_data_gen_server.py --connection --port 50030 --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_viola_bowlplatefork --reference_joint_angles_path /home/lawrence/xembody_followup/viola_dataset/ee_states.txt --start_id 13420 &
CUDA_VISIBLE_DEVICES=4 python paired_images_data_gen_client.py --connection --port 50030  --num_cam_poses_per_robot_pose 5 --save_paired_images_folder_path paired_images_viola_bowlplatefork --reference_joint_angles_path /home/lawrence/xembody_followup/viola_dataset/ee_states.txt --start_id 13420 &