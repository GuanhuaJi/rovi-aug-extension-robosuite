# conda activate mirage 
# python /home/jiguanhua/mirage/robot2robot/rendering/generate_video.py

import subprocess
import os

def main():
    # Set your dataset here
    robot_dataset = "stack"
    
    # Loop over the desired range of episodes
    for episode in range(0, 10):  # e.g., episodes 0 through 19
        print(f"Processing episode {episode}...")
        
        # Determine the source robot
        if robot_dataset in ["autolab_ur5", "asu_table_top_rlds"]:
            source_robot = "UR5e"
        else:
            source_robot = "Panda"

        # Construct the list of commands
        commands = [
            # 1) Overlay with Panda
            [
                "python", "/home/jiguanhua/mirage/robot2robot/rendering/overlay.py",
                "--original_folder",
                f"/home/jiguanhua/mirage/robot2robot/rendering/video_inpainting/{robot_dataset}/{episode}",
                "--mask_folder",
                f"/home/jiguanhua/mirage/robot2robot/rendering/paired_images/{robot_dataset}/Panda_mask/{episode}",
                "--overlay_folder",
                f"/home/jiguanhua/mirage/robot2robot/rendering/datasets/states/{robot_dataset}/episode_{episode}/images",
                "--output_folder",
                f"/home/jiguanhua/mirage/robot2robot/rendering/video_inpainting/{robot_dataset}/reverted_{episode}",
                "--reverse",
                "True"
            ],
            # 2) Convert the original folder to video
            [
                "python", "/home/jiguanhua/mirage/robot2robot/rendering/image_to_video.py",
                "--folder",
                f"/home/jiguanhua/mirage/robot2robot/rendering/datasets/states/{robot_dataset}/episode_{episode}/images",
                "--output_video",
                f"/home/jiguanhua/mirage/robot2robot/rendering/cross_inpainting/{robot_dataset}_{source_robot}/{robot_dataset}_{source_robot}_{episode}.mp4",
                "--fps",
                "30"
            ],
            [
                "python", "/home/jiguanhua/mirage/robot2robot/rendering/overlay.py",
                "--original_folder",
                f"/home/jiguanhua/mirage/robot2robot/rendering/video_inpainting/{robot_dataset}/reverted_{episode}",
                #f"/home/jiguanhua/mirage/robot2robot/rendering/video_inpainting/{robot_dataset}/{episode}",
                "--mask_folder",
                f"/home/jiguanhua/mirage/robot2robot/rendering/paired_images/{robot_dataset}/Panda_mask/{episode}",
                "--overlay_folder",
                f"/home/jiguanhua/mirage/robot2robot/rendering/paired_images/{robot_dataset}/Panda_rgb/{episode}",
                "--output_folder",
                f"/home/jiguanhua/mirage/robot2robot/rendering/cross_inpainting/{robot_dataset}_Panda/{episode}"
            ],
            # 4) Convert IIWA output folder to video
            [
                "python", "/home/jiguanhua/mirage/robot2robot/rendering/image_to_video.py",
                "--folder",
                f"/home/jiguanhua/mirage/robot2robot/rendering/cross_inpainting/{robot_dataset}_Panda/{episode}",
                "--output_video",
                f"/home/jiguanhua/mirage/robot2robot/rendering/cross_inpainting/{robot_dataset}_Panda/{robot_dataset}_Panda_{episode}.mp4",
                "--fps",
                "30"
            ],
            # 3) Overlay with IIWA
            [
                "python", "/home/jiguanhua/mirage/robot2robot/rendering/overlay.py",
                "--original_folder",
                f"/home/jiguanhua/mirage/robot2robot/rendering/video_inpainting/{robot_dataset}/reverted_{episode}",
                #f"/home/jiguanhua/mirage/robot2robot/rendering/video_inpainting/{robot_dataset}/{episode}",
                "--mask_folder",
                f"/home/jiguanhua/mirage/robot2robot/rendering/paired_images/{robot_dataset}/IIWA_mask/{episode}",
                "--overlay_folder",
                f"/home/jiguanhua/mirage/robot2robot/rendering/paired_images/{robot_dataset}/IIWA_rgb/{episode}",
                "--output_folder",
                f"/home/jiguanhua/mirage/robot2robot/rendering/cross_inpainting/{robot_dataset}_IIWA/{episode}"
            ],
            # 4) Convert IIWA output folder to video
            [
                "python", "/home/jiguanhua/mirage/robot2robot/rendering/image_to_video.py",
                "--folder",
                f"/home/jiguanhua/mirage/robot2robot/rendering/cross_inpainting/{robot_dataset}_IIWA/{episode}",
                "--output_video",
                f"/home/jiguanhua/mirage/robot2robot/rendering/cross_inpainting/{robot_dataset}_IIWA/{robot_dataset}_IIWA_{episode}.mp4",
                "--fps",
                "30"
            ],
            # 5) Overlay with Sawyer
            [
                "python", "/home/jiguanhua/mirage/robot2robot/rendering/overlay.py",
                "--original_folder",
                f"/home/jiguanhua/mirage/robot2robot/rendering/video_inpainting/{robot_dataset}/reverted_{episode}",
                #f"/home/jiguanhua/mirage/robot2robot/rendering/video_inpainting/{robot_dataset}/{episode}",
                "--mask_folder",
                f"/home/jiguanhua/mirage/robot2robot/rendering/paired_images/{robot_dataset}/Sawyer_mask/{episode}",
                "--overlay_folder",
                f"/home/jiguanhua/mirage/robot2robot/rendering/paired_images/{robot_dataset}/Sawyer_rgb/{episode}",
                "--output_folder",
                f"/home/jiguanhua/mirage/robot2robot/rendering/cross_inpainting/{robot_dataset}_Sawyer/{episode}"
            ],
            # 6) Convert Sawyer output folder to video
            [
                "python", "/home/jiguanhua/mirage/robot2robot/rendering/image_to_video.py",
                "--folder",
                f"/home/jiguanhua/mirage/robot2robot/rendering/cross_inpainting/{robot_dataset}_Sawyer/{episode}",
                "--output_video",
                f"/home/jiguanhua/mirage/robot2robot/rendering/cross_inpainting/{robot_dataset}_Sawyer/{robot_dataset}_Sawyer_{episode}.mp4",
                "--fps",
                "30"
            ],
            # 7) Overlay with Jaco
            [
                "python", "/home/jiguanhua/mirage/robot2robot/rendering/overlay.py",
                "--original_folder",
                f"/home/jiguanhua/mirage/robot2robot/rendering/video_inpainting/{robot_dataset}/reverted_{episode}",
                #f"/home/jiguanhua/mirage/robot2robot/rendering/video_inpainting/{robot_dataset}/{episode}",
                "--mask_folder",
                f"/home/jiguanhua/mirage/robot2robot/rendering/paired_images/{robot_dataset}/Jaco_mask/{episode}",
                "--overlay_folder",
                f"/home/jiguanhua/mirage/robot2robot/rendering/paired_images/{robot_dataset}/Jaco_rgb/{episode}",
                "--output_folder",
                f"/home/jiguanhua/mirage/robot2robot/rendering/cross_inpainting/{robot_dataset}_Jaco/{episode}"
            ],
            # 8) Convert Jaco output folder to video
            [
                "python", "/home/jiguanhua/mirage/robot2robot/rendering/image_to_video.py",
                "--folder",
                f"/home/jiguanhua/mirage/robot2robot/rendering/cross_inpainting/{robot_dataset}_Jaco/{episode}",
                "--output_video",
                f"/home/jiguanhua/mirage/robot2robot/rendering/cross_inpainting/{robot_dataset}_Jaco/{robot_dataset}_Jaco_{episode}.mp4",
                "--fps",
                "30"
            ],
            [
                "python", "/home/jiguanhua/mirage/robot2robot/rendering/overlay.py",
                "--original_folder",
                f"/home/jiguanhua/mirage/robot2robot/rendering/video_inpainting/{robot_dataset}/reverted_{episode}",
                #f"/home/jiguanhua/mirage/robot2robot/rendering/video_inpainting/{robot_dataset}/{episode}",
                "--mask_folder",
                f"/home/jiguanhua/mirage/robot2robot/rendering/paired_images/{robot_dataset}/UR5e_mask/{episode}",
                "--overlay_folder",
                f"/home/jiguanhua/mirage/robot2robot/rendering/paired_images/{robot_dataset}/UR5e_rgb/{episode}",
                "--output_folder",
                f"/home/jiguanhua/mirage/robot2robot/rendering/cross_inpainting/{robot_dataset}_UR5e/{episode}"
            ],
            # 10) Convert UR5e output folder to video
            [
                "python", "/home/jiguanhua/mirage/robot2robot/rendering/image_to_video.py",
                "--folder",
                f"/home/jiguanhua/mirage/robot2robot/rendering/cross_inpainting/{robot_dataset}_UR5e/{episode}",
                "--output_video",
                f"/home/jiguanhua/mirage/robot2robot/rendering/cross_inpainting/{robot_dataset}_UR5e/{robot_dataset}_UR5e_{episode}.mp4",
                "--fps",
                "30"
            ],
            [
                "python", "/home/jiguanhua/mirage/robot2robot/rendering/overlay.py",
                "--original_folder",
                f"/home/jiguanhua/mirage/robot2robot/rendering/video_inpainting/{robot_dataset}/reverted_{episode}",
                #f"/home/jiguanhua/mirage/robot2robot/rendering/video_inpainting/{robot_dataset}/{episode}",
                "--mask_folder",
                f"/home/jiguanhua/mirage/robot2robot/rendering/paired_images/{robot_dataset}/Kinova3_mask/{episode}",
                "--overlay_folder",
                f"/home/jiguanhua/mirage/robot2robot/rendering/paired_images/{robot_dataset}/Kinova3_rgb/{episode}",
                "--output_folder",
                f"/home/jiguanhua/mirage/robot2robot/rendering/cross_inpainting/{robot_dataset}_Kinova3/{episode}"
            ],
            # 10) Convert Kinova3 output folder to video
            [
                "python", "/home/jiguanhua/mirage/robot2robot/rendering/image_to_video.py",
                "--folder",
                f"/home/jiguanhua/mirage/robot2robot/rendering/cross_inpainting/{robot_dataset}_Kinova3/{episode}",
                "--output_video",
                f"/home/jiguanhua/mirage/robot2robot/rendering/cross_inpainting/{robot_dataset}_Kinova3/{robot_dataset}_Kinova3_{episode}.mp4",
                "--fps",
                "30"
            ],
        ]

        # Run each command in sequence
        for cmd in commands:
            subprocess.run(cmd, check=True)

        print(f"Episode {episode} completed.\n")

if __name__ == "__main__":
    main()