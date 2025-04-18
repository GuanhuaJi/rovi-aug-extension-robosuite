import os
import subprocess

def main():
    # Define variables
    robot_dataset = "can"
    # Change this to the source robot you want to run
    source_robot = "Panda"
    regenerate = False

    # Create a copy of the current environment variables
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"

    # Loop through episodes 0 to 19
    for episode in range(0, 6):
        print(f"Processing episode {episode}...")
        # 1. Run test_server.py
        if regenerate is False:
            subprocess.run([
                "python", "export_source_robot_states.py",
                "--robot_dataset", robot_dataset,
                "--episode", str(episode)
            ], env=env, check=True)

        # 2. Run test_client.py for each target robot
        for target_robot in ["Panda", "IIWA", "Sawyer", "Jaco", "UR5e", "Kinova3"]:
        #for target_robot in ["Panda"]:
            subprocess.run([
                "python", "generate_target_robot_images.py",
                "--robot_dataset", robot_dataset,
                "--target_robot", target_robot,
                "--episode", str(episode)
            ], env=env, check=True)

        if regenerate is False:
            input_folder = f"/home/jiguanhua/mirage/robot2robot/rendering/paired_images/{robot_dataset}/{source_robot}_mask/{episode}"
            output_folder = f"/home/jiguanhua/mirage/robot2robot/rendering/paired_images/{robot_dataset}/{source_robot}_mask/{episode}"
            subprocess.run([
                "python", "remove_small_points.py",
                input_folder,
                output_folder,
                "--min_size", "10"
            ], env=env, check=True)

            # 3. Run expand_mask.py
            input_folder = f"/home/jiguanhua/mirage/robot2robot/rendering/paired_images/{robot_dataset}/{source_robot}_mask/{episode}"
            output_folder = f"/home/jiguanhua/mirage/robot2robot/rendering/paired_images/{robot_dataset}/{source_robot}_mask_expanded/{episode}"
            subprocess.run([
                "python", "expand_mask.py",
                "--input_folder", input_folder,
                "--output_folder", output_folder,
                "--alpha", "2.0",
                "--use_8_connected"
            ], env=env, check=True)

        print(f"Completed episode {episode}.\n")

if __name__ == "__main__":
    main()