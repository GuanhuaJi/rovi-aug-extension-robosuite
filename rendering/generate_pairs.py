import os
import subprocess

def main():
    # Define variables
    robot_dataset = "can"
    # Change this to the source robot you want to run
    source_robot = "IIWA"

    # Create a copy of the current environment variables
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"

    # Loop through episodes 0 to 19
    for episode in range(1, 20):
        print(f"Processing episode {episode}...")

        # 1. Run test_server.py
        subprocess.run([
            "python", "test_server.py",
            "--robot_dataset", robot_dataset,
            "--episode", str(episode)
        ], env=env, check=True)

        # 2. Run test_client.py for each target robot
        for target_robot in ["IIWA", "Sawyer", "Jaco"]:
            subprocess.run([
                "python", "test_client.py",
                "--robot_dataset", robot_dataset,
                "--target_robot", target_robot,
                "--episode", str(episode)
            ], env=env, check=True)

        # 3. Run expand_mask.py
        input_folder = f"/home/jiguanhua/mirage/robot2robot/rendering/paired_images/{robot_dataset}/{source_robot}_mask/{episode}"
        output_folder = f"/home/jiguanhua/mirage/robot2robot/rendering/paired_images/{robot_dataset}/{source_robot}_mask_expanded/{episode}"
        subprocess.run([
            "python", "expand_mask.py",
            "--input_folder", input_folder,
            "--output_folder", output_folder,
            "--alpha", "5.0",
            "--use_8_connected"
        ], env=env, check=True)

        print(f"Completed episode {episode}.\n")

if __name__ == "__main__":
    main()