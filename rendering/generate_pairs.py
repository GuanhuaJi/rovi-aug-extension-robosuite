# conda activate mirage
# python /home/jiguanhua/mirage/robot2robot/rendering/generate_pairs.py --robot_dataset "lift" --target_robot "Panda"
# python /home/jiguanhua/mirage/robot2robot/rendering/generate_pairs.py --robot_dataset "lift" --target_robot "IIWA"
# python /home/jiguanhua/mirage/robot2robot/rendering/generate_pairs.py --robot_dataset "lift" --target_robot "Sawyer"
# python /home/jiguanhua/mirage/robot2robot/rendering/generate_pairs.py --robot_dataset "lift" --target_robot "Jaco"
# python /home/jiguanhua/mirage/robot2robot/rendering/generate_pairs.py --robot_dataset "lift" --target_robot "UR5e"
# python /home/jiguanhua/mirage/robot2robot/rendering/generate_pairs.py --robot_dataset "lift" --target_robot "Kinova3"

import os
import subprocess
import argparse

def generate_mask(robot_dataset, source_robot, episode, env):
    input_folder = f"/home/jiguanhua/mirage/robot2robot/rendering/paired_images/{robot_dataset}/{source_robot}_mask/{episode}"
    output_folder = f"/home/jiguanhua/mirage/robot2robot/rendering/paired_images/{robot_dataset}/{source_robot}_mask/{episode}"
    subprocess.run([
        "python", "/home/jiguanhua/mirage/robot2robot/rendering/remove_small_points.py",
        input_folder,
        output_folder,
        "--min_size", "10"
    ], env=env, check=True)

    # 3. Run expand_mask.py
    input_folder = f"/home/jiguanhua/mirage/robot2robot/rendering/paired_images/{robot_dataset}/{source_robot}_mask/{episode}"
    output_folder = f"/home/jiguanhua/mirage/robot2robot/rendering/paired_images/{robot_dataset}/{source_robot}_mask_expanded/{episode}"
    subprocess.run([
        "python", "/home/jiguanhua/mirage/robot2robot/rendering/expand_mask.py",
        "--input_folder", input_folder,
        "--output_folder", output_folder,
        "--alpha", "2.0",
        "--use_8_connected"
    ], env=env, check=True)

def main(robot_dataset, target_robot):
    source_robot = "Panda"
    regenerate = target_robot in ["Panda", "IIWA", "Sawyer", "Jaco", "UR5e", "Kinova3"]

    # Create a copy of the current environment variables
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"

    # Loop through episodes 0 to 19
    for episode in range(0, 200):
        print(f"Processing episode {episode}...")
        # 1. Run test_server.py
        if regenerate is False:
            subprocess.run([
                "python", "/home/jiguanhua/mirage/robot2robot/rendering/export_source_robot_states.py",
                "--robot_dataset", robot_dataset,
                "--episode", str(episode)
            ], env=env, check=True)

        # 2. Run test_client.py for each target robot
        if target_robot is None:
            for target_robot in ["Panda", "IIWA", "Sawyer", "Jaco", "UR5e", "Kinova3"]:
            #for target_robot in ["Panda"]:
                subprocess.run([
                    "python", "/home/jiguanhua/mirage/robot2robot/rendering/generate_target_robot_images.py",
                    "--robot_dataset", robot_dataset,
                    "--target_robot", target_robot,
                    "--episode", str(episode)
                ], env=env, check=True)
            generate_mask(robot_dataset, source_robot, episode, env)
        elif target_robot in ["Panda", "IIWA", "Sawyer", "Jaco", "UR5e", "Kinova3"]:
            subprocess.run([
                "python", "/home/jiguanhua/mirage/robot2robot/rendering/generate_target_robot_images.py",
                "--robot_dataset", robot_dataset,
                "--target_robot", target_robot,
                "--episode", str(episode)
            ], env=env, check=True)
            if target_robot == "Panda":
                generate_mask(robot_dataset, source_robot, episode, env)
        print(f"Completed episode {episode}.\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_robot", type=str, default=None)
    parser.add_argument("--robot_dataset", type=str, default=None)
    args = parser.parse_args()
    main(args.robot_dataset, args.target_robot)