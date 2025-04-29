# conda activate mirage
# python /home/jiguanhua/mirage/robot2robot/rendering/generate_pairs.py --robot_dataset "stack" --target_robot "Panda"
# python /home/jiguanhua/mirage/robot2robot/rendering/generate_pairs.py --robot_dataset "stack" --target_robot "Panda"
# python /home/jiguanhua/mirage/robot2robot/rendering/generate_pairs.py --robot_dataset "stack" --target_robot "IIWA"
# python /home/jiguanhua/mirage/robot2robot/rendering/generate_pairs.py --robot_dataset "stack" --target_robot "Sawyer"
# python /home/jiguanhua/mirage/robot2robot/rendering/generate_pairs.py --robot_dataset "stack" --target_robot "Jaco"
# python /home/jiguanhua/mirage/robot2robot/rendering/generate_pairs.py --robot_dataset "stack" --target_robot "UR5e"
# python /home/jiguanhua/mirage/robot2robot/rendering/generate_pairs.py --robot_dataset "stack" --target_robot "Kinova3"
# python /home/jiguanhua/mirage/robot2robot/rendering/generate_pairs.py --robot_dataset "stack" --blacklist True
# python /home/jiguanhua/mirage/robot2robot/rendering/generate_pairs.py --robot_dataset "stack"


'''
import os
import subprocess
import argparse
import json
from pathlib import Path

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

def load_blacklist(blacklist_path) -> dict:
    if blacklist_path.exists() and blacklist_path.stat().st_size > 0:
        with blacklist_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}                      # start empty the first time

def save_blacklist(blacklist_path, blk: dict) -> None:
    with blacklist_path.open("w", encoding="utf-8") as f:
        json.dump(blk, f, indent=2)

def main(robot_dataset, target_robot, blacklist):
    source_robot = "Panda"
    regenerate = target_robot in ["Panda", "IIWA", "Sawyer", "Jaco", "UR5e", "Kinova3"]

    # Create a copy of the current environment variables
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"

    # Loop through episodes 0 to 19
    if blacklist:
        blacklist_path = Path(f"/home/jiguanhua/mirage/robot2robot/rendering/paired_images/{robot_dataset}/blacklist.json")
        if target_robot in ["Panda", "IIWA", "Sawyer", "Jaco", "UR5e", "Kinova3"]:
            blacklist_dict = load_blacklist(blacklist_path)
            episodes = blacklist_dict.get(target_robot, [])
            blacklist_dict[target_robot] = []
            save_blacklist(blacklist_path, blacklist_dict)
            for episode in episodes:
                subprocess.run([
                    "python", "/home/jiguanhua/mirage/robot2robot/rendering/generate_target_robot_images.py",
                    "--robot_dataset", robot_dataset,
                    "--target_robot", target_robot,
                    "--episode", str(episode)
                ], env=env, check=True)
        else:
            for robot_dataset, episodes in blacklist_dict.items():
                blacklist_dict = load_blacklist(blacklist_path)
                episodes = blacklist_dict.get(target_robot, [])
                blacklist_dict[target_robot] = []
                save_blacklist(blacklist_path, blacklist_dict)
                for episode in episode:
                    subprocess.run([
                        "python", "/home/jiguanhua/mirage/robot2robot/rendering/generate_target_robot_images.py",
                        "--robot_dataset", robot_dataset,
                        "--target_robot", target_robot,
                        "--episode", str(episode)
                    ], env=env, check=True)

    else:
        for episode in range(0, 20):
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
    parser.add_argument("--blacklist", type=bool, default=False)
    args = parser.parse_args()
    main(args.robot_dataset, args.target_robot, args.blacklist)



'''
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# --------------------------------------------------
# User-editable constants
ROOT = Path("/home/jiguanhua/mirage/robot2robot/rendering")
PAIRED_DIR = ROOT / "paired_images"
SOURCE_ROBOT = "Panda"
TARGET_ROBOTS = ["Panda", "IIWA", "Sawyer", "Jaco", "UR5e", "Kinova3"]
MIN_MASK_SIZE = "10"
CUDA_DEVICES = "0"
# --------------------------------------------------


def _run(cmd: list[str], env: dict[str, str]) -> None:
    """Run command in a subprocess with error propagation."""
    cmd = list(map(str, cmd))
    print("➜", " ".join(cmd))
    subprocess.run(cmd, env=env, check=True)


def load_blacklist(path: Path) -> dict[str, list[int]]:
    if path.exists() and path.stat().st_size > 0:
        return json.loads(path.read_text())
    return {}


def save_blacklist(path: Path, data: dict[str, list[int]]) -> None:
    path.write_text(json.dumps(data, indent=2))


def main(robot_dataset: str, target_robot: str | None, use_blacklist: bool) -> None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = CUDA_DEVICES

    blacklist_path = PAIRED_DIR / robot_dataset / "blacklist.json"
    blacklist = load_blacklist(blacklist_path)

    if use_blacklist:
        # Process episodes recorded in blacklist, then clear them
        episodes = blacklist.get(target_robot, []) if target_robot else []
        for epi in episodes:
            _run(
                [
                    "python",
                    ROOT / "generate_target_robot_images.py",
                    "--robot_dataset",
                    robot_dataset,
                    "--target_robot",
                    target_robot,
                    "--episode",
                    str(epi),
                ],
                env,
            )
        if target_robot:
            blacklist[target_robot] = []
            save_blacklist(blacklist_path, blacklist)
        return

    # Regular generation loop

    # if target_robot not in TARGET_ROBOTS:
    #     _run(
    #         [
    #             "python",
    #             ROOT / "export_source_robot_states.py",
    #             "--robot_dataset",
    #             robot_dataset,
    #         ],
    #         env,
    #     )

    # Decide which target robots to render
    robots_to_render = (
        [target_robot]
        if target_robot in TARGET_ROBOTS
        else TARGET_ROBOTS
    )

    for trg in robots_to_render:
        _run(
            [
                "python",
                ROOT / "generate_target_robot_images.py",
                "--robot_dataset",
                robot_dataset,
                "--target_robot",
                trg,
            ],
            env,
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot_dataset", required=True, help="e.g. square")
    ap.add_argument("--target_robot", choices=TARGET_ROBOTS, help="Target robot")
    ap.add_argument("--blacklist", action="store_true", help="Process blacklist only")
    args = ap.parse_args()
    main(args.robot_dataset, args.target_robot, args.blacklist)
