#!/usr/bin/env python3
# conda activate mirage
# python /home/guanhuaji/mirage/robot2robot/rendering/generate_video_OXE.py

import subprocess
from pathlib import Path
from itertools import chain  # build one flat list of commands

ROOT = Path("/home/guanhuaji/mirage/robot2robot/rendering")
OVERLAY = ROOT / "overlay.py"
IMG2VID = ROOT / "image_to_video.py"

TARGET_ROBOTS = ["Panda", "IIWA", "Sawyer", "Jaco", "UR5e", "Kinova3"]
#TARGET_ROBOTS = ["Panda"]
FPS = 30

def overlay_cmd(dataset: str, robot: str, ep: int) -> list[str]:
    """Return the overlay command for one robot & episode."""
    base = ROOT
    return [
        "python", str(OVERLAY),
        "--original_folder", str(base / "video_inpainting" / dataset / f"{ep}"),
        "--mask_folder",     str(base / "paired_images" / dataset / f"{robot}_mask" / f"{ep}"),
        "--overlay_folder",  str(base / "paired_images" / dataset / f"{robot}_rgb"  / f"{ep}"),
        "--output_folder",   str(base / "cross_inpainting" / f"{dataset}_{robot}" / f"{ep}")
    ]

def img2vid_cmd(dataset: str, robot: str, ep: int) -> list[str]:
    """Return the folder-to-video command matching overlay_cmd output."""
    out_dir = ROOT / "cross_inpainting" / f"{dataset}_{robot}"
    return [
        "python", str(IMG2VID),
        "--folder",       str(out_dir / f"{ep}"),
        "--output_video", str(out_dir / f"{dataset}_{robot}_{ep}.mp4"),
        "--fps",          str(FPS)
    ]

def main() -> None:
    robot_dataset = "austin_mutex"        # or "asu_table_top_rlds"
    for episode in range(20):            # 0 .. 19
        print(f"▶ episode {episode}")
        # build one flat iterable of [cmd, cmd, cmd, ...]
        cmds = chain.from_iterable(
            (overlay_cmd(robot_dataset, r, episode),
             img2vid_cmd(robot_dataset, r, episode))
            for r in TARGET_ROBOTS
        )
        for cmd in cmds:
            subprocess.run(cmd)
        print(f"✓ episode {episode} done\n")

if __name__ == "__main__":
    main()
