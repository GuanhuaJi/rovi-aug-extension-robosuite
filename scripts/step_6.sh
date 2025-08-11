python overlay_sim.py --source-dir ../replay_videos/can \
  --robots Panda IIWA Jaco Sawyer Kinova3 UR5e --start 0 --end 199 --overwrite

python overlay_sim.py --source-dir ../replay_videos/lift \
  --robots Panda IIWA Jaco Sawyer Kinova3 UR5e --start 0 --end 199 --overwrite

python overlay_sim.py --source-dir ../replay_videos/square \
  --robots Panda IIWA Jaco Sawyer Kinova3 UR5e --start 0 --end 199 --overwrite

python overlay_sim.py --source-dir ../replay_videos/stack \
  --robots Panda IIWA Jaco Sawyer Kinova3 UR5e --start 0 --end 999 --overwrite

python overlay_sim.py --source-dir ../replay_videos/two_piece \
  --robots Panda IIWA Jaco Sawyer Kinova3 UR5e --start 0 --end 999 --overwrite