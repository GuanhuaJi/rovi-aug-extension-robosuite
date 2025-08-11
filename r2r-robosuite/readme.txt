# robot2robot

1. installation:
To install, run:

cd rovi-aug-extension-robosuite
conda env create -f environment.yml
conda activate r2r-robosuite

cd mirage-robosuite
pip install -e .

2. download original simulation data:
download simulation data here: LINK

3. extract properties and compute target eef
after you have simulation data, run 
cd rendering
python export_source_robot_states_sim.py --robot_dataset <DATASET> --hdf5_path <PATH TO .hdf5 FILE> # DATASET: can/lift/square/stack/two_piece
or 
python export_source_robot_states_sim.py --robot_dataset <DATASET> --hdf5_path <PATH TO .hdf5 FILE> # DATASET: can/lift/square/stack/two_piece --start START_EPISODE --end END_EPISODE
to process certain range of episodes instead of all episodes

4. replay on multiple robots
python generate_target_robot_images.py --robot_dataset <DATASET> --target_robots <LIST OF ROBOTS> # DATASET: can/lift/square/stack/two_piece ROBOT: Sawyer/UR5e/IIWA/Kinova3/Jaco
or
python generate_target_robot_images.py --robot_dataset <DATASET> --target_robots <LIST OF ROBOTS> # DATASET: can/lift/square/stack/two_piece ROBOT: Sawyer/UR5e/IIWA/Kinova3/Jaco --start START_EPISODE --end END_EPISODE
to process certain range of episodes instead of all episodes. be sure to always include Panda in the list of robots since we will use it to generate inpaintings. this will replay robots along the trajectory.

5. install e2fgvi inpaintings
cd e2fgvi_inpainting
conda create -n e2fgvi python=3.9 -y
conda activate e2fgvi
conda config --env --set channel_priority strict
conda install pytorch==2.4.1 torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html
conda install tensorboard matplotlib scikit-image
conda install tqdm
pip install -U imageio imageio-ffmpeg

After you run step 4, you should notice a folder named replay_videos is created under r2r-robosuite, we will be using replay_videos/DATASET/source_replays and replay_videos/DATASET/Panda_replay_mask for inpainting. 

python batched_inference_e2fgvi_mp4.py --bg_root <PATH TO source_replays> --mask_root /home/guanhuaji/mirage/robot2robot/rendering/paired_images/lift/Panda_replay_mask --output_root <PATH TO Panda_replay_mask> --dilution 1

python batched_inference_e2fgvi_mp4.py --bg_root <PATH TO source_replays> --mask_root <PATH TO Panda_replay_mask> --output_root <PATH TO output folder> --dilution 1

which will stored the replays under output folder

6. overlay

overlay replay image onto inpainting image 

python overlay_sim.py --source-dir replay_videos/DATASET \
  --robots ROBOTS --start XX --end XX --overwrite











