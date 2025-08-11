# robot2robot

End-to-end pipeline to replay Robosuite trajectories across multiple robots, generate background inpaintings with E2FGVI, and overlay replays for clean composites.

---

## 1) Installation

```bash
# repo layout:
# ├─ rovi-aug-extension-robosuite/
# └─ mirage-robosuite/

# create the main env
cd rovi-aug-extension-robosuite
conda env create -f environment.yml
conda activate r2r-robosuite

# install mirage-robosuite in editable mode
cd ../mirage-robosuite
pip install -e .
```

---

## 2) Download simulation data

Download the original simulation data here: **LINK**
(Place it wherever you prefer; you’ll point to the `.hdf5` file(s) in the next step.)

---

## 3) Extract properties & compute target EEF

From `rovi-aug-extension-robosuite/rendering`:

```bash
# Process all episodes
python export_source_robot_states_sim.py \
  --robot_dataset <DATASET> \
  --hdf5_path <PATH/TO/file.hdf5>
# DATASET ∈ {can, lift, square, stack, two_piece}

# Process a subset of episodes
python export_source_robot_states_sim.py \
  --robot_dataset <DATASET> \
  --hdf5_path <PATH/TO/file.hdf5> \
  --start <START_EPISODE> \
  --end <END_EPISODE>
```

Run these commands in the [step_3.sh](scripts/step_3.sh) script to replicate the paper dataset.

---

## 4) Replay on multiple robots

```bash
# Replay the trajectory on a set of target robots
python generate_target_robot_images.py \
  --robot_dataset <DATASET> \
  --target_robots <LIST OF ROBOTS>
# DATASET ∈ {can, lift, square, stack, two_piece}
# ROBOTS  ∈ {Sawyer, UR5e, IIWA, Kinova3, Jaco, Panda}
```

Optionally limit to a range of episodes:

```bash
python generate_target_robot_images.py \
  --robot_dataset <DATASET> \
  --target_robots <LIST OF ROBOTS> \
  --start <START_EPISODE> \
  --end <END_EPISODE>
```

**Important:** Always include **Panda** in `--target_robots`. Panda replays are used to build inpainting masks.

After this step, a folder `replay_videos/` is created under `rovi-aug-extension-robosuite/`. You’ll use:

* `replay_videos/<DATASET>/source_replays`
* `replay_videos/<DATASET>/Panda_replay_mask`


Run these commands in the [step_4.sh](scripts/step_4.sh) script to replicate the paper dataset.

---

## 5) Install E2FGVI & run inpainting

Create a separate environment for E2FGVI:

```bash
cd e2fgvi
conda create -n e2fgvi python=3.9 -y
conda activate e2fgvi

conda config --env --set channel_priority strict
conda install pytorch==2.4.1 torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html
conda install tensorboard matplotlib scikit-image tqdm
pip install -U imageio imageio-ffmpeg
```

Run batched inpainting (background = source replays; mask = Panda masks):

```bash
python batched_inference_e2fgvi_mp4.py \
  --bg_root   <PATH/TO/replay_videos/<DATASET>/source_replays> \
  --mask_root <PATH/TO/replay_videos/<DATASET>/Panda_replay_mask> \
  --output_root <PATH/TO/output_folder> \
  --dilution 1
```

The resulting inpainted videos are written under `<output_root>`.

Run these commands in the [step_5.sh](scripts/step_5.sh) script to replicate the paper dataset.

---

## 6) Overlay (compose replays onto inpaintings)

Overlay per-robot replay frames onto the inpainted backgrounds:

```bash
python overlay_sim.py \
  --source-dir replay_videos/<DATASET> \
  --robots <LIST OF ROBOTS> \
  --start <START_EPISODE> \
  --end <END_EPISODE> \
  --overwrite
```

Run these commands in the [step_6.sh](scripts/step_6.sh) script to replicate the paper dataset.

**Notes**

* `--robots` accepts the same robot names as above. Include only the robots you want to composite.
* Use `--overwrite` to replace existing outputs.

---

## Directory hints

After step 6, you should have a structure like:

```
replay_videos/
  <DATASET>/
    source_replays/
    inpaint/
    Panda/
    Panda_replay_mask/
    Panda_replay_video/
    Sawyer/
    Sawyer_overlay/            # overlay outputs for Sawyer
    Sawyer_replay_mask/
    Sawyer_replay_video/
    ...
    source_replays/
    source_robot_states/
    target_robot_states/
```

---

## Tips

* GPU acceleration (CUDA 12.1) is recommended for E2FGVI.
* If you only need a subset of episodes, always pass `--start/--end` to save time.
* Keep Panda in the target set to ensure mask generation for inpainting.
