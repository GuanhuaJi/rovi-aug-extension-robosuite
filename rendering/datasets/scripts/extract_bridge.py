import os
import numpy as np
import tensorflow_datasets as tfds
from PIL import Image

DATASET_PATH = "gs://gresearch/robotics/bridge/0.1.0"
def main():
    builder = tfds.builder_from_directory(builder_dir=DATASET_PATH)
    ds = builder.as_dataset(split="train[:20]", shuffle_files=False)
    for episode_num, episode in enumerate(ds):
        
        # 4) 遍历当前 episode 中的每个 step
        steps_dataset = episode["steps"]
        for step_idx, step in enumerate(steps_dataset):

            # 如果你想模仿原先“一次性取到 state 并切片”的做法：
            task_description = step["observation"]["natural_language_instruction"] # (8,)
            print(step_idx, task_description)

if __name__ == "__main__":
    main()
