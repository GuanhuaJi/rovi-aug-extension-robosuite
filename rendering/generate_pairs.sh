#!/usr/bin/env bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mirage

robot_dataset="autolab_ur5"

start=0    # 起始分区号
end=10     # 结束分区号（含）

for partition in $(seq "$start" "$((end-1))"); do        # 0‥7
  for robot in Panda Sawyer Jaco Kinova3 IIWA UR5e; do
    python /home/guanhuaji/mirage/robot2robot/rendering/generate_target_robot_images.py \
           --robot_dataset "${robot_dataset}" \
           --target_robot  "${robot}" \
           --partition     "${partition}" &
  done
done

wait
echo "✓ 分区 ${start}-${end} 全部处理完毕！"