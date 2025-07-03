#!/usr/bin/env bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mirage

robot_dataset="austin_mutex"  # 机器人数据集名称

start=19    # 起始分区号
end=20     # 结束分区号（含）

for partition in $(seq "$start" "$((end-1))"); do        # 0‥7
  #for robot in Panda Sawyer Jaco Kinova3 IIWA UR5e; do
  #for robot in Sawyer Jaco Kinova3 IIWA; do
  for robot in Kinova3; do
    python /home/guanhuaji/mirage/robot2robot/rendering/generate_target_robot_images.py \
           --robot_dataset "${robot_dataset}" \
           --target_robot  "${robot}" \
           --partition     "${partition}" &
  done
done

wait
echo "✓ 分区 ${start}-${end} 全部处理完毕！"