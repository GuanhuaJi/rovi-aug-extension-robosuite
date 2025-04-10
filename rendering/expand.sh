robot_dataset="can"
episode=0

if [ "$robot_dataset" = "autolab_ur5" ] || [ "$robot_dataset" = "asu_table_top_rlds" ]; then
    source_robot="UR5e"
else
    source_robot="Panda"
fi

python expand_mask.py \
    --input_folder /home/jiguanhua/mirage/robot2robot/rendering/paired_images/${robot_dataset}/${source_robot}_mask/${episode} \
    --output_folder /home/jiguanhua/mirage/robot2robot/rendering/paired_images/${robot_dataset}/${source_robot}_mask_expanded/${episode} \
    --alpha 1.0 \
    --use_8_connected


# python shift_mask.py \
#     --input_folder /home/jiguanhua/mirage/robot2robot/rendering/paired_images/${robot_dataset}/${source_robot}_mask_expanded/${episode} \
#     --output_folder /home/jiguanhua/mirage/robot2robot/rendering/paired_images/${robot_dataset}/${source_robot}_mask_expanded/${episode} \
#     --shift_x 20 \
#     --shift_y 0