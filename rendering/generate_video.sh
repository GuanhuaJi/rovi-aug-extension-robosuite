robot_dataset="can"
episode=0

if [ "$robot_dataset" = "autolab_ur5" ] || [ "$robot_dataset" = "asu_table_top_rlds" ]; then
    source_robot="UR5e"
else
    source_robot="Panda"
fi
python overlay.py \
    --original_folder "/home/jiguanhua/mirage/robot2robot/rendering/video_inpainting/${robot_dataset}/${episode}" \
    --mask_folder "/home/jiguanhua/mirage/robot2robot/rendering/paired_images/${robot_dataset}/Panda_mask/${episode}" \
    --overlay_folder "/home/jiguanhua/mirage/robot2robot/rendering/datasets/states/${robot_dataset}/episode_${episode}/images" \
    --output_folder "/home/jiguanhua/mirage/robot2robot/rendering/video_inpainting/${robot_dataset}/reverted_${episode}" \
    --reverse True

python image_to_video.py \
    --folder "/home/jiguanhua/mirage/robot2robot/rendering/datasets/states/${robot_dataset}/episode_${episode}/images" \
    --output_video /home/jiguanhua/mirage/robot2robot/rendering/cross_inpainting/${robot_dataset}_${source_robot}/${robot_dataset}_${source_robot}_${episode}.mp4 \
    --fps 30

python overlay.py \
    --original_folder "/home/jiguanhua/mirage/robot2robot/rendering/video_inpainting/${robot_dataset}/reverted_${episode}" \
    --mask_folder "/home/jiguanhua/mirage/robot2robot/rendering/paired_images/${robot_dataset}/IIWA_mask/${episode}" \
    --overlay_folder "/home/jiguanhua/mirage/robot2robot/rendering/paired_images/${robot_dataset}/IIWA_rgb/${episode}" \
    --output_folder "/home/jiguanhua/mirage/robot2robot/rendering/cross_inpainting/${robot_dataset}_IIWA/${episode}"

python image_to_video.py \
    --folder "/home/jiguanhua/mirage/robot2robot/rendering/cross_inpainting/${robot_dataset}_IIWA/${episode}" \
    --output_video /home/jiguanhua/mirage/robot2robot/rendering/cross_inpainting/${robot_dataset}_IIWA/${robot_dataset}_IIWA_${episode}.mp4 \
    --fps 30

# python overlay.py \
#     --original_folder "/home/jiguanhua/mirage/robot2robot/rendering/video_inpainting/${robot_dataset}/reverted_${episode}" \
#     --mask_folder "/home/jiguanhua/mirage/robot2robot/rendering/paired_images/${robot_dataset}/Sawyer_mask/${episode}" \
#     --overlay_folder "/home/jiguanhua/mirage/robot2robot/rendering/paired_images/${robot_dataset}/Sawyer_rgb/${episode}" \
#     --output_folder "/home/jiguanhua/mirage/robot2robot/rendering/cross_inpainting/${robot_dataset}_Sawyer/${episode}"

# python image_to_video.py \
#     --folder "/home/jiguanhua/mirage/robot2robot/rendering/cross_inpainting/${robot_dataset}_Sawyer/${episode}" \
#     --output_video /home/jiguanhua/mirage/robot2robot/rendering/cross_inpainting/${robot_dataset}_Sawyer/${robot_dataset}_Sawyer_${episode}.mp4 \
#     --fps 30

# python overlay.py \
#     --original_folder "/home/jiguanhua/mirage/robot2robot/rendering/video_inpainting/${robot_dataset}/reverted_${episode}" \
#     --mask_folder "/home/jiguanhua/mirage/robot2robot/rendering/paired_images/${robot_dataset}/Jaco_mask/${episode}" \
#     --overlay_folder "/home/jiguanhua/mirage/robot2robot/rendering/paired_images/${robot_dataset}/Jaco_rgb/${episode}" \
#     --output_folder "/home/jiguanhua/mirage/robot2robot/rendering/cross_inpainting/${robot_dataset}_Jaco/${episode}"

# python image_to_video.py \
#     --folder "/home/jiguanhua/mirage/robot2robot/rendering/cross_inpainting/${robot_dataset}_Jaco/${episode}" \
#     --output_video /home/jiguanhua/mirage/robot2robot/rendering/cross_inpainting/${robot_dataset}_Jaco/${robot_dataset}_Jaco_${episode}.mp4 \
#     --fps 30