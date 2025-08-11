import time
import multiprocessing as mp
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

import cv2
import matplotlib.pyplot as plt
from PIL import Image
import argparse
MUTE_OUTPUT = True



MODEL_CHECKPOINT = "./checkpoints/checkpoint_150.pt"
MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_b+.yaml"
# vid_predictor = build_sam2_video_predictor(model_config, model_checkoint, device=device)
#sam2_model = build_sam2(model_config, model_checkoint, device="cuda")
# img_predictor = SAM2ImagePredictor(sam2_model)

def _remove_jpgs(directory: Path) -> int:
    """Delete all *.jpg/*.jpeg (any case) in *directory*; return count."""
    removed = 0
    for pattern in ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG"):
        for img in directory.glob(pattern):
            img.unlink()
            removed += 1
    return removed

def extract_frames(video_path, frames_path):
    # Create output directory based on video filename
    os.makedirs(frames_path, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Stop if no frames left

        frame_count += 1
        frame_filename = os.path.join(frames_path, f"{frame_count:05d}.jpg")
        cv2.imwrite(frame_filename, frame)

    cap.release()
    #print(f"Extracted {frame_count} frames to '{frames_path}'.")


def best_image_to_video_validation(video_dir, replay_vidname, background_vidname,
                                   vid_predictor, img_predictor):

    # scan all the JPEG frame names in this directory
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    print(f"Found {len(frame_names)} frames in {video_dir}")

    # setup images
    inference_state = vid_predictor.init_state(video_path=video_dir)

    # take a look the first video frame
    num_frames = len(frame_names)
    weights = [0.5 + 0 * abs(i - (num_frames / 2)) / (num_frames / 2) for i in range(num_frames)]
    frame_diffs = []

    for i in range(1, len(frame_names)):
        prev_frame = np.array(Image.open(os.path.join(video_dir, frame_names[i-1]))).astype(np.float32)
        curr_frame = np.array(Image.open(os.path.join(video_dir, frame_names[i]))).astype(np.float32)
        diff = np.sum(np.abs(curr_frame - prev_frame) > 50) # Count pixels with differences greater than threshold
        weighted_diff = diff * weights[i]
        frame_diffs.append(weighted_diff)
        # Find the frame with the maximum weighted difference
    ann_frame_idx = np.argmax(frame_diffs) + 1

    # Promptless mask generation on first frame

    img_predictor.set_image(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx]))) # apply SAM image encoder to the image

    # prompt encoding

    #mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(input_point, input_label, box=None, mask_logits=None, normalize_coords=True)
    sparse_embeddings, dense_embeddings = img_predictor.model.sam_prompt_encoder(points=None,boxes=None,masks=None,)

    # mask decoder
    batched_mode = False  #unnorm_coords.shape[0] > 1 # multi object prediction
    high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in img_predictor._features["high_res_feats"]]
    low_res_masks, prd_scores, _, _ = img_predictor.model.sam_mask_decoder(image_embeddings=img_predictor._features["image_embed"][-1].unsqueeze(0),image_pe=img_predictor.model.sam_prompt_encoder.get_dense_pe(),sparse_prompt_embeddings=sparse_embeddings,dense_prompt_embeddings=dense_embeddings,multimask_output=True,repeat_image=batched_mode,high_res_features=high_res_features,)
    prd_masks = img_predictor._transforms.postprocess_masks(low_res_masks, img_predictor._orig_hw[-1])# Upscale the masks to the original im
    prd_mask = torch.sigmoid(prd_masks[:, 0])

    # Clean the mask:
    cleaned_mask = prd_mask.detach().cpu().squeeze()

    clean_squash = cleaned_mask / cleaned_mask.max()
    cleaned_mask = torch.where(clean_squash >= 0.5, 1, torch.tensor(0.0))

    # Add new mask for first frame(0 - first frame, 1 - one object)
    frame_idx, obj_ids, video_res_masks = vid_predictor.add_new_mask(inference_state, ann_frame_idx, 1, cleaned_mask)

    # Composite video
    composite_frames = {}
    background_frames = {}

    video_segments = {}  # video_segments contains the per-frame segmentation results
    
    for out_frame_idx, out_obj_ids, out_mask_logits in vid_predictor.propagate_in_video(inference_state, start_frame_idx=ann_frame_idx, reverse=True):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

        assert len(video_segments[out_frame_idx].items()) == 1

        out_obj_id, out_mask = list(video_segments[out_frame_idx].items())[0]
        source_path = os.path.join(video_dir, frame_names[out_frame_idx])
        source_image = Image.open(source_path).convert('RGB')
        white_background = Image.new('RGB', source_image.size, (255, 255, 255, 255))

        # Apply the mask to the source image
        masked_image = Image.composite(source_image, white_background, Image.fromarray(out_mask.squeeze()))
        background_image = Image.composite(source_image, white_background, Image.fromarray(~out_mask.squeeze()))
        composite_frames[out_frame_idx] = np.array(masked_image.convert("RGB"))
        background_frames[out_frame_idx] = np.array(background_image.convert("RGB"))


    for out_frame_idx, out_obj_ids, out_mask_logits in vid_predictor.propagate_in_video(inference_state, start_frame_idx=ann_frame_idx, reverse=False):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

        assert len(video_segments[out_frame_idx].items()) == 1

        out_obj_id, out_mask = list(video_segments[out_frame_idx].items())[0]
        source_path = os.path.join(video_dir, frame_names[out_frame_idx])
        source_image = Image.open(source_path).convert('RGB')
        white_background = Image.new('RGB', source_image.size, (255, 255, 255, 255))

        # Apply the mask to the source image
        masked_image = Image.composite(source_image, white_background, Image.fromarray(out_mask.squeeze()))
        background_image = Image.composite(source_image, white_background, Image.fromarray(~out_mask.squeeze()))
        composite_frames[out_frame_idx] = np.array(masked_image.convert("RGB"))
        background_frames[out_frame_idx] = np.array(background_image.convert("RGB"))

    composite_frames = sorted(composite_frames.items(), key=lambda x: x[0])
    background_frames = sorted(background_frames.items(), key=lambda x: x[0])

    size = composite_frames[0][1].shape[1], composite_frames[0][1].shape[0]
    fps = 10
    composite_out = cv2.VideoWriter(replay_vidname, cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[0], size[1]), True)
    background_out = cv2.VideoWriter(background_vidname, cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[0], size[1]), True)


    for i in range(len(composite_frames)):
        data = composite_frames[i][1].astype('uint8')
        rgb_image = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        composite_out.write(rgb_image)

        data = background_frames[i][1].astype('uint8')
        rgb_image = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        background_out.write(rgb_image)


    composite_out.release()
    background_out.release()

    return {
        "video_segments" : video_segments,
        "composites" : composite_frames,
    }

def convert_image(input_path, output_path):
    # Read the image
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not read image {input_path}")
        return
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Create a binary mask where white pixels become black and non-white become white
    _, binary = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)
    cv2.imwrite(output_path, binary)
def process_directory(input_dir, output_dir):
    """Process all images in the input directory and save converted images to output directory"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_extensions = ('.jpg', '.jpeg', '.png')
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(image_extensions)]
    
    for i, image_file in enumerate(sorted(image_files)):
        input_path = os.path.join(input_dir, image_file)
        output_path = os.path.join(output_dir, f"{int(Path(image_file).stem)}.jpg")
        #print(output_path)
        
        convert_image(input_path, output_path)
        
def worker_fn(episodes, directory, gpu_idx, super_resolution, counter):
    # 把子进程所有输出重定向到 /dev/null
    if MUTE_OUTPUT:
        devnull = open(os.devnull, "w")
        sys.stdout = devnull
        sys.stderr = devnull

    # 绑定指定 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)

    import torch
    globals()['torch'] = torch
    from sam2.build_sam import build_sam2_video_predictor, build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    device = torch.device("cuda")

    # PyTorch 淡入式设置
    if device.type == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32  = True

    # 构建模型
    vid_predictor = build_sam2_video_predictor(MODEL_CONFIG, MODEL_CHECKPOINT, device=device)
    sam2_model    = build_sam2(MODEL_CONFIG, MODEL_CHECKPOINT, device=device)
    img_predictor = SAM2ImagePredictor(sam2_model)

    # 逐 episode 处理
    for ep in episodes:
        ep_dir = Path(directory) / str(ep)
        merge_dir = ep_dir / ("super_resolution_mask_frames" if super_resolution else "mask_frames")

        replay_mp4 = ep_dir / "trajectory_replay.mp4"
        replay_dir = ep_dir / "trajectory_replay_frames"
        bk_mp4     = ep_dir / "trajectory_background.mp4"
        mask_dir   = ep_dir / ("super_resolution_mask_frames" if super_resolution else "mask_frames")

        # a) 分割 & 合成
        src_frames = ep_dir / ("super_resolution_frames" if super_resolution else "frames")
        best_image_to_video_validation(str(src_frames), str(replay_mp4), str(bk_mp4),
                                       vid_predictor, img_predictor)
        # b) 抽帧
        extract_frames(str(replay_mp4), str(replay_dir))
        # c) 生成二值 mask
        process_directory(str(replay_dir), str(mask_dir))

        # 更新全局计数器
        with counter.get_lock():
            counter.value += 1   # 已完成 episode +1

# ────────────────────────── 主入口 ──────────────────────────
def main():
    parser = argparse.ArgumentParser(description="按 episode 范围或自动发现方式并行调用 worker_fn")
    parser.add_argument('--start', type=int, default=None, help='起始 episode（含）')
    parser.add_argument('--end',   type=int, default=None, help='结束 episode（不含）')
    parser.add_argument('--directory', type=str, required=True,
                        help='数据根目录，每个 episode 是一个纯数字子目录')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='并行进程 / GPU 数量 (≤ 可见 GPU 数)')
    parser.add_argument('--super_resolution', action='store_true', help='是否启用超分辨率处理')
    parser.add_argument('--list-file', action='store_true',
                        help='从 {directory}/needs_update.txt 读取 episode 列表')
    args = parser.parse_args()

    # — 生成待处理 episode 列表 —
    if args.start is not None and args.end is not None:
        episodes = list(range(args.start, args.end))
    elif args.list_file:
        txt_path = Path(args.directory) / "needs_update.txt"
        if not txt_path.is_file():
            raise SystemExit(f"未找到列表文件 {txt_path}")
        episodes = sorted({int(l) for l in txt_path.read_text().splitlines() if l.strip().isdigit()})
    else:
        episodes = sorted(int(p.name) for p in Path(args.directory).iterdir()
                          if p.is_dir() and p.name.isdigit())
    if not episodes:
        raise SystemExit("未发现任何 episode 目录")

    total = len(episodes)
    chunks = [[] for _ in range(args.num_workers)]
    for idx, ep in enumerate(episodes):
        chunks[idx % args.num_workers].append(ep)

    # 共享计数器
    counter = mp.Value('i', 0)

    # 启动子进程
    procs = []
    for gpu_idx, ep_chunk in enumerate(chunks):
        if not ep_chunk:
            continue
        p = mp.Process(target=worker_fn,
                       args=(ep_chunk, args.directory, gpu_idx,
                             args.super_resolution, counter),
                       daemon=False)
        p.start()
        procs.append(p)

    # 单一 tqdm 进度条
    last = 0
    with tqdm(total=total, desc="Episodes 完成进度", unit="ep") as pbar:
        while any(p.is_alive() for p in procs):
            time.sleep(1)  # 刷新频率 1 s
            with counter.get_lock():
                done = counter.value
            if done > last:
                pbar.update(done - last)
                last = done
        # 避免极少数 race condition
        pbar.update(total - last)

    # 等待子进程退出
    for p in procs:
        p.join()

    print("✅ 全部 episode 处理完成")

if __name__ == "__main__":
    main()

'''
python /home/guanhuaji/oxeplusplus/sam2/sam2/inference.py --start 100 --end 1000 --directory /home/guanhuaji/load_datasets/austin_buds_dataset_converted_externally_to_rlds
python /home/guanhuaji/oxeplusplus/sam2/sam2/inference.py --start 100 --end 1000 --directory /home/guanhuaji/load_datasets/toto
python /home/guanhuaji/oxeplusplus/sam2/sam2/inference.py --start 0 --end 201 --directory /home/guanhuaji/load_datasets/kaist_nonprehensile_converted_externally_to_rlds
python /home/guanhuaji/oxeplusplus/sam2/sam2/inference.py --start 0 --end 0 --directory /home/guanhuaji/load_datasets/austin_buds_dataset_converted_externally_to_rlds

python /home/guanhuaji/oxeplusplus/sam2/sam2/inference.py --directory /home/guanhuaji/load_datasets/austin_sailor_dataset_converted_externally_to_rlds/train

python /home/guanhuaji/oxeplusplus/sam2/sam2/inference.py --directory /home/guanhuaji/load_datasets/language_table/train --start 18 --end 19
python /home/guanhuaji/oxeplusplus/sam2/sam2/inference.py --start 16000 --end 17000 --directory /home/guanhuaji/load_datasets/utaustin_mutex
python /home/guanhuaji/oxeplusplus/sam2/sam2/inference.py --directory /home/guanhuaji/load_datasets/berkeley_autolab_ur5/test
python /home/guanhuaji/oxeplusplus/sam2/sam2/inference.py --directory /home/guanhuaji/load_datasets/nyu_franka_play_dataset_converted_externally_to_rlds/val

python /home/guanhuaji/oxeplusplus/sam2/sam2/inference.py --directory /home/guanhuaji/load_datasets/nyu_franka_play_dataset_converted_externally_to_rlds/train --list-file

python /home/guanhuaji/oxeplusplus/sam2/sam2/inference.py --directory /home/guanhuaji/load_datasets/toto/test

python /home/guanhuaji/oxeplusplus/sam2/sam2/inference.py --start 0 --end 50 --directory /home/guanhuaji/load_datasets/austin_buds_dataset_converted_externally_to_rlds/train

python /home/guanhuaji/oxeplusplus/sam2/sam2/inference.py --directory /home/guanhuaji/load_datasets/ucsd_kitchen_dataset_converted_externally_to_rlds/train

python /home/guanhuaji/oxeplusplus/sam2/sam2/inference.py --directory /home/guanhuaji/load_datasets/taco_play/train

python /home/guanhuaji/oxeplusplus/sam2/sam2/inference.py --directory /home/guanhuaji/load_datasets/iamlab_cmu_pickup_insert_converted_externally_to_rlds/train
'''

'''
python /home/guanhuaji/oxeplusplus/sam2/sam2/inference.py --directory /home/guanhuaji/load_datasets/austin_sailor_dataset_converted_externally_to_rlds/train --list-file
python /home/guanhuaji/oxeplusplus/sam2/sam2/inference.py --directory /home/guanhuaji/load_datasets/utaustin_mutex/train --list-file
python /home/guanhuaji/oxeplusplus/sam2/sam2/inference.py --directory /home/guanhuaji/load_datasets/taco_play/train --list-file
python /home/guanhuaji/oxeplusplus/sam2/sam2/inference.py --directory /home/guanhuaji/load_datasets/taco_play/test

python /home/guanhuaji/oxeplusplus/sam2/sam2/inference.py --directory /home/guanhuaji/load_datasets/viola/train --list-file
python /home/guanhuaji/oxeplusplus/sam2/sam2/inference.py --directory /home/guanhuaji/load_datasets/bridge/train
python /home/guanhuaji/oxeplusplus/sam2/sam2/inference.py --directory /home/abrashid/OXE_inpainting/fractal20220817_data/train


'''