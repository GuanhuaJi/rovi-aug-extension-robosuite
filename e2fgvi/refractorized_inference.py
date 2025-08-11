import cv2
from PIL import Image
import numpy as np
import importlib
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import animation
import torch
import re
from core.utils import to_tensors
from torchvision.utils import save_image


class VideoProcessor:
    def __init__(self, args):
        self.args = args
        self.ref_length = args['step']
        self.num_ref = args['num_ref']
        self.neighbor_stride = args['neighbor_stride']
        self.default_fps = args['savefps']
        self.frame_save_dir = args['save_frame']

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.args["model"] == "e2fgvi":
            self.size = (432, 240)
        elif self.args["set_size"]:
            self.size = (self.args['width'], self.args['height'])
        else:
            self.size = None

        self.net = importlib.import_module('model.' + self.args['model'])
        self.model = self.net.InpaintGenerator().to(self.device)
        self.data = torch.load(self.args['ckpt'], map_location=self.device)
        self.model.load_state_dict(self.data)
        self.model.eval()


    @staticmethod
    def natural_sort_key(s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]

    def get_ref_index(self, f, neighbor_ids, length):
        ref_index = []
        if self.num_ref == -1:
            for i in range(0, length, self.ref_length):
                if i not in neighbor_ids:
                    ref_index.append(i)
        else:
            start_idx = max(0, f - self.ref_length * (self.num_ref // 2))
            end_idx = min(length, f + self.ref_length * (self.num_ref // 2))
            for i in range(start_idx, end_idx + 1, self.ref_length):
                if i not in neighbor_ids:
                    if len(ref_index) > self.num_ref:
                        break
                    ref_index.append(i)
        return ref_index

    def read_mask(self, mpath, size):
        masks = []
        mnames = os.listdir(mpath)
        mnames.sort(key=self.natural_sort_key)
        for mp in mnames:
            m = Image.open(os.path.join(mpath, mp))
            m = m.resize(size, Image.NEAREST)
            m = np.array(m.convert('L'))
            m = np.array(m > 0).astype(np.uint8)
            m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)), iterations=4)
            masks.append(Image.fromarray(m * 255))
        return masks

    def process_masks(self, masks):
        new_masks = []
        for i in range(masks.shape[0]):
            m = Image.fromarray(masks[i])
            m = m.resize(self.size, Image.NEAREST)
            m = np.array(m.convert('L'))
            m = np.array(m > 0).astype(np.uint8)
            m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)), iterations=4)
            new_masks.append(Image.fromarray(m * 255))
        return new_masks

    def read_frame_from_videos(self):
        vname = self.args.video
        frames = []
        if self.args.use_mp4:
            vidcap = cv2.VideoCapture(vname)
            success, image = vidcap.read()
            while success:
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                frames.append(image)
                success, image = vidcap.read()
        else:
            lst = os.listdir(vname)
            lst.sort(key=self.natural_sort_key)
            fr_lst = [os.path.join(vname, name) for name in lst]
            for fr in fr_lst:
                image = cv2.imread(fr)
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                frames.append(image)
        return frames

    def resize_frames(self, frames, size=None):
        if size is not None:
            frames = [f.resize(size) for f in frames]
        else:
            size = frames[0].size
        return frames, size

    def main_worker(self, imgs, masks):  # M x 3 x H x W, M x 1 x H x W
        with torch.no_grad():
            frames = imgs
            imgs = torch.from_numpy(imgs).permute(0, 3, 1, 2).float().unsqueeze(0) / 255.0
            imgs = imgs * 2 - 1
            processed_masks = self.process_masks(masks)
            masks = to_tensors()(processed_masks).unsqueeze(0)
            masks = masks.float()  # Ensure masks are float

            cpu_masks = masks.cpu().numpy().squeeze().astype(np.uint8)[..., None]
            
            imgs, masks = imgs.to(self.device), masks.to(self.device)
            video_length = imgs.shape[1]
            h, w = imgs.shape[3], imgs.shape[4]
            # comp_frames = [None] * video_length
            put_frame = [False] * video_length
            comp_frames = np.zeros((video_length, h, w, 3))

            for f in tqdm(range(0, video_length, self.neighbor_stride)):
                neighbor_ids = [i for i in range(max(0, f - self.neighbor_stride), min(video_length, f + self.neighbor_stride + 1))]
                ref_ids = self.get_ref_index(f, neighbor_ids, video_length)
                selected_imgs = imgs[:1, neighbor_ids + ref_ids]
                selected_masks = masks[:1, neighbor_ids + ref_ids]
                masked_imgs = selected_imgs * (1 - selected_masks)
                mod_size_h = 60
                mod_size_w = 108
                h_pad = (mod_size_h - h % mod_size_h) % mod_size_h
                w_pad = (mod_size_w - w % mod_size_w) % mod_size_w
                masked_imgs = torch.cat([masked_imgs, torch.flip(masked_imgs, [3])], 3)[:, :, :, :h + h_pad, :]
                masked_imgs = torch.cat([masked_imgs, torch.flip(masked_imgs, [4])], 4)[:, :, :, :, :w + w_pad]
                pred_imgs, _ = self.model(masked_imgs, len(neighbor_ids))
                pred_imgs = pred_imgs[:, :, :h, :w]
                pred_imgs = (pred_imgs + 1) / 2
                pred_imgs = pred_imgs.cpu().permute(0, 2, 3, 1).numpy() * 255
                for i in range(len(neighbor_ids)):
                    idx = neighbor_ids[i]
                    img = pred_imgs[i] * cpu_masks[idx] + frames[idx] * (1 - cpu_masks[idx])
                    # if comp_frames[idx] is None:
                    #     comp_frames[idx] = img
                    # else:
                    #     comp_frames[idx] = comp_frames[idx].astype(np.float32) * 0.5 + img.astype(np.float32) * 0.5
                    if put_frame[idx]:
                        comp_frames[idx] = comp_frames[idx].astype(np.float32) * 0.5 + img.astype(np.float32) * 0.5
                    else:
                        comp_frames[idx] = img
                        put_frame[idx] = True

            # self.save_video(comp_frames, (256, 256))
            return comp_frames.astype(np.uint8)
        
    def save_video(self, comp_frames, size):
        h, w = size[1], size[0]
        save_dir_name = 'results'
        ext_name = '_results.mp4'
        save_base_name = self.args['video.split']
        save_name = save_base_name.replace('.mp4', ext_name) if self.args['use_mp4'] else save_base_name + ext_name
        if not os.path.exists(save_dir_name):
            os.makedirs(save_dir_name)
        save_path = os.path.join(save_dir_name, save_name)
        writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), self.args['savefps'], (w, h))
        for f in range(len(comp_frames)):
            comp = comp_frames[f].astype(np.uint8)
            writer.write(cv2.cvtColor(comp, cv2.COLOR_RGB2BGR))
            if not os.path.exists(self.frame_save_dir):
                os.makedirs(self.frame_save_dir)
            frame_path = os.path.join(self.frame_save_dir, f'{f:04d}.png')
            cv2.imwrite(frame_path, cv2.cvtColor(comp, cv2.COLOR_RGB2BGR))
        writer.release()
        print(f'Finish test! The result video is saved in: {save_path}.')

        self.show_results(comp_frames)

    def show_results(self, comp_frames):
        fig = plt.figure('Let us enjoy the result')
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.axis('off')
        ax1.set_title('Original Video')
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.axis('off')
        ax2.set_title('Our Result')
        imdata1 = ax1.imshow(comp_frames[0].astype(np.uint8))
        imdata2 = ax2.imshow(comp_frames[0].astype(np.uint8))

        def update(idx):
            imdata1.set_data(comp_frames[idx].astype(np.uint8))
            imdata2.set_data(comp_frames[idx].astype(np.uint8))

        fig.tight_layout()
        anim = animation.FuncAnimation(fig, update, frames=len(comp_frames), interval=50)
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="E2FGVI")
    parser.add_argument("-v", "--video", type=str, required=True)
    parser.add_argument("-save", "--save_frame", type=str, required=True)
    parser.add_argument("-c", "--ckpt", type=str, required=True)
    parser.add_argument("-m", "--mask", type=str, required=True)
    parser.add_argument("--model", type=str, choices=['e2fgvi', 'e2fgvi_hq'])
    parser.add_argument("--step", type=int, default=10)
    parser.add_argument("--num_ref", type=int, default=-1)
    parser.add_argument("--neighbor_stride", type=int, default=1)
    parser.add_argument("--savefps", type=int, default=24)
    parser.add_argument("--set_size", action='store_true')
    parser.add_argument("--width", type=int, default=432)
    parser.add_argument("--height", type=int, default=240)
    args = parser.parse_args
