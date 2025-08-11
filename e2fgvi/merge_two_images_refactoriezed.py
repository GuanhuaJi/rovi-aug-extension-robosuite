import os
from PIL import Image
import numpy as np
import cv2
import re
import random

class ImageProcessor:
    def __init__(self, base_dir):
        self.base_dir = base_dir

    def change_brightness(self, img, mean_value=100, mask=None, randomness=0):
        value = mean_value + random.randint(-randomness, randomness)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        if mask is None:
            mask = np.ones_like(v)
        else:
            mask = mask.squeeze()
        
        if value > 0:
            lim = 255 - value
            v[(v > lim) & (mask == 1)] = 255
            v[(v <= lim) & (mask == 1)] += value
        else:
            lim = -value
            v[(v < lim) & (mask == 1)] = 0
            v[(v >= lim) & (mask == 1)] -= lim

        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return img

    @staticmethod
    def natural_sort_key(s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

    def segment_object(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_bound = np.array([0, 0, 230])
        upper_bound = np.array([180, 50, 255])
        
        color_mask = cv2.inRange(hsv, lower_bound, upper_bound)
        object_mask = cv2.bitwise_not(color_mask)
        
        edges = cv2.Canny(image, 100, 200)
        combined_mask = cv2.bitwise_or(object_mask, edges)
        
        kernel = np.ones((3, 3), np.uint8)
        combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        eroded_mask = cv2.erode(combined_mask, kernel=np.ones((5, 5), np.uint8), iterations=1)
        final_mask = cv2.GaussianBlur(eroded_mask, (3, 3), 0)
        
        return final_mask

    def paste_objects(self, background, object):
        obj_image = object
        background = background
        obj_image = Image.fromarray(obj_image)
        background = Image.fromarray(background)
        
        # obj_image = obj_image.crop((obj_image.width // 2, 0, obj_image.width, obj_image.height))
        mask = self.segment_object(np.array(obj_image))
        
        # brightened_obj_image = self.change_brightness(np.array(obj_image), mean_value=50, randomness=0)
        brightened_obj_image_augmented = self.change_brightness(np.array(obj_image), mean_value=50, randomness=30)
        
        mask = Image.fromarray(mask).convert("L")

        # Comment after debugging
        # mask.save('test_mask.png')

        # obj_image_brighted = Image.fromarray(brightened_obj_image).convert('RGBA')
        obj_image_augmented = Image.fromarray(brightened_obj_image_augmented).convert('RGBA')
        # obj_image_augmented.save('test_obj.png')

        if mask.size != obj_image_augmented.size:
            mask = mask.resize(obj_image_augmented.size, Image.ANTIALIAS)
        
        # Save brightened images
        # brighted_background = background.copy()
        # brighted_background.paste(obj_image_brighted, (0, 0), mask)
        # # output_filename_brighted = f"output_{i:04d}.png"
        # # output_path_brighted = os.path.join(output_dir_brighted, output_filename_brighted)
        # brighted_background.save('test_brighted.png')
        # print(f"Saved {output_path_brighted}")

        # Save augmented images
        augmented_background = background.copy()
        augmented_background.paste(obj_image_augmented, (0, 0), mask)
        # augmented_background.save('test_augmented.png')
        # output_filename_augmented = f"output_0.png"
        # output_path_augmented = os.path.join('./', output_filename_augmented)
        # augmented_background.save(output_path_augmented)
        # import pdb; pdb.set_trace()
        # print(f"Saved {output_path_augmented}")
        return np.array(augmented_background)

    def process_all_subfolders(self):
        for folder_name in os.listdir(self.base_dir):
            folder_path = os.path.join(self.base_dir, folder_name)
            if os.path.isdir(folder_path):
                background_dir = os.path.join(folder_path, 'inpainted_frames')
                object_dir = os.path.join(folder_path, 'r2r_transferred')
                output_dir_base = os.path.join(folder_path, 'final_images_brighted')
                if os.path.isdir(background_dir) and os.path.isdir(object_dir):
                    backgrounds = [np.array(Image.open(os.path.join(background_dir, f))) for f in sorted(os.listdir(background_dir), key=self.natural_sort_key) if f.endswith(('.png', '.jpg', '.jpeg'))]
                    objects = [np.array(Image.open(os.path.join(object_dir, f))) for f in sorted(os.listdir(object_dir), key=self.natural_sort_key) if f.endswith(('.png', '.jpg', '.jpeg'))]
                    self.paste_objects(backgrounds, objects, output_dir_base)
                else:
                    print(f"Missing inpainted_frames or r2r_transferred in {folder_path}")

if __name__ == "__main__":
    # Example usage
    base_dir = '/rscratch/cfxu/diffusion-RL/style-transfer/data/parsered_images_robo/2024-05-10-cup-franka-gripper_mask'
    processor = ImageProcessor(base_dir)

    test_background = np.array(Image.open('/home/kdharmarajan/mirage2/test/inpainted_background_78.png'))
    test_object = np.array(Image.open('/home/kdharmarajan/mirage2/test/robot_aug_imgs_78.png'))
    processor.paste_objects(test_background, test_object)
