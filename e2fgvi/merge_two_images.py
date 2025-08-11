import os
from PIL import Image
import numpy as np
import cv2
import re
import random

def change_brightness(img, mean_value=100, mask=None, randomness=0):
    value = mean_value + random.randint(-randomness, randomness)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    if mask is None:
        mask = np.ones_like(v)
    else:
        mask = mask.squeeze()
    # Apply mask to the brightness channel
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

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]

def segment_object(image):
    # Convert image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define the range for white color in HSV
    lower_bound = np.array([0, 0, 200])  # Lower bound for white
    upper_bound = np.array([180, 50, 255])  # Upper bound for white
    
    # Create a binary mask where white represents the background
    color_mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    # Invert the color mask to get the object mask
    object_mask = cv2.bitwise_not(color_mask)
    
    # Apply Canny edge detection
    edges = cv2.Canny(image, 100, 200)
    
    # Combine the object mask and edges
    combined_mask = cv2.bitwise_or(object_mask, edges)
    
    # Apply dilation to reduce the fuzziness of the boundary
    kernel = np.ones((3, 3), np.uint8)
    combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)
    
    # Optional: Apply morphological operations to clean up the mask
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    
    # Apply erosion to remove internal masks
    eroded_mask = cv2.erode(combined_mask, kernel=np.ones((5, 5), np.uint8), iterations=1)
    
    # Apply Gaussian blur to smooth the edges
    final_mask = cv2.GaussianBlur(eroded_mask, (3, 3), 0)
    
    return final_mask

def paste_objects(background_dir, object_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # List backgrounds and objects
    backgrounds = sorted([f for f in os.listdir(background_dir) if f.endswith(('.png', '.jpg', '.jpeg'))], key=natural_sort_key)
    objects = sorted([f for f in os.listdir(object_dir) if f.endswith(('.png', '.jpg', '.jpeg'))], key=natural_sort_key)

    # Ensure matching counts
    if len(backgrounds) != len(objects):
        print("Error: The number of background images does not match the number of object images.")
        return

    # Loop over paired images
    for bg_filename, obj_filename in zip(backgrounds, objects):
        background_path = os.path.join(background_dir, bg_filename)
        object_path = os.path.join(object_dir, obj_filename)
        
        # Load background image
        background = Image.open(background_path).convert('RGB')
        if background is None:
            print(f"Error: Failed to load background image {background_path}")
            continue

        # Load object image
        obj_image = Image.open(object_path).convert('RGB')
        if obj_image is None:
            print(f"Error: Failed to load object image {object_path}")
            continue
        
        # Crop the object image to use the right half
        obj_image = obj_image.crop((obj_image.width // 2, 0, obj_image.width, obj_image.height))
       
        # Create a mask from the object image
        mask = segment_object(np.array(obj_image))
        
        # Apply brightness change using the mask
        brightened_obj_image = change_brightness(np.array(obj_image), mean_value=50)

        # Convert mask to a format compatible with PIL (mode 'L' for 8-bit pixels, black and white)
        mask = Image.fromarray(mask).convert("L")
        
        # Convert the brightened object image back to PIL image
        obj_image = Image.fromarray(brightened_obj_image).convert('RGBA')

        # Ensure the mask and object image have the same size
        if mask.size != obj_image.size:
            mask = mask.resize(obj_image.size, Image.ANTIALIAS)
        
        # Paste object using mask at the location (0, 0)
        background.paste(obj_image, (0, 0), mask)

        # Save the composited image
        output_filename = f"{obj_filename}"
        output_path = os.path.join(output_dir, output_filename)
        background.save(output_path)
        print(f"Saved {output_path}")

def process_all_subfolders(base_dir):
    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)
        if os.path.isdir(folder_path):
            background_dir = os.path.join(folder_path, 'inpainted_frames')
            object_dir = os.path.join(folder_path, 'r2r_transferred')
            output_dir = os.path.join(folder_path, 'final_images_brighted')
            if os.path.isdir(background_dir) and os.path.isdir(object_dir):
                paste_objects(background_dir, object_dir, output_dir)
            else:
                print(f"Missing inpainted_frames or r2r_transferred in {folder_path}")

# Example usage
base_dir = '/rscratch/cfxu/diffusion-RL/style-transfer/data/parsered_images_robo/2024-05-10-cup-franka-gripper_mask'
process_all_subfolders(base_dir)
