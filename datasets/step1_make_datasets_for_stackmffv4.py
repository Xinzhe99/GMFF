#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Xinzhe Xie
# @University  : ZheJiang University

"""
This script generates multi-focus image stacks from single images and their depth maps.
It creates depth-based layers and generates randomized focus index maps. The process includes:
1. Depth map quantization into layers
2. Layer generation with varying blur levels
3. Layer order randomization
4. Focus index map generation (0 to N-1)
5. Data organization with npy format for focus indices

Input file structure:
- original_path/
  - *.jpg or *.png              # Original image files
- depth_path/
  - *.png                       # Corresponding depth map files (same name as original images)

Output file structure:
- output_path/
  - AiF/
    - *.jpg                     # Original image copies
  - depth/
    - *.png                     # Depth map copies
  - focus_stack/
    - [image_name]/
      - 0.jpg, 1.jpg, ..., N.jpg # Multi-focus image stack (N is randomly selected layer count)
      - layer_order.npy          # Layer order mapping information
  - focus_index_gt/
    - *.npy                      # Focus index ground truth maps (numpy array format)

Example usage:
    python 1_datasets_for_stackmffv4.py --original_path ./specific_dataset/imgs --depth_path ./specific_dataset_name/depths --output_path /path/to/stackmffv4_training_datasets/specific_dataset_name --max_workers 8
"""

import os
import glob
import cv2
import numpy as np
from tqdm import tqdm
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import argparse


def generate_focus_layers_and_index(img, img_depth, num_regions):
    """
    Generate focus layers from depth map and create randomized focus index map.
    
    Args:
        img (numpy.ndarray): Input image
        img_depth (numpy.ndarray): Corresponding depth map
        num_regions (int): Number of depth regions/layers
    
    Returns:
        tuple: (list of focus layers, focus_index_gt map, layer_order_mapping)
    """
    # Create blurred image layers with different blur levels
    blur_kernels = [2 * i + 1 for i in range(num_regions)]
    blurred_images = []
    for kernel_size in blur_kernels:
        blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
        blurred_images.append(blurred)
    
    # Quantize depth map into layers
    ref_points = np.linspace(0, 255, num_regions + 1)
    quantized_depth = np.digitize(img_depth, ref_points) - 1
    # Ensure values are within valid range
    quantized_depth = np.clip(quantized_depth, 0, num_regions - 1)
    
    # Create focused images for each depth layer
    focus_layers = []
    for focus_layer_idx in range(num_regions):
        layer_result = np.zeros_like(img)
        
        for depth_level in range(num_regions):
            mask = (quantized_depth == depth_level)
            # Calculate blur level relative to current focus layer
            blur_distance = abs(depth_level - focus_layer_idx)
            blur_idx = min(blur_distance, len(blurred_images) - 1)
            layer_result[mask] = blurred_images[blur_idx][mask]
        
        # Fill possible black areas
        black_mask = np.all(layer_result == 0, axis=2)
        layer_result[black_mask] = img[black_mask]
        
        focus_layers.append(layer_result)
    
    # Create mapping from original order to shuffled order
    original_order = list(range(num_regions))
    shuffled_order = original_order.copy()
    random.shuffle(shuffled_order)
    
    # Create mapping from original depth index to new order index
    depth_to_new_index = {original_idx: new_idx for new_idx, original_idx in enumerate(shuffled_order)}
    
    # Reorder layers according to shuffled order
    shuffled_layers = [focus_layers[i] for i in shuffled_order]
    
    # Create focus index map: map original depth indices to new layer order indices
    focus_index_gt = np.zeros_like(quantized_depth, dtype=np.int64)
    for original_depth_idx in range(num_regions):
        mask = (quantized_depth == original_depth_idx)
        new_layer_idx = depth_to_new_index[original_depth_idx]
        focus_index_gt[mask] = new_layer_idx
    
    return shuffled_layers, focus_index_gt, shuffled_order


def process_single_image(args):
    """
    Process a single image for multi-threading.
    
    Args:
        args (tuple): (pic_path, depth_path, output_path, num_regions_list)
    
    Returns:
        tuple: (success, image_name, message)
    """
    pic_path, depth_path, output_path, num_regions_list = args
    
    # Initialize name to avoid unbound variable error
    name = "unknown"
    
    try:
        filename = os.path.basename(pic_path)
        name, _ = os.path.splitext(filename)

        img = cv2.imread(pic_path)
        img_depth_path = os.path.join(depth_path, name + '.png')
        
        if not os.path.exists(img_depth_path):
            return False, name, f"Depth map not found: {img_depth_path}"
            
        img_depth = cv2.imread(img_depth_path, 0)
        
        if img is None or img_depth is None:
            return False, name, f"Failed to load image or depth map"

        num_regions = random.choice(num_regions_list)
        focus_layers, focus_index_gt, layer_order = generate_focus_layers_and_index(img, img_depth, num_regions)

        # Create separate folders for data organization
        original_folder = os.path.join(output_path, 'AiF')
        depth_folder = os.path.join(output_path, 'depth')
        focus_stack_folder = os.path.join(output_path, 'focus_stack')
        focus_index_folder = os.path.join(output_path, 'focus_index_gt')

        for folder in [original_folder, depth_folder, focus_stack_folder, focus_index_folder]:
            os.makedirs(folder, exist_ok=True)

        # Save the original image
        cv2.imwrite(os.path.join(original_folder, f'{name}.jpg'), img)

        # Save the depth image
        cv2.imwrite(os.path.join(depth_folder, f'{name}.png'), img_depth)

        # Save the focus index ground truth as npy file
        np.save(os.path.join(focus_index_folder, f'{name}.npy'), focus_index_gt)

        # Save the focus stack images
        focus_image_folder = os.path.join(focus_stack_folder, name)
        os.makedirs(focus_image_folder, exist_ok=True)
        for i, layer in enumerate(focus_layers):
            cv2.imwrite(os.path.join(focus_image_folder, f'{i}.jpg'), layer)
        
        # Save layer order mapping for reference as a numpy array
        order_array = np.array(layer_order, dtype=np.int32)
        np.save(os.path.join(focus_image_folder, 'layer_order.npy'), order_array)
        
        return True, name, f"Successfully processed {num_regions} layers"
        
    except Exception as e:
        return False, name, f"Error processing: {str(e)}"


def process_images(original_path, depth_path, output_path, max_workers=None):
    """
    Process a set of images and their depth maps to create multi-focus image stacks using multi-threading.
    
    Args:
        original_path (str): Path to original images
        depth_path (str): Path to depth maps
        output_path (str): Path to save generated data
        max_workers (int): Maximum number of worker threads. If None, uses CPU count.
    """
    if max_workers is None:
        max_workers = min(multiprocessing.cpu_count(), 8)  # Limit max threads to avoid overload
    
    num_regions_list = list(range(2, 25))  # All numbers from 2 to 24
    os.makedirs(output_path, exist_ok=True)

    # Get all image files
    original_images = glob.glob(os.path.join(original_path, '*.jpg')) + glob.glob(os.path.join(original_path, '*.png'))
    
    if not original_images:
        print(f"No images found in {original_path}")
        return
    
    print(f"Found {len(original_images)} images to process")
    print(f"Using {max_workers} worker threads")
    
    # Prepare argument list
    task_args = [(pic_path, depth_path, output_path, num_regions_list) for pic_path in original_images]
    
    # Statistics
    success_count = 0
    failed_count = 0
    failed_images = []
    
    # Process using thread pool
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_path = {executor.submit(process_single_image, args): args[0] for args in task_args}
        
        # Show progress with tqdm
        with tqdm(total=len(original_images), desc="Processing images") as pbar:
            for future in as_completed(future_to_path):
                pic_path = future_to_path[future]
                try:
                    success, name, message = future.result()
                    if success:
                        success_count += 1
                    else:
                        failed_count += 1
                        failed_images.append((name, message))
                        print(f"\nFailed: {message}")
                except Exception as e:
                    failed_count += 1
                    failed_images.append((os.path.basename(pic_path), str(e)))
                    print(f"\nException: {str(e)}")
                finally:
                    pbar.update(1)
    
    # Print processing statistics
    print(f"\n=== Processing Complete ===")
    print(f"Successfully processed: {success_count} images")
    print(f"Failed: {failed_count} images")
    
    if failed_images:
        print("\nFailed images:")
        for name, error in failed_images:
            print(f"  - {name}: {error}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate multi-focus image stacks from single images and depth maps.')
    parser.add_argument('--original_path', type=str, required=True, 
                        help='Path to original images')
    parser.add_argument('--depth_path', type=str, required=True, 
                        help='Path to depth maps')
    parser.add_argument('--output_path', type=str, required=True, 
                        help='Path to save generated data')
    parser.add_argument('--max_workers', type=int, default=None, 
                        help='Maximum number of worker threads. If None, uses CPU count (max 8).')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_images(args.original_path, args.depth_path, args.output_path, args.max_workers)