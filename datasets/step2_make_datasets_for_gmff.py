# -*- coding: utf-8 -*-
# @Author  : XinZhe Xie
# @University  : ZheJiang University

"""
The following file directory structure is required to run this script:

test_root/
    dataset1/           # Dataset root directory (e.g. NYU-V2, ADE, etc.)
        TR/             # Training/testing data directory
            focus_stack/ # Original image stack directory
                scene1/  # Scene directory
                    img1.png  # Image files (supports .png, .jpg, .jpeg, .bmp)
                    img2.png
                    ...
                scene2/
                    ...
            AiF_missing/ # Output directory (automatically created by the script to save fusion results)

The model file should be located at ../weights/stackmffv4.pth or specified via the --model_path parameter.

Command line arguments:
    --model_path: Path to the model weights file
    --test_root: Root directory of test data
    --test_datasets: List of dataset names to process
    --batch_size: Batch processing size
    --num_workers: Number of data loading worker threads
    --gpu_id: GPU ID to use (-1 for automatic selection)
    --max_image_size: Maximum image size limit

Usage examples:
    python 2_datasets_for_gmff.py --test_datasets NYU-V2 ADE
    python 2_datasets_for_gmff.py --test_root /path/to/datasets --test_datasets DIODE DUTS --gpu_id 0
"""

import argparse
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm
from stackmffv4.network import StackMFF_V4
import random
import re
import time
import cv2
import numpy as np
import torch.nn.functional as F
import os
import matplotlib
import matplotlib.colors
import matplotlib.cm
import torch
import torch.nn as nn

# python 2_datasets_gmff.py --test_datasets ADE DIODE DUTS NYU-V2 Cityscapes
def parse_args():
    """
    Parse command line arguments for batch evaluation of multiple datasets
    
    Returns:
        argparse.Namespace: Parsed arguments containing:
            - model_path: Path to the trained model weights
            - test_root: Root directory containing test datasets
            - test_datasets: List of dataset names to evaluate
            - batch_size: Batch size for evaluation
            - num_workers: Number of data loading workers
            - gpu_id: GPU ID to use for inference
            - max_image_size: Maximum allowed image size to prevent memory issues
    """
    parser = argparse.ArgumentParser(description="Make training datasets for GMFF")
    parser.add_argument('--model_path', type=str, default='../weights/stackmffv4.pth',
                        help='Path to the trained model weights')
    parser.add_argument('--test_root', type=str, default='/path/to/stackmffv4_training_datasets',
                        help='Path to test data root directory')
    parser.add_argument('--test_datasets', nargs='+',
                        default=['NYU-V2'],
                        help='datasets to process')
    parser.add_argument('--batch_size', type=int, default=1, 
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=0, 
                        help='Number of data loading workers')
    parser.add_argument('--gpu_id', type=int, default=-1,
                        help='GPU ID to use for inference (-1 for auto selection)')
    parser.add_argument('--max_image_size', type=int, default=1024,
                        help='Maximum allowed image size to prevent memory issues')
    return parser.parse_args()

def resize_to_multiple_of_32(image, max_size=1024):
    """
    Resize input image tensor to dimensions that are multiples of 32, with a maximum size limit
    
    Args:
        image (torch.Tensor): Input image tensor
        max_size (int): Maximum allowed size for the larger dimension
        
    Returns:
        tuple: (resized_image, (original_height, original_width))
            - resized_image: Image resized to multiple of 32
            - tuple of original dimensions
    """
    h, w = image.shape[-2:]
    
    # If image dimensions exceed the maximum limit, scale first
    if max(h, w) > max_size:
        scale_factor = max_size / max(h, w)
        h = int(h * scale_factor)
        w = int(w * scale_factor)
        
        # Ensure scaled dimensions are multiples of 32
        h = ((h - 1) // 32 + 1) * 32
        w = ((w - 1) // 32 + 1) * 32
    
    # Use stricter dimension constraints to ensure no odd dimensions after network processing
    new_h = ((h - 1) // 32 + 1) * 32
    new_w = ((w - 1) // 32 + 1) * 32
    
    # Ensure new dimensions are multiples of 64 to avoid odd dimension issues
    if new_h % 64 != 0:
        new_h = ((new_h - 1) // 64 + 1) * 64
    if new_w % 64 != 0:
        new_w = ((new_w - 1) // 64 + 1) * 64
        
    resized_image = F.interpolate(image, size=(new_h, new_w), mode='bilinear', align_corners=False)
    return resized_image, (h, w)

def gray_to_colormap(img, cmap='rainbow'):
    """
    Convert grayscale image to colormap visualization
    
    Args:
        img (numpy.ndarray): Input grayscale image (normalized to [0,1])
        cmap (str): Matplotlib colormap name
        
    Returns:
        numpy.ndarray: RGB colormap visualization
    """
    img = np.clip(img, 0, 1)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    cmap_m = matplotlib.cm.get_cmap(cmap)
    map_obj = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap_m)
    colormap = (map_obj.to_rgba(img)[:, :, :3] * 255).astype(np.uint8)
    return colormap

class TestDataset(Dataset):
    """
    Dataset class for loading and processing image stacks
    
    Args:
        root_dir (str): Root directory containing image stacks
        transform (callable, optional): Optional transform to be applied on images
        subset_fraction (float, optional): Fraction of total stacks to use
    """
    def __init__(self, root_dir, transform=None, subset_fraction=1.0):
        self.root_dir = root_dir
        self.transform = transform
        self.image_stacks = []
        self.stack_names = []
        self.stack_extensions = []  # Store the extension of the first image in each stack

        # Get all stack directories and optionally sample a subset
        all_stacks = sorted(os.listdir(root_dir))
        subset_size = int(len(all_stacks) * subset_fraction)
        selected_stacks = random.sample(all_stacks, subset_size)

        # Load image paths for each stack
        for stack_name in selected_stacks:
            stack_path = os.path.join(root_dir, stack_name)
            if os.path.isdir(stack_path):
                image_stack = []
                first_img_extension = None
                for img_name in sorted(os.listdir(stack_path), key=self.sort_key):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        img_path = os.path.join(stack_path, img_name)
                        image_stack.append(img_path)
                        # Store the extension of the first image
                        if first_img_extension is None:
                            first_img_extension = os.path.splitext(img_name)[1].lower()

                if image_stack:
                    self.image_stacks.append(image_stack)
                    self.stack_names.append(stack_name)
                    self.stack_extensions.append(first_img_extension if first_img_extension else '.png')  # Default to .png
                else:
                    print(f'Failed to read image stack: {stack_name}')

    def __len__(self):
        return len(self.image_stacks)

    def __getitem__(self, idx):
        """
        Get a single image stack
        
        Args:
            idx (int): Index of the stack
            
        Returns:
            tuple: (stack_tensor, stack_name, num_images, stack_extension)
                - stack_tensor: Tensor containing the image stack
                - stack_name: Name of the stack
                - num_images: Number of images in the stack
                - stack_extension: Extension of images in the stack
        """
        image_stack = self.image_stacks[idx]
        stack_name = self.stack_names[idx]
        stack_extension = self.stack_extensions[idx]

        # Randomly drop 0%-50% of images, keeping at least 1 image
        # But only apply random dropping if there are more than 2 images
        # For stacks with only 2 images, don't drop any images to ensure we have at least 2
        num_images = len(image_stack)
        
        if num_images > 2:
            # For stacks with more than 2 images, apply random dropping
            max_drop_count = min(int(num_images * 0.5), num_images - 2)  # Ensure at least 2 images remain
            drop_count = random.randint(0, max_drop_count)  # Randomly select how many images to drop
            
            # Randomly select which images to keep
            if drop_count > 0:
                keep_indices = random.sample(range(num_images), num_images - drop_count)
                keep_indices.sort()  # Keep the order
                selected_image_stack = [image_stack[i] for i in keep_indices]
            else:
                selected_image_stack = image_stack
        else:
            # For stacks with 2 or fewer images, don't drop any images
            selected_image_stack = image_stack

        images = []
        for img_path in selected_image_stack:
            # Read image and convert to grayscale
            bgr_img = cv2.imread(img_path)
            gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
            # Normalize to [0,1]
            gray_img = gray_img.astype(np.float32) / 255.0
            if self.transform:
                gray_img = self.transform(gray_img)
            images.append(gray_img.squeeze(0))

        stack_tensor = torch.stack(images)
        return stack_tensor, stack_name, len(images), stack_extension

    @staticmethod
    def sort_key(filename):
        """
        Extract numerical value from filename for sorting
        """
        numbers = re.findall(r'\d+\.?\d*', filename)
        return float(numbers[0]) if numbers else 0


def infer_dataset(model, dataset_loader, device, save_path, dataset_name, max_image_size=1024):
    """
    Perform inference on a dataset and save results
    
    Args:
        model: Neural network model
        dataset_loader: DataLoader containing the test dataset
        device: Computing device (CPU/GPU)
        save_path: Directory to save results (directly to AiF_missing)
        dataset_name: Name of the dataset being processed
        max_image_size: Maximum allowed image size to prevent memory issues
    
    Returns:
        float: Average inference time per stack
    """
    model.eval()

    # For this simplified version, we only need the save_path which is the AiF_missing directory
    # No need to create additional subdirectories

    # Initialize timing metrics
    total_inference_time = 0
    total_stacks = 0

    # Add progress bar for the dataset processing
    dataset_size = len(dataset_loader)
    progress_bar = tqdm(enumerate(dataset_loader), total=dataset_size, desc=f"Processing {dataset_name}")

    with torch.no_grad():
        for idx, (image_stack, stack_name, _, stack_extension) in progress_bar:
            # Save original dimensions
            original_size = image_stack.shape[-2:]

            # Resize input to multiple of 32 for network processing, with size limit
            resized_image_stack, _ = resize_to_multiple_of_32(image_stack, max_image_size)
            resized_image_stack = resized_image_stack.to(device)

            # Load original color images for color fusion
            color_stack = load_color_stack(dataset_loader.dataset.image_stacks[idx])

            # Warmup on first batch (not timed) to stabilize kernels
            if idx == 0:
                for _ in range(3):
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    _ = model(resized_image_stack)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()

            # Measure inference time with CUDA synchronization to ensure strict timing
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.time()
            # Model inference - StackMFF_V3 returns (fused_image, focus_indices)
            fused_image, focus_indices = model(resized_image_stack)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.time()

            inference_time = end_time - start_time
            total_inference_time += inference_time
            total_stacks += 1

            # Process model outputs - resize back to original dimensions
            fused_image, focus_indices = process_model_output(
                fused_image, focus_indices, original_size)

            # Create color fused image and focus colormap
            color_fused_bgr = create_fused_color_image(fused_image, focus_indices, color_stack)
            
            # Generate output filename with stack name and preserve the original extension
            # Ensure stack_name is properly formatted to avoid tuple formatting issues
            if isinstance(stack_name, tuple):
                # If stack_name is a tuple, extract the first element
                clean_stack_name = str(stack_name[0]) if stack_name and len(stack_name) > 0 else str(stack_name)
            elif isinstance(stack_name, str):
                # If stack_name is already a string, use it directly
                clean_stack_name = stack_name
            else:
                # For any other type, convert to string
                clean_stack_name = str(stack_name)
                
            # Ensure stack_extension is properly formatted
            if isinstance(stack_extension, tuple):
                # If stack_extension is a tuple, extract the first element
                clean_extension = str(stack_extension[0]) if stack_extension and len(stack_extension) > 0 else '.png'
            elif isinstance(stack_extension, str):
                # If stack_extension is already a string, use it directly
                clean_extension = stack_extension
            else:
                # For any other type, convert to string
                clean_extension = str(stack_extension)
                
            # Remove any unwanted characters that might be in the string representation
            clean_stack_name = clean_stack_name.replace("(", "").replace(")", "").replace("'", "").replace(",", "")
            clean_extension = clean_extension.replace("(", "").replace(")", "").replace("'", "").replace(",", "").replace(" ", "")
            
            # Ensure the extension starts with a dot
            if not clean_extension.startswith('.'):
                clean_extension = '.' + clean_extension if clean_extension else '.png'
                
            filename = f'{clean_stack_name}{clean_extension}'
            
            # Save only the color fused image directly to the AiF_missing directory
            try:
                cv2.imwrite(os.path.join(save_path, filename), color_fused_bgr)
            except Exception as e:
                print(f"Error saving image: {str(e)}")
                continue

            # Update progress bar with current inference time
            progress_bar.set_postfix({'Inference Time': f'{inference_time:.4f}s'})

    # Calculate average inference time
    avg_inference_time = total_inference_time / total_stacks if total_stacks > 0 else 0
    return avg_inference_time


def main():
    """
    Main function to run batch evaluation
    
    Directory structure should be:
    test_root/
        dataset1/
            TR/
                focus_stack/
                    scene1/
                        img1.png
                        img2.png
                        ...
                    scene2/
                        ...
                AiF_missing/
    """
    # Parse command line arguments
    args = parse_args()
    
    # Create output directory: test_root/dataset_name/TR/AiF_missing
    # We'll save directly to AiF_missing without additional subdirectories
    output_dirs = {}
    
    for dataset_name in args.test_datasets:
        # Updated path construction to avoid duplication
        dataset_root = os.path.join(args.test_root, dataset_name)
        # Create AiF_missing directory under TR
        output_dir = os.path.join(dataset_root, 'TR', 'AiF_missing')
        os.makedirs(output_dir, exist_ok=True)
        output_dirs[dataset_name] = output_dir

    # Set up device and model
    if torch.cuda.is_available():
        if args.gpu_id >= 0:
            # Check if the specified GPU exists
            if args.gpu_id < torch.cuda.device_count():
                device = torch.device(f"cuda:{args.gpu_id}")
                print(f"Using GPU {args.gpu_id}: {torch.cuda.get_device_name(args.gpu_id)}")
            else:
                print(f"Warning: GPU {args.gpu_id} not available. Available GPUs: {torch.cuda.device_count()}")
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    
    model = StackMFF_V4()
    # Load model state dict and handle potential 'module.' prefix from DataParallel
    state_dict = torch.load(args.model_path, map_location=device)
    # Remove 'module.' prefix if it exists (from DataParallel training)
    if any(key.startswith('module.') for key in state_dict.keys()):
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    else:
        new_state_dict = state_dict
    model.load_state_dict(new_state_dict)
    
    model.to(device)
    # Only use DataParallel if CUDA is available, multiple GPUs exist, and no specific GPU is selected
    if torch.cuda.is_available() and torch.cuda.device_count() > 1 and args.gpu_id < 0:
        model = nn.DataParallel(model)

    # Initialize test data loaders for each dataset
    test_loaders = {}
    for dataset_name in args.test_datasets:
        # Updated path construction to avoid duplication
        dataset_root = os.path.join(args.test_root, dataset_name)
        if not os.path.exists(dataset_root):
            print(f"Warning: Dataset directory {dataset_root} not found. Skipping...")
            continue

        # Set up data transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        # Create dataset and dataloader
        # Updated path to match new structure: dataset_name/TR/focus_stack
        dataset = TestDataset(
            root_dir=os.path.join(dataset_root, 'TR', 'focus_stack'),
            transform=transform
        )

        test_loaders[dataset_name] = DataLoader(
            dataset, 
            shuffle=False, 
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )

    # Process each dataset
    dataset_avg_times = {}
    for dataset, loader in test_loaders.items():
        print(f"Processing dataset: {dataset}")
        # Use the specific output directory for this dataset
        avg_time = infer_dataset(model, loader, device, output_dirs[dataset], dataset, args.max_image_size)
        dataset_avg_times[dataset] = avg_time

    # Print results summary
    print("\nAverage inference times per image stack:")
    for dataset, avg_time in dataset_avg_times.items():
        print(f"{dataset}: {avg_time:.4f} seconds")
        print(f"Results saved to: {output_dirs[dataset]}")

def create_fused_color_image(fused_image, focus_indices, color_stack):
    """
    Create color fused image based on focus indices (vectorized implementation)
    
    Args:
        fused_image: Fused grayscale image
        focus_indices: Index map indicating which image to sample from
        color_stack: List of original color images
    
    Returns:
        numpy.ndarray: Color fused image in BGR format
    """
    height, width = fused_image.shape
    num_images = len(color_stack)

    # Ensure indices are within valid range
    focus_indices = np.clip(focus_indices, 0, num_images - 1).astype(int)

    # Convert color stack to single numpy array (N, H, W, 3)
    color_array = np.stack(color_stack, axis=0)

    # Use advanced indexing to get corresponding color values
    fused_color = color_array[focus_indices, np.arange(height)[:, None], np.arange(width)]

    # Convert RGB to BGR for saving
    fused_color_bgr = cv2.cvtColor(fused_color.astype(np.uint8), cv2.COLOR_RGB2BGR)
    return fused_color_bgr


def load_color_stack(image_paths):
    """
    Load color image stack from file paths
    
    Args:
        image_paths: List of paths to image files
    
    Returns:
        list: List of RGB images
    """
    color_images = []
    for img_path in image_paths:
        # Read BGR image and convert to RGB
        bgr_img = cv2.imread(img_path)
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        color_images.append(rgb_img)
    return color_images


def process_model_output(fused_image, focus_indices, original_size):
    """
    Process model outputs and resize to original dimensions
    
    Args:
        fused_image: Fused image from model
        focus_indices: Focus index map from model
        original_size: Original image dimensions
    
    Returns:
        tuple: (fused_image, focus_indices) resized to original dimensions
    """
    # Convert to numpy and resize back to original size
    fused_image = cv2.resize(fused_image.cpu().numpy().squeeze(),
                             (original_size[1], original_size[0]))
    focus_indices = cv2.resize(focus_indices.cpu().numpy().squeeze().astype(np.float32),
                               (original_size[1], original_size[0]),
                               interpolation=cv2.INTER_NEAREST).astype(int)

    return fused_image, focus_indices


if __name__ == "__main__":
    main()