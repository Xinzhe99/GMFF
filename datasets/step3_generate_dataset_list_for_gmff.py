import os
import glob
import platform
import argparse

"""
This script generates a list of image paths from multiple datasets with the same structure and saves them to an output file.

Required directory structure:
base_dir/
    dataset1/
        TR/
            AiF/
                image1.png
                image2.jpg
                ...
    dataset2/
        TR/
            AiF/
                image1.png
                image2.jpg
                ...
    ...

The script will recursively search for images in the specified subdirectory (TR/AiF by default) of each dataset and 
generate a list of all image paths. The output file will contain one image path per line.

Usage:
1. Set the base_directory to the root directory containing all datasets
2. Specify the dataset names in the dataset_names list
3. Set the output_file_path for the generated list
4. Run the script

Example:
    python 3_generate_dataset_list_for_gmff.py --base_dir /path/to/stackmffv4_training_datasets --datasets ADE DIODE DUTS NYU-V2 Cityscapes --output datasets/gmff_training_datasets.txt
"""

def parse_args():
    """
    Parse command line arguments
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Generate a list of image paths from multiple datasets")
    parser.add_argument('--base_dir', '-b', type=str, required=True,
                        help='Base directory containing all datasets')
    parser.add_argument('--datasets', '-d', type=str, nargs='+', required=True,
                        help='Dataset names to process')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output file path for the list')
    parser.add_argument('--sub_dir', type=str, default="TR/AiF",
                        help='Subdirectory structure within each dataset (default: "TR/AiF")')
    parser.add_argument('--clear', action='store_true',
                        help='Clear output file before writing (default: False)')
    
    return parser.parse_args()

def generate_dataset_list(base_dir, dataset_names, output_file, sub_dir="TR/AiF", clear=False):
    """
    Generate a list of image paths from multiple datasets with same structure and save to output file.
    
    Args:
        base_dir (str): Base directory containing all datasets
        dataset_names (list or str): Dataset name or list of dataset names
        output_file (str): Output file path for the list
        sub_dir (str): Subdirectory structure within each dataset (default: "TR/AiF")
        clear (bool): Whether to clear the output file before writing (default: False)
    """
    # Supported image formats
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
    
    # Ensure dataset_names is a list
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]
    
    # Get all image file paths
    image_paths = []
    for dataset_name in dataset_names:
        # Build full path
        source_dir = os.path.join(base_dir, dataset_name, sub_dir)
        print(f"Processing dataset: {dataset_name} at {source_dir}")
        
        # Check if directory exists
        if not os.path.exists(source_dir):
            print(f"Warning: Directory {source_dir} does not exist, skipping...")
            continue
            
        for extension in image_extensions:
            # Find images in current directory
            pattern = os.path.join(source_dir, extension)
            image_paths.extend(glob.glob(pattern))
            # Find images in subdirectories
            pattern = os.path.join(source_dir, '**', extension)
            image_paths.extend(glob.glob(pattern, recursive=True))
    
    # Deduplicate and sort
    image_paths = sorted(list(set(image_paths)))
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Clear file if requested
    if clear:
        open(output_file, 'w').close()
    
    # Append to file
    with open(output_file, 'a', encoding='utf-8') as f:
        for path in image_paths:
            # Decide path separator based on operating system
            if platform.system() == "Windows":
                # Convert to Unix style path on Windows (maintain original path style)
                unix_path = path.replace('\\', '/')
            else:
                # Keep original path on Linux/Mac
                unix_path = path
            f.write(f"{unix_path}\n")
    
    print(f"Generated list with {len(image_paths)} images from {len(dataset_names)} datasets")
    print(f"Output file: {output_file}")

def main():
    """Main function to parse arguments and generate dataset list"""
    args = parse_args()
    generate_dataset_list(args.base_dir, args.datasets, args.output, args.sub_dir, args.clear)

if __name__ == "__main__":
    main()