"""
Data Preprocessing Script

Prepares OCTA-500 or custom datasets for training
"""

import os
import sys
from pathlib import Path
import numpy as np
import cv2
import argparse
from tqdm import tqdm
import json
import SimpleITK as sitk
from typing import List, Dict
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from data.preprocess import VesselPreprocessor, PreprocessingConfig
# Add project root





def prepare_octa500(input_dir: str, output_dir: str):
    """
    Prepare OCTA-500 dataset
    
    OCTA-500 structure:
    OCTA-500/
    ├── OCTA_3M/
    │   ├── ProjectionMaps/
    │   │   ├── OCTA/  (3D OCTA volumes)
    │   │   └── GT_Masks/  (Ground truth segmentations)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    print(f"Preparing OCTA-500 dataset...")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        (output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_path / 'masks' / split).mkdir(parents=True, exist_ok=True)
    
    # Find OCTA volumes and masks
    octa_dirs = list((input_path / 'OCTA_3M' / 'ProjectionMaps' / 'OCTA').glob('*'))
    print(f"Found {len(octa_dirs)} samples")
    
    # Split data
    n_samples = len(octa_dirs)
    n_train = int(0.7 * n_samples)
    n_val = int(0.15 * n_samples)
    
    splits = {
        'train': octa_dirs[:n_train],
        'val': octa_dirs[n_train:n_train+n_val],
        'test': octa_dirs[n_train+n_val:]
    }
    
    preprocessor = VesselPreprocessor(PreprocessingConfig())
    
    # Process each split
    for split_name, sample_dirs in splits.items():
        print(f"\nProcessing {split_name} split ({len(sample_dirs)} samples)...")
        
        for sample_dir in tqdm(sample_dirs):
            sample_id = sample_dir.name
            
            # Load OCTA volume (assuming .mhd or .nii format)
            volume_file = list(sample_dir.glob('*.mhd'))
            if not volume_file:
                volume_file = list(sample_dir.glob('*.nii*'))
            
            if not volume_file:
                print(f"Warning: No volume found for {sample_id}")
                continue
            
            # Load volume
            volume = sitk.ReadImage(str(volume_file[0]))
            volume_np = sitk.GetArrayFromImage(volume)  # (D, H, W)
            
            # Load corresponding mask
            mask_file = input_path / 'OCTA_3M' / 'ProjectionMaps' / 'GT_Masks' / sample_id
            mask_files = list(mask_file.glob('*.png'))
            
            if not mask_files:
                print(f"Warning: No mask found for {sample_id}")
                continue
            
            mask = cv2.imread(str(mask_files[0]), cv2.IMREAD_GRAYSCALE)
            
            # Process each slice in volume
            for slice_idx in range(volume_np.shape[0]):
                slice_img = volume_np[slice_idx]
                
                # Normalize
                slice_img = slice_img.astype(np.float32)
                if slice_img.max() > 0:
                    slice_img = slice_img / slice_img.max()
                
                # Preprocess
                processed, metadata = preprocessor.process_single_frame(slice_img, mask / 255.0)
                
                # Save
                output_name = f"{sample_id}_slice{slice_idx:03d}"
                
                # Save image
                img_save = (processed * 255).astype(np.uint8)
                cv2.imwrite(
                    str(output_path / 'images' / split_name / f"{output_name}.png"),
                    img_save
                )
                
                # Save mask
                cv2.imwrite(
                    str(output_path / 'masks' / split_name / f"{output_name}.png"),
                    mask
                )
    
    # Create split files
    for split_name in splits.keys():
        split_file = output_path / 'splits' / f'{split_name}.txt'
        split_file.parent.mkdir(exist_ok=True)
        
        image_files = sorted((output_path / 'images' / split_name).glob('*.png'))
        sample_ids = [f.stem for f in image_files]
        
        with open(split_file, 'w') as f:
            f.write('\n'.join(sample_ids))
    
    print(f"\nPreprocessing complete!")
    print(f"Output saved to {output_dir}")


def prepare_custom_dataset(input_dir: str, output_dir: str, has_masks: bool = False):
    """
    Prepare custom dataset
    
    Expected structure:
    input_dir/
    ├── images/
    └── masks/  (optional)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    print(f"Preparing custom dataset...")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        (output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        if has_masks:
            (output_path / 'masks' / split).mkdir(parents=True, exist_ok=True)
    
    # Get all images
    image_files = sorted((input_path / 'images').glob('*.*'))
    print(f"Found {len(image_files)} images")
    
    # Split data
    n_total = len(image_files)
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)
    
    splits = {
        'train': image_files[:n_train],
        'val': image_files[n_train:n_train+n_val],
        'test': image_files[n_train+n_val:]
    }
    
    preprocessor = VesselPreprocessor(PreprocessingConfig())
    
    # Process each split
    for split_name, img_files in splits.items():
        print(f"\nProcessing {split_name} split ({len(img_files)} images)...")
        
        for img_file in tqdm(img_files):
            # Load image
            img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
            
            # Load mask if available
            mask = None
            if has_masks:
                mask_file = input_path / 'masks' / img_file.name
                if mask_file.exists():
                    mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE) / 255.0
            
            # Preprocess
            processed, metadata = preprocessor.process_single_frame(img / 255.0, mask)
            
            # Save
            output_name = img_file.stem
            
            # Save image
            img_save = (processed * 255).astype(np.uint8)
            cv2.imwrite(
                str(output_path / 'images' / split_name / f"{output_name}.png"),
                img_save
            )
            
            # Save mask
            if mask is not None:
                mask_save = (mask * 255).astype(np.uint8)
                cv2.imwrite(
                    str(output_path / 'masks' / split_name / f"{output_name}.png"),
                    mask_save
                )
    
    # Create split files
    for split_name, img_files in splits.items():
        split_file = output_path / 'splits' / f'{split_name}.txt'
        split_file.parent.mkdir(exist_ok=True)
        
        sample_ids = [f.stem for f in img_files]
        
        with open(split_file, 'w') as f:
            f.write('\n'.join(sample_ids))
    
    print(f"\nPreprocessing complete!")
    print(f"Output saved to {output_dir}")


def generate_synthetic_data(output_dir: str, num_samples: int = 100):
    """
    Generate synthetic vessel data for testing
    
    Creates synthetic vessel-like structures using random walks
    """
    output_path = Path(output_dir)
    print(f"Generating {num_samples} synthetic samples...")
    
    # Create directories
    for split in ['train', 'val', 'test']:
        (output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_path / 'masks' / split).mkdir(parents=True, exist_ok=True)
    
    # Split
    n_train = int(0.7 * num_samples)
    n_val = int(0.15 * num_samples)
    
    splits = {
        'train': n_train,
        'val': n_val,
        'test': num_samples - n_train - n_val
    }
    
    sample_idx = 0
    
    for split_name, n_samples_split in splits.items():
        print(f"Generating {split_name} split ({n_samples_split} samples)...")
        
        for i in tqdm(range(n_samples_split)):
            # Generate synthetic vessel-like structure
            img_size = 512
            img = np.zeros((img_size, img_size), dtype=np.float32)
            mask = np.zeros((img_size, img_size), dtype=np.float32)
            
            # Random walk to create vessel-like structures
            num_vessels = np.random.randint(5, 15)
            
            for _ in range(num_vessels):
                # Start point
                x, y = np.random.randint(0, img_size, 2)
                
                # Vessel thickness
                thickness = np.random.randint(2, 8)
                
                # Random walk
                for step in range(np.random.randint(50, 200)):
                    # Move
                    angle = np.random.randn() * 0.3  # Somewhat smooth
                    if step > 0:
                        dx = int(np.cos(angle) * 5)
                        dy = int(np.sin(angle) * 5)
                    else:
                        dx, dy = 0, 0
                    
                    x = np.clip(x + dx, 0, img_size - 1)
                    y = np.clip(y + dy, 0, img_size - 1)
                    
                    # Draw
                    cv2.circle(mask, (x, y), thickness, 1, -1)
                    cv2.circle(img, (x, y), thickness, np.random.rand() * 0.5 + 0.5, -1)
            
            # Add noise
            img += np.random.randn(img_size, img_size) * 0.1
            img = np.clip(img, 0, 1)
            
            # Blur
            img = cv2.GaussianBlur(img, (5, 5), 1.0)
            
            # Save
            sample_name = f"synthetic_{sample_idx:04d}"
            
            img_save = (img * 255).astype(np.uint8)
            mask_save = (mask * 255).astype(np.uint8)
            
            cv2.imwrite(
                str(output_path / 'images' / split_name / f"{sample_name}.png"),
                img_save
            )
            cv2.imwrite(
                str(output_path / 'masks' / split_name / f"{sample_name}.png"),
                mask_save
            )
            
            sample_idx += 1
    
    # Create split files
    for split_name, n_samples_split in splits.items():
        split_file = output_path / 'splits' / f'{split_name}.txt'
        split_file.parent.mkdir(exist_ok=True)
        
        start_idx = sum(list(splits.values())[:list(splits.keys()).index(split_name)])
        sample_ids = [f"synthetic_{i:04d}" for i in range(start_idx, start_idx + n_samples_split)]
        
        with open(split_file, 'w') as f:
            f.write('\n'.join(sample_ids))
    
    print(f"\nSynthetic data generation complete!")
    print(f"Output saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Preprocess vessel image datasets')
    
    # 1. 修改了 --dataset: 去掉了 required=True，增加了默认值 'synthetic'
    # 如果你想跑真实数据，把 default='synthetic' 改为 'octa500' 或 'custom'
    parser.add_argument('--dataset', type=str, default='synthetic',
                        choices=['octa500', 'custom', 'synthetic'],
                        help='Dataset type')

    # 2. 修改了 --input_dir: 增加了默认路径
    # 如果是用 custom 或 octa500，请把下面的路径改成你电脑上实际的文件夹路径
    parser.add_argument('--input_dir', type=str, default=r'D:\babba\data\input_images',
                        help='Input directory (not needed for synthetic)')

    # 3. 修改了 --output_dir: 去掉了 required=True，增加了默认输出路径
    parser.add_argument('--output_dir', type=str, default=r'./data/processed',
                        help='Output directory')

    parser.add_argument('--has_masks', action='store_true',
                        help='Whether custom dataset has masks')
    
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of synthetic samples to generate')
    
    # 解析参数
    args = parser.parse_args()
    
    # 打印当前使用的参数，方便确认
    print(f"Running with configuration:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Input:   {args.input_dir}")
    print(f"  Output:  {args.output_dir}")
    print("-" * 30)

    # 逻辑判断部分保持不变
    if args.dataset == 'octa500':
        # 检查路径是否存在
        if not args.input_dir or not os.path.exists(args.input_dir):
            raise ValueError(f"Error: Input directory not found: {args.input_dir}")
        prepare_octa500(args.input_dir, args.output_dir)
    
    elif args.dataset == 'custom':
        if not args.input_dir or not os.path.exists(args.input_dir):
            raise ValueError(f"Error: Input directory not found: {args.input_dir}")
        prepare_custom_dataset(args.input_dir, args.output_dir, args.has_masks)
    
    elif args.dataset == 'synthetic':
        generate_synthetic_data(args.output_dir, args.num_samples)

if __name__ == "__main__":
    main()

