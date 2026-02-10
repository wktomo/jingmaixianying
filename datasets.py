# """
# Dataset Loaders for Vessel Images
# Supports OCTA-500, DRIVE, STARE, and custom video sequences
# """

# import os
# import numpy as np
# import torch
# from torch.utils.data import Dataset, DataLoader
# from typing import Tuple, List, Dict, Optional
# import cv2
# from pathlib import Path
# import SimpleITK as sitk
# import albumentations as A
# from albumentations.pytorch import ToTensorV2


# class VesselSegmentationDataset(Dataset):
#     """
#     Dataset for vessel segmentation training (Step 2)
    
#     Returns:
#         image: (C, H, W) tensor
#         mask: (1, H, W) tensor
#         skeleton: (1, H, W) tensor (optional)
#     """
    
#     def __init__(self,
#                  data_root: str,
#                  split: str = 'train',
#                  image_size: Tuple[int, int] = (512, 512),
#                  augmentation: bool = True):
#         """
#         Args:
#             data_root: Path to dataset root
#             split: 'train', 'val', or 'test'
#             image_size: Target image size (H, W)
#             augmentation: Whether to apply augmentation
#         """
#         self.data_root = Path(data_root)
#         self.split = split
#         self.image_size = image_size
#         self.augmentation = augmentation and (split == 'train')
        
#         # Load file lists
#         self.image_paths, self.mask_paths = self._load_file_lists()
        
#         # Setup augmentation pipeline
#         self.transform = self._get_transforms()
        
#     def _load_file_lists(self) -> Tuple[List[Path], List[Path]]:
#         """Load image and mask file paths"""
#         split_file = self.data_root / 'splits' / f'{self.split}.txt'
        
#         if split_file.exists():
#             # Load from split file
#             with open(split_file, 'r') as f:
#                 sample_ids = [line.strip() for line in f]
#         else:
#             # Auto-generate from directory
#             image_dir = self.data_root / 'images' / self.split
#             sample_ids = [f.stem for f in sorted(image_dir.glob('*.*'))]
        
#         image_paths = []
#         mask_paths = []
        
#         for sample_id in sample_ids:
#             # Support multiple image formats
#             img_path = None
#             for ext in ['.png', '.jpg', '.tif', '.npy']:
#                 candidate = self.data_root / 'images' / self.split / f'{sample_id}{ext}'
#                 if candidate.exists():
#                     img_path = candidate
#                     break
            
#             # Support multiple mask formats
#             mask_path = None
#             for ext in ['.png', '.jpg', '.tif', '.npy']:
#                 candidate = self.data_root / 'masks' / self.split / f'{sample_id}{ext}'
#                 if candidate.exists():
#                     mask_path = candidate
#                     break
            
#             if img_path and mask_path:
#                 image_paths.append(img_path)
#                 mask_paths.append(mask_path)
        
#         return image_paths, mask_paths
    
#     def _get_transforms(self):
#         """Get augmentation transforms"""
#         if self.augmentation:
#             return A.Compose([
#                 A.Resize(*self.image_size),
#                 A.HorizontalFlip(p=0.5),
#                 A.VerticalFlip(p=0.5),
#                 A.RandomRotate90(p=0.5),
#                 A.ShiftScaleRotate(
#                     shift_limit=0.1,
#                     scale_limit=0.1,
#                     rotate_limit=15,
#                     p=0.5
#                 ),
#                 A.ElasticTransform(p=0.3),
#                 A.GaussianBlur(blur_limit=(3, 7), p=0.3),
#                 A.RandomBrightnessContrast(p=0.3),
#                 A.Normalize(mean=[0.5], std=[0.5]),
#                 ToTensorV2()
#             ])
#         else:
#             return A.Compose([
#                 A.Resize(*self.image_size),
#                 A.Normalize(mean=[0.5], std=[0.5]),
#                 ToTensorV2()
#             ])
    
#     def _load_image(self, path: Path) -> np.ndarray:
#         """Load image from various formats"""
#         if path.suffix == '.npy':
#             img = np.load(path)
#         else:
#             img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        
#         # Ensure float32 and normalize to [0, 1]
#         img = img.astype(np.float32)
#         if img.max() > 1.0:
#             img = img / 255.0
        
#         return img
    
#     def _load_mask(self, path: Path) -> np.ndarray:
#         """Load mask and binarize"""
#         if path.suffix == '.npy':
#             mask = np.load(path)
#         else:
#             mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        
#         # Binarize
#         mask = (mask > 127).astype(np.float32)
        
#         return mask
    
#     def __len__(self) -> int:
#         return len(self.image_paths)
    
#     def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
#         # Load data
#         image = self._load_image(self.image_paths[idx])
#         mask = self._load_mask(self.mask_paths[idx])
        
#         # Ensure 2D
#         if len(image.shape) > 2:
#             image = image[..., 0]
#         if len(mask.shape) > 2:
#             mask = mask[..., 0]
        
#         # Apply transforms
#         transformed = self.transform(image=image, mask=mask)
#         image = transformed['image']
#         mask = transformed['mask']
        
#         # Ensure correct dimensions
#         if len(image.shape) == 2:
#             image = image.unsqueeze(0)
#         if len(mask.shape) == 2:
#             mask = mask.unsqueeze(0)
        
#         return {
#             'image': image,
#             'mask': mask,
#             'filename': self.image_paths[idx].stem
#         }


# class VesselEnhancementDataset(Dataset):
#     """
#     Dataset for diffusion enhancement training (Step 3)
    
#     Returns paired noisy/clean images with vessel masks
#     """
    
#     def __init__(self,
#                  data_root: str,
#                  split: str = 'train',
#                  image_size: Tuple[int, int] = (512, 512),
#                  use_synthetic_pairs: bool = True,
#                  segmentation_model: Optional[torch.nn.Module] = None):
#         """
#         Args:
#             data_root: Path to dataset root
#             split: 'train', 'val', or 'test'
#             image_size: Target image size
#             use_synthetic_pairs: Generate synthetic degraded images
#             segmentation_model: Model to generate vessel masks on-the-fly
#         """
#         self.data_root = Path(data_root)
#         self.split = split
#         self.image_size = image_size
#         self.use_synthetic_pairs = use_synthetic_pairs
#         self.segmentation_model = segmentation_model
        
#         # Load clean images
#         self.clean_paths = self._load_clean_images()
        
#         # Degradation transforms (for synthetic pairs)
#         if use_synthetic_pairs:
#             self.degradation = A.Compose([
#                 A.GaussNoise(var_limit=(10, 50), p=0.7),
#                 A.MotionBlur(blur_limit=(3, 7), p=0.5),
#                 A.RandomBrightnessContrast(
#                     brightness_limit=0.2,
#                     contrast_limit=0.2,
#                     p=0.5
#                 ),
#                 A.Downscale(scale_min=0.5, scale_max=0.9, p=0.3),
#             ])
        
#         self.transform = A.Compose([
#             A.Resize(*image_size),
#             A.Normalize(mean=[0.5], std=[0.5]),
#             ToTensorV2()
#         ])
    
#     def _load_clean_images(self) -> List[Path]:
#         """Load list of clean images"""
#         image_dir = self.data_root / 'images' / self.split
#         return sorted(image_dir.glob('*.*'))
    
#     def __len__(self) -> int:
#         return len(self.clean_paths)
    
#     def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
#         # Load clean image
#         clean = cv2.imread(str(self.clean_paths[idx]), cv2.IMREAD_GRAYSCALE)
#         clean = clean.astype(np.float32) / 255.0
        
#         # Generate or load noisy version
#         if self.use_synthetic_pairs:
#             noisy = self.degradation(image=clean)['image']
#         else:
#             # Load paired noisy image (if available)
#             noisy_path = self.data_root / 'noisy' / self.split / self.clean_paths[idx].name
#             if noisy_path.exists():
#                 noisy = cv2.imread(str(noisy_path), cv2.IMREAD_GRAYSCALE)
#                 noisy = noisy.astype(np.float32) / 255.0
#             else:
#                 noisy = clean  # Fallback
        
#         # Apply transforms
#         clean_t = self.transform(image=clean)['image']
#         noisy_t = self.transform(image=noisy)['image']
        
#         # Generate vessel mask (on-the-fly or load)
#         if self.segmentation_model is not None:
#             with torch.no_grad():
#                 mask = self.segmentation_model(clean_t.unsqueeze(0))['mask']
#                 mask = mask.squeeze(0)
#         else:
#             # Try to load pre-computed mask
#             mask_path = self.data_root / 'masks' / self.split / self.clean_paths[idx].name
#             if mask_path.exists():
#                 mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
#                 mask = mask.astype(np.float32) / 255.0
#                 mask = self.transform(image=mask)['image']
#             else:
#                 mask = torch.zeros_like(clean_t)
        
#         return {
#             'noisy': noisy_t,
#             'clean': clean_t,
#             'mask': mask,
#             'filename': self.clean_paths[idx].stem
#         }


# class VesselVideoDataset(Dataset):
#     """
#     Dataset for video sequences (Step 4 SLAM)
    
#     Returns video frames with camera poses
#     """
    
#     def __init__(self,
#                  video_path: str,
#                  image_size: Tuple[int, int] = (512, 512),
#                  frame_stride: int = 1):
#         """
#         Args:
#             video_path: Path to video file or image sequence directory
#             image_size: Target image size
#             frame_stride: Sample every Nth frame
#         """
#         self.video_path = Path(video_path)
#         self.image_size = image_size
#         self.frame_stride = frame_stride
        
#         # Load frames
#         self.frames = self._load_frames()
        
#         self.transform = A.Compose([
#             A.Resize(*image_size),
#             A.Normalize(mean=[0.5], std=[0.5]),
#             ToTensorV2()
#         ])
    
#     def _load_frames(self) -> List[np.ndarray]:
#         """Load video frames"""
#         frames = []
        
#         if self.video_path.is_file():
#             # Load from video file
#             cap = cv2.VideoCapture(str(self.video_path))
#             frame_idx = 0
            
#             while True:
#                 ret, frame = cap.read()
#                 if not ret:
#                     break
                
#                 if frame_idx % self.frame_stride == 0:
#                     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#                     frames.append(frame)
                
#                 frame_idx += 1
            
#             cap.release()
#         else:
#             # Load from image directory
#             image_paths = sorted(self.video_path.glob('*.*'))
#             for i, path in enumerate(image_paths):
#                 if i % self.frame_stride == 0:
#                     frame = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
#                     frames.append(frame)
        
#         return frames
    
#     def __len__(self) -> int:
#         return len(self.frames)
    
#     def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
#         frame = self.frames[idx].astype(np.float32) / 255.0
#         frame_t = self.transform(image=frame)['image']
        
#         return {
#             'image': frame_t,
#             'frame_idx': torch.tensor(idx)
#         }


# def create_dataloaders(config: Dict,
#                       segmentation_model: Optional[torch.nn.Module] = None
#                       ) -> Dict[str, DataLoader]:
#     """
#     Create data loaders for all splits
    
#     Args:
#         config: Configuration dict
#         segmentation_model: Optional segmentation model for enhancement dataset
        
#     Returns:
#         Dict of DataLoaders
#     """
#     dataset_type = config['data']['dataset']
    
#     if 'step2' in config.get('experiment_name', ''):
#         # Segmentation dataset
#         train_dataset = VesselSegmentationDataset(
#             data_root=config['data']['data_root'],
#             split='train',
#             image_size=config['data']['image_size'],
#             augmentation=config['data'].get('augmentation', {}).get('enabled', True)
#         )
#         val_dataset = VesselSegmentationDataset(
#             data_root=config['data']['data_root'],
#             split='val',
#             image_size=config['data']['image_size'],
#             augmentation=False
#         )
    
#     elif 'step3' in config.get('experiment_name', ''):
#         # Enhancement dataset
#         train_dataset = VesselEnhancementDataset(
#             data_root=config['data']['data_root'],
#             split='train',
#             image_size=config['data']['image_size'],
#             use_synthetic_pairs=config['data'].get('use_synthetic_pairs', True),
#             segmentation_model=segmentation_model
#         )
#         val_dataset = VesselEnhancementDataset(
#             data_root=config['data']['data_root'],
#             split='val',
#             image_size=config['data']['image_size'],
#             use_synthetic_pairs=False,
#             segmentation_model=segmentation_model
#         )
    
#     # Create data loaders
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=config['training']['batch_size'],
#         shuffle=True,
#         num_workers=config['data'].get('num_workers', 4),
#         pin_memory=config['data'].get('pin_memory', True),
#         drop_last=True
#     )
    
#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=config['training']['batch_size'],
#         shuffle=False,
#         num_workers=config['data'].get('num_workers', 4),
#         pin_memory=config['data'].get('pin_memory', True)
#     )
    
#     return {
#         'train': train_loader,
#         'val': val_loader
#     }


# if __name__ == "__main__":
#     # Test datasets
#     print("Testing VesselSegmentationDataset...")
#     # Create dummy data for testing
#     test_root = Path("test_data")
#     test_root.mkdir(exist_ok=True)
#     (test_root / "images" / "train").mkdir(parents=True, exist_ok=True)
#     (test_root / "masks" / "train").mkdir(parents=True, exist_ok=True)
    
#     # Create dummy image and mask
#     dummy_img = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
#     dummy_mask = np.random.randint(0, 2, (512, 512), dtype=np.uint8) * 255
#     cv2.imwrite(str(test_root / "images" / "train" / "test.png"), dummy_img)
#     cv2.imwrite(str(test_root / "masks" / "train" / "test.png"), dummy_mask)
    
#     dataset = VesselSegmentationDataset(
#         data_root=str(test_root),
#         split='train',
#         image_size=(256, 256)
#     )
    
#     print(f"Dataset size: {len(dataset)}")
#     sample = dataset[0]
#     print(f"Image shape: {sample['image'].shape}")
#     print(f"Mask shape: {sample['mask'].shape}")










# """
# 优化后的数据集加载器
# 主要改进:
# 1. 造影图像专用增强策略
# 2. 小血管保护增强
# 3. 边界保护机制
# """

# import os
# import numpy as np
# import torch
# from torch.utils.data import Dataset, DataLoader
# from typing import Tuple, List, Dict, Optional
# import cv2
# from pathlib import Path
# import SimpleITK as sitk
# import albumentations as A
# from albumentations.pytorch import ToTensorV2


# # ========== 新增: 自定义增强变换 ==========
# class VesselContrastEnhancement(A.ImageOnlyTransform):
#     """
#     血管对比度增强变换
#     功能: 动态调整血管区域的对比度
#     """
    
#     def __init__(self, alpha_range=(1.0, 1.5), always_apply=False, p=0.5):
#         super().__init__(always_apply, p)
#         self.alpha_range = alpha_range
    
#     def apply(self, img, alpha=1.2, **params):
#         # 计算全局均值
#         mean = np.mean(img)
        
#         # 对比度增强: img = mean + alpha * (img - mean)
#         enhanced = mean + alpha * (img - mean)
#         enhanced = np.clip(enhanced, 0, 1)
        
#         return enhanced.astype(np.float32)
    
#     def get_params(self):
#         return {
#             "alpha": np.random.uniform(self.alpha_range[0], self.alpha_range[1])
#         }


# class SmallVesselPreservingRotation(A.Rotate):
#     """
#     小血管保护旋转
#     功能: 使用更高质量的插值方法，保护细小血管
#     """
    
#     def __init__(self, limit=15, interpolation=cv2.INTER_CUBIC, **kwargs):
#         super().__init__(limit=limit, interpolation=interpolation, **kwargs)


# class AdaptiveCLAHE(A.ImageOnlyTransform):
#     """
#     自适应CLAHE增强
#     """
    
#     def __init__(self, clip_limit_range=(2.0, 4.0), tile_grid_size=(8, 8), 
#                  always_apply=False, p=0.5):
#         super().__init__(always_apply, p)
#         self.clip_limit_range = clip_limit_range
#         self.tile_grid_size = tile_grid_size
    
#     def apply(self, img, clip_limit=3.0, **params):
#         # 转换为uint8
#         if img.dtype != np.uint8:
#             img_uint8 = (img * 255).astype(np.uint8)
#         else:
#             img_uint8 = img
        
#         # 应用CLAHE
#         clahe = cv2.createCLAHE(
#             clipLimit=clip_limit,
#             tileGridSize=self.tile_grid_size
#         )
#         enhanced = clahe.apply(img_uint8)
        
#         # 转回float32
#         if img.dtype != np.uint8:
#             enhanced = enhanced.astype(np.float32) / 255.0
        
#         return enhanced
    
#     def get_params(self):
#         return {
#             "clip_limit": np.random.uniform(
#                 self.clip_limit_range[0], 
#                 self.clip_limit_range[1]
#             )
#         }


# class VesselSegmentationDataset(Dataset):
#     """
#     优化后的血管分割数据集
    
#     改进:
#     1. 造影图像专用增强
#     2. 小血管保护策略
#     3. 边界保护机制
#     """
    
#     def __init__(self,
#                  data_root: str,
#                  split: str = 'train',
#                  image_size: Tuple[int, int] = (512, 512),
#                  augmentation: bool = True):
#         """
#         Args:
#             data_root: Path to dataset root
#             split: 'train', 'val', or 'test'
#             image_size: Target image size (H, W)
#             augmentation: Whether to apply augmentation
#         """
#         self.data_root = Path(data_root)
#         self.split = split
#         self.image_size = image_size
#         self.augmentation = augmentation and (split == 'train')
        
#         # Load file lists
#         self.image_paths, self.mask_paths = self._load_file_lists()
        
#         # Setup augmentation pipeline
#         self.transform = self._get_transforms()
        
#     def _load_file_lists(self) -> Tuple[List[Path], List[Path]]:
#         """Load image and mask file paths"""
#         split_file = self.data_root / 'splits' / f'{self.split}.txt'
        
#         if split_file.exists():
#             with open(split_file, 'r') as f:
#                 sample_ids = [line.strip() for line in f]
#         else:
#             image_dir = self.data_root / 'images' / self.split
#             sample_ids = [f.stem for f in sorted(image_dir.glob('*.*'))]
        
#         image_paths = []
#         mask_paths = []
        
#         for sample_id in sample_ids:
#             img_path = None
#             for ext in ['.png', '.jpg', '.tif', '.npy']:
#                 candidate = self.data_root / 'images' / self.split / f'{sample_id}{ext}'
#                 if candidate.exists():
#                     img_path = candidate
#                     break
            
#             mask_path = None
#             for ext in ['.png', '.jpg', '.tif', '.npy']:
#                 candidate = self.data_root / 'masks' / self.split / f'{sample_id}{ext}'
#                 if candidate.exists():
#                     mask_path = candidate
#                     break
            
#             if img_path and mask_path:
#                 image_paths.append(img_path)
#                 mask_paths.append(mask_path)
        
#         return image_paths, mask_paths
    
#     def _get_transforms(self):
#         """
#         改进后的数据增强流程
        
#         针对造影静脉图像优化:
#         1. 使用更高质量插值保护小血管
#         2. 添加造影特异性增强
#         3. 保护边界信息
#         """
#         if self.augmentation:
#             return A.Compose([
#                 A.Resize(*self.image_size, interpolation=cv2.INTER_CUBIC),  # 改进: 使用三次插值
                
#                 # ========== 造影图像专用增强 ==========
#                 AdaptiveCLAHE(clip_limit_range=(2.0, 4.0), p=0.5),  # 自适应CLAHE
#                 VesselContrastEnhancement(alpha_range=(1.0, 1.3), p=0.4),  # 血管对比度增强
                
#                 # ========== 几何变换 (保护小血管) ==========
#                 A.HorizontalFlip(p=0.5),
#                 A.VerticalFlip(p=0.5),
#                 A.RandomRotate90(p=0.5),
#                 SmallVesselPreservingRotation(
#                     limit=15, 
#                     interpolation=cv2.INTER_CUBIC,  # 改进: 高质量插值
#                     border_mode=cv2.BORDER_REFLECT,  # 改进: 反射边界
#                     p=0.5
#                 ),
                
#                 # ========== 缩放和平移 ==========
#                 A.ShiftScaleRotate(
#                     shift_limit=0.1,
#                     scale_limit=0.1,
#                     rotate_limit=15,
#                     interpolation=cv2.INTER_CUBIC,  # 改进
#                     border_mode=cv2.BORDER_REFLECT,  # 改进
#                     p=0.5
#                 ),
                
#                 # ========== 弹性变形 (小血管友好) ==========
#                 A.ElasticTransform(
#                     alpha=50,  # 改进: 降低变形强度 (120→50)
#                     sigma=5,   # 改进: 增加平滑度 (3→5)
#                     interpolation=cv2.INTER_CUBIC,
#                     p=0.3
#                 ),
                
#                 # ========== 光学增强 (适度) ==========
#                 A.GaussianBlur(blur_limit=(3, 5), p=0.2),  # 改进: 降低概率和强度
#                 A.RandomBrightnessContrast(
#                     brightness_limit=0.15,  # 改进: 降低亮度变化 (0.2→0.15)
#                     contrast_limit=0.15,
#                     p=0.3
#                 ),
                
#                 # ========== 归一化 ==========
#                 A.Normalize(mean=[0.5], std=[0.5]),
#                 ToTensorV2()
#             ])
#         else:
#             # 验证集: 仅调整大小和归一化
#             return A.Compose([
#                 A.Resize(*self.image_size, interpolation=cv2.INTER_CUBIC),
#                 A.Normalize(mean=[0.5], std=[0.5]),
#                 ToTensorV2()
#             ])
    
#     def _load_image(self, path: Path) -> np.ndarray:
#         """Load image from various formats"""
#         if path.suffix == '.npy':
#             img = np.load(path)
#         else:
#             img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        
#         img = img.astype(np.float32)
#         if img.max() > 1.0:
#             img = img / 255.0
        
#         return img
    
#     def _load_mask(self, path: Path) -> np.ndarray:
#         """Load mask and binarize"""
#         if path.suffix == '.npy':
#             mask = np.load(path)
#         else:
#             mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        
#         mask = (mask > 127).astype(np.float32)
        
#         return mask
    
#     def __len__(self) -> int:
#         return len(self.image_paths)
    
#     def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
#         # Load data
#         image = self._load_image(self.image_paths[idx])
#         mask = self._load_mask(self.mask_paths[idx])
        
#         # Ensure 2D
#         if len(image.shape) > 2:
#             image = image[..., 0]
#         if len(mask.shape) > 2:
#             mask = mask[..., 0]
        
#         # Apply transforms
#         transformed = self.transform(image=image, mask=mask)
#         image = transformed['image']
#         mask = transformed['mask']
        
#         # Ensure correct dimensions
#         if len(image.shape) == 2:
#             image = image.unsqueeze(0)
#         if len(mask.shape) == 2:
#             mask = mask.unsqueeze(0)
        
#         return {
#             'image': image,
#             'mask': mask,
#             'filename': self.image_paths[idx].stem
#         }


# class VesselEnhancementDataset(Dataset):
#     """
#     Dataset for diffusion enhancement training (Step 3)
#     """
    
#     def __init__(self,
#                  data_root: str,
#                  split: str = 'train',
#                  image_size: Tuple[int, int] = (512, 512),
#                  use_synthetic_pairs: bool = True,
#                  segmentation_model: Optional[torch.nn.Module] = None):
#         self.data_root = Path(data_root)
#         self.split = split
#         self.image_size = image_size
#         self.use_synthetic_pairs = use_synthetic_pairs
#         self.segmentation_model = segmentation_model
        
#         self.clean_paths = self._load_clean_images()
        
#         if use_synthetic_pairs:
#             self.degradation = A.Compose([
#                 A.GaussNoise(var_limit=(10, 50), p=0.7),
#                 A.MotionBlur(blur_limit=(3, 7), p=0.5),
#                 A.RandomBrightnessContrast(
#                     brightness_limit=0.2,
#                     contrast_limit=0.2,
#                     p=0.5
#                 ),
#                 A.Downscale(scale_min=0.5, scale_max=0.9, p=0.3),
#             ])
        
#         self.transform = A.Compose([
#             A.Resize(*image_size),
#             A.Normalize(mean=[0.5], std=[0.5]),
#             ToTensorV2()
#         ])
    
#     def _load_clean_images(self) -> List[Path]:
#         image_dir = self.data_root / 'images' / self.split
#         return sorted(image_dir.glob('*.*'))
    
#     def __len__(self) -> int:
#         return len(self.clean_paths)
    
#     def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
#         clean = cv2.imread(str(self.clean_paths[idx]), cv2.IMREAD_GRAYSCALE)
#         clean = clean.astype(np.float32) / 255.0
        
#         if self.use_synthetic_pairs:
#             noisy = self.degradation(image=clean)['image']
#         else:
#             noisy_path = self.data_root / 'noisy' / self.split / self.clean_paths[idx].name
#             if noisy_path.exists():
#                 noisy = cv2.imread(str(noisy_path), cv2.IMREAD_GRAYSCALE)
#                 noisy = noisy.astype(np.float32) / 255.0
#             else:
#                 noisy = clean
        
#         clean_t = self.transform(image=clean)['image']
#         noisy_t = self.transform(image=noisy)['image']
        
#         if self.segmentation_model is not None:
#             with torch.no_grad():
#                 mask = self.segmentation_model(clean_t.unsqueeze(0))['mask']
#                 mask = mask.squeeze(0)
#         else:
#             mask_path = self.data_root / 'masks' / self.split / self.clean_paths[idx].name
#             if mask_path.exists():
#                 mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
#                 mask = mask.astype(np.float32) / 255.0
#                 mask = self.transform(image=mask)['image']
#             else:
#                 mask = torch.zeros_like(clean_t)
        
#         return {
#             'noisy': noisy_t,
#             'clean': clean_t,
#             'mask': mask,
#             'filename': self.clean_paths[idx].stem
#         }


# class VesselVideoDataset(Dataset):
#     """
#     Dataset for video sequences (Step 4 SLAM)
#     """
    
#     def __init__(self,
#                  video_path: str,
#                  image_size: Tuple[int, int] = (512, 512),
#                  frame_stride: int = 1):
#         self.video_path = Path(video_path)
#         self.image_size = image_size
#         self.frame_stride = frame_stride
        
#         self.frames = self._load_frames()
        
#         self.transform = A.Compose([
#             A.Resize(*image_size),
#             A.Normalize(mean=[0.5], std=[0.5]),
#             ToTensorV2()
#         ])
    
#     def _load_frames(self) -> List[np.ndarray]:
#         frames = []
        
#         if self.video_path.is_file():
#             cap = cv2.VideoCapture(str(self.video_path))
#             frame_idx = 0
            
#             while True:
#                 ret, frame = cap.read()
#                 if not ret:
#                     break
                
#                 if frame_idx % self.frame_stride == 0:
#                     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#                     frames.append(frame)
                
#                 frame_idx += 1
            
#             cap.release()
#         else:
#             image_paths = sorted(self.video_path.glob('*.*'))
#             for i, path in enumerate(image_paths):
#                 if i % self.frame_stride == 0:
#                     frame = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
#                     frames.append(frame)
        
#         return frames
    
#     def __len__(self) -> int:
#         return len(self.frames)
    
#     def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
#         frame = self.frames[idx].astype(np.float32) / 255.0
#         frame_t = self.transform(image=frame)['image']
        
#         return {
#             'image': frame_t,
#             'frame_idx': torch.tensor(idx)
#         }


# def create_dataloaders(config: Dict,
#                       segmentation_model: Optional[torch.nn.Module] = None
#                       ) -> Dict[str, DataLoader]:
#     """
#     Create data loaders for all splits
#     """
#     if 'step2' in config.get('experiment_name', ''):
#         train_dataset = VesselSegmentationDataset(
#             data_root=config['data']['processed'],
#             split='train',
#             image_size=config['data']['image_size'],
#             augmentation=config['data'].get('augmentation', {}).get('enabled', True)
#         )
#         val_dataset = VesselSegmentationDataset(
#             data_root=config['data']['processed'],
#             split='val',
#             image_size=config['data']['image_size'],
#             augmentation=False
#         )
    
#     elif 'step3' in config.get('experiment_name', ''):
#         train_dataset = VesselEnhancementDataset(
#             data_root=config['data']['processed'],
#             split='train',
#             image_size=config['data']['image_size'],
#             use_synthetic_pairs=config['data'].get('use_synthetic_pairs', True),
#             segmentation_model=segmentation_model
#         )
#         val_dataset = VesselEnhancementDataset(
#             data_root=config['data']['processed'],
#             split='val',
#             image_size=config['data']['image_size'],
#             use_synthetic_pairs=False,
#             segmentation_model=segmentation_model
#         )
    
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=config['training']['batch_size'],
#         shuffle=True,
#         num_workers=config['data'].get('num_workers', 4),
#         pin_memory=config['data'].get('pin_memory', True),
#         drop_last=True
#     )
    
#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=config['training']['batch_size'],
#         shuffle=False,
#         num_workers=config['data'].get('num_workers', 4),
#         pin_memory=config['data'].get('pin_memory', True)
#     )
    
#     return {
#         'train': train_loader,
#         'val': val_loader
#     }


# if __name__ == "__main__":
#     print("Testing optimized dataset...")
    
#     test_root = Path("test_data")
#     test_root.mkdir(exist_ok=True)
#     (test_root / "images" / "train").mkdir(parents=True, exist_ok=True)
#     (test_root / "masks" / "train").mkdir(parents=True, exist_ok=True)
    
#     dummy_img = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
#     dummy_mask = np.random.randint(0, 2, (512, 512), dtype=np.uint8) * 255
#     cv2.imwrite(str(test_root / "images" / "train" / "test.png"), dummy_img)
#     cv2.imwrite(str(test_root / "masks" / "train" / "test.png"), dummy_mask)
    
#     dataset = VesselSegmentationDataset(
#         data_root=str(test_root),
#         split='train',
#         image_size=(256, 256)
#     )
    
#     print(f"Dataset size: {len(dataset)}")
#     sample = dataset[0]
#     print(f"Image shape: {sample['image'].shape}")
#     print(f"Mask shape: {sample['mask'].shape}")

















"""
优化后的数据集加载器
主要改进:
1. 造影图像专用增强策略
2. 小血管保护增强
3. 边界保护机制
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Dict, Optional
import cv2
from pathlib import Path
import SimpleITK as sitk
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ========== 新增: 自定义增强变换 ==========
class VesselContrastEnhancement(A.ImageOnlyTransform):
    """
    血管对比度增强变换
    功能: 动态调整血管区域的对比度
    """
    
    def __init__(self, alpha_range=(1.0, 1.5), always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.alpha_range = alpha_range
    
    def apply(self, img, alpha=1.2, **params):
        # 计算全局均值
        mean = np.mean(img)
        
        # 对比度增强: img = mean + alpha * (img - mean)
        enhanced = mean + alpha * (img - mean)
        enhanced = np.clip(enhanced, 0, 1)
        
        return enhanced.astype(np.float32)
    
    def get_params(self):
        return {
            "alpha": np.random.uniform(self.alpha_range[0], self.alpha_range[1])
        }


class SmallVesselPreservingRotation(A.Rotate):
    """
    小血管保护旋转
    功能: 使用更高质量的插值方法，保护细小血管
    """
    
    def __init__(self, limit=15, interpolation=cv2.INTER_CUBIC, **kwargs):
        super().__init__(limit=limit, interpolation=interpolation, **kwargs)


class AdaptiveCLAHE(A.ImageOnlyTransform):
    """
    自适应CLAHE增强
    """
    
    def __init__(self, clip_limit_range=(2.0, 4.0), tile_grid_size=(8, 8), 
                 always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.clip_limit_range = clip_limit_range
        self.tile_grid_size = tile_grid_size
    
    def apply(self, img, clip_limit=3.0, **params):
        # 转换为uint8
        if img.dtype != np.uint8:
            img_uint8 = (img * 255).astype(np.uint8)
        else:
            img_uint8 = img
        
        # 应用CLAHE
        clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=self.tile_grid_size
        )
        enhanced = clahe.apply(img_uint8)
        
        # 转回float32
        if img.dtype != np.uint8:
            enhanced = enhanced.astype(np.float32) / 255.0
        
        return enhanced
    
    def get_params(self):
        return {
            "clip_limit": np.random.uniform(
                self.clip_limit_range[0], 
                self.clip_limit_range[1]
            )
        }


class VesselSegmentationDataset(Dataset):
    """
    优化后的血管分割数据集
    
    改进:
    1. 造影图像专用增强
    2. 小血管保护策略
    3. 边界保护机制
    """
    
    def __init__(self,
                 data_root: str,
                 split: str = 'train',
                 image_size: Tuple[int, int] = (512, 512),
                 augmentation: bool = True):
        """
        Args:
            data_root: Path to dataset root
            split: 'train', 'val', or 'test'
            image_size: Target image size (H, W)
            augmentation: Whether to apply augmentation
        """
        self.data_root = Path(data_root)
        self.split = split
        self.image_size = image_size
        self.augmentation = augmentation and (split == 'train')
        
        # Load file lists
        self.image_paths, self.mask_paths = self._load_file_lists()
        
        # Setup augmentation pipeline
        self.transform = self._get_transforms()
        
    def _load_file_lists(self) -> Tuple[List[Path], List[Path]]:
        """Load image and mask file paths"""
        split_file = self.data_root / 'splits' / f'{self.split}.txt'
        
        if split_file.exists():
            with open(split_file, 'r') as f:
                sample_ids = [line.strip() for line in f]
        else:
            image_dir = self.data_root / 'images' / self.split
            sample_ids = [f.stem for f in sorted(image_dir.glob('*.*'))]
        
        image_paths = []
        mask_paths = []
        
        for sample_id in sample_ids:
            img_path = None
            for ext in ['.png', '.jpg', '.tif', '.npy']:
                candidate = self.data_root / 'images' / self.split / f'{sample_id}{ext}'
                if candidate.exists():
                    img_path = candidate
                    break
            
            mask_path = None
            for ext in ['.png', '.jpg', '.tif', '.npy']:
                candidate = self.data_root / 'masks' / self.split / f'{sample_id}{ext}'
                if candidate.exists():
                    mask_path = candidate
                    break
            
            if img_path and mask_path:
                image_paths.append(img_path)
                mask_paths.append(mask_path)
        
        return image_paths, mask_paths
    
    def _get_transforms(self):
        """
        改进后的数据增强流程
        
        针对造影静脉图像优化:
        1. 使用更高质量插值保护小血管
        2. 添加造影特异性增强
        3. 保护边界信息
        """
        if self.augmentation:
            return A.Compose([
                A.Resize(*self.image_size, interpolation=cv2.INTER_CUBIC),  # 改进: 使用三次插值
                
                # ========== 造影图像专用增强 ==========
                AdaptiveCLAHE(clip_limit_range=(2.0, 4.0), p=0.5),  # 自适应CLAHE
                VesselContrastEnhancement(alpha_range=(1.0, 1.3), p=0.4),  # 血管对比度增强
                
                # ========== 几何变换 (保护小血管) ==========
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                SmallVesselPreservingRotation(
                    limit=15, 
                    interpolation=cv2.INTER_CUBIC,  # 改进: 高质量插值
                    border_mode=cv2.BORDER_REFLECT,  # 改进: 反射边界
                    p=0.5
                ),
                
                # ========== 缩放和平移 ==========
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1,
                    rotate_limit=15,
                    interpolation=cv2.INTER_CUBIC,  # 改进
                    border_mode=cv2.BORDER_REFLECT,  # 改进
                    p=0.5
                ),
                
                # ========== 弹性变形 (小血管友好) ==========
                A.ElasticTransform(
                    alpha=50,  # 改进: 降低变形强度 (120→50)
                    sigma=5,   # 改进: 增加平滑度 (3→5)
                    interpolation=cv2.INTER_CUBIC,
                    p=0.3
                ),
                
                # ========== 光学增强 (适度) ==========
                A.GaussianBlur(blur_limit=(3, 5), p=0.2),  # 改进: 降低概率和强度
                A.RandomBrightnessContrast(
                    brightness_limit=0.15,  # 改进: 降低亮度变化 (0.2→0.15)
                    contrast_limit=0.15,
                    p=0.3
                ),
                
                # ========== 归一化 ==========
                A.Normalize(mean=[0.5], std=[0.5]),
                ToTensorV2()
            ])
        else:
            # 验证集: 仅调整大小和归一化
            return A.Compose([
                A.Resize(*self.image_size, interpolation=cv2.INTER_CUBIC),
                A.Normalize(mean=[0.5], std=[0.5]),
                ToTensorV2()
            ])
    
    def _load_image(self, path: Path) -> np.ndarray:
        """Load image from various formats"""
        if path.suffix == '.npy':
            img = np.load(path)
        else:
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        
        img = img.astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0
        
        return img
    
    def _load_mask(self, path: Path) -> np.ndarray:
        """Load mask and binarize"""
        if path.suffix == '.npy':
            mask = np.load(path)
        else:
            mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        
        mask = (mask > 127).astype(np.float32)
        
        return mask
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Load data
        image = self._load_image(self.image_paths[idx])
        mask = self._load_mask(self.mask_paths[idx])
        
        # Ensure 2D
        if len(image.shape) > 2:
            image = image[..., 0]
        if len(mask.shape) > 2:
            mask = mask[..., 0]
        
        # Apply transforms
        transformed = self.transform(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask']
        
        # Ensure correct dimensions
        if len(image.shape) == 2:
            image = image.unsqueeze(0)
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)
        
        return {
            'image': image,
            'mask': mask,
            'filename': self.image_paths[idx].stem
        }


class VesselEnhancementDataset(Dataset):
    """
    Dataset for diffusion enhancement training (Step 3)
    """
    
    def __init__(self,
                 data_root: str,
                 split: str = 'train',
                 image_size: Tuple[int, int] = (512, 512),
                 use_synthetic_pairs: bool = True,
                 segmentation_model: Optional[torch.nn.Module] = None):
        self.data_root = Path(data_root)
        self.split = split
        self.image_size = image_size
        self.use_synthetic_pairs = use_synthetic_pairs
        self.segmentation_model = segmentation_model
        
        self.clean_paths = self._load_clean_images()
        
        if use_synthetic_pairs:
            self.degradation = A.Compose([
                A.GaussNoise(p=0.5),
                A.MotionBlur(blur_limit=(3, 7), p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5
                ),
                A.Downscale(p=0.3),
            ])
        
        self.transform = A.Compose([
            A.Resize(*image_size),
            A.Normalize(mean=[0.5], std=[0.5]),
            ToTensorV2()
        ])
    
    def _load_clean_images(self) -> List[Path]:
        image_dir = self.data_root / 'images' / self.split
        return sorted(image_dir.glob('*.*'))
    
    def __len__(self) -> int:
        return len(self.clean_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        clean = cv2.imread(str(self.clean_paths[idx]), cv2.IMREAD_GRAYSCALE)
        clean = clean.astype(np.float32) / 255.0
        
        if self.use_synthetic_pairs:
            noisy = self.degradation(image=clean)['image']
        else:
            noisy_path = self.data_root / 'noisy' / self.split / self.clean_paths[idx].name
            if noisy_path.exists():
                noisy = cv2.imread(str(noisy_path), cv2.IMREAD_GRAYSCALE)
                noisy = noisy.astype(np.float32) / 255.0
            else:
                noisy = clean
        
        clean_t = self.transform(image=clean)['image']
        noisy_t = self.transform(image=noisy)['image']
        
        if self.segmentation_model is not None:
            with torch.no_grad():
                mask = self.segmentation_model(clean_t.unsqueeze(0))['mask']
                mask = mask.squeeze(0)
        else:
            mask_path = self.data_root / 'masks' / self.split / self.clean_paths[idx].name
            if mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                mask = mask.astype(np.float32) / 255.0
                mask = self.transform(image=mask)['image']
            else:
                mask = torch.zeros_like(clean_t)
        
        return {
            'noisy': noisy_t,
            'clean': clean_t,
            'mask': mask,
            'filename': self.clean_paths[idx].stem
        }


class VesselVideoDataset(Dataset):
    """
    Dataset for video sequences (Step 4 SLAM)
    """
    
    def __init__(self,
                 video_path: str,
                 image_size: Tuple[int, int] = (512, 512),
                 frame_stride: int = 1):
        self.video_path = Path(video_path)
        self.image_size = image_size
        self.frame_stride = frame_stride
        
        self.frames = self._load_frames()
        
        self.transform = A.Compose([
            A.Resize(*image_size),
            A.Normalize(mean=[0.5], std=[0.5]),
            ToTensorV2()
        ])
    
    def _load_frames(self) -> List[np.ndarray]:
        frames = []
        
        if self.video_path.is_file():
            cap = cv2.VideoCapture(str(self.video_path))
            frame_idx = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % self.frame_stride == 0:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frames.append(frame)
                
                frame_idx += 1
            
            cap.release()
        else:
            image_paths = sorted(self.video_path.glob('*.*'))
            for i, path in enumerate(image_paths):
                if i % self.frame_stride == 0:
                    frame = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                    frames.append(frame)
        
        return frames
    
    def __len__(self) -> int:
        return len(self.frames)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        frame = self.frames[idx].astype(np.float32) / 255.0
        frame_t = self.transform(image=frame)['image']
        
        return {
            'image': frame_t,
            'frame_idx': torch.tensor(idx)
        }


def create_dataloaders(config: Dict,
                      segmentation_model: Optional[torch.nn.Module] = None
                      ) -> Dict[str, DataLoader]:
    """
    Create data loaders for all splits
    """
    if 'step2' in config.get('experiment_name', ''):
        train_dataset = VesselSegmentationDataset(
            data_root=config['data']['processed'],
            split='train',
            image_size=config['data']['image_size'],
            augmentation=config['data'].get('augmentation', {}).get('enabled', True)
        )
        val_dataset = VesselSegmentationDataset(
            data_root=config['data']['processed'],
            split='val',
            image_size=config['data']['image_size'],
            augmentation=False
        )
    
    elif 'step3' in config.get('experiment_name', ''):
        train_dataset = VesselEnhancementDataset(
            data_root=config['data']['processed'],
            split='train',
            image_size=config['data']['image_size'],
            use_synthetic_pairs=config['data'].get('use_synthetic_pairs', True),
            segmentation_model=segmentation_model
        )
        val_dataset = VesselEnhancementDataset(
            data_root=config['data']['processed'],
            split='val',
            image_size=config['data']['image_size'],
            use_synthetic_pairs=False,
            segmentation_model=segmentation_model
        )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data'].get('num_workers', 4),
        pin_memory=config['data'].get('pin_memory', True),
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data'].get('num_workers', 4),
        pin_memory=config['data'].get('pin_memory', True)
    )
    
    return {
        'train': train_loader,
        'val': val_loader
    }


if __name__ == "__main__":
    print("Testing optimized dataset...")
    
    test_root = Path("test_data")
    test_root.mkdir(exist_ok=True)
    (test_root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (test_root / "masks" / "train").mkdir(parents=True, exist_ok=True)
    
    dummy_img = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
    dummy_mask = np.random.randint(0, 2, (512, 512), dtype=np.uint8) * 255
    cv2.imwrite(str(test_root / "images" / "train" / "test.png"), dummy_img)
    cv2.imwrite(str(test_root / "masks" / "train" / "test.png"), dummy_mask)
    
    dataset = VesselSegmentationDataset(
        data_root=str(test_root),
        split='train',
        image_size=(256, 256)
    )
    
    print(f"Dataset size: {len(dataset)}")
    sample = dataset[0]
    print(f"Image shape: {sample['image'].shape}")
    print(f"Mask shape: {sample['mask'].shape}")

