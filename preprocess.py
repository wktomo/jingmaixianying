# """
# Data Preprocessing Module
# Handles denoising, normalization, keyframe selection, and ROI extraction
# """

# import numpy as np
# import cv2
# from typing import Tuple, List, Dict, Optional
# from dataclasses import dataclass
# import torch
# from skimage import exposure, filters
# from scipy.ndimage import gaussian_filter


# @dataclass
# class PreprocessingConfig:
#     """Configuration for preprocessing"""
#     denoise_sigma: float = 1.0
#     clahe_clip_limit: float = 2.0
#     clahe_grid_size: Tuple[int, int] = (8, 8)
#     normalize_range: Tuple[float, float] = (0.0, 1.0)
#     keyframe_threshold: float = 0.3  # Optical flow threshold
#     roi_padding: int = 20


# class VesselPreprocessor:
#     """
#     Preprocessing pipeline for vessel images
    
#     Pipeline:
#     1. Gaussian denoising
#     2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
#     3. Normalization
#     4. Keyframe selection (based on optical flow)
#     5. ROI extraction
#     """
    
#     def __init__(self, config: PreprocessingConfig):
#         self.config = config
        
#     def denoise(self, image: np.ndarray) -> np.ndarray:
#         """
#         Apply Gaussian denoising
        
#         Args:
#             image: Input image (H, W) or (H, W, C)
            
#         Returns:
#             Denoised image with same shape
#         """
#         return gaussian_filter(image, sigma=self.config.denoise_sigma)
    
#     def apply_clahe(self, image: np.ndarray) -> np.ndarray:
#         """
#         Apply CLAHE for adaptive contrast enhancement
        
#         Args:
#             image: Input grayscale image (H, W), values in [0, 255]
            
#         Returns:
#             Enhanced image (H, W), values in [0, 255]
#         """
#         # Convert to uint8 if needed
#         if image.dtype != np.uint8:
#             image = (image * 255).astype(np.uint8)
        
#         clahe = cv2.createCLAHE(
#             clipLimit=self.config.clahe_clip_limit,
#             tileGridSize=self.config.clahe_grid_size
#         )
#         return clahe.apply(image)
    
#     def normalize(self, image: np.ndarray) -> np.ndarray:
#         """
#         Normalize image to specified range
        
#         Args:
#             image: Input image (H, W) or (H, W, C)
            
#         Returns:
#             Normalized image in [min_val, max_val]
#         """
#         min_val, max_val = self.config.normalize_range
#         image = image.astype(np.float32)
        
#         # Min-max normalization
#         img_min, img_max = image.min(), image.max()
#         if img_max > img_min:
#             image = (image - img_min) / (img_max - img_min)
#             image = image * (max_val - min_val) + min_val
        
#         return image
    
#     def compute_optical_flow(self, 
#                             frame1: np.ndarray, 
#                             frame2: np.ndarray) -> Tuple[np.ndarray, float]:
#         """
#         Compute optical flow between two frames
        
#         Args:
#             frame1, frame2: Grayscale images (H, W)
            
#         Returns:
#             flow: Optical flow (H, W, 2)
#             magnitude: Mean flow magnitude
#         """
#         # Convert to uint8 if needed
#         if frame1.dtype != np.uint8:
#             frame1 = (frame1 * 255).astype(np.uint8)
#         if frame2.dtype != np.uint8:
#             frame2 = (frame2 * 255).astype(np.uint8)
        
#         # Compute optical flow using Farneback method
#         flow = cv2.calcOpticalFlowFarneback(
#             frame1, frame2,
#             None,
#             pyr_scale=0.5,
#             levels=3,
#             winsize=15,
#             iterations=3,
#             poly_n=5,
#             poly_sigma=1.2,
#             flags=0
#         )
        
#         # Compute magnitude
#         magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2).mean()
        
#         return flow, magnitude
    
#     def select_keyframes(self, 
#                         frames: List[np.ndarray],
#                         min_interval: int = 5) -> List[int]:
#         """
#         Select keyframes based on optical flow
        
#         Args:
#             frames: List of frames (each H, W)
#             min_interval: Minimum frames between keyframes
            
#         Returns:
#             List of keyframe indices
#         """
#         keyframe_indices = [0]  # Always include first frame
        
#         for i in range(min_interval, len(frames), min_interval):
#             if i >= len(frames):
#                 break
                
#             # Compute flow from last keyframe to current
#             last_kf_idx = keyframe_indices[-1]
#             _, magnitude = self.compute_optical_flow(
#                 frames[last_kf_idx], 
#                 frames[i]
#             )
            
#             # Add as keyframe if motion is significant
#             if magnitude > self.config.keyframe_threshold:
#                 keyframe_indices.append(i)
        
#         return keyframe_indices
    
#     def extract_roi(self, 
#                    image: np.ndarray,
#                    mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
#         """
#         Extract ROI based on vessel mask or intensity
        
#         Args:
#             image: Input image (H, W)
#             mask: Optional vessel mask (H, W)
            
#         Returns:
#             cropped_image: ROI image
#             roi_info: Dict with crop coordinates
#         """
#         if mask is None:
#             # Use Otsu thresholding to find foreground
#             thresh = filters.threshold_otsu(image)
#             mask = image > thresh
        
#         # Find bounding box of non-zero regions
#         coords = np.argwhere(mask)
#         if len(coords) == 0:
#             # No foreground found, return original
#             return image, {'x': 0, 'y': 0, 'w': image.shape[1], 'h': image.shape[0]}
        
#         y_min, x_min = coords.min(axis=0)
#         y_max, x_max = coords.max(axis=0)
        
#         # Add padding
#         pad = self.config.roi_padding
#         h, w = image.shape[:2]
#         y_min = max(0, y_min - pad)
#         y_max = min(h, y_max + pad)
#         x_min = max(0, x_min - pad)
#         x_max = min(w, x_max + pad)
        
#         # Crop
#         cropped = image[y_min:y_max, x_min:x_max]
        
#         roi_info = {
#             'x': x_min,
#             'y': y_min,
#             'w': x_max - x_min,
#             'h': y_max - y_min
#         }
        
#         return cropped, roi_info
    
#     def process_single_frame(self, 
#                             frame: np.ndarray,
#                             mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
#         """
#         Complete preprocessing pipeline for single frame
        
#         Args:
#             frame: Input frame (H, W) or (H, W, C)
#             mask: Optional vessel mask for ROI extraction
            
#         Returns:
#             processed_frame: Preprocessed frame
#             metadata: Dict with preprocessing info
#         """
#         # Convert to grayscale if needed
#         if len(frame.shape) == 3:
#             frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
#         original_shape = frame.shape
        
#         # Step 1: Denoise
#         frame = self.denoise(frame)
        
#         # Step 2: CLAHE
#         frame = self.apply_clahe(frame)
        
#         # Step 3: Normalize
#         frame = self.normalize(frame)
        
#         # Step 4: Extract ROI
#         frame, roi_info = self.extract_roi(frame, mask)
        
#         metadata = {
#             'original_shape': original_shape,
#             'roi': roi_info,
#             'preprocessed_shape': frame.shape
#         }
        
#         return frame, metadata
    
#     def process_sequence(self,
#                         frames: List[np.ndarray],
#                         masks: Optional[List[np.ndarray]] = None,
#                         select_keyframes: bool = True) -> Dict:
#         """
#         Process entire sequence
        
#         Args:
#             frames: List of frames
#             masks: Optional list of vessel masks
#             select_keyframes: Whether to select keyframes
            
#         Returns:
#             Dict containing:
#                 - processed_frames: List of preprocessed frames
#                 - keyframe_indices: List of keyframe indices (if enabled)
#                 - metadata: List of metadata dicts
#         """
#         processed_frames = []
#         metadata_list = []
        
#         # Process each frame
#         for i, frame in enumerate(frames):
#             mask = masks[i] if masks is not None else None
#             proc_frame, meta = self.process_single_frame(frame, mask)
#             processed_frames.append(proc_frame)
#             metadata_list.append(meta)
        
#         result = {
#             'processed_frames': processed_frames,
#             'metadata': metadata_list
#         }
        
#         # Select keyframes if requested
#         if select_keyframes:
#             kf_indices = self.select_keyframes(processed_frames)
#             result['keyframe_indices'] = kf_indices
        
#         return result


# def get_camera_intrinsics(image_size: Tuple[int, int],
#                           fov: float = 60.0) -> Dict[str, float]:
#     """
#     Estimate camera intrinsics from image size and field of view
    
#     Args:
#         image_size: (height, width)
#         fov: Field of view in degrees
        
#     Returns:
#         Dict with fx, fy, cx, cy
#     """
#     h, w = image_size
#     focal_length = w / (2 * np.tan(np.radians(fov / 2)))
    
#     return {
#         'fx': focal_length,
#         'fy': focal_length,
#         'cx': w / 2,
#         'cy': h / 2
#     }


# if __name__ == "__main__":
#     # Test preprocessing
#     config = PreprocessingConfig()
#     preprocessor = VesselPreprocessor(config)
    
#     # Create synthetic test image
#     test_image = np.random.rand(512, 512).astype(np.float32)
    
#     # Process single frame
#     processed, metadata = preprocessor.process_single_frame(test_image)
#     print(f"Original shape: {metadata['original_shape']}")
#     print(f"Processed shape: {metadata['preprocessed_shape']}")
#     print(f"ROI: {metadata['roi']}")

"""
优化后的数据预处理模块
主要改进:
1. 动态CLAHE参数调整
2. 血管对比度自适应增强
3. 小血管保护增强策略
"""

import numpy as np
import cv2
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import torch
from skimage import exposure, filters
from scipy.ndimage import gaussian_filter


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing"""
    denoise_sigma: float = 1.0
    clahe_clip_limit: float = 3.0  # 改进: 2.0 → 3.0，增强对比度
    clahe_grid_size: Tuple[int, int] = (8, 8)
    normalize_range: Tuple[float, float] = (0.0, 1.0)
    keyframe_threshold: float = 0.3
    roi_padding: int = 20
    
    # ========== 新增参数 ==========
    adaptive_clahe: bool = True  # 自适应CLAHE
    vessel_contrast_enhance: bool = True  # 血管对比度增强
    small_vessel_preserve: bool = True  # 小血管保护


class VesselPreprocessor:
    """
    优化后的血管图像预处理流程
    
    Pipeline:
    1. 自适应去噪
    2. 动态CLAHE
    3. 血管对比度增强
    4. 归一化
    5. 关键帧选择
    6. ROI提取
    """
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
    
    def denoise(self, image: np.ndarray) -> np.ndarray:
        """
        改进: 使用双边滤波代替高斯滤波，保护边缘
        """
        if image.dtype != np.uint8:
            image_uint8 = (image * 255).astype(np.uint8)
        else:
            image_uint8 = image
        
        # 双边滤波: 平滑噪声，保护边缘
        denoised = cv2.bilateralFilter(
            image_uint8,
            d=5,
            sigmaColor=50,
            sigmaSpace=50
        )
        
        # 转回原始类型
        if image.dtype == np.float32 or image.dtype == np.float64:
            denoised = denoised.astype(np.float32) / 255.0
        
        return denoised
    
    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        改进: 动态调整CLAHE参数
        """
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # ========== 改进: 自适应CLAHE ==========
        if self.config.adaptive_clahe:
            # 根据图像对比度动态调整clipLimit
            img_std = np.std(image)
            
            if img_std < 30:  # 低对比度
                clip_limit = 4.0
            elif img_std < 60:  # 中等对比度
                clip_limit = 3.0
            else:  # 高对比度
                clip_limit = 2.0
        else:
            clip_limit = self.config.clahe_clip_limit
        
        clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=self.config.clahe_grid_size
        )
        enhanced = clahe.apply(image)
        
        return enhanced
    
    # ========== 新增方法: 血管对比度增强 ==========
    def enhance_vessel_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        血管对比度自适应增强
        功能: 使用Frangi滤波器增强管状结构
        """
        if not self.config.vessel_contrast_enhance:
            return image
        
        if image.dtype != np.uint8:
            image_float = image.copy()
        else:
            image_float = image.astype(np.float32) / 255.0
        
        # Frangi滤波器参数 (针对不同尺度血管)
        scales = [1, 2, 3, 4]  # 多尺度检测
        vesselness = np.zeros_like(image_float)
        
        for scale in scales:
            # Hessian矩阵特征值分析
            sigma = scale
            
            # 简化的Frangi滤波器实现
            # 使用二阶导数检测管状结构
            kernel_size = int(6 * sigma + 1)
            if kernel_size % 2 == 0:
                kernel_size += 1
            
            # X方向二阶导数
            kernel_xx = cv2.getGaussianKernel(kernel_size, sigma)
            kernel_xx = kernel_xx @ kernel_xx.T
            kernel_xx = cv2.Sobel(kernel_xx, cv2.CV_64F, 2, 0)
            
            # Y方向二阶导数
            kernel_yy = cv2.getGaussianKernel(kernel_size, sigma)
            kernel_yy = kernel_yy @ kernel_yy.T
            kernel_yy = cv2.Sobel(kernel_yy, cv2.CV_64F, 0, 2)
            
            # 计算Hessian响应
            Ixx = cv2.filter2D(image_float, -1, kernel_xx)
            Iyy = cv2.filter2D(image_float, -1, kernel_yy)
            
            # 简化的vesselness度量 (基于Hessian特征值)
            vesselness_scale = np.abs(Ixx + Iyy)
            vesselness = np.maximum(vesselness, vesselness_scale)
        
        # 归一化
        vesselness = (vesselness - vesselness.min()) / (vesselness.max() - vesselness.min() + 1e-7)
        
        # 与原图融合
        enhanced = image_float + 0.3 * vesselness
        enhanced = np.clip(enhanced, 0, 1)
        
        # 转回uint8
        if image.dtype == np.uint8:
            enhanced = (enhanced * 255).astype(np.uint8)
        
        return enhanced
    
    def normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image to specified range
        """
        min_val, max_val = self.config.normalize_range
        image = image.astype(np.float32)
        
        # Min-max normalization
        img_min, img_max = image.min(), image.max()
        if img_max > img_min:
            image = (image - img_min) / (img_max - img_min)
            image = image * (max_val - min_val) + min_val
        
        return image
    
    def compute_optical_flow(self, 
                            frame1: np.ndarray, 
                            frame2: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Compute optical flow between two frames
        """
        if frame1.dtype != np.uint8:
            frame1 = (frame1 * 255).astype(np.uint8)
        if frame2.dtype != np.uint8:
            frame2 = (frame2 * 255).astype(np.uint8)
        
        flow = cv2.calcOpticalFlowFarneback(
            frame1, frame2,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
        
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2).mean()
        
        return flow, magnitude
    
    def select_keyframes(self, 
                        frames: List[np.ndarray],
                        min_interval: int = 5) -> List[int]:
        """
        Select keyframes based on optical flow
        """
        keyframe_indices = [0]
        
        for i in range(min_interval, len(frames), min_interval):
            if i >= len(frames):
                break
                
            last_kf_idx = keyframe_indices[-1]
            _, magnitude = self.compute_optical_flow(
                frames[last_kf_idx], 
                frames[i]
            )
            
            if magnitude > self.config.keyframe_threshold:
                keyframe_indices.append(i)
        
        return keyframe_indices
    
    def extract_roi(self, 
                   image: np.ndarray,
                   mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
        """
        Extract ROI based on vessel mask or intensity
        """
        if mask is None:
            thresh = filters.threshold_otsu(image)
            mask = image > thresh
        
        coords = np.argwhere(mask)
        if len(coords) == 0:
            return image, {'x': 0, 'y': 0, 'w': image.shape[1], 'h': image.shape[0]}
        
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        pad = self.config.roi_padding
        h, w = image.shape[:2]
        y_min = max(0, y_min - pad)
        y_max = min(h, y_max + pad)
        x_min = max(0, x_min - pad)
        x_max = min(w, x_max + pad)
        
        cropped = image[y_min:y_max, x_min:x_max]
        
        roi_info = {
            'x': x_min,
            'y': y_min,
            'w': x_max - x_min,
            'h': y_max - y_min
        }
        
        return cropped, roi_info
    
    def process_single_frame(self, 
                            frame: np.ndarray,
                            mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
        """
        改进后的完整预处理流程
        
        Pipeline:
        1. 去噪 (双边滤波)
        2. 动态CLAHE
        3. 血管对比度增强
        4. 归一化
        5. ROI提取
        """
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        original_shape = frame.shape
        
        # Step 1: Denoise (双边滤波)
        frame = self.denoise(frame)
        
        # Step 2: 动态CLAHE
        frame = self.apply_clahe(frame)
        
        # Step 3: 血管对比度增强
        frame = self.enhance_vessel_contrast(frame)
        
        # Step 4: Normalize
        frame = self.normalize(frame)
        
        # Step 5: Extract ROI
        frame, roi_info = self.extract_roi(frame, mask)
        
        metadata = {
            'original_shape': original_shape,
            'roi': roi_info,
            'preprocessed_shape': frame.shape
        }
        
        return frame, metadata
    
    def process_sequence(self,
                        frames: List[np.ndarray],
                        masks: Optional[List[np.ndarray]] = None,
                        select_keyframes: bool = True) -> Dict:
        """
        Process entire sequence
        """
        processed_frames = []
        metadata_list = []
        
        for i, frame in enumerate(frames):
            mask = masks[i] if masks is not None else None
            proc_frame, meta = self.process_single_frame(frame, mask)
            processed_frames.append(proc_frame)
            metadata_list.append(meta)
        
        result = {
            'processed_frames': processed_frames,
            'metadata': metadata_list
        }
        
        if select_keyframes:
            kf_indices = self.select_keyframes(processed_frames)
            result['keyframe_indices'] = kf_indices
        
        return result


def get_camera_intrinsics(image_size: Tuple[int, int],
                          fov: float = 60.0) -> Dict[str, float]:
    """
    Estimate camera intrinsics from image size and field of view
    """
    h, w = image_size
    focal_length = w / (2 * np.tan(np.radians(fov / 2)))
    
    return {
        'fx': focal_length,
        'fy': focal_length,
        'cx': w / 2,
        'cy': h / 2
    }


if __name__ == "__main__":
    print("Testing optimized preprocessing...")
    
    config = PreprocessingConfig(
        adaptive_clahe=True,
        vessel_contrast_enhance=True
    )
    preprocessor = VesselPreprocessor(config)
    
    # Create synthetic test image
    test_image = np.random.rand(512, 512).astype(np.float32)
    
    # Process single frame
    processed, metadata = preprocessor.process_single_frame(test_image)
    print(f"Original shape: {metadata['original_shape']}")
    print(f"Processed shape: {metadata['preprocessed_shape']}")
    print(f"ROI: {metadata['roi']}")