# """
# End-to-End Inference Pipeline

# Runs complete vessel 3D reconstruction pipeline:
# Input Video → Preprocessing → Segmentation → Enhancement → 3DGS-SLAM → Export
# """

# import os
# import sys
# from pathlib import Path
# import torch
# import torch.nn as nn
# import numpy as np
# import cv2
# from tqdm import tqdm
# import argparse
# import yaml
# from typing import Dict, List

# # Add project root
# project_root = Path(__file__).parent
# sys.path.insert(0, str(project_root))

# from data.preprocess import VesselPreprocessor, PreprocessingConfig, get_camera_intrinsics
# from models.segmentation.unet_plusplus import create_segmentation_model
# from models.reconstruction.vessel_3dgs import VesselGaussianSLAM, Camera


# class VesselReconstructionPipeline:
#     """Complete vessel 3D reconstruction pipeline"""
    
#     def __init__(self,
#                  segmentation_checkpoint: str,
#                  enhancement_checkpoint: str = None,
#                  config: Dict = None):
#         """
#         Args:
#             segmentation_checkpoint: Path to segmentation model checkpoint
#             enhancement_checkpoint: Path to enhancement model checkpoint (optional)
#             config: Configuration dict
#         """
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.config = config or {}
        
#         # Load models
#         print("Loading models...")
#         self.segmentation_model = self._load_segmentation_model(segmentation_checkpoint)
        
#         # Enhancement model (optional)
#         if enhancement_checkpoint:
#             self.enhancement_model = self._load_enhancement_model(enhancement_checkpoint)
#         else:
#             self.enhancement_model = None
#             print("No enhancement model provided, using original images")
        
#         # Preprocessor
#         self.preprocessor = VesselPreprocessor(PreprocessingConfig())
        
#         print(f"Pipeline initialized on {self.device}")
    
#     def _load_segmentation_model(self, checkpoint_path: str):
#         """Load segmentation model"""
#         checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
#         # Create model from config
#         model_config = checkpoint.get('config', {})
#         model = create_segmentation_model(model_config)
        
#         # Load weights
#         model.load_state_dict(checkpoint['model_state_dict'])
#         model = model.to(self.device)
#         model.eval()
        
#         print(f"Loaded segmentation model from {checkpoint_path}")
#         return model
    
#     def _load_enhancement_model(self, checkpoint_path: str):
#         """Load enhancement model (placeholder)"""
#         # TODO: Implement when enhancement model is ready
#         print(f"Enhancement model loading not yet implemented")
#         return None
#     @torch.no_grad()
#     def segment_frame(self, image: np.ndarray) -> Dict[str, np.ndarray]:
#         # 1. 记录原始尺寸以便后面恢复
#         orig_h, orig_w = image.shape[:2]
        
#         # 2. 计算最接近的 32 的倍数
#         new_h = (orig_h // 32) * 32
#         new_w = (orig_w // 32) * 32
        
#         # 3. 如果尺寸不符合，进行缩放
#         if orig_h != new_h or orig_w != new_w:
#             image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
#         # ... 原有的预处理代码 (灰度转换, 归一化) ...
#         if len(image.shape) == 3:
#             image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
#         image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(self.device)
#         if image_tensor.max() > 1.0:
#             image_tensor = image_tensor / 255.0

#         # 4. 模型推理
#         outputs = self.segmentation_model(image_tensor)
        
#         # 5. 获取结果并还原到原始尺寸
#         mask = outputs['mask'].squeeze().cpu().numpy()
#         if mask.shape != (orig_h, orig_w):
#             mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        
#         # 骨架和置信度也同步还原
#         skeleton = outputs.get('skeleton', outputs['mask']).squeeze().cpu().numpy()
#         if skeleton.shape != (orig_h, orig_w):
#             skeleton = cv2.resize(skeleton, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        
#         return {
#             'mask': mask,
#             'skeleton': skeleton,
#             'confidence': np.ones_like(mask) # 简化处理
#         }
#     # @torch.no_grad()
#     # def segment_frame(self, image: np.ndarray) -> Dict[str, np.ndarray]:
#     #     """
#     #     Segment vessel in single frame
        
#     #     Args:
#     #         image: Input image (H, W) or (H, W, 3)
            
#     #     Returns:
#     #         Dict with mask, skeleton, confidence
#     #     """
#     #     # Preprocess
#     #     if len(image.shape) == 3:
#     #         image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
#     #     # Normalize
#     #     image = image.astype(np.float32)
#     #     if image.max() > 1.0:
#     #         image = image / 255.0
        
#     #     # To tensor
#     #     image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(self.device)
        
#     #     # Segment
#     #     outputs = self.segmentation_model(image_tensor)
        
#     #     # Convert to numpy
#     #     mask = outputs['mask'].squeeze().cpu().numpy()
#     #     skeleton = outputs.get('skeleton', mask).squeeze().cpu().numpy()
#     #     confidence = outputs.get('confidence', np.ones_like(mask)).squeeze().cpu().numpy()
        
#     #     return {
#     #         'mask': mask,
#     #         'skeleton': skeleton,
#     #         'confidence': confidence
#     #     }
    
#     def enhance_frame(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
#         """
#         Enhance single frame
        
#         Args:
#             image: Input image (H, W)
#             mask: Vessel mask (H, W)
            
#         Returns:
#             Enhanced image (H, W)
#         """
#         if self.enhancement_model is None:
#             return image
        
#         # TODO: Implement enhancement
#         return image
    
#     def run_pipeline(self,
#                     input_path: str,
#                     output_dir: str,
#                     max_frames: int = None):
#         """
#         Run complete pipeline
        
#         Args:
#             input_path: Path to input video or image sequence
#             output_dir: Output directory
#             max_frames: Maximum frames to process (None for all)
#         """
#         output_path = Path(output_dir)
#         output_path.mkdir(parents=True, exist_ok=True)
        
#         # Create output subdirectories
#         (output_path / 'frames').mkdir(exist_ok=True)
#         (output_path / 'masks').mkdir(exist_ok=True)
#         (output_path / 'enhanced').mkdir(exist_ok=True)
#         (output_path / 'visualization').mkdir(exist_ok=True)
        
#         # Load video
#         print(f"Loading video from {input_path}...")
#         frames = self._load_video(input_path, max_frames)
#         print(f"Loaded {len(frames)} frames")
        
#         # Step 1: Preprocess
#         print("\nStep 1: Preprocessing...")
#         preprocessed = self.preprocessor.process_sequence(frames, select_keyframes=False)
#         processed_frames = preprocessed['processed_frames']
#         keyframe_indices = preprocessed.get('keyframe_indices', list(range(len(frames))))
        
#         print(f"Selected {len(keyframe_indices)} keyframes")
        
#         # Step 2: Segment all frames
#         print("\nStep 2: Segmenting vessels...")
#         segmentation_results = []
        
#         for i, frame in enumerate(tqdm(processed_frames, desc='Segmentation')):
#             result = self.segment_frame(frame)
#             segmentation_results.append(result)
            
#             # Save mask
#             mask_img = (result['mask'] * 255).astype(np.uint8)
#             cv2.imwrite(str(output_path / 'masks' / f'mask_{i:04d}.png'), mask_img)
        
#         # Step 3: Enhance frames
#         print("\nStep 3: Enhancing images...")
#         enhanced_frames = []
        
#         for i, (frame, seg_result) in enumerate(tqdm(
#             zip(processed_frames, segmentation_results),
#             desc='Enhancement',
#             total=len(processed_frames)
#         )):
#             enhanced = self.enhance_frame(frame, seg_result['mask'])
#             enhanced_frames.append(enhanced)
            
#             # Save enhanced
#             enhanced_img = (enhanced * 255).astype(np.uint8)
#             cv2.imwrite(str(output_path / 'enhanced' / f'enhanced_{i:04d}.png'), enhanced_img)
        
#         # Step 4: 3D Reconstruction (only on keyframes)
#         print("\nStep 4: 3D Reconstruction...")
        
#         # Setup camera
#         h, w = processed_frames[0].shape[:2]
#         intrinsics = get_camera_intrinsics((h, w), fov=60.0)
#         camera = Camera(
#             width=w,
#             height=h,
#             fx=intrinsics['fx'],
#             fy=intrinsics['fy'],
#             cx=intrinsics['cx'],
#             cy=intrinsics['cy']
#         )
        
#         # Create SLAM system
#         slam = VesselGaussianSLAM(
#             camera=camera,
#             initial_points=5000,
#             vessel_weight=2.0
#         )
        
#         # Process keyframes
#         for kf_idx in tqdm(keyframe_indices[:min(10, len(keyframe_indices))], desc='SLAM'):
#             # Get frame and mask
#             frame_rgb = np.stack([enhanced_frames[kf_idx]] * 3, axis=0)
#             frame_tensor = torch.from_numpy(frame_rgb).float().to(self.device)
            
#             mask_tensor = torch.from_numpy(segmentation_results[kf_idx]['mask']).unsqueeze(0).to(self.device)
            
#             # Add keyframe
#             slam.add_keyframe(kf_idx, frame_tensor, mask_tensor)
        
#         # Step 5: Export results
#         print("\nStep 5: Exporting results...")
        
#         # Export point cloud
#         point_cloud_path = output_path / 'reconstruction.ply'
#         slam.export_point_cloud(str(point_cloud_path))
        
#         # Save camera trajectory
#         trajectory = []
#         for kf in slam.keyframes:
#             pose = kf['pose_param'].detach().cpu().numpy()
#             trajectory.append(pose)
        
#         trajectory = np.array(trajectory)
#         np.save(output_path / 'trajectory.npy', trajectory)
        
#         # Create visualization video
#         self._create_visualization_video(
#             processed_frames,
#             segmentation_results,
#             enhanced_frames,
#             output_path
#         )
        
#         print(f"\nPipeline completed! Results saved to {output_dir}")
        
#         # Print summary
#         self._print_summary(output_path, len(frames), len(keyframe_indices))
    
#     def _load_video(self, video_path: str, max_frames: int = None) -> List[np.ndarray]:
#         """Load video frames"""
#         video_path = Path(video_path)
#         frames = []
        
#         if video_path.is_file() and video_path.suffix in ['.mp4', '.avi', '.mov']:
#             # Load from video file
#             cap = cv2.VideoCapture(str(video_path))
            
#             frame_idx = 0
#             target_size = None
#             while True:
#                 ret, frame = cap.read()
#                 if not ret:
#                     break
                
#                 frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#                 if target_size is None:
#                     target_size=(frame.shape[1],frame.shape[0])
#                 else:
#                     if (frame.shape[1],frame.shape[0]) != target_size:
#                         frame=cv2.resize(frame,target_size)
#                 frames.append(frame)
                
#                 frame_idx += 1
#                 if max_frames and frame_idx >= max_frames:
#                     break
            
#             cap.release()
        
#         elif video_path.is_dir():
#             # Load from image directory
#             image_files = sorted(video_path.glob('*.*'))
#             for i, img_path in enumerate(image_files):
#                 if max_frames and i >= max_frames:
#                     break
                
#                 frame = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
#                 frames.append(frame)
        
#         else:
#             raise ValueError(f"Invalid input path: {video_path}")
        
#         return frames
    
#     def _create_visualization_video(self,
#                                    original_frames: List[np.ndarray],
#                                    segmentation_results: List[Dict],
#                                    enhanced_frames: List[np.ndarray],
#                                    output_path: Path):
#         """Create visualization video"""
#         print("Creating visualization video...")
        
#         h, w = original_frames[0].shape[:2]
        
#         # Create video writer
#         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#         out = cv2.VideoWriter(
#             str(output_path / 'visualization.mp4'),
#             fourcc,
#             10.0,  # FPS
#             (w * 3, h)
#         )
        
#         for orig, seg, enh in zip(original_frames, segmentation_results, enhanced_frames):
#             # Normalize to 0-255
#             orig_vis = (orig * 255).astype(np.uint8)
#             mask_vis = (seg['mask'] * 255).astype(np.uint8)
#             enh_vis = (enh * 255).astype(np.uint8)
            
#             # Convert to color
#             orig_vis = cv2.cvtColor(orig_vis, cv2.COLOR_GRAY2BGR)
#             mask_vis = cv2.applyColorMap(mask_vis, cv2.COLORMAP_JET)
#             enh_vis = cv2.cvtColor(enh_vis, cv2.COLOR_GRAY2BGR)
            
#             # Concatenate
#             vis = np.hstack([orig_vis, mask_vis, enh_vis])
            
#             out.write(vis)
        
#         out.release()
#         print(f"Visualization saved to {output_path / 'visualization.mp4'}")
    
#     def _print_summary(self, output_path: Path, num_frames: int, num_keyframes: int):
#         """Print pipeline summary"""
#         print("\n" + "="*50)
#         print("PIPELINE SUMMARY")
#         print("="*50)
#         print(f"Total frames processed: {num_frames}")
#         print(f"Keyframes selected: {num_keyframes}")
#         print(f"\nOutputs:")
#         print(f"  - Segmentation masks: {output_path / 'masks'}")
#         print(f"  - Enhanced images: {output_path / 'enhanced'}")
#         print(f"  - 3D point cloud: {output_path / 'reconstruction.ply'}")
#         print(f"  - Camera trajectory: {output_path / 'trajectory.npy'}")
#         print(f"  - Visualization video: {output_path / 'visualization.mp4'}")
#         print("="*50)

# def main():
#     # ================= 配置区域 (请在这里修改你的路径) =================
#     # 输入视频路径或图片文件夹路径
#     input_path = "22.mp4" 
    
#     # 输出结果保存的文件夹路径
#     output_dir = "output_results"
    
#     # 训练好的分割模型权重文件路径 (.pth)
#     seg_checkpoint_path = r"checkpoints\step2_segmentation\best_model.pth"
    
#     # (可选) 增强模型路径，如果没有就设为 None
#     enh_checkpoint_path = None 
    
#     # (可选) 最大处理帧数，None 表示处理全部
#     max_frames_process = None 
#     # =================================================================

#     print(f"正在运行推理...")
#     print(f"输入: {input_path}")
#     print(f"输出: {output_dir}")
#     print(f"模型: {seg_checkpoint_path}")

#     # 创建输出目录
#     os.makedirs(output_dir, exist_ok=True)

#     # 创建推理管线
#     # 注意：这里我们直接使用上面定义的变量，不再使用 argparse
#     pipeline = VesselReconstructionPipeline(
#         segmentation_checkpoint=seg_checkpoint_path,
#         enhancement_checkpoint=enh_checkpoint_path,
#         config=None
#     )
    
#     # 运行管线
#     pipeline.run_pipeline(
#         input_path=input_path,
#         output_dir=output_dir,
#         max_frames=max_frames_process
#     )

# if __name__ == "__main__":
#     main()

"""
End-to-End Inference Pipeline (优化版)

Runs complete vessel 3D reconstruction pipeline:
Input Video → Preprocessing → Segmentation → Enhancement → 3DGS-SLAM → Export

优化内容：
- 添加详细调试信息
- 改进尺寸处理（padding替代resize）
- 模型输出验证
- 异常处理
- 中间结果保存
"""

import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import cv2
from tqdm import tqdm
import argparse
import yaml
from typing import Dict, List
import traceback

# Add project root
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from data.preprocess import VesselPreprocessor, PreprocessingConfig, get_camera_intrinsics
from models.segmentation.unet_plusplus import create_segmentation_model
from models.reconstruction.vessel_3dgs import VesselGaussianSLAM, Camera


class VesselReconstructionPipeline:
    """Complete vessel 3D reconstruction pipeline"""
    
    def __init__(self,
                 segmentation_checkpoint: str,
                 enhancement_checkpoint: str = None,
                 config: Dict = None,
                 debug: bool = True):
        """
        Args:
            segmentation_checkpoint: Path to segmentation model checkpoint
            enhancement_checkpoint: Path to enhancement model checkpoint (optional)
            config: Configuration dict
            debug: Enable debug mode
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config or {}
        self.debug = debug
        
        # Load models
        print("=" * 60)
        print("初始化推理管线...")
        print("=" * 60)
        print(f"设备: {self.device}")
        print(f"调试模式: {'开启' if debug else '关闭'}")
        
        self.segmentation_model = self._load_segmentation_model(segmentation_checkpoint)
        
        # Enhancement model (optional)
        if enhancement_checkpoint:
            self.enhancement_model = self._load_enhancement_model(enhancement_checkpoint)
        else:
            self.enhancement_model = None
            print("⚠️  未提供增强模型，使用原始图像")
        
        # Preprocessor
        self.preprocessor = VesselPreprocessor(PreprocessingConfig())
        
        print(f"✓ 管线初始化完成")
        print("=" * 60)
    
    def _load_segmentation_model(self, checkpoint_path: str):
        """加载分割模型（带验证）"""
        print(f"\n正在加载分割模型: {checkpoint_path}")
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"模型文件不存在: {checkpoint_path}")
        
        # 检查文件大小
        file_size = os.path.getsize(checkpoint_path) / (1024 * 1024)  # MB
        print(f"模型文件大小: {file_size:.2f} MB")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # 打印checkpoint信息
            print(f"Checkpoint 包含的键: {list(checkpoint.keys())}")
            
            if 'epoch' in checkpoint:
                print(f"训练轮数: {checkpoint['epoch']}")
            if 'best_dice' in checkpoint:
                print(f"最佳 Dice: {checkpoint['best_dice']:.4f}")
            if 'best_iou' in checkpoint:
                print(f"最佳 IoU: {checkpoint['best_iou']:.4f}")
            
            # Create model from config
            model_config = checkpoint.get('config', {})
            print(f"模型配置: {model_config}")
            
            model = create_segmentation_model(model_config)
            
            # Load weights
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                raise KeyError("Checkpoint中找不到模型权重 (需要 'model_state_dict' 或 'state_dict')")
            
            model = model.to(self.device)
            model.eval()
            
            # 验证模型
            self._validate_model(model)
            
            print(f"✓ 分割模型加载成功")
            return model
            
        except Exception as e:
            print(f"❌ 加载模型失败: {str(e)}")
            traceback.print_exc()
            raise
    
    def _validate_model(self, model):
        """验证模型是否正常工作"""
        print("验证模型...")
        
        # 创建测试输入
        test_input = torch.randn(1, 1, 256, 256).to(self.device)
        
        try:
            with torch.no_grad():
                outputs = model(test_input)
            
            print(f"模型输出键: {outputs.keys()}")
            
            if 'mask' in outputs:
                mask = outputs['mask']
                print(f"Mask shape: {mask.shape}")
                print(f"Mask range: [{mask.min():.3f}, {mask.max():.3f}]")
                
                # 检查异常值
                if torch.isnan(mask).any():
                    print("⚠️  警告: Mask包含NaN值")
                if torch.isinf(mask).any():
                    print("⚠️  警告: Mask包含Inf值")
                
                if mask.min() < -1 or mask.max() > 2:
                    print("⚠️  警告: Mask值范围异常，应该在[0, 1]之间")
            
            print("✓ 模型验证通过")
            
        except Exception as e:
            print(f"❌ 模型验证失败: {str(e)}")
            raise
    
    def _load_enhancement_model(self, checkpoint_path: str):
        """Load enhancement model (placeholder)"""
        print(f"增强模型加载功能尚未实现")
        return None
    
    @torch.no_grad()
    def segment_frame(self, image: np.ndarray, frame_idx: int = -1) -> Dict[str, np.ndarray]:
        """
        分割单帧图像（优化版）
        
        Args:
            image: 输入图像 (H, W) or (H, W, 3)
            frame_idx: 帧索引（用于调试）
            
        Returns:
            Dict with mask, skeleton, confidence
        """
        try:
            orig_h, orig_w = image.shape[:2]
            
            if self.debug and frame_idx == 0:
                print(f"\n首帧分割调试信息:")
                print(f"原始图像尺寸: {orig_h} x {orig_w}")
            
            # 计算padding（不使用resize）
            pad_h = (32 - orig_h % 32) % 32
            pad_w = (32 - orig_w % 32) % 32
            
            # 转灰度
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Padding
            if pad_h > 0 or pad_w > 0:
                image_padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode='reflect')
                if self.debug and frame_idx == 0:
                    print(f"Padding后尺寸: {image_padded.shape}")
            else:
                image_padded = image
            
            # 归一化
            image_tensor = torch.from_numpy(image_padded).float().unsqueeze(0).unsqueeze(0).to(self.device)
            if image_tensor.max() > 1.0:
                image_tensor = image_tensor / 255.0
            
            if self.debug and frame_idx == 0:
                print(f"输入tensor - shape: {image_tensor.shape}, range: [{image_tensor.min():.3f}, {image_tensor.max():.3f}]")
            
            # 模型推理
            outputs = self.segmentation_model(image_tensor)
            
            # 检查输出
            if self.debug and frame_idx == 0:
                print(f"模型输出键: {outputs.keys()}")
                if 'mask' in outputs:
                    print(f"Mask - shape: {outputs['mask'].shape}, range: [{outputs['mask'].min():.3f}, {outputs['mask'].max():.3f}]")
            
            # 获取mask并移除padding
            mask = outputs['mask'].squeeze().cpu().numpy()
            mask = mask[:orig_h, :orig_w]
            
            # 数值验证
            if np.isnan(mask).any() or np.isinf(mask).any():
                print(f"⚠️  警告: 帧{frame_idx} mask包含异常值 (NaN或Inf)")
                mask = np.nan_to_num(mask, nan=0.0, posinf=1.0, neginf=0.0)
            
            # Clip到合理范围
            mask = np.clip(mask, 0, 1)
            
            if self.debug and frame_idx == 0:
                print(f"最终Mask - shape: {mask.shape}, range: [{mask.min():.3f}, {mask.max():.3f}], mean: {mask.mean():.3f}")
            
            # 获取skeleton
            skeleton = outputs.get('skeleton', outputs['mask']).squeeze().cpu().numpy()
            skeleton = skeleton[:orig_h, :orig_w]
            skeleton = np.clip(skeleton, 0, 1)
            
            # 获取confidence（如果有）
            if 'confidence' in outputs:
                confidence = outputs['confidence'].squeeze().cpu().numpy()
                confidence = confidence[:orig_h, :orig_w]
                confidence = np.clip(confidence, 0, 1)
            else:
                confidence = np.ones_like(mask)
            
            return {
                'mask': mask,
                'skeleton': skeleton,
                'confidence': confidence
            }
            
        except Exception as e:
            print(f"❌ 分割帧{frame_idx}失败: {str(e)}")
            traceback.print_exc()
            # 返回空mask
            return {
                'mask': np.zeros_like(image),
                'skeleton': np.zeros_like(image),
                'confidence': np.zeros_like(image)
            }
    
    def enhance_frame(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Enhance single frame
        
        Args:
            image: Input image (H, W)
            mask: Vessel mask (H, W)
            
        Returns:
            Enhanced image (H, W)
        """
        if self.enhancement_model is None:
            return image
        
        # TODO: Implement enhancement
        return image
    
    def run_pipeline(self,
                    input_path: str,
                    output_dir: str,
                    max_frames: int = None,
                    save_debug: bool = True):
        """
        运行完整管线
        
        Args:
            input_path: 输入视频或图像序列路径
            output_dir: 输出目录
            max_frames: 最大处理帧数 (None表示全部)
            save_debug: 保存调试信息
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 创建输出子目录
        subdirs = ['frames', 'masks', 'enhanced', 'visualization', 'debug']
        for subdir in subdirs:
            (output_path / subdir).mkdir(exist_ok=True)
        
        print("\n" + "=" * 60)
        print("开始处理")
        print("=" * 60)
        
        # 加载视频
        print(f"正在加载视频: {input_path}")
        frames = self._load_video(input_path, max_frames)
        print(f"✓ 加载了 {len(frames)} 帧")
        
        if len(frames) == 0:
            print("❌ 没有加载到任何帧，请检查输入路径")
            return
        
        # 保存第一帧用于调试
        if save_debug:
            cv2.imwrite(str(output_path / 'debug' / 'first_frame.png'), frames[0])
        
        # Step 1: 预处理
        print("\n" + "-" * 60)
        print("步骤 1: 预处理")
        print("-" * 60)
        
        try:
            preprocessed = self.preprocessor.process_sequence(frames, select_keyframes=False)
            processed_frames = preprocessed['processed_frames']
            keyframe_indices = preprocessed.get('keyframe_indices', list(range(len(frames))))
            print(f"✓ 选择了 {len(keyframe_indices)} 个关键帧")
        except Exception as e:
            print(f"⚠️  预处理失败，使用原始帧: {str(e)}")
            processed_frames = frames
            keyframe_indices = list(range(len(frames)))
        
        # Step 2: 分割所有帧
        print("\n" + "-" * 60)
        print("步骤 2: 血管分割")
        print("-" * 60)
        
        segmentation_results = []
        failed_frames = []
        
        for i, frame in enumerate(tqdm(processed_frames, desc='分割进度')):
            result = self.segment_frame(frame, frame_idx=i)
            segmentation_results.append(result)
            
            # 检查分割质量
            if result['mask'].max() < 0.01:  # 几乎全黑
                failed_frames.append(i)
            
            # 保存mask
            mask_img = (result['mask'] * 255).astype(np.uint8)
            cv2.imwrite(str(output_path / 'masks' / f'mask_{i:04d}.png'), mask_img)
            
            # 保存调试信息（前5帧）
            if save_debug and i < 5:
                # 保存原始帧
                cv2.imwrite(str(output_path / 'debug' / f'frame_{i:04d}.png'), 
                           (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame)
                
                # 保存叠加可视化
                if len(frame.shape) == 2:
                    frame_vis = cv2.cvtColor((frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame, 
                                            cv2.COLOR_GRAY2BGR)
                else:
                    frame_vis = frame
                
                mask_colored = cv2.applyColorMap(mask_img, cv2.COLORMAP_JET)
                overlay = cv2.addWeighted(frame_vis, 0.6, mask_colored, 0.4, 0)
                cv2.imwrite(str(output_path / 'debug' / f'overlay_{i:04d}.png'), overlay)
        
        if failed_frames:
            print(f"⚠️  警告: {len(failed_frames)} 帧分割结果几乎为空: {failed_frames[:10]}{'...' if len(failed_frames) > 10 else ''}")
        
        print(f"✓ 分割完成")
        
        # Step 3: 增强帧
        print("\n" + "-" * 60)
        print("步骤 3: 图像增强")
        print("-" * 60)
        
        enhanced_frames = []
        for i, (frame, seg_result) in enumerate(tqdm(
            zip(processed_frames, segmentation_results),
            desc='增强进度',
            total=len(processed_frames)
        )):
            enhanced = self.enhance_frame(frame, seg_result['mask'])
            enhanced_frames.append(enhanced)
            
            # 保存增强结果
            enhanced_img = (enhanced * 255).astype(np.uint8) if enhanced.max() <= 1.0 else enhanced
            cv2.imwrite(str(output_path / 'enhanced' / f'enhanced_{i:04d}.png'), enhanced_img)
        
        print(f"✓ 增强完成")
        
        # Step 4: 3D重建
        print("\n" + "-" * 60)
        print("步骤 4: 3D 重建")
        print("-" * 60)
        
        try:
            self._run_3d_reconstruction(
                processed_frames,
                enhanced_frames,
                segmentation_results,
                keyframe_indices,
                output_path
            )
            print(f"✓ 3D重建完成")
        except Exception as e:
            print(f"⚠️  3D重建失败: {str(e)}")
            traceback.print_exc()
        
        # Step 5: 创建可视化视频
        print("\n" + "-" * 60)
        print("步骤 5: 生成可视化")
        print("-" * 60)
        
        try:
            self._create_visualization_video(
                processed_frames,
                segmentation_results,
                enhanced_frames,
                output_path
            )
            print(f"✓ 可视化完成")
        except Exception as e:
            print(f"⚠️  可视化失败: {str(e)}")
            traceback.print_exc()
        
        # 打印总结
        self._print_summary(output_path, len(frames), len(keyframe_indices), failed_frames)
    
    def _run_3d_reconstruction(self,
                              processed_frames,
                              enhanced_frames,
                              segmentation_results,
                              keyframe_indices,
                              output_path):
        """运行3D重建"""
        # Setup camera
        h, w = processed_frames[0].shape[:2]
        intrinsics = get_camera_intrinsics((h, w), fov=60.0)
        camera = Camera(
            width=w,
            height=h,
            fx=intrinsics['fx'],
            fy=intrinsics['fy'],
            cx=intrinsics['cx'],
            cy=intrinsics['cy']
        )
        
        # Create SLAM system
        slam = VesselGaussianSLAM(
            camera=camera,
            initial_points=5000,
            vessel_weight=2.0
        )
        
        # Process keyframes (限制数量避免内存问题)
        max_keyframes = min(10, len(keyframe_indices))
        for kf_idx in tqdm(keyframe_indices[:max_keyframes], desc='SLAM处理'):
            # Get frame and mask
            frame_rgb = np.stack([enhanced_frames[kf_idx]] * 3, axis=0)
            frame_tensor = torch.from_numpy(frame_rgb).float().to(self.device)
            
            mask_tensor = torch.from_numpy(segmentation_results[kf_idx]['mask']).unsqueeze(0).to(self.device)
            
            # Add keyframe
            slam.add_keyframe(kf_idx, frame_tensor, mask_tensor)
        
        # Export point cloud
        point_cloud_path = output_path / 'reconstruction.ply'
        slam.export_point_cloud(str(point_cloud_path))
        
        # Save camera trajectory
        trajectory = []
        for kf in slam.keyframes:
            pose = kf['pose_param'].detach().cpu().numpy()
            trajectory.append(pose)
        
        if trajectory:
            trajectory = np.array(trajectory)
            np.save(output_path / 'trajectory.npy', trajectory)
    
    def _load_video(self, video_path: str, max_frames: int = None) -> List[np.ndarray]:
        """加载视频帧"""
        video_path = Path(video_path)
        frames = []
        
        if video_path.is_file() and video_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
            # 从视频文件加载
            cap = cv2.VideoCapture(str(video_path))
            
            if not cap.isOpened():
                raise ValueError(f"无法打开视频文件: {video_path}")
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"视频信息: {total_frames} 帧, {fps:.2f} FPS")
            
            frame_idx = 0
            target_size = None
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 转灰度
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # 统一尺寸
                if target_size is None:
                    target_size = (frame.shape[1], frame.shape[0])
                else:
                    if (frame.shape[1], frame.shape[0]) != target_size:
                        frame = cv2.resize(frame, target_size)
                
                frames.append(frame)
                
                frame_idx += 1
                if max_frames and frame_idx >= max_frames:
                    break
            
            cap.release()
        
        elif video_path.is_dir():
            # 从图像目录加载
            image_files = sorted(video_path.glob('*.*'))
            image_files = [f for f in image_files if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp']]
            
            print(f"找到 {len(image_files)} 张图像")
            
            for i, img_path in enumerate(image_files):
                if max_frames and i >= max_frames:
                    break
                
                frame = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if frame is not None:
                    frames.append(frame)
        
        else:
            raise ValueError(f"无效的输入路径: {video_path}")
        
        return frames
    
    def _create_visualization_video(self,
                                   original_frames: List[np.ndarray],
                                   segmentation_results: List[Dict],
                                   enhanced_frames: List[np.ndarray],
                                   output_path: Path):
        """创建可视化视频"""
        print("正在创建可视化视频...")
        
        h, w = original_frames[0].shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            str(output_path / 'visualization.mp4'),
            fourcc,
            10.0,  # FPS
            (w * 3, h)
        )
        
        for i, (orig, seg, enh) in enumerate(zip(original_frames, segmentation_results, enhanced_frames)):
            # 归一化到0-255
            orig_vis = (orig * 255).astype(np.uint8) if orig.max() <= 1.0 else orig.astype(np.uint8)
            mask_vis = (seg['mask'] * 255).astype(np.uint8)
            enh_vis = (enh * 255).astype(np.uint8) if enh.max() <= 1.0 else enh.astype(np.uint8)
            
            # 转彩色
            orig_vis = cv2.cvtColor(orig_vis, cv2.COLOR_GRAY2BGR)
            mask_vis = cv2.applyColorMap(mask_vis, cv2.COLORMAP_JET)
            enh_vis = cv2.cvtColor(enh_vis, cv2.COLOR_GRAY2BGR)
            
            # 添加文本标签
            cv2.putText(orig_vis, 'Original', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(mask_vis, 'Segmentation', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(enh_vis, 'Enhanced', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # 添加帧号
            frame_text = f'Frame: {i}/{len(original_frames)}'
            cv2.putText(orig_vis, frame_text, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # 拼接
            vis = np.hstack([orig_vis, mask_vis, enh_vis])
            out.write(vis)
        
        out.release()
        print(f"✓ 可视化视频已保存")
    
    def _print_summary(self, output_path: Path, num_frames: int, num_keyframes: int, failed_frames: List[int]):
        """打印管线总结"""
        print("\n" + "=" * 60)
        print("处理完成总结")
        print("=" * 60)
        print(f"总处理帧数: {num_frames}")
        print(f"关键帧数量: {num_keyframes}")
        
        if failed_frames:
            print(f"⚠️  分割失败帧: {len(failed_frames)} 帧")
        
        print(f"\n输出文件:")
        print(f"  📁 分割masks: {output_path / 'masks'}")
        print(f"  📁 增强图像: {output_path / 'enhanced'}")
        print(f"  📁 调试信息: {output_path / 'debug'}")
        print(f"  🎬 可视化视频: {output_path / 'visualization.mp4'}")
        
        if (output_path / 'reconstruction.ply').exists():
            print(f"  🗿 3D点云: {output_path / 'reconstruction.ply'}")
        if (output_path / 'trajectory.npy').exists():
            print(f"  📊 相机轨迹: {output_path / 'trajectory.npy'}")
        
        print("=" * 60)
        print("\n💡 调试建议:")
        print("1. 检查 debug 文件夹中的前5帧overlay图像")
        print("2. 查看 visualization.mp4 确认分割效果")
        print("3. 如果分割全黑，检查模型权重是否正确")
        print("=" * 60)


def main():
    # ================= 配置区域 =================
    # 输入视频路径或图片文件夹路径
    input_path = "22.mp4"
    
    # 输出结果保存的文件夹路径
    output_dir = "output_results"
    
    # 训练好的分割模型权重文件路径 (.pth)
    seg_checkpoint_path = r"checkpoints\step2_segmentation\best_model.pth"
    
    # (可选) 增强模型路径，如果没有就设为 None
    enh_checkpoint_path = None
    
    # (可选) 最大处理帧数，None 表示处理全部
    max_frames_process = None
    
    # 是否启用调试模式（会输出详细信息）
    debug_mode = True
    
    # 是否保存调试文件
    save_debug_files = True
    # ===========================================

    print("=" * 60)
    print("血管3D重建推理管线")
    print("=" * 60)
    print(f"输入: {input_path}")
    print(f"输出: {output_dir}")
    print(f"模型: {seg_checkpoint_path}")
    print(f"调试: {'开启' if debug_mode else '关闭'}")
    print("=" * 60)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    try:
        # 创建推理管线
        pipeline = VesselReconstructionPipeline(
            segmentation_checkpoint=seg_checkpoint_path,
            enhancement_checkpoint=enh_checkpoint_path,
            config=None,
            debug=debug_mode
        )
        
        # 运行管线
        pipeline.run_pipeline(
            input_path=input_path,
            output_dir=output_dir,
            max_frames=max_frames_process,
            save_debug=save_debug_files
        )
        
        print("\n✅ 所有任务完成!")
        
    except Exception as e:
        print(f"\n❌ 程序执行出错: {str(e)}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()