# """
# Vessel Segmentation Models (Step 2)
# Implements U-Net++ with attention mechanisms
# """

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import List, Dict, Tuple, Optional
# from skimage.morphology import skeletonize
# import numpy as np


# class SCSEBlock(nn.Module):
#     """Spatial and Channel Squeeze & Excitation"""
    
#     def __init__(self, channels: int, reduction: int = 16):
#         super().__init__()
        
#         # Channel SE
#         self.cse = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(channels, channels // reduction, 1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channels // reduction, channels, 1),
#             nn.Sigmoid()
#         )
        
#         # Spatial SE
#         self.sse = nn.Sequential(
#             nn.Conv2d(channels, 1, 1),
#             nn.Sigmoid()
#         )
    
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # Channel attention
#         cse_out = self.cse(x) * x
        
#         # Spatial attention
#         sse_out = self.sse(x) * x
        
#         # Combine
#         return cse_out + sse_out


# class ConvBlock(nn.Module):
#     """Convolution Block with BatchNorm and ReLU"""
    
#     def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1):
#         super().__init__()
        
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, 3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Dropout2d(dropout),
#             nn.Conv2d(out_channels, out_channels, 3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )
    
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.conv(x)


# class UNetPlusPlus(nn.Module):
#     """
#     U-Net++ architecture for vessel segmentation
    
#     Features:
#     - Nested skip connections
#     - Deep supervision
#     - SCSE attention
#     - Multiple output heads (mask, skeleton, confidence)
#     """
    
#     def __init__(self,
#                  in_channels: int = 1,
#                  out_channels: int = 1,
#                  features: List[int] = [32, 64, 128, 256, 512],
#                  deep_supervision: bool = True,
#                  attention: str = 'scse',
#                  output_skeleton: bool = True,
#                  output_confidence: bool = True):
#         """
#         Args:
#             in_channels: Number of input channels
#             out_channels: Number of output channels
#             features: Feature channels at each level
#             deep_supervision: Enable deep supervision
#             attention: Attention type ('scse' or None)
#             output_skeleton: Output skeleton prediction
#             output_confidence: Output confidence map
#         """
#         super().__init__()
        
#         self.deep_supervision = deep_supervision
#         self.output_skeleton = output_skeleton
#         self.output_confidence = output_confidence
        
#         # Encoder
#         self.encoders = nn.ModuleList()
#         self.pools = nn.ModuleList()
        
#         for i, feat in enumerate(features):
#             in_ch = in_channels if i == 0 else features[i-1]
#             self.encoders.append(ConvBlock(in_ch, feat))
#             if i < len(features) - 1:
#                 self.pools.append(nn.MaxPool2d(2))
        
#         # Nested decoder
#         self.decoders = nn.ModuleDict()
        
#         # Build nested skip connections
#         for i in range(len(features) - 1):
#             for j in range(len(features) - 1 - i):
#                 # X^{i,j} where i is depth, j is skip connection index
#                 in_ch = features[j+1]  # From lower level
#                 skip_ch = features[j] * (i + 1)  # From skip connections
#                 out_ch = features[j]
                
#                 self.decoders[f'decoder_{i}_{j}'] = nn.Sequential(
#                     nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1),
#                     nn.BatchNorm2d(out_ch),
#                     nn.ReLU(inplace=True),
#                     nn.Conv2d(out_ch, out_ch, 3, padding=1),
#                     nn.BatchNorm2d(out_ch),
#                     nn.ReLU(inplace=True)
#                 )
                
#                 # Add attention
#                 if attention == 'scse':
#                     self.decoders[f'attention_{i}_{j}'] = SCSEBlock(out_ch)
        
#         # Upsampling
#         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
#         # Output heads
#         if deep_supervision:
#             # Multiple output heads for deep supervision
#             self.outputs = nn.ModuleList([
#                 nn.Conv2d(features[0], out_channels, 1)
#                 for _ in range(len(features) - 1)
#             ])
#         else:
#             # Single output head
#             self.outputs = nn.ModuleList([
#                 nn.Conv2d(features[0], out_channels, 1)
#             ])
        
#         # Skeleton output head
#         if output_skeleton:
#             self.skeleton_head = nn.Conv2d(features[0], 1, 1)
        
#         # Confidence output head
#         if output_confidence:
#             self.confidence_head = nn.Sequential(
#                 nn.Conv2d(features[0], features[0] // 2, 1),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(features[0] // 2, 1, 1),
#                 nn.Sigmoid()
#             )
    
#     def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
#         """
#         Forward pass
        
#         Args:
#             x: Input tensor (B, C, H, W)
            
#         Returns:
#             Dict with keys:
#                 - mask: Main segmentation output (B, 1, H, W)
#                 - mask_deep: List of deep supervision outputs (optional)
#                 - skeleton: Skeleton prediction (B, 1, H, W) (optional)
#                 - confidence: Confidence map (B, 1, H, W) (optional)
#         """
#         # Store encoder outputs
#         encoder_outputs = []
        
#         # Encoder path
#         for i, encoder in enumerate(self.encoders):
#             if i == 0:
#                 enc_out = encoder(x)
#             else:
#                 enc_out = encoder(self.pools[i-1](encoder_outputs[-1]))
#             encoder_outputs.append(enc_out)
        
#         # Nested decoder path
#         decoder_outputs = {f'0_{i}': encoder_outputs[i] for i in range(len(encoder_outputs))}
        
#         # Build decoder pyramid
#         for i in range(len(self.encoders) - 1):
#             for j in range(len(self.encoders) - 1 - i):
#                 # Get inputs
#                 lower = self.upsample(decoder_outputs[f'{i}_{j+1}'])
                
#                 # Collect skip connections
#                 skips = [decoder_outputs[f'{k}_{j}'] for k in range(i + 1)]
                
#                 # Concatenate
#                 dec_input = torch.cat([lower] + skips, dim=1)
                
#                 # Decode
#                 dec_out = self.decoders[f'decoder_{i}_{j}'](dec_input)
                
#                 # Apply attention
#                 if f'attention_{i}_{j}' in self.decoders:
#                     dec_out = self.decoders[f'attention_{i}_{j}'](dec_out)
                
#                 decoder_outputs[f'{i+1}_{j}'] = dec_out
        
#         # Generate outputs
#         result = {}
        
#         if self.deep_supervision:
#             # Deep supervision: use all nested outputs
#             masks = []
#             for i in range(len(self.encoders) - 1):
#                 mask = self.outputs[i](decoder_outputs[f'{i+1}_0'])
#                 masks.append(torch.sigmoid(mask))
            
#             result['mask'] = masks[-1]  # Use deepest as main output
#             result['mask_deep'] = masks
#         else:
#             # Single output
#             final_features = decoder_outputs[f'{len(self.encoders)-1}_0']
#             mask = self.outputs[0](final_features)
#             result['mask'] = torch.sigmoid(mask)
        
#         # Skeleton output
#         if self.output_skeleton:
#             final_features = decoder_outputs[f'{len(self.encoders)-1}_0']
#             skeleton = self.skeleton_head(final_features)
#             result['skeleton'] = torch.sigmoid(skeleton)
        
#         # Confidence output
#         if self.output_confidence:
#             final_features = decoder_outputs[f'{len(self.encoders)-1}_0']
#             confidence = self.confidence_head(final_features)
#             result['confidence'] = confidence
        
#         return result


# class VesselSegmentor(nn.Module):
#     """
#     Vessel segmentation module with post-processing
    
#     Includes:
#     - U-Net++ backbone
#     - Skeleton extraction
#     - Confidence estimation
#     """
    
#     def __init__(self, model: UNetPlusPlus):
#         super().__init__()
#         self.model = model
    
#     @torch.no_grad()
#     def extract_skeleton(self, mask: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
#         """
#         Extract vessel skeleton from binary mask
        
#         Args:
#             mask: Probability mask (B, 1, H, W)
#             threshold: Binarization threshold
            
#         Returns:
#             Skeleton (B, 1, H, W)
#         """
#         batch_size = mask.size(0)
#         skeletons = []
        
#         for i in range(batch_size):
#             # Binarize
#             binary_mask = (mask[i, 0].cpu().numpy() > threshold).astype(np.uint8)
            
#             # Skeletonize
#             skeleton = skeletonize(binary_mask).astype(np.float32)
            
#             # Convert back to tensor
#             skeleton = torch.from_numpy(skeleton).unsqueeze(0).to(mask.device)
#             skeletons.append(skeleton)
        
#         return torch.stack(skeletons, dim=0)
    
#     def forward(self, x: torch.Tensor, 
#                 extract_skeleton_post: bool = False) -> Dict[str, torch.Tensor]:
#         """
#         Forward pass with optional post-processing
        
#         Args:
#             x: Input image (B, C, H, W)
#             extract_skeleton_post: Extract skeleton via post-processing
            
#         Returns:
#             Dict with segmentation outputs
#         """
#         outputs = self.model(x)
        
#         # Post-process skeleton if requested
#         if extract_skeleton_post and 'skeleton' not in outputs:
#             outputs['skeleton'] = self.extract_skeleton(outputs['mask'])
        
#         return outputs


# def create_segmentation_model(config: Dict) -> VesselSegmentor:
#     """
#     Create segmentation model from config
    
#     Args:
#         config: Model configuration dict
        
#     Returns:
#         VesselSegmentor instance
#     """
#     model_config = config['model']
    
#     backbone = UNetPlusPlus(
#         in_channels=model_config['in_channels'],
#         out_channels=model_config['out_channels'],
#         features=model_config['features'],
#         deep_supervision=model_config['deep_supervision'],
#         attention=model_config.get('attention', 'scse'),
#         output_skeleton=model_config.get('output_skeleton', True),
#         output_confidence=model_config.get('output_confidence', True)
#     )
    
#     model = VesselSegmentor(backbone)
    
#     return model


# if __name__ == "__main__":
#     # Test model
#     print("Testing U-Net++ model...")
    
#     model = UNetPlusPlus(
#         in_channels=1,
#         out_channels=1,
#         features=[32, 64, 128, 256, 512],
#         deep_supervision=True,
#         attention='scse'
#     )
    
#     # Test forward pass
#     x = torch.randn(2, 1, 256, 256)
#     outputs = model(x)
    
#     print(f"Input shape: {x.shape}")
#     print(f"Mask shape: {outputs['mask'].shape}")
#     if 'mask_deep' in outputs:
#         print(f"Deep supervision outputs: {len(outputs['mask_deep'])}")
#     if 'skeleton' in outputs:
#         print(f"Skeleton shape: {outputs['skeleton'].shape}")
#     if 'confidence' in outputs:
#         print(f"Confidence shape: {outputs['confidence'].shape}")
    
#     # Count parameters
#     total_params = sum(p.numel() for p in model.parameters())
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"\nTotal parameters: {total_params:,}")
#     print(f"Trainable parameters: {trainable_params:,}")
"""
优化后的血管分割模型 (针对小静脉检出和边缘清晰度)
主要改进:
1. 多尺度边界感知注意力 (MSBA)
2. 小血管增强模块 (SVEM)
3. 高分辨率特征保留 (减少下采样层级)
4. 空洞卷积多尺度特征提取
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from skimage.morphology import skeletonize
import numpy as np


class SCSEBlock(nn.Module):
    """Spatial and Channel Squeeze & Excitation"""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        
        # Channel SE
        self.cse = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
        
        # Spatial SE
        self.sse = nn.Sequential(
            nn.Conv2d(channels, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel attention
        cse_out = self.cse(x) * x
        
        # Spatial attention
        sse_out = self.sse(x) * x
        
        # Combine
        return cse_out + sse_out


# ========== 新增模块 1: 边界感知注意力 ==========
class BoundaryAwareAttention(nn.Module):
    """
    边界感知注意力模块
    功能: 使用Sobel算子提取边缘特征，增强边界区域的特征表达
    """
    
    def __init__(self, channels: int):
        super().__init__()
        
        # Sobel算子用于边缘检测
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3).repeat(channels, 1, 1, 1))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3).repeat(channels, 1, 1, 1))
        
        # 边界特征融合
        self.boundary_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征 (B, C, H, W)
        Returns:
            边界增强后的特征 (B, C, H, W)
        """
        # 计算梯度
        grad_x = F.conv2d(x, self.sobel_x, padding=1, groups=x.size(1))
        grad_y = F.conv2d(x, self.sobel_y, padding=1, groups=x.size(1))
        
        # 合并梯度信息
        grad_mag = torch.cat([grad_x, grad_y], dim=1)
        
        # 生成边界注意力图
        boundary_attention = self.boundary_conv(grad_mag)
        
        # 应用注意力
        return x * (1 + boundary_attention)


# ========== 新增模块 2: 小血管增强模块 ==========
class SmallVesselEnhancementModule(nn.Module):
    """
    小血管增强模块 (SVEM)
    功能: 使用多尺度空洞卷积捕获不同尺度的血管特征
    """
    
    def __init__(self, channels: int):
        super().__init__()
        
        # 多尺度空洞卷积 (dilation rates: 1, 2, 4)
        self.atrous_conv1 = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 3, padding=1, dilation=1),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True)
        )
        self.atrous_conv2 = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 3, padding=2, dilation=2),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True)
        )
        self.atrous_conv4 = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 3, padding=4, dilation=4),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True)
        )
        
        # 1x1卷积分支
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True)
        )
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征 (B, C, H, W)
        Returns:
            多尺度增强后的特征 (B, C, H, W)
        """
        # 多尺度并行卷积
        feat1 = self.atrous_conv1(x)
        feat2 = self.atrous_conv2(x)
        feat4 = self.atrous_conv4(x)
        feat_1x1 = self.conv1x1(x)
        
        # 拼接并融合
        multi_scale_feat = torch.cat([feat1, feat2, feat4, feat_1x1], dim=1)
        output = self.fusion(multi_scale_feat)
        
        # 残差连接
        return x + output


class ConvBlock(nn.Module):
    """Convolution Block with BatchNorm and ReLU"""
    
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNetPlusPlus(nn.Module):
    """
    优化后的U-Net++架构
    
    改进点:
    1. 减少下采样层级 (5层→4层，保留更多细节)
    2. 每层解码器添加边界感知注意力
    3. 高分辨率层添加小血管增强模块
    4. 增加边界预测头
    """
    
    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 1,
                 features: List[int] = [32, 64, 128, 256],  # 改进: 减少到4层
                 deep_supervision: bool = True,
                 attention: str = 'scse',
                 output_skeleton: bool = True,
                 output_confidence: bool = True,
                 output_boundary: bool = True):  # 新增: 边界输出
        """
        Args:
            features: 改进 - 减少下采样层级 [32,64,128,256] 代替 [32,64,128,256,512]
            output_boundary: 是否输出边界预测
        """
        super().__init__()
        
        self.deep_supervision = deep_supervision
        self.output_skeleton = output_skeleton
        self.output_confidence = output_confidence
        self.output_boundary = output_boundary
        
        # Encoder
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        for i, feat in enumerate(features):
            in_ch = in_channels if i == 0 else features[i-1]
            self.encoders.append(ConvBlock(in_ch, feat))
            if i < len(features) - 1:
                self.pools.append(nn.MaxPool2d(2))
        
        # ========== 改进: 添加小血管增强模块到高分辨率层 ==========
        self.svem_modules = nn.ModuleDict()
        for i in range(2):  # 仅在前2层添加
            self.svem_modules[f'svem_{i}'] = SmallVesselEnhancementModule(features[i])
        
        # Nested decoder
        self.decoders = nn.ModuleDict()
        
        # ========== 改进: 添加边界感知注意力 ==========
        self.boundary_attention = nn.ModuleDict()
        
        # Build nested skip connections
        for i in range(len(features) - 1):
            for j in range(len(features) - 1 - i):
                in_ch = features[j+1]
                skip_ch = features[j] * (i + 1)
                out_ch = features[j]
                
                self.decoders[f'decoder_{i}_{j}'] = nn.Sequential(
                    nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True)
                )
                
                # Add attention
                if attention == 'scse':
                    self.decoders[f'attention_{i}_{j}'] = SCSEBlock(out_ch)
                
                # ========== 改进: 添加边界感知注意力 ==========
                self.boundary_attention[f'boundary_attn_{i}_{j}'] = BoundaryAwareAttention(out_ch)
        
        # Upsampling - 改进: 使用转置卷积代替双线性插值，保留边缘细节
        self.upsample_layers = nn.ModuleDict()
        for i in range(len(features) - 1):
            for j in range(len(features) - 1 - i):
                self.upsample_layers[f'up_{i}_{j}'] = nn.ConvTranspose2d(
                    features[j+1], features[j+1], kernel_size=2, stride=2
                )
        
        # Output heads
        if deep_supervision:
            self.outputs = nn.ModuleList([
                nn.Conv2d(features[0], out_channels, 1)
                for _ in range(len(features) - 1)
            ])
        else:
            self.outputs = nn.ModuleList([
                nn.Conv2d(features[0], out_channels, 1)
            ])
        
        # Skeleton output head
        if output_skeleton:
            self.skeleton_head = nn.Conv2d(features[0], 1, 1)
        
        # Confidence output head
        if output_confidence:
            self.confidence_head = nn.Sequential(
                nn.Conv2d(features[0], features[0] // 2, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(features[0] // 2, 1, 1),
                nn.Sigmoid()
            )
        
        # ========== 新增: 边界输出头 ==========
        if output_boundary:
            self.boundary_head = nn.Sequential(
                nn.Conv2d(features[0], features[0] // 2, 3, padding=1),
                nn.BatchNorm2d(features[0] // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(features[0] // 2, 1, 1)
            )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Returns:
            Dict with keys:
                - mask: 主分割输出
                - boundary: 边界预测 (新增)
                - mask_deep: 深度监督输出
                - skeleton: 骨架预测
                - confidence: 置信度图
        """
        # Store encoder outputs
        encoder_outputs = []
        
        # Encoder path
        for i, encoder in enumerate(self.encoders):
            if i == 0:
                enc_out = encoder(x)
            else:
                enc_out = encoder(self.pools[i-1](encoder_outputs[-1]))
            
            # ========== 改进: 在高分辨率层应用小血管增强 ==========
            if f'svem_{i}' in self.svem_modules:
                enc_out = self.svem_modules[f'svem_{i}'](enc_out)
            
            encoder_outputs.append(enc_out)
        
        # Nested decoder path
        decoder_outputs = {f'0_{i}': encoder_outputs[i] for i in range(len(encoder_outputs))}
        
        # Build decoder pyramid
        for i in range(len(self.encoders) - 1):
            for j in range(len(self.encoders) - 1 - i):
                # ========== 改进: 使用转置卷积上采样 ==========
                lower = self.upsample_layers[f'up_{i}_{j}'](decoder_outputs[f'{i}_{j+1}'])
                
                # Collect skip connections
                skips = [decoder_outputs[f'{k}_{j}'] for k in range(i + 1)]
                
                # Concatenate
                dec_input = torch.cat([lower] + skips, dim=1)
                
                # Decode
                dec_out = self.decoders[f'decoder_{i}_{j}'](dec_input)
                
                # Apply SCSE attention
                if f'attention_{i}_{j}' in self.decoders:
                    dec_out = self.decoders[f'attention_{i}_{j}'](dec_out)
                
                # ========== 改进: 应用边界感知注意力 ==========
                dec_out = self.boundary_attention[f'boundary_attn_{i}_{j}'](dec_out)
                
                decoder_outputs[f'{i+1}_{j}'] = dec_out
        
        # Generate outputs
        result = {}
        final_features = decoder_outputs[f'{len(self.encoders)-1}_0']
        
        if self.deep_supervision:
            masks = []
            for i in range(len(self.encoders) - 1):
                mask = self.outputs[i](decoder_outputs[f'{i+1}_0'])
                masks.append(mask)  # 注意: 不在这里sigmoid，留给损失函数处理
            
            result['mask'] = masks[-1]
            result['mask_deep'] = masks
        else:
            mask = self.outputs[0](final_features)
            result['mask'] = mask
        
        # Skeleton output
        if self.output_skeleton:
            skeleton = self.skeleton_head(final_features)
            result['skeleton'] = skeleton
        
        # Confidence output
        if self.output_confidence:
            confidence = self.confidence_head(final_features)
            result['confidence'] = confidence
        
        # ========== 新增: 边界输出 ==========
        if self.output_boundary:
            boundary = self.boundary_head(final_features)
            result['boundary'] = boundary
        
        return result


class VesselSegmentor(nn.Module):
    """
    Vessel segmentation module with post-processing
    """
    
    def __init__(self, model: UNetPlusPlus):
        super().__init__()
        self.model = model
    
    @torch.no_grad()
    def extract_skeleton(self, mask: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Extract vessel skeleton from binary mask
        """
        batch_size = mask.size(0)
        skeletons = []
        
        for i in range(batch_size):
            binary_mask = (mask[i, 0].cpu().numpy() > threshold).astype(np.uint8)
            skeleton = skeletonize(binary_mask).astype(np.float32)
            skeleton = torch.from_numpy(skeleton).unsqueeze(0).to(mask.device)
            skeletons.append(skeleton)
        
        return torch.stack(skeletons, dim=0)
    
    def forward(self, x: torch.Tensor, 
                extract_skeleton_post: bool = False) -> Dict[str, torch.Tensor]:
        outputs = self.model(x)
        
        if extract_skeleton_post and 'skeleton' not in outputs:
            outputs['skeleton'] = self.extract_skeleton(outputs['mask'].sigmoid())
        
        return outputs


def create_segmentation_model(config: Dict) -> VesselSegmentor:
    """
    Create segmentation model from config
    """
    model_config = config['model']
    
    backbone = UNetPlusPlus(
        in_channels=model_config['in_channels'],
        out_channels=model_config['out_channels'],
        features=model_config.get('features', [32, 64, 128, 256]),  # 改进: 默认4层
        deep_supervision=model_config['deep_supervision'],
        attention=model_config.get('attention', 'scse'),
        output_skeleton=model_config.get('output_skeleton', True),
        output_confidence=model_config.get('output_confidence', True),
        output_boundary=True  # 改进: 启用边界预测
    )
    
    model = VesselSegmentor(backbone)
    
    return model


if __name__ == "__main__":
    print("Testing Optimized U-Net++ model...")
    
    model = UNetPlusPlus(
        in_channels=1,
        out_channels=1,
        features=[32, 64, 128, 256],  # 改进: 4层
        deep_supervision=True,
        attention='scse',
        output_boundary=True
    )
    
    x = torch.randn(2, 1, 512, 512)
    outputs = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Mask shape: {outputs['mask'].shape}")
    if 'boundary' in outputs:
        print(f"Boundary shape: {outputs['boundary'].shape}")
    if 'mask_deep' in outputs:
        print(f"Deep supervision outputs: {len(outputs['mask_deep'])}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")