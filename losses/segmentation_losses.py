# """
# Loss Functions for Vessel Segmentation and Topology Preservation
# """

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import Optional, List
# import numpy as np
# from scipy.ndimage import distance_transform_edt


# class DiceLoss(nn.Module):
#     """Dice Loss for segmentation"""
    
#     def __init__(self, smooth: float = 1.0):
#         super().__init__()
#         self.smooth = smooth
    
#     def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             pred: Predictions (B, C, H, W), already passed through sigmoid
#             target: Ground truth (B, C, H, W), values in [0, 1]
            
#         Returns:
#             Dice loss (scalar)
#         """
#         pred = pred.contiguous().view(-1)
#         target = target.contiguous().view(-1)
        
#         intersection = (pred * target).sum()
#         dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
#         return 1 - dice


# class DiceBCELoss(nn.Module):
#     """Combined Dice + Binary Cross Entropy Loss"""
    
#     def __init__(self, 
#                  dice_weight: float = 0.7,
#                  bce_weight: float = 0.3,
#                  smooth: float = 1.0):
#         super().__init__()
#         self.dice_weight = dice_weight
#         self.bce_weight = bce_weight
#         self.dice_loss = DiceLoss(smooth=smooth)
#         self.bce_loss = nn.BCEWithLogitsLoss()
    
#     def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#         dice = self.dice_loss(pred, target)
#         bce = self.bce_loss(pred, target)
        
#         return self.dice_weight * dice + self.bce_weight * bce


# class ClDiceLoss(nn.Module):
#     """
#     Centerline Dice Loss (ClDice)
    
#     Evaluates segmentation quality based on skeleton/centerline connectivity
#     Reference: "clDice - A Novel Topology-Preserving Loss Function for 
#                Tubular Structure Segmentation" (CVPR 2021)
#     """
    
#     def __init__(self, smooth: float = 1.0, iter_: int = 3):
#         super().__init__()
#         self.smooth = smooth
#         self.iter = iter_
    
#     def soft_skeletonize(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Soft skeletonization using iterative morphological operations
        
#         Args:
#             x: Binary mask (B, 1, H, W)
            
#         Returns:
#             Soft skeleton (B, 1, H, W)
#         """
#         # Morphological operations
#         for i in range(self.iter):
#             # Erosion
#             min_pool = F.max_pool2d(x * -1, kernel_size=3, stride=1, padding=1) * -1
            
#             # Dilation of eroded
#             max_pool = F.max_pool2d(min_pool, kernel_size=3, stride=1, padding=1)
            
#             # Subtract to get boundary
#             boundary = x - max_pool
            
#             # Accumulate skeleton
#             if i == 0:
#                 skeleton = boundary
#             else:
#                 skeleton = torch.maximum(skeleton, boundary)
            
#             x = min_pool
        
#         return skeleton
    
#     def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             pred: Predicted mask (B, 1, H, W)
#             target: Ground truth mask (B, 1, H, W)
            
#         Returns:
#             ClDice loss
#         """
#         # Get soft skeletons
#         pred_skel = self.soft_skeletonize(pred)
#         target_skel = self.soft_skeletonize(target)
        
#         # Precision: skeleton pixels in prediction that match target
#         tprec = (pred_skel * target).sum() / (pred_skel.sum() + self.smooth)
        
#         # Recall: skeleton pixels in target that are predicted
#         tsens = (target_skel * pred).sum() / (target_skel.sum() + self.smooth)
        
#         # ClDice
#         cl_dice = 2 * tprec * tsens / (tprec + tsens + self.smooth)
        
#         return 1 - cl_dice


# class TopologyPreservingLoss(nn.Module):
#     """
#     Topology-preserving loss for vessel segmentation
    
#     Combines:
#     1. Standard segmentation loss (Dice + BCE)
#     2. Skeleton connectivity loss (ClDice)
#     3. Optional deep supervision
#     """
    
#     def __init__(self,
#                  dice_weight: float = 0.7,
#                  bce_weight: float = 0.3,
#                  cldice_weight: float = 0.1,
#                  deep_supervision: bool = True,
#                  ds_weights: Optional[List[float]] = None):
#         super().__init__()
        
#         self.dice_bce = DiceBCELoss(dice_weight, bce_weight)
#         self.cldice = ClDiceLoss()
#         self.cldice_weight = cldice_weight
#         self.deep_supervision = deep_supervision
        
#         # Deep supervision weights (from deep to shallow)
#         if ds_weights is None:
#             ds_weights = [1.0, 0.5, 0.25, 0.125]
#         self.ds_weights = ds_weights
    
#     def forward(self, 
#                 outputs: dict,
#                 target: torch.Tensor) -> dict:
#         """
#         Args:
#             outputs: Model outputs dict with keys:
#                 - mask: Main prediction (B, 1, H, W)
#                 - mask_deep: List of deep supervision outputs (optional)
#                 - skeleton: Skeleton prediction (optional)
#             target: Ground truth mask (B, 1, H, W)
            
#         Returns:
#             Dict with total loss and components
#         """
#         losses = {}
        
#         # Main segmentation loss
#         main_mask = outputs['mask']
#         seg_loss = self.dice_bce(main_mask, target)
#         losses['segmentation'] = seg_loss
        
#         # Topology loss (ClDice)
#         topo_loss = self.cldice(main_mask, target)
#         losses['topology'] = topo_loss
        
#         # Deep supervision
#         if self.deep_supervision and 'mask_deep' in outputs:
#             ds_loss = 0
#             for i, mask in enumerate(outputs['mask_deep']):
#                 weight = self.ds_weights[i] if i < len(self.ds_weights) else 0.1
                
#                 # Resize target to match deep supervision output
#                 target_resized = F.interpolate(
#                     target,
#                     size=mask.shape[-2:],
#                     mode='bilinear',
#                     align_corners=True
#                 )
                
#                 ds_loss += weight * self.dice_bce(mask, target_resized)
            
#             losses['deep_supervision'] = ds_loss / len(outputs['mask_deep'])
        
#         # Total loss
#         total_loss = seg_loss + self.cldice_weight * topo_loss
        
#         if 'deep_supervision' in losses:
#             total_loss += losses['deep_supervision']
        
#         losses['total'] = total_loss
        
#         return losses


# class VesselWeightedPhotometricLoss(nn.Module):
#     """
#     Vessel-weighted photometric loss for 3DGS-SLAM (Step 4)
    
#     Applies higher loss weight to vessel regions
#     """
    
#     def __init__(self,
#                  vessel_weight: float = 2.0,
#                  background_weight: float = 1.0,
#                  mask_threshold: float = 0.5,
#                  ssim_weight: float = 0.2):
#         super().__init__()
#         self.vessel_weight = vessel_weight
#         self.background_weight = background_weight
#         self.mask_threshold = mask_threshold
#         self.ssim_weight = ssim_weight
    
#     def ssim(self, 
#              img1: torch.Tensor, 
#              img2: torch.Tensor,
#              window_size: int = 11) -> torch.Tensor:
#         """
#         Compute SSIM loss
        
#         Args:
#             img1, img2: Images (B, C, H, W)
#             window_size: Window size for SSIM
            
#         Returns:
#             SSIM loss
#         """
#         # Simplified SSIM implementation
#         C1 = 0.01 ** 2
#         C2 = 0.03 ** 2
        
#         mu1 = F.avg_pool2d(img1, window_size, 1, window_size // 2)
#         mu2 = F.avg_pool2d(img2, window_size, 1, window_size // 2)
        
#         mu1_sq = mu1 ** 2
#         mu2_sq = mu2 ** 2
#         mu1_mu2 = mu1 * mu2
        
#         sigma1_sq = F.avg_pool2d(img1 ** 2, window_size, 1, window_size // 2) - mu1_sq
#         sigma2_sq = F.avg_pool2d(img2 ** 2, window_size, 1, window_size // 2) - mu2_sq
#         sigma12 = F.avg_pool2d(img1 * img2, window_size, 1, window_size // 2) - mu1_mu2
        
#         ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
#                    ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
#         return 1 - ssim_map.mean()
    
#     def forward(self,
#                 pred: torch.Tensor,
#                 target: torch.Tensor,
#                 vessel_mask: torch.Tensor) -> dict:
#         """
#         Args:
#             pred: Predicted/rendered image (B, C, H, W)
#             target: Target/ground truth image (B, C, H, W)
#             vessel_mask: Vessel probability mask (B, 1, H, W)
            
#         Returns:
#             Dict with loss components
#         """
#         # Create weight map
#         is_vessel = (vessel_mask > self.mask_threshold).float()
#         weight_map = (is_vessel * self.vessel_weight + 
#                      (1 - is_vessel) * self.background_weight)
        
#         # L1 loss with vessel weighting
#         l1_loss = torch.abs(pred - target)
#         weighted_l1 = (l1_loss * weight_map).mean()
        
#         # SSIM loss
#         ssim_loss = self.ssim(pred, target)
        
#         # Total loss
#         total_loss = weighted_l1 + self.ssim_weight * ssim_loss
        
#         return {
#             'total': total_loss,
#             'l1': weighted_l1,
#             'ssim': ssim_loss,
#             'vessel_weight_mean': weight_map.mean()
#         }


# class ClosedLoopLoss(nn.Module):
#     """
#     Closed-loop loss for end-to-end optimization (Step 5)
    
#     Combines:
#     1. Photometric consistency (rendering vs enhanced)
#     2. Segmentation consistency (rendered mask vs original mask)
#     3. Temporal smoothness (pose and appearance)
#     """
    
#     def __init__(self,
#                  photometric_weight: float = 1.0,
#                  segmentation_weight: float = 0.3,
#                  temporal_weight: float = 0.05):
#         super().__init__()
        
#         self.photometric_weight = photometric_weight
#         self.segmentation_weight = segmentation_weight
#         self.temporal_weight = temporal_weight
        
#         self.vessel_photometric = VesselWeightedPhotometricLoss()
    
#     def forward(self,
#                 rendered_image: torch.Tensor,
#                 enhanced_image: torch.Tensor,
#                 rendered_mask: torch.Tensor,
#                 target_mask: torch.Tensor,
#                 poses: Optional[torch.Tensor] = None) -> dict:
#         """
#         Args:
#             rendered_image: Rendered from 3DGS (B, C, H, W)
#             enhanced_image: Enhanced by diffusion (B, C, H, W)
#             rendered_mask: Mask from rendered image (B, 1, H, W)
#             target_mask: Original vessel mask (B, 1, H, W)
#             poses: Camera poses for temporal smoothness (B, T, 7) [optional]
            
#         Returns:
#             Dict with loss components
#         """
#         losses = {}
        
#         # Photometric consistency
#         photo_losses = self.vessel_photometric(
#             rendered_image,
#             enhanced_image,
#             target_mask
#         )
#         losses['photometric'] = photo_losses['total']
        
#         # Segmentation consistency
#         seg_loss = F.binary_cross_entropy(rendered_mask, target_mask)
#         losses['segmentation'] = seg_loss
        
#         # Temporal smoothness (if poses provided)
#         if poses is not None and poses.size(1) > 1:
#             # Compute pose differences between consecutive frames
#             pose_diff = torch.diff(poses, dim=1)
#             temporal_loss = torch.mean(pose_diff ** 2)
#             losses['temporal'] = temporal_loss
#         else:
#             losses['temporal'] = torch.tensor(0.0, device=rendered_image.device)
        
#         # Total loss
#         total_loss = (self.photometric_weight * losses['photometric'] +
#                      self.segmentation_weight * losses['segmentation'] +
#                      self.temporal_weight * losses['temporal'])
        
#         losses['total'] = total_loss
        
#         return losses


# if __name__ == "__main__":
#     # Test losses
#     print("Testing loss functions...")
    
#     # Create dummy data
#     B, C, H, W = 2, 1, 256, 256
#     pred = torch.sigmoid(torch.randn(B, C, H, W))
#     target = torch.randint(0, 2, (B, C, H, W)).float()
    
#     # Test Dice Loss
#     dice_loss = DiceLoss()
#     loss = dice_loss(pred, target)
#     print(f"Dice Loss: {loss.item():.4f}")
    
#     # Test DiceBCE Loss
#     dice_bce_loss = DiceBCELoss()
#     loss = dice_bce_loss(pred, target)
#     print(f"DiceBCE Loss: {loss.item():.4f}")
    
#     # Test ClDice Loss
#     cldice_loss = ClDiceLoss()
#     loss = cldice_loss(pred, target)
#     print(f"ClDice Loss: {loss.item():.4f}")
    
#     # Test Topology Preserving Loss
#     topo_loss = TopologyPreservingLoss()
#     outputs = {
#         'mask': pred,
#         'mask_deep': [pred, pred, pred]
#     }
#     losses = topo_loss(outputs, target)
#     print(f"Total Topology Loss: {losses['total'].item():.4f}")
#     print(f"  - Segmentation: {losses['segmentation'].item():.4f}")
#     print(f"  - Topology: {losses['topology'].item():.4f}")
#     print(f"  - Deep Supervision: {losses['deep_supervision'].item():.4f}")
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
import numpy as np
from scipy.ndimage import distance_transform_edt


class DiceLoss(nn.Module):
    """Dice Loss for segmentation"""
    
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predictions (B, C, H, W), logits (未经sigmoid)
            target: Ground truth (B, C, H, W), values in [0, 1]
        """
        pred = torch.sigmoid(pred)  # 先sigmoid
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice


class DiceBCELoss(nn.Module):
    """Combined Dice + Binary Cross Entropy Loss"""
    
    def __init__(self, 
                 dice_weight: float = 0.7,
                 bce_weight: float = 0.3,
                 smooth: float = 1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss(smooth=smooth)
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dice = self.dice_loss(pred, target)
        bce = self.bce_loss(pred, target)
        
        return self.dice_weight * dice + self.bce_weight * bce


# ========== 新增损失 1: 边界敏感损失 ==========
class BoundaryLoss(nn.Module):
    """
    边界敏感损失
    功能: 通过距离变换对边界区域施加更高权重
    理论: 使用距离变换生成权重图，边界附近像素权重更高
    """
    
    def __init__(self, theta0: float = 3.0, theta: float = 5.0):
        super().__init__()
        self.theta0 = theta0
        self.theta = theta
    
    def compute_distance_transform(self, mask: torch.Tensor) -> torch.Tensor:
        """
        计算距离变换
        Args:
            mask: 二值mask (B, 1, H, W)
        Returns:
            距离图 (B, 1, H, W)
        """
        batch_size = mask.size(0)
        dist_maps = []
        
        for i in range(batch_size):
            binary_mask = mask[i, 0].cpu().numpy()
            
            # 计算到前景的距离
            dist_fg = distance_transform_edt(binary_mask)
            # 计算到背景的距离
            dist_bg = distance_transform_edt(1 - binary_mask)
            
            # 合并距离（边界处距离小）
            dist_map = np.minimum(dist_fg, dist_bg)
            
            dist_tensor = torch.from_numpy(dist_map).unsqueeze(0).to(mask.device)
            dist_maps.append(dist_tensor)
        
        return torch.stack(dist_maps, dim=0)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: 预测logits (B, 1, H, W)
            target: 真值 (B, 1, H, W)
        """
        # 计算距离变换
        with torch.no_grad():
            dist_map = self.compute_distance_transform(target)
            
            # 生成权重图: 边界附近权重高
            weight_map = torch.exp(-dist_map / self.theta0)
            weight_map = weight_map / weight_map.max()  # 归一化
            
            # 计算最终权重
            final_weight = 1 + self.theta * weight_map
        
        # 修复: 使用 binary_cross_entropy_with_logits (AMP安全)
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        weighted_loss = (bce_loss * final_weight).mean()
        
        return weighted_loss


# ========== 新增损失 2: Hausdorff距离损失 ==========
class HausdorffDistanceLoss(nn.Module):
    """
    Hausdorff距离损失 (HD95)
    功能: 惩罚边界预测的最大误差，确保边界清晰
    """
    
    def __init__(self, alpha: float = 2.0):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: 预测logits (B, 1, H, W)
            target: 真值 (B, 1, H, W)
        """
        pred_sigmoid = torch.sigmoid(pred)
        
        # 计算边界
        pred_boundary = self._get_boundary(pred_sigmoid)
        target_boundary = self._get_boundary(target)
        
        # 计算Hausdorff距离 (简化版)
        if pred_boundary.sum() > 0 and target_boundary.sum() > 0:
            # 计算预测边界到真值边界的距离
            dist_pred_to_target = self._boundary_distance(pred_boundary, target_boundary)
            dist_target_to_pred = self._boundary_distance(target_boundary, pred_boundary)
            
            # 双向Hausdorff距离
            hd_loss = torch.max(dist_pred_to_target, dist_target_to_pred)
        else:
            hd_loss = torch.tensor(0.0, device=pred.device)
        
        return hd_loss
    
    def _get_boundary(self, mask: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """提取边界"""
        binary_mask = (mask > threshold).float()
        
        # 形态学梯度: 膨胀 - 腐蚀
        kernel = torch.ones(1, 1, 3, 3, device=mask.device)
        dilated = F.conv2d(binary_mask, kernel, padding=1)
        eroded = -F.conv2d(-binary_mask, kernel, padding=1)
        
        boundary = (dilated - eroded > 0).float()
        return boundary
    
    def _boundary_distance(self, boundary1: torch.Tensor, boundary2: torch.Tensor) -> torch.Tensor:
        """计算边界间距离"""
        # 简化实现: 使用L2距离
        coords1 = torch.nonzero(boundary1[0, 0], as_tuple=False).float()
        coords2 = torch.nonzero(boundary2[0, 0], as_tuple=False).float()
        
        if len(coords1) == 0 or len(coords2) == 0:
            return torch.tensor(0.0, device=boundary1.device)
        
        # 计算最小距离
        dist_matrix = torch.cdist(coords1, coords2)
        min_dist = dist_matrix.min(dim=1)[0].mean()
        
        return min_dist


class ClDiceLoss(nn.Module):
    """
    Centerline Dice Loss (ClDice)
    改进: 增加迭代次数，更好地保持拓扑
    """
    
    def __init__(self, smooth: float = 1.0, iter_: int = 5):  # 改进: 迭代次数 3→5
        super().__init__()
        self.smooth = smooth
        self.iter = iter_
    
    def soft_skeletonize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Soft skeletonization using iterative morphological operations
        """
        for i in range(self.iter):
            min_pool = F.max_pool2d(x * -1, kernel_size=3, stride=1, padding=1) * -1
            max_pool = F.max_pool2d(min_pool, kernel_size=3, stride=1, padding=1)
            boundary = x - max_pool
            
            if i == 0:
                skeleton = boundary
            else:
                skeleton = torch.maximum(skeleton, boundary)
            
            x = min_pool
        
        return skeleton
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: 预测logits (B, 1, H, W)
            target: 真值 (B, 1, H, W)
        """
        pred_sigmoid = torch.sigmoid(pred)
        
        # Get soft skeletons
        pred_skel = self.soft_skeletonize(pred_sigmoid)
        target_skel = self.soft_skeletonize(target)
        
        # Precision & Recall
        tprec = (pred_skel * target).sum() / (pred_skel.sum() + self.smooth)
        tsens = (target_skel * pred_sigmoid).sum() / (target_skel.sum() + self.smooth)
        
        # ClDice
        cl_dice = 2 * tprec * tsens / (tprec + tsens + self.smooth)
        
        return 1 - cl_dice


# ========== 新增损失 3: 小血管加权损失 ==========
class SmallVesselWeightedLoss(nn.Module):
    """
    小血管加权损失
    功能: 对细小血管区域施加更高权重
    """
    
    def __init__(self, small_vessel_threshold: int = 5):
        super().__init__()
        self.threshold = small_vessel_threshold
    
    def compute_vessel_width(self, mask: torch.Tensor) -> torch.Tensor:
        """
        估算血管宽度 (使用距离变换)
        """
        batch_size = mask.size(0)
        width_maps = []
        
        for i in range(batch_size):
            binary_mask = mask[i, 0].cpu().numpy()
            
            # 距离变换: 像素到边界的距离 ≈ 半径
            dist = distance_transform_edt(binary_mask)
            width = dist * 2  # 直径 = 半径 × 2
            
            width_tensor = torch.from_numpy(width).unsqueeze(0).to(mask.device)
            width_maps.append(width_tensor)
        
        return torch.stack(width_maps, dim=0)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: 预测logits (B, 1, H, W)
            target: 真值 (B, 1, H, W)
        """
        # 计算血管宽度
        with torch.no_grad():
            width_map = self.compute_vessel_width(target)
            
            # 小血管区域: 宽度 < threshold
            small_vessel_mask = (width_map > 0) & (width_map < self.threshold)
            
            # 权重图: 小血管区域权重 = 3, 其他 = 1
            weight_map = torch.ones_like(target)
            weight_map[small_vessel_mask] = 3.0
        
        # 修复: 使用 binary_cross_entropy_with_logits (AMP安全)
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        weighted_loss = (bce_loss * weight_map).mean()
        
        return weighted_loss


class TopologyPreservingLoss(nn.Module):
    """
    优化后的拓扑保持损失
    
    结合:
    1. DiceBCE (基础分割)
    2. ClDice (拓扑连通性)
    3. Boundary Loss (边界清晰度) - 新增
    4. Hausdorff Loss (边界精度) - 新增
    5. Small Vessel Weighted Loss (小血管增强) - 新增
    """
    
    def __init__(self,
                 dice_weight: float = 0.5,  # 改进: 降低基础损失权重
                 bce_weight: float = 0.2,
                 cldice_weight: float = 0.1,
                 boundary_weight: float = 0.15,  # 新增
                 hausdorff_weight: float = 0.05,  # 新增
                 small_vessel_weight: float = 0.1,  # 新增
                 deep_supervision: bool = True,
                 ds_weights: Optional[List[float]] = None):
        super().__init__()
        
        self.dice_bce = DiceBCELoss(dice_weight / (dice_weight + bce_weight), 
                                     bce_weight / (dice_weight + bce_weight))
        self.cldice = ClDiceLoss()
        self.boundary_loss = BoundaryLoss()  # 新增
        self.hausdorff_loss = HausdorffDistanceLoss()  # 新增
        self.small_vessel_loss = SmallVesselWeightedLoss()  # 新增
        
        self.cldice_weight = cldice_weight
        self.boundary_weight = boundary_weight
        self.hausdorff_weight = hausdorff_weight
        self.small_vessel_weight = small_vessel_weight
        self.deep_supervision = deep_supervision
        
        if ds_weights is None:
            ds_weights = [1.0, 0.5, 0.25]  # 改进: 适配4层网络
        self.ds_weights = ds_weights
    
    def forward(self, 
                outputs: dict,
                target: torch.Tensor) -> dict:
        """
        Args:
            outputs: 模型输出
                - mask: 主预测 (B, 1, H, W)
                - boundary: 边界预测 (可选)
                - mask_deep: 深度监督输出
            target: 真值 (B, 1, H, W)
        """
        losses = {}
        main_mask = outputs['mask']
        
        # 1. 基础分割损失
        seg_loss = self.dice_bce(main_mask, target)
        losses['segmentation'] = seg_loss
        
        # 2. 拓扑损失
        topo_loss = self.cldice(main_mask, target)
        losses['topology'] = topo_loss
        
        # ========== 新增损失 ==========
        # 3. 边界敏感损失
        boundary_loss = self.boundary_loss(main_mask, target)
        losses['boundary'] = boundary_loss
        
        # 4. Hausdorff距离损失
        hd_loss = self.hausdorff_loss(main_mask, target)
        losses['hausdorff'] = hd_loss
        
        # 5. 小血管加权损失
        sv_loss = self.small_vessel_loss(main_mask, target)
        losses['small_vessel'] = sv_loss
        
        # 6. 边界预测损失 (如果有边界输出)
        if 'boundary' in outputs:
            # 计算真值边界
            target_boundary = self._extract_boundary(target)
            boundary_pred_loss = F.binary_cross_entropy_with_logits(
                outputs['boundary'], target_boundary
            )
            losses['boundary_pred'] = boundary_pred_loss
        
        # 7. 深度监督
        if self.deep_supervision and 'mask_deep' in outputs:
            ds_loss = 0
            for i, mask in enumerate(outputs['mask_deep']):
                weight = self.ds_weights[i] if i < len(self.ds_weights) else 0.1
                
                target_resized = F.interpolate(
                    target,
                    size=mask.shape[-2:],
                    mode='bilinear',
                    align_corners=True
                )
                
                ds_loss += weight * self.dice_bce(mask, target_resized)
            
            losses['deep_supervision'] = ds_loss / len(outputs['mask_deep'])
        
        # 总损失
        total_loss = (
            seg_loss +
            self.cldice_weight * topo_loss +
            self.boundary_weight * boundary_loss +
            self.hausdorff_weight * hd_loss +
            self.small_vessel_weight * sv_loss
        )
        
        if 'boundary_pred' in losses:
            total_loss += 0.1 * losses['boundary_pred']
        
        if 'deep_supervision' in losses:
            total_loss += losses['deep_supervision']
        
        losses['total'] = total_loss
        
        return losses
    
    def _extract_boundary(self, mask: torch.Tensor) -> torch.Tensor:
        """提取边界"""
        kernel = torch.ones(1, 1, 3, 3, device=mask.device)
        dilated = F.conv2d(mask, kernel, padding=1)
        eroded = -F.conv2d(-mask, kernel, padding=1)
        boundary = (dilated - eroded > 0).float()
        return boundary


class VesselWeightedPhotometricLoss(nn.Module):
    """
    Vessel-weighted photometric loss for 3DGS-SLAM (Step 4)
    """
    
    def __init__(self,
                 vessel_weight: float = 2.0,
                 background_weight: float = 1.0,
                 mask_threshold: float = 0.5,
                 ssim_weight: float = 0.2):
        super().__init__()
        self.vessel_weight = vessel_weight
        self.background_weight = background_weight
        self.mask_threshold = mask_threshold
        self.ssim_weight = ssim_weight
    
    def ssim(self, 
             img1: torch.Tensor, 
             img2: torch.Tensor,
             window_size: int = 11) -> torch.Tensor:
        """Compute SSIM loss"""
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        mu1 = F.avg_pool2d(img1, window_size, 1, window_size // 2)
        mu2 = F.avg_pool2d(img2, window_size, 1, window_size // 2)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.avg_pool2d(img1 ** 2, window_size, 1, window_size // 2) - mu1_sq
        sigma2_sq = F.avg_pool2d(img2 ** 2, window_size, 1, window_size // 2) - mu2_sq
        sigma12 = F.avg_pool2d(img1 * img2, window_size, 1, window_size // 2) - mu1_mu2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return 1 - ssim_map.mean()
    
    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor,
                vessel_mask: torch.Tensor) -> dict:
        is_vessel = (vessel_mask > self.mask_threshold).float()
        weight_map = (is_vessel * self.vessel_weight + 
                     (1 - is_vessel) * self.background_weight)
        
        l1_loss = torch.abs(pred - target)
        weighted_l1 = (l1_loss * weight_map).mean()
        
        ssim_loss = self.ssim(pred, target)
        
        total_loss = weighted_l1 + self.ssim_weight * ssim_loss
        
        return {
            'total': total_loss,
            'l1': weighted_l1,
            'ssim': ssim_loss,
            'vessel_weight_mean': weight_map.mean()
        }


if __name__ == "__main__":
    print("Testing optimized loss functions...")
    
    B, C, H, W = 2, 1, 256, 256
    pred = torch.randn(B, C, H, W)  # logits
    target = torch.randint(0, 2, (B, C, H, W)).float()
    
    # Test Boundary Loss
    boundary_loss = BoundaryLoss()
    loss = boundary_loss(pred, target)
    print(f"Boundary Loss: {loss.item():.4f}")
    
    # Test Hausdorff Loss
    hd_loss = HausdorffDistanceLoss()
    loss = hd_loss(pred, target)
    print(f"Hausdorff Loss: {loss.item():.4f}")
    
    # Test Small Vessel Loss
    sv_loss = SmallVesselWeightedLoss()
    loss = sv_loss(pred, target)
    print(f"Small Vessel Loss: {loss.item():.4f}")
    
    # Test Total Loss
    topo_loss = TopologyPreservingLoss()
    outputs = {
        'mask': pred,
        'mask_deep': [pred, pred, pred],
        'boundary': pred
    }
    losses = topo_loss(outputs, target)
    print(f"\nTotal Loss: {losses['total'].item():.4f}")
    for key, val in losses.items():
        if key != 'total':
            print(f"  - {key}: {val.item():.4f}")
