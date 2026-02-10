"""
Vessel-Aware 3D Gaussian Splatting SLAM (Step 4)

Key Innovation: Vessel-weighted photometric loss in 3DGS
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import numpy as np
from dataclasses import dataclass


@dataclass
class GaussianParams:
    """3D Gaussian parameters"""
    means: torch.Tensor          # (N, 3) - 3D positions
    scales: torch.Tensor         # (N, 3) - Scales
    rotations: torch.Tensor      # (N, 4) - Quaternions
    opacities: torch.Tensor      # (N, 1) - Opacities
    features: torch.Tensor       # (N, F) - SH features or colors
    

class Camera:
    """Camera parameters"""
    
    def __init__(self,
                 width: int,
                 height: int,
                 fx: float,
                 fy: float,
                 cx: float,
                 cy: float):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        
        # Intrinsic matrix
        self.K = torch.tensor([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=torch.float32)
    
    def project(self, points_3d: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
        """
        Project 3D points to 2D image plane
        
        Args:
            points_3d: (N, 3) 3D points in world frame
            pose: (4, 4) camera pose [R|t]
            
        Returns:
            points_2d: (N, 2) 2D projections
        """
        # Transform to camera frame
        R = pose[:3, :3]
        t = pose[:3, 3]
        
        points_cam = (R @ points_3d.T).T + t
        
        # Project using intrinsics
        points_2d = points_cam[:, :2] / points_cam[:, 2:3]
        points_2d[:, 0] = points_2d[:, 0] * self.fx + self.cx
        points_2d[:, 1] = points_2d[:, 1] * self.fy + self.cy
        
        return points_2d, points_cam[:, 2]  # Return depth as well


class GaussianRenderer:
    """
    Simplified Gaussian Splatting Renderer
    
    Note: Full implementation requires CUDA kernels from gaussian-splatting repo
    This is a PyTorch-only approximation for demonstration
    """
    
    def __init__(self, camera: Camera):
        self.camera = camera
    
    def render(self,
               gaussians: GaussianParams,
               pose: torch.Tensor,
               vessel_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Render image from 3D Gaussians
        
        Args:
            gaussians: Gaussian parameters
            pose: Camera pose (4, 4)
            vessel_mask: Vessel mask for weighted rendering (1, H, W)
            
        Returns:
            Dict with rendered outputs
        """
        device = gaussians.means.device
        H, W = self.camera.height, self.camera.width
        
        # Project Gaussians to image plane
        points_2d, depths = self.camera.project(gaussians.means, pose)
        
        # Filter points outside image
        valid_mask = (
            (points_2d[:, 0] >= 0) & (points_2d[:, 0] < W) &
            (points_2d[:, 1] >= 0) & (points_2d[:, 1] < H) &
            (depths > 0)
        )
        
        points_2d = points_2d[valid_mask]
        depths = depths[valid_mask]
        opacities = gaussians.opacities[valid_mask]
        features = gaussians.features[valid_mask]
        scales_2d = gaussians.scales[valid_mask, :2]  # Project scales
        
        # Initialize output
        rendered_image = torch.zeros(3, H, W, device=device)
        rendered_depth = torch.zeros(1, H, W, device=device)
        rendered_alpha = torch.zeros(1, H, W, device=device)
        
        # Sort by depth (painter's algorithm)
        depth_order = torch.argsort(depths, descending=True)
        
        # Simplified splatting (not optimal - real impl uses CUDA)
        for idx in depth_order:
            x, y = points_2d[idx]
            x_int, y_int = int(x), int(y)
            
            if 0 <= x_int < W and 0 <= y_int < H:
                # Simplified Gaussian weight (should be proper 2D Gaussian)
                weight = opacities[idx]
                
                # Alpha blending
                alpha = rendered_alpha[0, y_int, x_int]
                rendered_image[:, y_int, x_int] = (
                    rendered_image[:, y_int, x_int] * (1 - weight) +
                    features[idx, :3] * weight
                )
                rendered_depth[0, y_int, x_int] = (
                    rendered_depth[0, y_int, x_int] * (1 - weight) +
                    depths[idx] * weight
                )
                rendered_alpha[0, y_int, x_int] = alpha + (1 - alpha) * weight
        
        return {
            'image': rendered_image,
            'depth': rendered_depth,
            'alpha': rendered_alpha
        }


class VesselGaussianSLAM(nn.Module):
    """
    Vessel-Aware Gaussian Splatting SLAM System
    
    Combines:
    1. 3D Gaussian representation
    2. Camera pose tracking
    3. Vessel-weighted photometric loss
    4. Keyframe-based mapping
    """
    
    def __init__(self,
                 camera: Camera,
                 initial_points: int = 10000,
                 max_points: int = 100000,
                 vessel_weight: float = 2.0):
        """
        Args:
            camera: Camera parameters
            initial_points: Initial number of Gaussians
            max_points: Maximum number of Gaussians
            vessel_weight: Weight for vessel regions in photometric loss
        """
        super().__init__()
        
        self.camera = camera
        self.max_points = max_points
        self.vessel_weight = vessel_weight
        
        # Initialize Gaussians
        self.gaussians = self._initialize_gaussians(initial_points)
        
        # Renderer
        self.renderer = GaussianRenderer(camera)
        
        # Pose parameters (will be optimized)
        self.poses = {}  # Dict of {frame_idx: nn.Parameter(pose)}
        
        # Keyframes
        self.keyframes = []
    
    def _initialize_gaussians(self, num_points: int) -> GaussianParams:
        """
        Initialize random Gaussians
        
        Args:
            num_points: Number of points
            
        Returns:
            GaussianParams
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Random initialization in a volume
        means = torch.randn(num_points, 3, device=device) * 2.0
        
        # Small initial scales
        scales = torch.ones(num_points, 3, device=device) * 0.01
        
        # Identity rotations (quaternions)
        rotations = torch.zeros(num_points, 4, device=device)
        rotations[:, 0] = 1.0  # w component
        
        # Low initial opacity
        opacities = torch.sigmoid(torch.randn(num_points, 1, device=device) - 2.0)
        
        # Random colors (will be optimized)
        features = torch.rand(num_points, 3, device=device)
        
        return GaussianParams(
            means=nn.Parameter(means),
            scales=nn.Parameter(scales),
            rotations=nn.Parameter(rotations),
            opacities=nn.Parameter(opacities),
            features=nn.Parameter(features)
        )
    
    def add_keyframe(self,
                    frame_idx: int,
                    image: torch.Tensor,
                    vessel_mask: torch.Tensor,
                    initial_pose: Optional[torch.Tensor] = None):
        """
        Add new keyframe
        
        Args:
            frame_idx: Frame index
            image: Image tensor (C, H, W)
            vessel_mask: Vessel mask (1, H, W)
            initial_pose: Initial pose estimate (4, 4)
        """
        if initial_pose is None:
            # Initialize as identity
            initial_pose = torch.eye(4, device=image.device)
        
        # Store as optimizable parameter
        pose_param = nn.Parameter(initial_pose)
        self.poses[frame_idx] = pose_param
        
        # Store keyframe data
        self.keyframes.append({
            'frame_idx': frame_idx,
            'image': image,
            'vessel_mask': vessel_mask,
            'pose_param': pose_param
        })
    
    def track_frame(self,
                   image: torch.Tensor,
                   vessel_mask: torch.Tensor,
                   initial_pose: torch.Tensor,
                   num_iterations: int = 20) -> Tuple[torch.Tensor, Dict]:
        """
        Track camera pose for new frame
        
        Args:
            image: Input image (C, H, W)
            vessel_mask: Vessel mask (1, H, W)
            initial_pose: Initial pose estimate (4, 4)
            num_iterations: Optimization iterations
            
        Returns:
            optimized_pose: Refined pose (4, 4)
            info: Dict with tracking info
        """
        pose = nn.Parameter(initial_pose.clone())
        optimizer = torch.optim.Adam([pose], lr=1e-3)
        
        losses = []
        
        for i in range(num_iterations):
            optimizer.zero_grad()
            
            # Render from current pose
            rendered = self.renderer.render(self.gaussians, pose, vessel_mask)
            
            # Photometric loss (vessel-weighted)
            photo_loss = self._vessel_weighted_loss(
                rendered['image'],
                image,
                vessel_mask
            )
            
            photo_loss.backward()
            optimizer.step()
            
            losses.append(photo_loss.item())
        
        return pose.detach(), {'losses': losses}
    
    def _vessel_weighted_loss(self,
                             rendered: torch.Tensor,
                             target: torch.Tensor,
                             vessel_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute vessel-weighted photometric loss
        
        Args:
            rendered: Rendered image (C, H, W)
            target: Target image (C, H, W)
            vessel_mask: Vessel mask (1, H, W)
            
        Returns:
            Weighted loss
        """
        # Create weight map
        weight_map = torch.ones_like(vessel_mask)
        weight_map[vessel_mask > 0.5] = self.vessel_weight
        
        # L1 loss
        l1_loss = torch.abs(rendered - target)
        
        # Apply weights
        weighted_loss = (l1_loss * weight_map).mean()
        
        return weighted_loss
    
    def densify(self, gradient_threshold: float = 0.0002):
        """
        Densify Gaussians in high-gradient regions
        
        This is a simplified version - full implementation in gaussian-splatting repo
        """
        # Check gradients
        if not hasattr(self.gaussians.means, 'grad') or self.gaussians.means.grad is None:
            return
        
        grads = self.gaussians.means.grad.norm(dim=1)
        
        # Find high-gradient Gaussians
        high_grad_mask = grads > gradient_threshold
        
        if high_grad_mask.sum() == 0:
            return
        
        # Split high-gradient Gaussians
        # (Simplified - actual implementation is more complex)
        num_new = min(high_grad_mask.sum().item(), 
                     self.max_points - len(self.gaussians.means))
        
        if num_new > 0:
            # Clone parameters for new Gaussians
            new_means = self.gaussians.means[high_grad_mask][:num_new]
            new_scales = self.gaussians.scales[high_grad_mask][:num_new] * 0.5
            new_rotations = self.gaussians.rotations[high_grad_mask][:num_new]
            new_opacities = self.gaussians.opacities[high_grad_mask][:num_new]
            new_features = self.gaussians.features[high_grad_mask][:num_new]
            
            # Append
            self.gaussians.means = nn.Parameter(
                torch.cat([self.gaussians.means, new_means])
            )
            self.gaussians.scales = nn.Parameter(
                torch.cat([self.gaussians.scales, new_scales])
            )
            self.gaussians.rotations = nn.Parameter(
                torch.cat([self.gaussians.rotations, new_rotations])
            )
            self.gaussians.opacities = nn.Parameter(
                torch.cat([self.gaussians.opacities, new_opacities])
            )
            self.gaussians.features = nn.Parameter(
                torch.cat([self.gaussians.features, new_features])
            )
    
    def prune(self, opacity_threshold: float = 0.005):
        """
        Remove low-opacity Gaussians
        """
        # Keep Gaussians with opacity > threshold
        keep_mask = self.gaussians.opacities.squeeze() > opacity_threshold
        
        self.gaussians.means = nn.Parameter(self.gaussians.means[keep_mask])
        self.gaussians.scales = nn.Parameter(self.gaussians.scales[keep_mask])
        self.gaussians.rotations = nn.Parameter(self.gaussians.rotations[keep_mask])
        self.gaussians.opacities = nn.Parameter(self.gaussians.opacities[keep_mask])
        self.gaussians.features = nn.Parameter(self.gaussians.features[keep_mask])
    
    def export_point_cloud(self, filepath: str):
        """
        Export Gaussians as point cloud (PLY format)
        """
        import open3d as o3d
        
        # Convert to numpy
        points = self.gaussians.means.detach().cpu().numpy()
        colors = self.gaussians.features.detach().cpu().numpy()
        
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Save
        o3d.io.write_point_cloud(filepath, pcd)
        print(f"Point cloud saved to {filepath}")


if __name__ == "__main__":
    print("Testing Vessel-Aware 3DGS-SLAM...")
    
    # Create camera
    camera = Camera(
        width=512,
        height=512,
        fx=500.0,
        fy=500.0,
        cx=256.0,
        cy=256.0
    )
    
    # Create SLAM system
    slam = VesselGaussianSLAM(
        camera=camera,
        initial_points=1000,
        vessel_weight=2.0
    )
    
    print(f"Initialized with {len(slam.gaussians.means)} Gaussians")
    
    # Test adding keyframe
    test_image = torch.rand(3, 512, 512)
    test_mask = torch.rand(1, 512, 512)
    
    slam.add_keyframe(0, test_image, test_mask)
    print(f"Added keyframe 0")
    print(f"Number of keyframes: {len(slam.keyframes)}")
