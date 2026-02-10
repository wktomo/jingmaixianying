"""
Utility Functions for Vessel 3D Reconstruction
"""

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
from pathlib import Path


def normalize_image(image: np.ndarray, 
                   target_range: Tuple[float, float] = (0, 1)) -> np.ndarray:
    """
    Normalize image to target range
    
    Args:
        image: Input image
        target_range: Target (min, max) range
        
    Returns:
        Normalized image
    """
    min_val, max_val = target_range
    img_min, img_max = image.min(), image.max()
    
    if img_max > img_min:
        normalized = (image - img_min) / (img_max - img_min)
        normalized = normalized * (max_val - min_val) + min_val
    else:
        normalized = np.full_like(image, min_val)
    
    return normalized


def visualize_segmentation(image: np.ndarray,
                           mask: np.ndarray,
                           skeleton: Optional[np.ndarray] = None,
                           save_path: Optional[str] = None):
    """
    Visualize segmentation results
    
    Args:
        image: Original image (H, W)
        mask: Segmentation mask (H, W)
        skeleton: Vessel skeleton (H, W) [optional]
        save_path: Path to save visualization
    """
    # Normalize for display
    image_vis = normalize_image(image, (0, 255)).astype(np.uint8)
    mask_vis = (mask * 255).astype(np.uint8)
    
    # Create overlay
    overlay = cv2.cvtColor(image_vis, cv2.COLOR_GRAY2BGR)
    mask_colored = cv2.applyColorMap(mask_vis, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(overlay, 0.6, mask_colored, 0.4, 0)
    
    # Add skeleton if provided
    if skeleton is not None:
        skeleton_vis = (skeleton * 255).astype(np.uint8)
        skeleton_pts = np.argwhere(skeleton_vis > 127)
        for pt in skeleton_pts:
            cv2.circle(overlay, (pt[1], pt[0]), 1, (0, 255, 0), -1)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image_vis, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(mask_vis, cmap='gray')
    axes[1].set_title('Vessel Mask')
    axes[1].axis('off')
    
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_enhancement(original: np.ndarray,
                         enhanced: np.ndarray,
                         mask: np.ndarray,
                         save_path: Optional[str] = None):
    """
    Visualize enhancement results
    
    Args:
        original: Original noisy image (H, W)
        enhanced: Enhanced image (H, W)
        mask: Vessel mask (H, W)
        save_path: Path to save visualization
    """
    # Normalize
    orig_vis = normalize_image(original, (0, 255)).astype(np.uint8)
    enh_vis = normalize_image(enhanced, (0, 255)).astype(np.uint8)
    mask_vis = (mask * 255).astype(np.uint8)
    
    # Compute difference
    diff = np.abs(enhanced - original)
    diff_vis = normalize_image(diff, (0, 255)).astype(np.uint8)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    axes[0, 0].imshow(orig_vis, cmap='gray')
    axes[0, 0].set_title('Original (Noisy)')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(enh_vis, cmap='gray')
    axes[0, 1].set_title('Enhanced')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(mask_vis, cmap='jet')
    axes[1, 0].set_title('Vessel Mask')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(diff_vis, cmap='hot')
    axes[1, 1].set_title('Enhancement Difference')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def visualize_trajectory(poses: np.ndarray,
                        save_path: Optional[str] = None):
    """
    Visualize camera trajectory in 3D
    
    Args:
        poses: Camera poses (N, 4, 4)
        save_path: Path to save visualization
    """
    # Extract positions
    positions = poses[:, :3, 3]
    
    # Create 3D plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot trajectory
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
           'b-', linewidth=2, label='Trajectory')
    
    # Mark start and end
    ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2],
              c='g', s=100, marker='o', label='Start')
    ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2],
              c='r', s=100, marker='*', label='End')
    
    # Plot camera orientations
    for i in range(0, len(poses), max(1, len(poses) // 10)):
        R = poses[i, :3, :3]
        t = poses[i, :3, 3]
        
        # Camera forward direction (negative Z)
        forward = R @ np.array([0, 0, -1]) * 0.5
        ax.quiver(t[0], t[1], t[2], 
                 forward[0], forward[1], forward[2],
                 color='r', alpha=0.6)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Camera Trajectory')
    ax.legend()
    
    # Equal aspect ratio
    max_range = np.array([
        positions[:, 0].max() - positions[:, 0].min(),
        positions[:, 1].max() - positions[:, 1].min(),
        positions[:, 2].max() - positions[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (positions[:, 0].max() + positions[:, 0].min()) * 0.5
    mid_y = (positions[:, 1].max() + positions[:, 1].min()) * 0.5
    mid_z = (positions[:, 2].max() + positions[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def save_metrics_plot(metrics: dict,
                     save_path: str,
                     title: str = 'Training Metrics'):
    """
    Plot and save training metrics
    
    Args:
        metrics: Dict of metric lists {name: [values]}
        save_path: Path to save plot
        title: Plot title
    """
    num_metrics = len(metrics)
    fig, axes = plt.subplots(num_metrics, 1, figsize=(10, 4 * num_metrics))
    
    if num_metrics == 1:
        axes = [axes]
    
    for ax, (name, values) in zip(axes, metrics.items()):
        ax.plot(values, linewidth=2)
        ax.set_xlabel('Epoch' if len(values) < 1000 else 'Iteration')
        ax.set_ylabel(name)
        ax.set_title(f'{title}: {name}')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def compute_gradient_magnitude(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute gradient magnitude for visualization
    
    Args:
        tensor: Input tensor (B, C, H, W)
        
    Returns:
        Gradient magnitude (B, 1, H, W)
    """
    # Sobel filters
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                           dtype=tensor.dtype, device=tensor.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                           dtype=tensor.dtype, device=tensor.device).view(1, 1, 3, 3)
    
    # Compute gradients for each channel
    grad_x = torch.conv2d(tensor, sobel_x.repeat(tensor.size(1), 1, 1, 1), 
                         padding=1, groups=tensor.size(1))
    grad_y = torch.conv2d(tensor, sobel_y.repeat(tensor.size(1), 1, 1, 1), 
                         padding=1, groups=tensor.size(1))
    
    # Magnitude
    magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)
    
    # Average over channels
    magnitude = magnitude.mean(dim=1, keepdim=True)
    
    return magnitude


def load_checkpoint(checkpoint_path: str, 
                   model: torch.nn.Module,
                   optimizer: Optional[torch.optim.Optimizer] = None,
                   device: str = 'cuda') -> dict:
    """
    Load model checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint
        model: Model to load weights into
        optimizer: Optimizer to load state (optional)
        device: Device to load to
        
    Returns:
        Dict with epoch, metrics, etc.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer if provided
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Loaded checkpoint from {checkpoint_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Best metric: {checkpoint.get('best_metric', 'N/A')}")
    
    return checkpoint


def create_video_from_images(image_dir: Path,
                            output_path: str,
                            fps: int = 30,
                            pattern: str = '*.png'):
    """
    Create video from image sequence
    
    Args:
        image_dir: Directory containing images
        output_path: Output video path
        fps: Frames per second
        pattern: File pattern to match
    """
    # Get sorted image files
    image_files = sorted(image_dir.glob(pattern))
    
    if len(image_files) == 0:
        print(f"No images found in {image_dir} with pattern {pattern}")
        return
    
    # Read first image to get dimensions
    first_img = cv2.imread(str(image_files[0]))
    h, w = first_img.shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    # Write frames
    for img_path in image_files:
        img = cv2.imread(str(img_path))
        out.write(img)
    
    out.release()
    print(f"Video created: {output_path}")
    print(f"  Frames: {len(image_files)}")
    print(f"  Resolution: {w}x{h}")
    print(f"  FPS: {fps}")


if __name__ == "__main__":
    # Test utility functions
    print("Testing utility functions...")
    
    # Create dummy data
    image = np.random.rand(256, 256)
    mask = (np.random.rand(256, 256) > 0.7).astype(np.float32)
    skeleton = (mask > 0) & (np.random.rand(256, 256) > 0.9)
    
    # Test visualization
    visualize_segmentation(image, mask, skeleton.astype(float), 'test_seg.png')
    print("Segmentation visualization saved")
    
    # Test trajectory visualization
    poses = np.array([np.eye(4) for _ in range(20)])
    poses[:, 0, 3] = np.linspace(0, 10, 20)
    poses[:, 1, 3] = np.sin(np.linspace(0, 4*np.pi, 20))
    poses[:, 2, 3] = np.linspace(0, 5, 20)
    
    visualize_trajectory(poses, 'test_trajectory.png')
    print("Trajectory visualization saved")
