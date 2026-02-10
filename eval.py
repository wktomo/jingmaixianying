"""
Evaluation Script for Vessel 3D Reconstruction Pipeline

Evaluates:
1. Segmentation quality (Dice, IoU, ClDice)
2. Enhancement quality (PSNR, SSIM, Vessel Visibility)
3. 3D reconstruction quality (Depth error, ATE, Rendering quality)
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
import json
from typing import Dict, List
from skimage.morphology import skeletonize
from skimage.metrics import structural_similarity as ssim

# Add project root
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


class SegmentationEvaluator:
    """Evaluate segmentation quality"""
    
    @staticmethod
    def dice_coefficient(pred: np.ndarray, gt: np.ndarray) -> float:
        """Compute Dice coefficient"""
        pred = pred.flatten()
        gt = gt.flatten()
        
        intersection = (pred * gt).sum()
        return (2.0 * intersection + 1e-7) / (pred.sum() + gt.sum() + 1e-7)
    
    @staticmethod
    def iou_score(pred: np.ndarray, gt: np.ndarray) -> float:
        """Compute IoU"""
        pred = pred.flatten()
        gt = gt.flatten()
        
        intersection = (pred * gt).sum()
        union = (pred + gt > 0).sum()
        
        return (intersection + 1e-7) / (union + 1e-7)
    
    @staticmethod
    def cldice_score(pred: np.ndarray, gt: np.ndarray) -> float:
        """Compute Centerline Dice (ClDice)"""
        # Binarize
        pred_binary = (pred > 0.5).astype(np.uint8)
        gt_binary = (gt > 0.5).astype(np.uint8)
        
        # Skeletonize
        pred_skel = skeletonize(pred_binary)
        gt_skel = skeletonize(gt_binary)
        
        # Precision: pred skeleton matched by gt mask
        tprec = (pred_skel * gt_binary).sum() / (pred_skel.sum() + 1e-7)
        
        # Recall: gt skeleton matched by pred mask
        tsens = (gt_skel * pred_binary).sum() / (gt_skel.sum() + 1e-7)
        
        # ClDice
        cl_dice = 2 * tprec * tsens / (tprec + tsens + 1e-7)
        
        return cl_dice
    
    @staticmethod
    def evaluate_batch(pred_masks: List[np.ndarray],
                      gt_masks: List[np.ndarray]) -> Dict[str, float]:
        """Evaluate a batch of predictions"""
        dice_scores = []
        iou_scores = []
        cldice_scores = []
        
        for pred, gt in zip(pred_masks, gt_masks):
            dice_scores.append(SegmentationEvaluator.dice_coefficient(pred, gt))
            iou_scores.append(SegmentationEvaluator.iou_score(pred, gt))
            cldice_scores.append(SegmentationEvaluator.cldice_score(pred, gt))
        
        return {
            'Dice': np.mean(dice_scores),
            'IoU': np.mean(iou_scores),
            'ClDice': np.mean(cldice_scores),
            'Dice_std': np.std(dice_scores),
            'IoU_std': np.std(iou_scores),
            'ClDice_std': np.std(cldice_scores)
        }


class EnhancementEvaluator:
    """Evaluate enhancement quality"""
    
    @staticmethod
    def psnr(pred: np.ndarray, gt: np.ndarray) -> float:
        """Compute PSNR"""
        mse = np.mean((pred - gt) ** 2)
        if mse == 0:
            return 100.0
        
        max_pixel = 1.0 if pred.max() <= 1.0 else 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        
        return psnr
    
    @staticmethod
    def ssim_score(pred: np.ndarray, gt: np.ndarray) -> float:
        """Compute SSIM"""
        # Ensure grayscale
        if len(pred.shape) == 3:
            pred = cv2.cvtColor(pred, cv2.COLOR_RGB2GRAY)
        if len(gt.shape) == 3:
            gt = cv2.cvtColor(gt, cv2.COLOR_RGB2GRAY)
        
        # Compute SSIM
        score, _ = ssim(pred, gt, full=True)
        
        return score
    
    @staticmethod
    def vessel_visibility_boost(enhanced: np.ndarray,
                                noisy: np.ndarray,
                                vessel_mask: np.ndarray) -> float:
        """
        Compute vessel visibility boost
        
        Measures how much clearer vessels become after enhancement
        """
        # Get vessel regions
        vessel_pixels_enhanced = enhanced[vessel_mask > 0.5]
        vessel_pixels_noisy = noisy[vessel_mask > 0.5]
        
        # Get background regions
        bg_pixels_enhanced = enhanced[vessel_mask <= 0.5]
        bg_pixels_noisy = noisy[vessel_mask <= 0.5]
        
        # Compute contrast (vessel vs background)
        contrast_enhanced = np.abs(vessel_pixels_enhanced.mean() - bg_pixels_enhanced.mean())
        contrast_noisy = np.abs(vessel_pixels_noisy.mean() - bg_pixels_noisy.mean())
        
        # Visibility boost
        boost = (contrast_enhanced - contrast_noisy) / (contrast_noisy + 1e-7)
        
        return boost
    
    @staticmethod
    def evaluate_batch(enhanced_images: List[np.ndarray],
                      gt_images: List[np.ndarray],
                      noisy_images: List[np.ndarray],
                      vessel_masks: List[np.ndarray]) -> Dict[str, float]:
        """Evaluate a batch of enhanced images"""
        psnr_scores = []
        ssim_scores = []
        visibility_boosts = []
        
        for enhanced, gt, noisy, mask in zip(enhanced_images, gt_images, 
                                              noisy_images, vessel_masks):
            psnr_scores.append(EnhancementEvaluator.psnr(enhanced, gt))
            ssim_scores.append(EnhancementEvaluator.ssim_score(enhanced, gt))
            visibility_boosts.append(
                EnhancementEvaluator.vessel_visibility_boost(enhanced, noisy, mask)
            )
        
        return {
            'PSNR': np.mean(psnr_scores),
            'SSIM': np.mean(ssim_scores),
            'VesselVisibilityBoost': np.mean(visibility_boosts),
            'PSNR_std': np.std(psnr_scores),
            'SSIM_std': np.std(ssim_scores)
        }


class ReconstructionEvaluator:
    """Evaluate 3D reconstruction quality"""
    
    @staticmethod
    def depth_error(pred_depth: np.ndarray, gt_depth: np.ndarray, mask: np.ndarray = None) -> Dict:
        """Compute depth errors"""
        if mask is not None:
            pred_depth = pred_depth[mask]
            gt_depth = gt_depth[mask]
        
        # Absolute error
        abs_error = np.abs(pred_depth - gt_depth)
        
        # Relative error
        rel_error = abs_error / (gt_depth + 1e-7)
        
        return {
            'MAE': np.mean(abs_error),
            'RMSE': np.sqrt(np.mean(abs_error ** 2)),
            'MedAE': np.median(abs_error),
            'MeanRelError': np.mean(rel_error)
        }
    
    @staticmethod
    def trajectory_error(pred_poses: np.ndarray, gt_poses: np.ndarray) -> Dict:
        """
        Compute trajectory errors (ATE)
        
        Args:
            pred_poses: (N, 4, 4) predicted poses
            gt_poses: (N, 4, 4) ground truth poses
        """
        # Extract translations
        pred_trans = pred_poses[:, :3, 3]
        gt_trans = gt_poses[:, :3, 3]
        
        # Absolute Trajectory Error (ATE)
        ate = np.linalg.norm(pred_trans - gt_trans, axis=1)
        
        # Rotation errors
        rotation_errors = []
        for i in range(len(pred_poses)):
            R_pred = pred_poses[i, :3, :3]
            R_gt = gt_poses[i, :3, :3]
            
            # Relative rotation
            R_rel = R_gt.T @ R_pred
            
            # Angle
            trace = np.trace(R_rel)
            angle = np.arccos((trace - 1) / 2)
            rotation_errors.append(np.rad2deg(angle))
        
        return {
            'ATE_mean': np.mean(ate),
            'ATE_median': np.median(ate),
            'ATE_std': np.std(ate),
            'RotError_mean': np.mean(rotation_errors),
            'RotError_median': np.median(rotation_errors)
        }
    
    @staticmethod
    def rendering_quality(rendered_images: List[np.ndarray],
                         gt_images: List[np.ndarray]) -> Dict:
        """Evaluate rendering quality"""
        psnr_scores = []
        ssim_scores = []
        
        for rendered, gt in zip(rendered_images, gt_images):
            psnr_scores.append(EnhancementEvaluator.psnr(rendered, gt))
            ssim_scores.append(EnhancementEvaluator.ssim_score(rendered, gt))
        
        return {
            'Rendering_PSNR': np.mean(psnr_scores),
            'Rendering_SSIM': np.mean(ssim_scores)
        }


def evaluate_pipeline(pred_dir: Path,
                     gt_dir: Path,
                     metrics: List[str] = ['all']) -> Dict:
    """
    Evaluate complete pipeline
    
    Args:
        pred_dir: Directory with predictions
        gt_dir: Directory with ground truth
        metrics: List of metrics to compute ['segmentation', 'enhancement', 'reconstruction', 'all']
    """
    results = {}
    
    # Evaluate segmentation
    if 'segmentation' in metrics or 'all' in metrics:
        print("Evaluating segmentation...")
        
        pred_masks = sorted((pred_dir / 'masks').glob('*.png'))
        gt_masks = sorted((gt_dir / 'masks').glob('*.png'))
        
        pred_masks_np = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) / 255.0 
                         for p in pred_masks]
        gt_masks_np = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) / 255.0 
                       for p in gt_masks]
        
        seg_results = SegmentationEvaluator.evaluate_batch(pred_masks_np, gt_masks_np)
        results['segmentation'] = seg_results
        
        print(f"  Dice: {seg_results['Dice']:.4f} ± {seg_results['Dice_std']:.4f}")
        print(f"  IoU: {seg_results['IoU']:.4f} ± {seg_results['IoU_std']:.4f}")
        print(f"  ClDice: {seg_results['ClDice']:.4f} ± {seg_results['ClDice_std']:.4f}")
    
    # Evaluate enhancement
    if 'enhancement' in metrics or 'all' in metrics:
        print("\nEvaluating enhancement...")
        
        enhanced_images = sorted((pred_dir / 'enhanced').glob('*.png'))
        gt_images = sorted((gt_dir / 'clean').glob('*.png'))
        noisy_images = sorted((gt_dir / 'noisy').glob('*.png'))
        vessel_masks = sorted((gt_dir / 'masks').glob('*.png'))
        
        enhanced_np = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) / 255.0 
                      for p in enhanced_images]
        gt_np = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) / 255.0 
                for p in gt_images]
        noisy_np = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) / 255.0 
                   for p in noisy_images]
        masks_np = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) / 255.0 
                   for p in vessel_masks]
        
        enh_results = EnhancementEvaluator.evaluate_batch(
            enhanced_np, gt_np, noisy_np, masks_np
        )
        results['enhancement'] = enh_results
        
        print(f"  PSNR: {enh_results['PSNR']:.2f} ± {enh_results['PSNR_std']:.2f} dB")
        print(f"  SSIM: {enh_results['SSIM']:.4f} ± {enh_results['SSIM_std']:.4f}")
        print(f"  Visibility Boost: {enh_results['VesselVisibilityBoost']:.2%}")
    
    # Save results
    results_file = pred_dir / 'evaluation_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate Vessel 3D Reconstruction Pipeline')
    parser.add_argument('--pred_dir', type=str, required=True,
                       help='Directory with predictions')
    parser.add_argument('--gt_dir', type=str, required=True,
                       help='Directory with ground truth')
    parser.add_argument('--metrics', nargs='+', default=['all'],
                       choices=['segmentation', 'enhancement', 'reconstruction', 'all'],
                       help='Metrics to compute')
    
    args = parser.parse_args()
    
    # Run evaluation
    results = evaluate_pipeline(
        pred_dir=Path(args.pred_dir),
        gt_dir=Path(args.gt_dir),
        metrics=args.metrics
    )


if __name__ == "__main__":
    main()
