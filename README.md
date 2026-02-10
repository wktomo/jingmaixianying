# Vessel-Aware 3D Reconstruction Pipeline

A complete pipeline for vessel image enhancement and 3D reconstruction using ControlNet-guided diffusion and Gaussian Splatting SLAM.

## System Overview

```
Raw Images → Preprocessing → U-Net Segmentation → ControlNet Enhancement → 3DGS-SLAM → Closed-Loop Optimization
```

### Five-Step Pipeline:

1. **Preprocessing**: Denoising, CLAHE normalization, keyframe selection
2. **Coarse Segmentation**: U-Net++ for vessel probability maps + skeleton extraction
3. **Vessel-Aware Diffusion Enhancement**: ControlNet with vessel mask conditioning
4. **3DGS-SLAM Reconstruction**: Vessel-weighted photometric SLAM
5. **Closed-Loop Optimization**: End-to-end differentiable refinement

## Key Innovations

- **Vessel-conditioned ControlNet**: Multi-scale vessel mask injection into diffusion models
- **Weighted Gaussian Splatting**: Vessel region-aware photometric loss
- **End-to-end Optimization**: Differentiable pipeline from enhancement to 3D reconstruction

## Installation

```bash
# Create conda environment
conda create -n vessel3d python=3.10
conda activate vessel3d

# Install PyTorch (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt

# Install Gaussian Splatting submodule (if using)
cd third_party
git clone https://github.com/graphdeco-inria/gaussian-splatting.git --recursive
cd gaussian-splatting
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
```

## Data Preparation

### Supported Datasets:
- **OCTA-500**: 3D OCTA volumetric data
- **DRIVE**: 2D retinal fundus images
- **STARE**: Retinal vessel segmentation
- **Custom**: Video sequences with camera calibration

### Data Structure:
```
data/
├── raw/
│   ├── OCTA-500/
│   │   ├── OCTA_3M/
│   │   └── OCTA_6M/
│   └── DRIVE/
├── processed/
└── splits/
    ├── train.txt
    ├── val.txt
    └── test.txt
```

### Preprocessing:
```bash
python scripts/preprocess_data.py --dataset OCTA-500 --output_dir data/processed/octa500
```

## Training

### Stage 1: Train Segmentation Model
```bash
python train_step2_segmentation.py \
    --config configs/step2_segmentation.yaml \
    experiment_name=unet_vessel
```

### Stage 2: Train ControlNet Enhancement
```bash
python train_step3_diffusion.py \
    --config configs/step3_diffusion.yaml \
    segmentation_ckpt=checkpoints/best_segmentation.pth
```

### Stage 3: End-to-End Fine-tuning
```bash
python train_end2end.py \
    --config configs/end2end.yaml \
    enhancement_ckpt=checkpoints/best_enhancement.pth
```

## Inference

### Run Full Pipeline:
```bash
python inference.py \
    --input_video data/test_sequence.mp4 \
    --output_dir outputs/reconstruction \
    --config configs/inference.yaml
```

### Outputs:
- Enhanced video frames
- 3D vessel reconstruction (PLY/OBJ format)
- Camera trajectory
- Evaluation metrics

## Evaluation

```bash
python eval.py \
    --gt_masks data/test/masks \
    --pred_dir outputs/reconstruction \
    --metrics all
```

### Metrics:
- **Segmentation**: Dice, IoU, ClDice
- **Enhancement**: PSNR, SSIM, Vessel Visibility Boost
- **3D Reconstruction**: Depth Error, ATE, Rendering PSNR

## Project Structure

```
vessel_3d_recon/
├── configs/              # Hydra configuration files
├── data/                 # Data loading and preprocessing
├── models/               # Core models
│   ├── segmentation/     # U-Net variants
│   ├── enhancement/      # ControlNet + Diffusion
│   └── reconstruction/   # 3DGS-SLAM
├── losses/               # Custom loss functions
├── utils/                # Utilities
├── scripts/              # Helper scripts
├── train_*.py            # Training scripts
└── eval_*.py             # Evaluation scripts
```

## Hardware Requirements

- **Minimum**: RTX 3090 (24GB VRAM)
- **Recommended**: RTX 4090 or A100
- **Multi-GPU**: Supported via DDP

Memory optimization:
- Gradient checkpointing
- Mixed precision (FP16)
- Gradient accumulation

## Citation

```bibtex
@article{vessel3d2024,
  title={Vessel-Aware 3D Reconstruction via Diffusion-Guided Gaussian Splatting},
  author={Your Name},
  journal={IEEE Transactions on Medical Imaging},
  year={2024}
}
```

## License

MIT License

## Acknowledgments

- Stable Diffusion by Stability AI
- 3D Gaussian Splatting by GRAPHDECO-INRIA
- ControlNet by Lvmin Zhang
