# Quick Start Guide

This guide will help you get started with the Vessel 3D Reconstruction pipeline quickly.

## Prerequisites

- Python 3.10+
- CUDA 11.8+ (for GPU support)
- 24GB+ GPU memory (RTX 3090/4090 or better)

## Installation

### 1. Create Environment

```bash
conda create -n vessel3d python=3.10
conda activate vessel3d
```

### 2. Install PyTorch

```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio
```

### 3. Install Dependencies

```bash
cd vessel_3d_recon
pip install -r requirements.txt
```

### 4. (Optional) Install 3D Gaussian Splatting

For full 3DGS functionality, install the CUDA kernels:

```bash
git clone https://github.com/graphdeco-inria/gaussian-splatting.git --recursive
cd gaussian-splatting
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
```

## Data Preparation

### Option 1: Use OCTA-500 Dataset

1. Download OCTA-500 from: [https://ieee-dataport.org/open-access/octa-500](https://ieee-dataport.org/open-access/octa-500)

2. Extract and organize:
```bash
python scripts/prepare_octa500.py \
    --input_dir /path/to/OCTA-500 \
    --output_dir data/processed/octa500
```

### Option 2: Use Your Own Data

Organize your data as follows:

```
data/custom/
├── images/
│   ├── train/
│   │   ├── image_001.png
│   │   └── ...
│   ├── val/
│   └── test/
└── masks/  (optional, for training)
    ├── train/
    ├── val/
    └── test/
```

## Training

### Stage 1: Train Segmentation Model

```bash
python train_step2_segmentation.py \
    data.data_root=data/processed/octa500 \
    training.batch_size=8 \
    training.max_epochs=200
```

**Expected time**: ~4-6 hours on RTX 4090

**Output**: `checkpoints/step2_segmentation/best_model.pth`

### Stage 2: Train Enhancement Model (Optional)

```bash
python train_step3_diffusion.py \
    data.data_root=data/processed/octa500 \
    segmentation.checkpoint=checkpoints/step2_segmentation/best_model.pth \
    training.batch_size=4 \
    training.max_epochs=100
```

**Expected time**: ~12-16 hours on RTX 4090 (diffusion models are slow!)

**Output**: `checkpoints/step3_enhancement/best_model.pth`

### Memory Optimization Tips

If you run out of memory:

1. Reduce batch size:
```bash
python train_step2_segmentation.py training.batch_size=4
```

2. Enable gradient accumulation:
```bash
python train_step2_segmentation.py \
    training.batch_size=4 \
    training.accumulate_grad_batches=8
```

3. Enable gradient checkpointing (for diffusion):
```bash
python train_step3_diffusion.py \
    memory.gradient_checkpointing=true
```

## Inference

### Run Complete Pipeline

```bash
python inference.py \
    --input data/test_video.mp4 \
    --output outputs/test_results \
    --seg_checkpoint checkpoints/step2_segmentation/best_model.pth \
    --max_frames 100
```

### With Enhancement

```bash
python inference.py \
    --input data/test_video.mp4 \
    --output outputs/test_results \
    --seg_checkpoint checkpoints/step2_segmentation/best_model.pth \
    --enh_checkpoint checkpoints/step3_enhancement/best_model.pth
```

### Output Structure

```
outputs/test_results/
├── frames/              # Processed frames
├── masks/               # Segmentation masks
├── enhanced/            # Enhanced images
├── visualization.mp4    # Side-by-side comparison
├── reconstruction.ply   # 3D point cloud
└── trajectory.npy       # Camera poses
```

## Evaluation

### Evaluate Segmentation

```bash
python eval.py \
    --pred_dir outputs/test_results \
    --gt_dir data/test/ground_truth \
    --metrics segmentation
```

### Evaluate Enhancement

```bash
python eval.py \
    --pred_dir outputs/test_results \
    --gt_dir data/test/ground_truth \
    --metrics enhancement
```

### View Results

Results are saved as JSON:

```bash
cat outputs/test_results/evaluation_results.json
```

## Visualization

### View Point Cloud

```bash
# Using Open3D
python -c "import open3d as o3d; pcd = o3d.io.read_point_cloud('outputs/test_results/reconstruction.ply'); o3d.visualization.draw_geometries([pcd])"
```

### View Trajectory

```bash
python -c "
import numpy as np
from utils.visualization import visualize_trajectory
poses = np.load('outputs/test_results/trajectory.npy')
visualize_trajectory(poses, 'trajectory_plot.png')
"
```

## Common Issues

### Issue: CUDA Out of Memory

**Solution 1**: Reduce batch size
```bash
python train_step2_segmentation.py training.batch_size=2
```

**Solution 2**: Use gradient checkpointing
```yaml
# In config file
memory:
  gradient_checkpointing: true
```

### Issue: Slow Training

**Solution 1**: Enable mixed precision (FP16)
```bash
python train_step2_segmentation.py training.use_amp=true
```

**Solution 2**: Use multiple GPUs
```bash
CUDA_VISIBLE_DEVICES=0,1 python train_step2_segmentation.py num_gpus=2
```

### Issue: Poor Segmentation Quality

**Solutions**:
1. Train longer: `training.max_epochs=300`
2. Increase model capacity: `model.features=[64,128,256,512,1024]`
3. Add more augmentation
4. Check your data quality

### Issue: Enhancement Model Too Slow

For faster inference:
1. Reduce DDIM steps: `model.num_inference_steps=10`
2. Use smaller base model: `model.base_model=runwayml/stable-diffusion-v1-5`

## Next Steps

- **Fine-tune on your data**: Adapt the pre-trained models to your specific dataset
- **Experiment with hyperparameters**: Adjust loss weights, learning rates, etc.
- **Try different architectures**: Swap U-Net++ for nnU-Net, try different diffusion models
- **Add temporal consistency**: Implement video-specific enhancements
- **Improve 3D reconstruction**: Integrate full 3DGS CUDA kernels for better quality

## Resources

- [Full Documentation](README.md)
- [Model Architecture Details](docs/architecture.md)
- [Training Tips](docs/training_tips.md)
- [API Reference](docs/api.md)

## Support

For questions and issues:
1. Check the [FAQ](docs/faq.md)
2. Open an issue on GitHub
3. Contact the authors

## Citation

If you use this code for research, please cite:

```bibtex
@article{vessel3d2024,
  title={Vessel-Aware 3D Reconstruction via Diffusion-Guided Gaussian Splatting},
  author={Your Name},
  journal={IEEE Transactions on Medical Imaging},
  year={2024}
}
```
