"""
Training Script for Vessel Segmentation (Step 2)

Train U-Net++ model for vessel segmentation with topology-preserving loss
Optimized for memory efficiency
"""
# $env:HF_ENDPOINT = "https://hf-mirror.com"    训练时记得使用
import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from tqdm import tqdm
import numpy as np
import gc

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from data.datasets import VesselSegmentationDataset
from models.segmentation.unet_plusplus import create_segmentation_model
from losses.segmentation_losses import TopologyPreservingLoss


class SegmentationTrainer:
    """Trainer for vessel segmentation"""
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        
        # 优化 1: 清理 GPU 缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        # Setup logging
        if config.logging.use_wandb:
            wandb.init(
                project=config.logging.wandb_project,
                name=config.experiment_name,
                config=OmegaConf.to_container(config, resolve=True)
            )
        
        # Create model
        self.model = create_segmentation_model(config)
        self.model = self.model.to(self.device)
        
        # 优化 2: 使用 torch.compile 加速（PyTorch 2.0+）
        if hasattr(torch, 'compile') and config.training.get('use_compile', False):
            try:
                self.model = torch.compile(self.model)
                print("Model compiled successfully")
            except Exception as e:
                print(f"Compilation failed: {e}, using regular model")
        
        # Create loss
        self.criterion = TopologyPreservingLoss(
            dice_weight=config.loss.segmentation.dice_weight,
            bce_weight=config.loss.segmentation.bce_weight,
            cldice_weight=config.loss.topology.weight if config.loss.topology.enabled else 0,
            deep_supervision=config.model.deep_supervision,
            ds_weights=config.loss.get('deep_supervision_weights', None)
        )
        
        # Create optimizer
        self.optimizer = self._create_optimizer()
        
        # Create scheduler
        self.scheduler = self._create_scheduler()
        
        # Create data loaders
        self.train_loader, self.val_loader = self._create_dataloaders()
        
        # 优化 3: 使用 GradScaler，确保正确初始化
        self.use_amp = config.training.get('use_amp', True) and torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
        
        # 优化 4: 梯度累积
        self.accumulation_steps = config.training.get('gradient_accumulation_steps', 1)
        
        # Training state
        self.current_epoch = 0
        self.best_metric = 0.0
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint.dirpath)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def _create_optimizer(self):
        """Create optimizer"""
        opt_config = self.config.training.optimizer
        
        # 优化 5: 添加参数分组，对 BatchNorm 参数不使用 weight decay
        bn_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if 'bn' in name or 'norm' in name:
                bn_params.append(param)
            else:
                other_params.append(param)
        
        param_groups = [
            {'params': other_params, 'weight_decay': opt_config.weight_decay},
            {'params': bn_params, 'weight_decay': 0.0}
        ]
        
        if opt_config.type == 'AdamW':
            return torch.optim.AdamW(
                param_groups,
                lr=opt_config.lr,
                betas=opt_config.betas
            )
        elif opt_config.type == 'Adam':
            return torch.optim.Adam(
                param_groups,
                lr=opt_config.lr
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_config.type}")
    
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        sched_config = self.config.training.scheduler
        
        if sched_config.type == 'CosineAnnealingLR':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=sched_config.T_max,
                eta_min=sched_config.eta_min
            )
        elif sched_config.type == 'ReduceLROnPlateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                patience=10,
                factor=0.5
            )
        else:
            return None
    
    def _create_dataloaders(self):
        """Create data loaders"""
        # Training dataset
        train_dataset = VesselSegmentationDataset(
            data_root=self.config.data.processed,
            split='train',
            image_size=tuple(self.config.data.image_size),
            augmentation=self.config.data.augmentation.enabled
        )
        
        # Validation dataset
        val_dataset = VesselSegmentationDataset(
            data_root=self.config.data.processed,
            split='val',
            image_size=tuple(self.config.data.image_size),
            augmentation=False
        )
        
        # Create loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory,
            drop_last=True,
            # 优化 6: 使用 persistent_workers 减少进程启动开销
            persistent_workers=True if self.config.data.num_workers > 0 else False,
            # 优化 7: 预取数据
            prefetch_factor=2 if self.config.data.num_workers > 0 else None
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory,
            persistent_workers=True if self.config.data.num_workers > 0 else False,
            prefetch_factor=2 if self.config.data.num_workers > 0 else None
        )
        
        return train_loader, val_loader
    
    def train_epoch(self):
        """Train one epoch"""
        self.model.train()
        
        # FIX: Use empty dict to dynamically collect all loss keys
        epoch_losses = {}
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            images = batch['image'].to(self.device, non_blocking=True)
            masks = batch['mask'].to(self.device, non_blocking=True)
            
            # Forward pass with mixed precision
            if self.use_amp:
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    outputs = self.model(images)
                    losses = self.criterion(outputs, masks.float())
                    # 优化 8: 梯度累积，除以累积步数
                    loss = losses['total'] / self.accumulation_steps
                
                # Backward with gradient scaling
                self.scaler.scale(loss).backward()
                
                # 优化 9: 只在累积步数到达时更新参数
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    # Gradient clipping
                    if self.config.training.gradient_clip_val > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.training.gradient_clip_val
                        )
                    
                    # Optimizer step
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)  # 优化 10: 使用 set_to_none
            else:
                outputs = self.model(images)
                losses = self.criterion(outputs, masks.float())
                loss = losses['total'] / self.accumulation_steps
                
                loss.backward()
                
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    if self.config.training.gradient_clip_val > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.training.gradient_clip_val
                        )
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
            
            # FIX: Dynamically initialize and append to epoch_losses
            for key in losses:
                if key not in epoch_losses:
                    epoch_losses[key] = []
                epoch_losses[key].append(losses[key].item())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': losses['total'].item(),
                'lr': self.optimizer.param_groups[0]['lr']
            })
            
            # Log to wandb
            if self.config.logging.use_wandb and batch_idx % self.config.logging.log_every_n_steps == 0:
                wandb.log({
                    f'train/{key}': losses[key].item()
                    for key in losses
                })
            
            # 优化 11: 定期清理 GPU 缓存
            if batch_idx % 50 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Return average losses
        return {key: np.mean(vals) for key, vals in epoch_losses.items()}
    
    @torch.no_grad()
    def validate(self):
        """Validate model"""
        self.model.eval()
        
        # FIX: Use empty dict to dynamically collect all loss keys
        val_losses = {}
        
        # Metrics
        dice_scores = []
        iou_scores = []
        
        for batch in tqdm(self.val_loader, desc='Validation'):
            images = batch['image'].to(self.device, non_blocking=True)
            masks = batch['mask'].to(self.device, non_blocking=True)
            
            # 优化 12: 验证时也使用混合精度
            if self.use_amp:
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    outputs = self.model(images)
                    losses = self.criterion(outputs, masks.float())
            else:
                outputs = self.model(images)
                losses = self.criterion(outputs, masks.float())
            
            # FIX: Dynamically initialize and append to val_losses
            for key in losses:
                if key not in val_losses:
                    val_losses[key] = []
                val_losses[key].append(losses[key].item())
            
            # Compute metrics
            pred_masks = outputs['mask'] if isinstance(outputs, dict) else outputs
            
            # 血管分割必须先过 Sigmoid 再二值化
            pred_binary = (pred_masks.sigmoid() > 0.5).float()
            
            # Dice score
            intersection = (pred_binary * masks).sum(dim=(2, 3))
            union = pred_binary.sum(dim=(2, 3)) + masks.sum(dim=(2, 3))
            dice = (2 * intersection + 1e-7) / (union + 1e-7)
            dice_scores.extend(dice.cpu().numpy())
            
            # IoU
            intersection = (pred_binary * masks).sum(dim=(2, 3))
            union = (pred_binary + masks).clamp(0, 1).sum(dim=(2, 3))
            iou = (intersection + 1e-7) / (union + 1e-7)
            iou_scores.extend(iou.cpu().numpy())
            
            # 优化 13: 及时释放不需要的张量
            del outputs, pred_masks, pred_binary
        
        # Average metrics
        metrics = {
            'val/loss': np.mean(val_losses['total']),
            'val/Dice': np.mean(dice_scores),
            'val/IoU': np.mean(iou_scores)
        }
        
        # Log to wandb
        if self.config.logging.use_wandb:
            wandb.log(metrics)
        
        return metrics
    
    def save_checkpoint(self, filename: str):
        """Save checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_metric': self.best_metric,
            'config': OmegaConf.to_container(self.config, resolve=True)
        }
        
        filepath = self.checkpoint_dir / filename
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")
    
    def train(self):
        """Main training loop"""
        for epoch in range(self.current_epoch, self.config.training.max_epochs):
            self.current_epoch = epoch
            
            # Train
            train_losses = self.train_epoch()
            print(f"Epoch {epoch}: Train Loss = {train_losses['total']:.4f}")
            
            # Validate
            val_metrics = self.validate()
            print(f"Epoch {epoch}: Val Dice = {val_metrics['val/Dice']:.4f}")
            
            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['val/Dice'])
                else:
                    self.scheduler.step()
            
            # Save best model
            metric = val_metrics[f"val/{self.config.logging.monitor_metric.split('/')[-1]}"]
            if metric > self.best_metric:
                self.best_metric = metric
                self.save_checkpoint('best_model.pth')
            
            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth')
            
            # 优化 14: 每个 epoch 后清理缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
        
        print("Training completed!")


@hydra.main(version_base=None, config_path="configs", config_name="step2_segmentation")
def main(config: DictConfig):
    """Main training function"""
    
    # Set random seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
        # 优化 15: 设置 cudnn benchmark
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
    # 优化 16: 设置内存分配器配置
    if torch.cuda.is_available():
        # 启用内存碎片整理
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Create trainer
    trainer = SegmentationTrainer(config)
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()