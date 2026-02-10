# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# import hydra
# from omegaconf import DictConfig, OmegaConf
# import wandb
# from tqdm import tqdm
# from pathlib import Path
# import numpy as np

# # 导入你的模型和损失函数组件
# from data.datasets import VesselEnhancementDataset
# from models.enhancement.vessel_controlnet import VesselDiffusionEnhancer
# from losses.segmentation_losses import VesselWeightedPhotometricLoss

# class EnhancementTrainer:
#     def __init__(self, config: DictConfig):
#         self.config = config
#         self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        
#         # 1. 初始化增强模型 (内部加载 ControlNet 参数)
#         self.model = VesselDiffusionEnhancer(config).to(self.device)
        
#         # 2. 血管加权损失函数：对血管区域施加更高权重
#         self.criterion = VesselWeightedPhotometricLoss(
#             # vessel_weight=config.loss.vessel_weight, # 建议设为 2.0
#             # ssim_weight=0.5
#             vessel_weight=2.0,  
#             ssim_weight=0.5
#         )
        
#         # 3. 优化器：按照文档建议，优先训练 ControlNet 部分以保证稳定性
#         self.optimizer = torch.optim.AdamW(
#             self.model.controlnet.parameters(), 
#             lr=config.training.optimizer.lr
#         )
        
#         # 4. 加载数据集：读取你生成的 3通道中间产物
#         self.train_loader = self._create_dataloader()
        
#         # 5. 显存优化：使用混合精度训练 (AMP)
#         self.scaler = torch.amp.GradScaler('cuda') if config.training.use_amp else None

#     def _create_dataloader(self):
#         # 使用你之前定义的 Dataset 加载器
#         my_data_path = r"D:/babba/xxx/jingmaixianying/vessel_3d_recon/data/processed/"
#         dataset = VesselEnhancementDataset(
#             data_root=my_data_path,
#             split='train',
#             image_size=tuple(self.config.data.image_size)
#         )
#         return DataLoader(
#             dataset, 
#             batch_size=self.config.training.batch_size,
#             shuffle=True, 
#             num_workers=4,
#             pin_memory=True
#         )

#     def train_epoch(self, epoch):
#         self.model.train()
#         pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
#         for batch in pbar:
#             # 获取 3通道中间产物 (condition) 和 原始图 (image)
#             # low_res = batch['images'].to(self.device)
#             # target = batch['images'].to(self.device)
#             # condition = batch['condition'].to(self.device) # 掩码+骨架+边缘
#             # vessel_mask = batch['mask'].to(self.device)

#             low_res = batch['noisy'].to(self.device)   # 输入：噪声图
#             target = batch['clean'].to(self.device)    # 目标：高清图
#             vessel_mask = batch['mask'].to(self.device) # 掩码：计算 Loss 用
            
#             # 特别注意: 你的 Dataset 并没有生成专门的 3 通道 condition。
#             # 这里如果模型需要 condition，通常直接传入 mask。
#             # 如果模型报错通道不匹配 (1 vs 3)，请参考下方的“维度处理”部分。
#             condition = batch['mask'].to(self.device)
#             with torch.amp.autocast('cuda', enabled=self.scaler is not None):
#                 # 模型根据条件引导生成增强图像
#                 enhanced = self.model(low_res, condition)
                
#                 # 计算 Loss
#                 loss = self.criterion(enhanced, target, vessel_mask)
#             # 反向传播与梯度更新
#             self.optimizer.zero_grad(set_to_none=True)
#             if self.scaler:
#                 self.scaler.scale(loss).backward()
#                 self.scaler.step(self.optimizer)
#                 self.scaler.update()
#             else:
#                 loss.backward()
#                 self.optimizer.step()

#             pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
#             if self.config.logging.use_wandb:
#                 wandb.log({"train/loss": loss.item()})

#     def save_checkpoint(self, epoch):
#         save_dir = Path(self.config.checkpoint.dirpath)
#         save_dir.mkdir(parents=True, exist_ok=True)
#         path = save_dir / f"enhancement_epoch_{epoch}.pth"
#         torch.save({
#             'epoch': epoch,
#             'model_state_dict': self.model.state_dict(),
#             'config': OmegaConf.to_container(self.config)
#         }, path)

# @hydra.main(version_base=None, config_path="configs", config_name="step3_diffusion")
# def main(config: DictConfig):
#     # 初始化日志系统
#     if config.logging.use_wandb:
#         wandb.init(project="Vessel-Enhancement-Step3", config=OmegaConf.to_container(config))
    
#     trainer = EnhancementTrainer(config)
    
#     # 开始训练循环
#     for epoch in range(config.training.max_epochs):
#         trainer.train_epoch(epoch)
        
#         # 定期保存模型
#         if (epoch + 1) % 10 == 0:
#             trainer.save_checkpoint(epoch)

# if __name__ == "__main__":
#     main()
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from tqdm import tqdm
from pathlib import Path
import numpy as np

# 导入你的模型和损失函数组件
from data.datasets import VesselEnhancementDataset
from models.enhancement.vessel_controlnet import VesselDiffusionEnhancer
from losses.segmentation_losses import VesselWeightedPhotometricLoss

class EnhancementTrainer:
    def __init__(self, config: DictConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        
        # 1. 初始化增强模型 (内部加载 ControlNet 参数)
        self.model = VesselDiffusionEnhancer(config).to(self.device)
        
        # 2. 血管加权损失函数：对血管区域施加更高权重
        self.criterion = VesselWeightedPhotometricLoss(
            # vessel_weight=config.loss.vessel_weight, # 建议设为 2.0
            # ssim_weight=0.5
            vessel_weight=2.0,  
            ssim_weight=0.5
        )
        
        # 3. 优化器：按照文档建议，优先训练 ControlNet 部分以保证稳定性
        self.optimizer = torch.optim.AdamW(
            self.model.controlnet.parameters(), 
            lr=config.training.optimizer.lr
        )
        
        # 4. 加载数据集：读取你生成的 3通道中间产物
        self.train_loader = self._create_dataloader()
        
        # 5. 显存优化：使用混合精度训练 (AMP)
        self.scaler = torch.amp.GradScaler('cuda') if config.training.use_amp else None

    def _create_dataloader(self):
        # 使用你之前定义的 Dataset 加载器
        my_data_path = r"D:/babba/xxx/jingmaixianying/vessel_3d_recon/data/processed/"
        dataset = VesselEnhancementDataset(
            data_root=my_data_path,
            split='train',
            image_size=tuple(self.config.data.image_size)
        )
        return DataLoader(
            dataset, 
            batch_size=self.config.training.batch_size,
            shuffle=True, 
            num_workers=4,
            pin_memory=True
        )

    @staticmethod
    def _to_3channel(img: torch.Tensor) -> torch.Tensor:
        """将单通道图像转换为3通道（修复VAE输入要求）"""
        if img.shape[1] == 1:
            return img.repeat(1, 3, 1, 1)
        return img

    def train_epoch(self, epoch):
        self.model.train()
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch in pbar:
            # 获取 3通道中间产物 (condition) 和 原始图 (image)
            # low_res = batch['images'].to(self.device)
            # target = batch['images'].to(self.device)
            # condition = batch['condition'].to(self.device) # 掩码+骨架+边缘
            # vessel_mask = batch['mask'].to(self.device)

            low_res = batch['noisy'].to(self.device)   # 输入：噪声图
            target = batch['clean'].to(self.device)    # 目标：高清图
            vessel_mask = batch['mask'].to(self.device) # 掩码：计算 Loss 用
            
            # 特别注意: 你的 Dataset 并没有生成专门的 3 通道 condition。
            # 这里如果模型需要 condition，通常直接传入 mask。
            # 如果模型报错通道不匹配 (1 vs 3)，请参考下方的“维度处理”部分。
            condition = batch['mask'].to(self.device)
            
            # 🔧 修复：将单通道图像转换为3通道以满足VAE输入要求
            low_res = self._to_3channel(low_res)
            target = self._to_3channel(target)
            condition = self._to_3channel(condition)

            with torch.amp.autocast('cuda', enabled=self.scaler is not None):
                # 模型根据条件引导生成增强图像
                enhanced = self.model(low_res, condition)
                
                # 计算 Loss
                loss = self.criterion(enhanced, target, vessel_mask)
            # 反向传播与梯度更新
            self.optimizer.zero_grad(set_to_none=True)
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            if self.config.logging.use_wandb:
                wandb.log({"train/loss": loss.item()})

    def save_checkpoint(self, epoch):
        save_dir = Path(self.config.checkpoint.dirpath)
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / f"enhancement_epoch_{epoch}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'config': OmegaConf.to_container(self.config)
        }, path)

@hydra.main(version_base=None, config_path="configs", config_name="step3_diffusion")
def main(config: DictConfig):
    # 初始化日志系统
    if config.logging.use_wandb:
        wandb.init(project="Vessel-Enhancement-Step3", config=OmegaConf.to_container(config))
    
    trainer = EnhancementTrainer(config)
    
    # 开始训练循环
    for epoch in range(config.training.max_epochs):
        trainer.train_epoch(epoch)
        
        # 定期保存模型
        if (epoch + 1) % 10 == 0:
            trainer.save_checkpoint(epoch)

if __name__ == "__main__":
    main()