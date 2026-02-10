
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL
from diffusers.models import UNet2DConditionModel
import numpy as np


class VesselConditionEncoder(nn.Module):
    """
    Encode vessel masks and skeletons into conditioning signals
    
    Takes multiple vessel-related inputs and produces multi-scale features
    """
    
    def __init__(self,
                 conditioning_channels: int = 3,
                 output_channels: int = 320,
                 num_scales: int = 3):
        """
        Args:
            conditioning_channels: Number of input channels (mask + skeleton + edge)
            output_channels: Output feature channels
            num_scales: Number of scale levels for multi-scale injection
        """
        super().__init__()
        
        self.num_scales = num_scales
        
        # Initial convolution
        self.init_conv = nn.Sequential(
            nn.Conv2d(conditioning_channels, 64, 7, padding=3),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
        )
        
        # Multi-scale encoder
        self.scale_encoders = nn.ModuleList()
        in_ch = 64
        out_ch = 64
        
        for i in range(num_scales):
            out_ch = min(output_channels, 64 * (2 ** i))
            
            self.scale_encoders.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.GroupNorm(min(32, out_ch // 4), out_ch),
                nn.SiLU(),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.GroupNorm(min(32, out_ch // 4), out_ch),
                nn.SiLU(),
            ))
            
            in_ch = out_ch
        
        # Downsampling
        self.downsample = nn.MaxPool2d(2)
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: Conditioning input (B, C, H, W)
                Channel 0: Vessel probability mask
                Channel 1: Vessel skeleton
                Channel 2: Vessel edges
                
        Returns:
            List of multi-scale features
        """
        features = []
        
        # Initial encoding
        x = self.init_conv(x)
        
        # Multi-scale encoding
        for i, encoder in enumerate(self.scale_encoders):
            x = encoder(x)
            features.append(x)
            
            if i < len(self.scale_encoders) - 1:
                x = self.downsample(x)
        
        return features


class ControlNetBlock(nn.Module):
    """
    Single ControlNet conditioning block
    
    Injects vessel conditioning into U-Net features
    """
    
    def __init__(self,
                 in_channels: int,
                 conditioning_channels: int):
        super().__init__()
        
        # Zero convolution for stable training
        self.zero_conv = nn.Conv2d(conditioning_channels, in_channels, 1)
        nn.init.zeros_(self.zero_conv.weight)
        nn.init.zeros_(self.zero_conv.bias)
    
    def forward(self, 
                x: torch.Tensor,
                conditioning: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: U-Net features (B, C, H, W)
            conditioning: Conditioning features (B, C', H, W)
            
        Returns:
            Conditioned features (B, C, H, W)
        """
        # Resize conditioning to match features
        if conditioning.shape[-2:] != x.shape[-2:]:
            conditioning = F.interpolate(
                conditioning,
                size=x.shape[-2:],
                mode='bilinear',
                align_corners=True
            )
        
        # Add conditioning via zero convolution
        control_signal = self.zero_conv(conditioning)
        
        return x + control_signal


class VesselControlNet(nn.Module):
    """
    Vessel-aware ControlNet
    
    Integrates vessel structure information into Stable Diffusion
    """
    
    def __init__(self,
                 base_model_id: str = "stabilityai/stable-diffusion-2-1-base",
                 conditioning_channels: int = 3,
                 num_control_scales: int = 3,
                 control_scales: List[float] = [1.0, 0.5, 0.25]):
        """
        Args:
            base_model_id: HuggingFace model ID for base diffusion model
            conditioning_channels: Number of conditioning input channels
            num_control_scales: Number of scale levels
            control_scales: Scaling factors for control signals at different depths
        """
        super().__init__()
        
        self.control_scales = control_scales
        
        # Vessel condition encoder
        self.condition_encoder = VesselConditionEncoder(
            conditioning_channels=conditioning_channels,
            output_channels=320,
            num_scales=num_control_scales
        )
        
        # Control blocks (match U-Net architecture)
        # For SD 2.1, typical channel configs: [320, 640, 1280, 1280]
        self.control_blocks = nn.ModuleList([
            ControlNetBlock(320, 64),
            ControlNetBlock(640, 128),
            ControlNetBlock(1280, 256),
        ])
    
    def prepare_vessel_conditioning(self,
                                   mask: torch.Tensor,
                                   skeleton: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Prepare multi-channel vessel conditioning
        
        Args:
            mask: Vessel probability mask (B, 1, H, W)
            skeleton: Vessel skeleton (B, 1, H, W) [optional]
            
        Returns:
            Conditioning tensor (B, 3, H, W)
        """
        # Compute edges from mask
        edges = self._compute_edges(mask)
        
        # Use skeleton if provided, otherwise use mask
        if skeleton is None:
            skeleton = mask
        
        # Concatenate: [mask, skeleton, edges]
        conditioning = torch.cat([mask, skeleton, edges], dim=1)
        
        return conditioning
    
    def _compute_edges(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute edges using Sobel filter
        
        Args:
            mask: Input mask (B, 1, H, W)
            
        Returns:
            Edge map (B, 1, H, W)
        """
        # Sobel kernels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=mask.dtype, device=mask.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                               dtype=mask.dtype, device=mask.device).view(1, 1, 3, 3)
        
        # Compute gradients
        grad_x = F.conv2d(mask, sobel_x, padding=1)
        grad_y = F.conv2d(mask, sobel_y, padding=1)
        
        # Magnitude
        edges = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        
        return edges
    
    def forward(self,
                unet_features: List[torch.Tensor],
                mask: torch.Tensor,
                skeleton: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        """
        Forward pass
        
        Args:
            unet_features: List of U-Net intermediate features
            mask: Vessel mask (B, 1, H, W)
            skeleton: Vessel skeleton (B, 1, H, W) [optional]
            
        Returns:
            List of conditioned features with same structure as input
        """
        # Prepare conditioning
        conditioning = self.prepare_vessel_conditioning(mask, skeleton)
        
        # Encode conditioning at multiple scales
        condition_features = self.condition_encoder(conditioning)
        
        # Apply control to U-Net features
        conditioned_features = []
        
        for i, (feat, control_block, scale) in enumerate(
            zip(unet_features, self.control_blocks, self.control_scales)
        ):
            # Select appropriate conditioning scale
            cond_feat = condition_features[min(i, len(condition_features) - 1)]
            
            # Apply control with scaling
            conditioned = control_block(feat, cond_feat) * scale
            conditioned_features.append(conditioned)
        
        return conditioned_features


class VesselDiffusionEnhancer(nn.Module):
    """
    Complete vessel enhancement pipeline using ControlNet + Stable Diffusion
    
    Pipeline:
    1. Input noisy vessel image
    2. Generate vessel mask/skeleton via segmentation model
    3. Enhance using vessel-conditioned diffusion
    """
    
    def __init__(self,
                 config,
                 base_model_id: str = "stabilityai/stable-diffusion-2-1-base",
                 segmentation_model: Optional[nn.Module] = None,
                 num_inference_steps: int = 20,
                 guidance_scale: float = 7.5):
        """
        Args:
            base_model_id: HuggingFace model ID
            segmentation_model: Pretrained vessel segmentation model
            num_inference_steps: Number of DDIM steps
            guidance_scale: Classifier-free guidance scale
        """
        super().__init__()
        
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        
        # Load base diffusion model components
        # self.vae = AutoencoderKL.from_pretrained(
        #     base_model_id, 
        #     subfolder="vae"
        # )
        model_id = config.model.base_model  # 确保这里拿到的是 'stabilityai/stable-diffusion-2-1-base'
        custom_path = r"D:\babba\xxx\jingmaixianying\vessel_3d_recon\diff"
        # self.vae = AutoencoderKL.from_pretrained(
        #     model_id, 
        #     subfolder="vae",
        #     use_safetensors=True
        # )
        # self.unet = UNet2DConditionModel.from_pretrained(
        #     base_model_id,
        #     subfolder="unet"
        # )
        
        # # Scheduler
        # self.scheduler = DDIMScheduler.from_pretrained(
        #     base_model_id,
        #     subfolder="scheduler"
        # )
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae",cache_dir=custom_path)
        # 加载 UNet
        self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet",cache_dir=custom_path)
        # 加载 Scheduler (Step 3 通常使用 DDIM)
        self.scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler",cache_dir=custom_path)
        # ControlNet
        self.controlnet = VesselControlNet(
            base_model_id=base_model_id,
            conditioning_channels=3,
            num_control_scales=3
        )
        
        # Segmentation model for generating masks
        self.segmentation_model = segmentation_model
        if segmentation_model is not None:
            self.segmentation_model.eval()
            for param in self.segmentation_model.parameters():
                param.requires_grad = False
        
        # 🔧 修复：初始化空文本嵌入（Stable Diffusion 需要 text embeddings）
        self.null_text_embeds = self._init_null_text_embeddings(base_model_id)
    
    def _init_null_text_embeddings(self, model_id: str) -> torch.Tensor:
        """
        生成空文本的嵌入向量
        
        Stable Diffusion 是 text-to-image 模型，UNet 需要 text embeddings。
        但图像增强任务不需要文本，所以使用空文本（""）的嵌入作为占位符。
        
        Args:
            model_id: HuggingFace model ID
            
        Returns:
            空文本嵌入 (1, 77, 768)
        """
        try:
            from transformers import CLIPTokenizer, CLIPTextModel
            
            print("正在加载 CLIP text encoder 生成空文本嵌入...")
            
            # 加载 tokenizer 和 text encoder
            custom_path = r"D:\babba\xxx\jingmaixianying\vessel_3d_recon\diff"
            tokenizer = CLIPTokenizer.from_pretrained(
                model_id, 
                subfolder="tokenizer",
                cache_dir=custom_path
            )
            text_encoder = CLIPTextModel.from_pretrained(
                model_id,
                subfolder="text_encoder",
                cache_dir=custom_path
            )
            
            # 编码空文本
            with torch.no_grad():
                text_input = tokenizer(
                    "",  # 空文本
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                null_embeds = text_encoder(text_input.input_ids)[0]
            
            print(f"✓ 空文本嵌入生成成功，形状: {null_embeds.shape}")
            
            # 释放 text encoder 节省显存
            del text_encoder
            del tokenizer
            torch.cuda.empty_cache()
            
            return null_embeds
            
        except Exception as e:
            print(f"⚠️ 警告：无法生成空文本嵌入 - {e}")
            print("   将使用零向量作为替代")
            # 返回零向量作为后备方案 (1, 77, 768)
            return torch.zeros(1, 77, 768)
    
    @torch.no_grad()
    def generate_vessel_mask(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate vessel mask and skeleton using segmentation model
        
        Args:
            image: Input image (B, C, H, W)
            
        Returns:
            mask, skeleton: Both (B, 1, H, W)
        """
        if self.segmentation_model is None:
            # Return dummy masks
            return torch.zeros_like(image[:, :1]), torch.zeros_like(image[:, :1])
        
        outputs = self.segmentation_model(image)
        mask = outputs['mask']
        skeleton = outputs.get('skeleton', mask)
        
        return mask, skeleton
    
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Encode image to latent space
        
        Args:
            image: Image in [-1, 1] range (B, C, H, W)
            
        Returns:
            Latent code (B, 4, H//8, W//8)
        """
        latent = self.vae.encode(image).latent_dist.sample()
        latent = latent * self.vae.config.scaling_factor
        return latent
    
    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to image
        
        Args:
            latent: Latent code (B, 4, H//8, W//8)
            
        Returns:
            Image in [-1, 1] range (B, C, H, W)
        """
        latent = latent / self.vae.config.scaling_factor
        image = self.vae.decode(latent).sample
        return image
    
    def forward(self,
                noisy_image: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                skeleton: Optional[torch.Tensor] = None,
                return_intermediates: bool = False) -> Dict[str, torch.Tensor]:
        """
        Enhance vessel image using vessel-conditioned diffusion
        
        Args:
            noisy_image: Noisy input (B, C, H, W) in [-1, 1]
            mask: Vessel mask (B, 1, H, W) [optional, auto-generated if None]
            skeleton: Vessel skeleton (B, 1, H, W) [optional]
            return_intermediates: Return intermediate denoising steps
            
        Returns:
            Dict with:
                - enhanced: Enhanced image (B, C, H, W)
                - mask: Vessel mask used
                - intermediates: List of intermediate images [optional]
        """
        device = noisy_image.device
        B = noisy_image.size(0)
        
        # Generate vessel conditioning if not provided
        if mask is None:
            mask, skeleton = self.generate_vessel_mask(noisy_image)
        
        # Encode to latent space
        latent = self.encode_image(noisy_image)
        
        # Add noise (for training) or start from noisy latent
        timesteps = self.scheduler.timesteps
        
        # Denoising loop
        intermediates = []
        
        for t in timesteps:
            # Expand timestep for batch
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            
            # Predict noise
            latent_input = self.scheduler.scale_model_input(latent, t)
            
            # 🔧 修复：准备文本嵌入（复制到当前 batch size）
            encoder_hidden_states = self.null_text_embeds.repeat(B, 1, 1).to(device)
            
            # Get U-Net features (simplified - actual implementation needs proper hooks)
            noise_pred = self.unet(
                latent_input,
                t_batch,
                encoder_hidden_states=encoder_hidden_states,  # 使用空文本嵌入
                return_dict=False
            )[0]
            
            # Apply ControlNet conditioning (simplified)
            # In full implementation, inject into U-Net's down/up blocks
            
            # Scheduler step
            latent = self.scheduler.step(noise_pred, t, latent).prev_sample
            
            # Save intermediate
            if return_intermediates:
                intermediate_image = self.decode_latent(latent)
                intermediates.append(intermediate_image)
        
        # Decode final latent
        enhanced = self.decode_latent(latent)
        
        result = {
            'enhanced': enhanced,
            'mask': mask,
            'skeleton': skeleton
        }
        
        if return_intermediates:
            result['intermediates'] = intermediates
        
        return result


if __name__ == "__main__":
    print("Testing Vessel-Aware ControlNet...")
    
    # Create models
    controlnet = VesselControlNet()
    
    # Test conditioning preparation
    mask = torch.rand(2, 1, 256, 256)
    skeleton = torch.rand(2, 1, 256, 256)
    
    conditioning = controlnet.prepare_vessel_conditioning(mask, skeleton)
    print(f"Conditioning shape: {conditioning.shape}")
    
    # Test condition encoder
    features = controlnet.condition_encoder(conditioning)
    print(f"Number of scale features: {len(features)}")
    for i, feat in enumerate(features):
        print(f"  Scale {i}: {feat.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in controlnet.parameters())
    print(f"\nTotal ControlNet parameters: {total_params:,}")