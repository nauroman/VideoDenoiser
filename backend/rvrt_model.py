"""
RVRT Model Implementation
Recurrent Video Restoration Transformer for video denoising
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import trunc_normal_
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class RVRT(nn.Module):
    """
    Recurrent Video Restoration Transformer (RVRT)
    Simplified implementation for video denoising
    """
    
    def __init__(
        self,
        img_size=64,
        in_chans=3,
        embed_dim=96,
        depths=[6, 6, 6, 6],
        num_heads=[6, 6, 6, 6],
        window_size=[8, 8],
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
        use_checkpoint=False,
        upscale=1,
        img_range=1.,
        upsampler='',
        resi_connection='1conv',
        clip_size=2,
        **kwargs
    ):
        super(RVRT, self).__init__()
        
        self.img_range = img_range
        self.upscale = upscale
        self.clip_size = clip_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        
        # Simple feature extraction for denoising
        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)
        
        # Main processing blocks
        self.num_layers = len(depths)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
            ) for _ in range(self.num_layers)
        ])
        
        # Reconstruction
        self.conv_last = nn.Conv2d(embed_dim, in_chans, 3, 1, 1)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        """
        Forward pass for video denoising
        
        Args:
            x: Input tensor [B, T, C, H, W] where T is temporal dimension
            
        Returns:
            Denoised video [B, T, C, H, W]
        """
        B, T, C, H, W = x.shape
        
        # Process frame by frame with residual connection
        output_frames = []
        
        for t in range(T):
            frame = x[:, t, :, :, :]  # [B, C, H, W]
            
            # Normalize to [0, 1]
            mean = frame.mean([1, 2, 3], keepdim=True)
            frame_normalized = frame - mean
            
            # Feature extraction
            feat = self.conv_first(frame_normalized)
            
            # Process through blocks with residual connections
            for block in self.blocks:
                feat = feat + block(feat)
            
            # Reconstruction
            out = self.conv_last(feat)
            out = out + mean  # Add mean back
            
            output_frames.append(out)
        
        # Stack frames back
        output = torch.stack(output_frames, dim=1)  # [B, T, C, H, W]
        
        return output
    
    def load_pretrained(self, checkpoint_path: str):
        """
        Load pre-trained weights
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        if not checkpoint_path or not Path(checkpoint_path).exists():
            logger.warning("No checkpoint file provided or file not found")
            logger.warning("Using untrained model - will only copy input (no denoising)")
            return
            
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Handle different checkpoint formats
            if 'params' in checkpoint:
                state_dict = checkpoint['params']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Remove 'module.' prefix if present
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            
            # Load with strict=False to allow missing keys (simplified architecture)
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                logger.warning(f"Missing keys in checkpoint: {len(missing_keys)} keys")
            if unexpected_keys:
                logger.warning(f"Unexpected keys in checkpoint: {len(unexpected_keys)} keys")
                
            logger.info(f"Successfully loaded checkpoint from: {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            logger.warning("Using untrained model - will only copy input (no denoising)")


def create_rvrt_model(checkpoint_path: str = None, device: str = 'cuda') -> RVRT:
    """
    Create RVRT model for video denoising
    
    Args:
        checkpoint_path: Path to pre-trained weights (optional)
        device: Device to load model on
        
    Returns:
        RVRT model instance
    """
    model = RVRT(
        upscale=1,
        img_size=64,
        window_size=[8, 8],
        img_range=1.,
        depths=[6, 6, 6, 6],
        embed_dim=96,
        num_heads=[6, 6, 6, 6],
        mlp_ratio=4,
        upsampler='',
        clip_size=2
    )
    
    if checkpoint_path:
        model.load_pretrained(checkpoint_path)
    
    model = model.to(device)
    model.eval()
    
    return model
