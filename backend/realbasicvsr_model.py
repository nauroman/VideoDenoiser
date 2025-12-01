"""
RealBasicVSR Model Implementation
Based on official repository: https://github.com/ckkelvinchan/RealBasicVSR
"""

import torch
import torch.nn as nn
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def create_realbasicvsr_model(checkpoint_path: str, device: str = 'cuda'):
    """
    Create RealBasicVSR model for video restoration
    
    Args:
        checkpoint_path: Path to pre-trained weights
        device: Device to load model on
        
    Returns:
        RealBasicVSR model instance
    """
    logger.info("Creating RealBasicVSR model...")
    
    try:
        # Try to use mmedit (MMEditing)
        try:
            from mmedit.apis import init_model
            logger.info("Using MMEditing framework")
            
            # RealBasicVSR config
            config_dict = {
                'type': 'BasicVSR',
                'generator': {
                    'type': 'BasicVSRNet',
                    'mid_channels': 64,
                    'num_blocks': 30,
                    'spynet_pretrained': None
                }
            }
            
            # Load model with MMEditing
            model = init_model(config_dict, checkpoint_path, device=device)
            logger.info("RealBasicVSR model loaded successfully with MMEditing")
            return model
            
        except ImportError:
            logger.warning("MMEditing not available, using BasicSR")
            
            # Fallback to BasicSR
            from basicsr.archs.basicvsr_arch import BasicVSRNet
            
            # Create model
            model = BasicVSRNet(
                num_feat=64,
                num_block=30,
                spynet_path=None
            )
            
            # Load weights
            if Path(checkpoint_path).exists():
                logger.info(f"Loading weights from: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                
                # Handle different checkpoint formats
                if 'params_ema' in checkpoint:
                    state_dict = checkpoint['params_ema']
                elif 'params' in checkpoint:
                    state_dict = checkpoint['params']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
                
                # Load state dict
                model.load_state_dict(state_dict, strict=False)
                logger.info("RealBasicVSR weights loaded successfully")
            
            model.to(device)
            model.eval()
            
            logger.info("RealBasicVSR model loaded successfully with BasicSR")
            return model
            
    except Exception as e:
        logger.error(f"Failed to create RealBasicVSR model: {e}")
        logger.exception(e)
        raise


class RealBasicVSRWrapper:
    """Wrapper for RealBasicVSR model to provide consistent interface"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        
    def __call__(self, x):
        """
        Process video frames
        
        Args:
            x: Input tensor of shape [B, T, C, H, W]
            
        Returns:
            Output tensor of shape [B, T, C, H, W]
        """
        with torch.no_grad():
            # RealBasicVSR expects [B, T, C, H, W]
            output = self.model(x)
            return output
    
    def eval(self):
        """Set model to evaluation mode"""
        self.model.eval()
        return self
    
    def to(self, device):
        """Move model to device"""
        self.model.to(device)
        self.device = device
        return self
