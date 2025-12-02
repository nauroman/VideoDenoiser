"""
Model Manager for NAFNet
Handles automatic downloading and management of pre-trained weights
"""

import os
import requests
from pathlib import Path
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

class ModelManager:
    """Manages NAFNet model weights download and storage"""
    
    # NAFNet models
    # Official repository: https://github.com/megvii-research/NAFNet
    MODELS = {
        'baseline_sidd_width64': {
            # Baseline model trained on SIDD dataset
            'url': 'https://github.com/megvii-research/NAFNet/releases/download/v1.0/Baseline-SIDD-width64.pth',
            'filename': 'Baseline-SIDD-width64.pth',
            'description': 'Baseline SIDD width64 ⭐ Recommended'
        },
        'nafnet_sidd_width64': {
            # NAFNet trained on SIDD dataset (larger model)
            'url': 'https://github.com/megvii-research/NAFNet/releases/download/v1.0/NAFNet-SIDD-width64.pth',
            'filename': 'NAFNet-SIDD-width64.pth',
            'description': 'NAFNet SIDD width64 (Best quality)'
        },
        'nafnet_sidd_width32': {
            # NAFNet trained on SIDD dataset for image denoising
            'url': 'https://github.com/megvii-research/NAFNet/releases/download/v1.0/NAFNet-SIDD-width32.pth',
            'filename': 'NAFNet-SIDD-width32.pth',
            'description': 'NAFNet SIDD width32 (Faster)'
        }
    }
    
    def __init__(self, models_dir: str = 'models'):
        """
        Initialize model manager
        
        Args:
            models_dir: Directory to store model weights
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
    def get_model_path(self, model_name: str = 'rvrt_denoising') -> Path:
        """
        Get path to model file, download if not exists
        
        Args:
            model_name: Name of the model to get
            
        Returns:
            Path to model file (or None if download fails)
        """
        if model_name not in self.MODELS:
            raise ValueError(f"Unknown model: {model_name}")
            
        model_info = self.MODELS[model_name]
        model_path = self.models_dir / model_info['filename']
        
        if not model_path.exists():
            logger.info(f"Model not found, attempting download: {model_info['description']}")
            try:
                self._download_model(model_info['url'], model_path)
            except Exception as e:
                logger.error(f"Failed to download model: {e}")
                logger.warning("Will proceed without pre-trained weights (output will be same as input)")
                return None
        else:
            logger.info(f"Model found: {model_path}")
            
        return model_path
    
    def _download_model(self, url: str, destination: Path) -> None:
        """
        Download model file with progress bar
        
        Args:
            url: URL to download from
            destination: Path to save file
        """
        try:
            logger.info(f"Downloading from: {url}")
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            block_size = 8192
            
            with open(destination, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=destination.name) as pbar:
                    for chunk in response.iter_content(block_size):
                        f.write(chunk)
                        pbar.update(len(chunk))
                        
            logger.info(f"Successfully downloaded: {destination}")
            
        except Exception as e:
            if destination.exists():
                destination.unlink()
            logger.error(f"Failed to download model: {e}")
            raise RuntimeError(f"Failed to download model: {e}")
    
    def list_available_models(self) -> dict:
        """
        List all available models with their status
        
        Returns:
            Dictionary with model info and download status
        """
        models_status = {}
        for name, info in self.MODELS.items():
            model_path = self.models_dir / info['filename']
            file_size = model_path.stat().st_size if model_path.exists() else 0
            models_status[name] = {
                'description': info['description'],
                'downloaded': model_path.exists(),
                'path': str(model_path) if model_path.exists() else None,
                'filename': info['filename'],
                'url': info['url'],
                'size_mb': file_size / (1024 * 1024) if file_size > 0 else 0
            }
        return models_status
    
    def download_model_by_name(self, model_name: str) -> bool:
        """
        Download a specific model by name
        
        Args:
            model_name: Name of the model to download
            
        Returns:
            True if successful, False otherwise
        """
        if model_name not in self.MODELS:
            logger.error(f"Unknown model: {model_name}")
            return False
        
        model_info = self.MODELS[model_name]
        model_path = self.models_dir / model_info['filename']
        
        if model_path.exists():
            logger.info(f"Model already exists: {model_path}")
            return True
        
        try:
            logger.info(f"Downloading {model_info['description']}...")
            self._download_model(model_info['url'], model_path)
            return True
        except Exception as e:
            logger.error(f"Failed to download {model_name}: {e}")
            return False
