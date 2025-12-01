"""
Setup script for Video Denoiser
Installs all dependencies and downloads required models
"""

import os
import sys
import subprocess
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_python_version():
    """Verify Python version is 3.8 or higher"""
    if sys.version_info < (3, 8):
        logger.error("Python 3.8 or higher is required")
        return False
    logger.info(f"Python version: {sys.version}")
    return True


def check_cuda():
    """Check if CUDA is available"""
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"CUDA is available: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA version: {torch.version.cuda}")
            return True
        else:
            logger.warning("CUDA is not available, will use CPU (slower)")
            return False
    except ImportError:
        logger.warning("PyTorch not installed yet")
        return None


def install_pytorch():
    """Install PyTorch with CUDA support"""
    logger.info("Installing PyTorch with CUDA support...")
    
    try:
        # Install PyTorch with CUDA 11.8 (compatible with RTX 4090)
        subprocess.check_call([
            sys.executable,
            "-m",
            "pip",
            "install",
            "torch",
            "torchvision",
            "--index-url",
            "https://download.pytorch.org/whl/cu118"
        ])
        logger.info("PyTorch installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install PyTorch: {e}")
        return False


def install_requirements():
    """Install all dependencies from requirements.txt"""
    logger.info("Installing dependencies...")
    
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if not requirements_file.exists():
        logger.error("requirements.txt not found")
        return False
    
    try:
        # Upgrade pip first
        subprocess.check_call([
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "pip"
        ])
        
        # Install requirements
        subprocess.check_call([
            sys.executable,
            "-m",
            "pip",
            "install",
            "-r",
            str(requirements_file)
        ])
        logger.info("All dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        return False


def check_ffmpeg():
    """Check if FFmpeg is installed"""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            logger.info("FFmpeg is installed")
            return True
        else:
            logger.warning("FFmpeg check returned non-zero code")
            return False
    except FileNotFoundError:
        logger.warning("FFmpeg is not installed or not in PATH")
        logger.info("Please install FFmpeg:")
        logger.info("  Download from: https://ffmpeg.org/download.html")
        logger.info("  Or use: winget install ffmpeg")
        return False


def create_directories():
    """Create necessary directories"""
    logger.info("Creating directories...")
    
    dirs = ['models', 'uploads', 'outputs', 'temp']
    for dir_name in dirs:
        dir_path = Path(__file__).parent / dir_name
        dir_path.mkdir(exist_ok=True)
        
        # Create .gitkeep files
        gitkeep = dir_path / '.gitkeep'
        gitkeep.touch(exist_ok=True)
    
    logger.info("Directories created")


def download_models():
    """Download RVRT pre-trained models"""
    logger.info("Checking models...")
    
    try:
        # Import after installation
        from backend.model_manager import ModelManager
        
        model_manager = ModelManager()
        
        # This will download the model if not present
        logger.info("Downloading RVRT model (this may take a few minutes)...")
        model_path = model_manager.get_model_path('rvrt_denoising')
        
        logger.info(f"Model ready at: {model_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to download models: {e}")
        logger.info("Models will be downloaded on first run")
        return False


def main():
    """Main setup function"""
    logger.info("=" * 60)
    logger.info("Video Denoiser Setup")
    logger.info("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install PyTorch with CUDA
    if not install_pytorch():
        logger.error("Failed to install PyTorch")
        sys.exit(1)
    
    # Install other requirements
    if not install_requirements():
        logger.error("Failed to install requirements")
        sys.exit(1)
    
    # Check CUDA
    check_cuda()
    
    # Check FFmpeg
    ffmpeg_ok = check_ffmpeg()
    
    # Create directories
    create_directories()
    
    # Download models
    download_models()
    
    logger.info("=" * 60)
    logger.info("Setup completed successfully!")
    logger.info("=" * 60)
    
    if not ffmpeg_ok:
        logger.warning("Please install FFmpeg before running the application")
    
    logger.info("\nTo start the application, run:")
    logger.info("  python main.py")
    logger.info("\nOr double-click: run.bat")


if __name__ == "__main__":
    main()
