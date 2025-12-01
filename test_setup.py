"""
Test script to verify installation and setup
Run this to check if everything is working correctly
"""

import sys
import subprocess
from pathlib import Path


def test_python_version():
    """Test Python version"""
    print("Testing Python version...")
    version = sys.version_info
    if version >= (3, 8):
        print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"✗ Python version too old: {version.major}.{version.minor}.{version.micro}")
        print("  Required: Python 3.8+")
        return False


def test_pytorch():
    """Test PyTorch installation"""
    print("\nTesting PyTorch...")
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        return True
    except ImportError:
        print("✗ PyTorch not installed")
        print("  Run: python setup.py")
        return False


def test_cuda():
    """Test CUDA availability"""
    print("\nTesting CUDA...")
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            print(f"✓ CUDA available: {device_name}")
            print(f"  CUDA version: {cuda_version}")
            return True
        else:
            print("⚠ CUDA not available (will use CPU)")
            print("  This is OK but processing will be slower")
            return True
    except ImportError:
        print("✗ Cannot test CUDA (PyTorch not installed)")
        return False


def test_dependencies():
    """Test required dependencies"""
    print("\nTesting dependencies...")
    
    required = [
        'torch',
        'torchvision',
        'fastapi',
        'uvicorn',
        'opencv-python',
        'numpy',
        'Pillow',
        'ffmpeg-python',
        'einops',
        'timm'
    ]
    
    all_ok = True
    for package in required:
        try:
            __import__(package.replace('-', '_'))
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} not installed")
            all_ok = False
    
    if not all_ok:
        print("\n  Run: python setup.py")
    
    return all_ok


def test_ffmpeg():
    """Test FFmpeg installation"""
    print("\nTesting FFmpeg...")
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            print(f"✓ {version_line}")
            return True
        else:
            print("✗ FFmpeg installed but returned error")
            return False
    except FileNotFoundError:
        print("✗ FFmpeg not found in PATH")
        print("  Install: winget install ffmpeg")
        print("  Or download from: https://ffmpeg.org")
        return False
    except Exception as e:
        print(f"✗ Error testing FFmpeg: {e}")
        return False


def test_directories():
    """Test directory structure"""
    print("\nTesting directories...")
    
    required_dirs = ['models', 'uploads', 'outputs', 'backend', 'frontend']
    all_ok = True
    
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"✓ {dir_name}/")
        else:
            print(f"✗ {dir_name}/ not found")
            all_ok = False
    
    return all_ok


def test_model_download():
    """Test model download capability"""
    print("\nTesting model availability...")
    try:
        from backend.model_manager import ModelManager
        
        manager = ModelManager()
        models = manager.list_available_models()
        
        for name, info in models.items():
            status = "✓" if info['downloaded'] else "⚠"
            print(f"{status} {info['description']}")
            if not info['downloaded']:
                print(f"  Will be downloaded on first use")
        
        return True
    except Exception as e:
        print(f"✗ Error checking models: {e}")
        return False


def test_import_backend():
    """Test backend imports"""
    print("\nTesting backend imports...")
    try:
        from backend import VideoProcessor, ModelManager
        print("✓ Backend modules import successfully")
        return True
    except Exception as e:
        print(f"✗ Backend import failed: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("Video Denoiser - Setup Test")
    print("=" * 60)
    
    tests = [
        ("Python Version", test_python_version),
        ("PyTorch", test_pytorch),
        ("CUDA", test_cuda),
        ("Dependencies", test_dependencies),
        ("FFmpeg", test_ffmpeg),
        ("Directories", test_directories),
        ("Backend", test_import_backend),
        ("Models", test_model_download),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ Test '{name}' crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed! You're ready to use Video Denoiser.")
        print("\nTo start the application:")
        print("  - Double-click: run.bat")
        print("  - Or run: python main.py")
    else:
        print("\n⚠ Some tests failed. Please:")
        print("  1. Review the errors above")
        print("  2. Check INSTALL.md for help")
        print("  3. Run: python setup.py")
    
    print("=" * 60)
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
