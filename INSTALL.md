# Installation Guide for Video Denoiser

## Prerequisites Installation

### 1. Install Python

1. Download Python 3.8+ from [python.org](https://www.python.org/downloads/)
2. **IMPORTANT**: Check "Add Python to PATH" during installation
3. Verify installation:
   ```cmd
   python --version
   ```

### 2. Install FFmpeg

#### Option A: Using winget (Easiest)
```cmd
winget install ffmpeg
```

#### Option B: Manual Installation
1. Download from [ffmpeg.org](https://www.ffmpeg.org/download.html)
2. Extract to `C:\ffmpeg`
3. Add to PATH:
   - Open "Environment Variables"
   - Edit "Path" under System Variables
   - Add `C:\ffmpeg\bin`
4. Restart Command Prompt
5. Verify:
   ```cmd
   ffmpeg -version
   ```

### 3. Install CUDA (Optional but Recommended)

If you have an NVIDIA GPU:

1. Check your GPU driver version:
   ```cmd
   nvidia-smi
   ```

2. If driver is old, download latest from [NVIDIA](https://www.nvidia.com/download/index.aspx)

3. CUDA toolkit (optional, PyTorch includes CUDA):
   - Download from [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads)

## Application Setup

### Method 1: Automated (Recommended)

Simply double-click `run.bat`. It will:
- Create virtual environment
- Install all dependencies
- Download AI models
- Start the application

### Method 2: Manual

1. **Create virtual environment** (optional but recommended):
   ```cmd
   python -m venv .venv
   .venv\Scripts\activate
   ```

2. **Run setup script**:
   ```cmd
   python setup.py
   ```

3. **Start application**:
   ```cmd
   python main.py
   ```

## First Run

1. On first run, the application will:
   - Download RVRT model (~30-120 MB)
   - Create necessary directories
   - Initialize backend

2. Your browser will open automatically to `http://localhost:8000`

3. If browser doesn't open, manually navigate to the URL

## Verification

### Check Python
```cmd
python --version
# Should show Python 3.8 or higher
```

### Check FFmpeg
```cmd
ffmpeg -version
# Should show FFmpeg version and configuration
```

### Check CUDA (if using GPU)
```cmd
nvidia-smi
# Should show your GPU (RTX 4090)
```

### Test PyTorch CUDA
```python
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
# Should print: CUDA available: True
```

## Common Installation Issues

### Python not found
- Reinstall Python with "Add to PATH" checked
- Or manually add Python to PATH

### FFmpeg not found
- Verify FFmpeg is in PATH
- Restart Command Prompt after adding to PATH
- Try running `ffmpeg` directly

### CUDA not available
- Update NVIDIA drivers
- Verify GPU compatibility
- Application will still work with CPU (slower)

### Permission errors
- Run Command Prompt as Administrator
- Check antivirus isn't blocking

### Module not found errors
- Activate virtual environment first
- Run `python setup.py` again
- Try `pip install -r requirements.txt` manually

## Updating

To update the application:

1. Pull latest changes (if using git)
2. Activate virtual environment
3. Run:
   ```cmd
   pip install -r requirements.txt --upgrade
   ```

## Uninstallation

To remove the application:

1. Delete the project folder
2. Optionally remove virtual environment
3. FFmpeg and Python can remain for other applications

## System Requirements Check

Run this to verify your system:

```cmd
python -c "import sys; import platform; print(f'Python: {sys.version}'); print(f'OS: {platform.system()} {platform.release()}')"
```

Expected output:
```
Python: 3.x.x ...
OS: Windows 10/11
```

## Getting Help

If installation fails:

1. Check error messages carefully
2. Verify all prerequisites are installed
3. Review this guide again
4. Check README.md troubleshooting section
5. Create an issue with:
   - Error message
   - Python version
   - FFmpeg version
   - GPU model (if applicable)

## Next Steps

After successful installation:

1. Read [README.md](README.md) for usage guide
2. Try processing a test video
3. Adjust settings for your use case
4. Check performance benchmarks

---

**Installation Time**: ~10-15 minutes (including downloads)

**Disk Space**: ~5GB (including models and dependencies)
