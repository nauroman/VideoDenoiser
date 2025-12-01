# Quick Start Guide 🚀

Get up and running with Video Denoiser in 5 minutes!

## Prerequisites ✅

Before starting, make sure you have:
- [ ] Windows 10/11
- [ ] Python 3.8+ installed ([Download](https://www.python.org/downloads/))
- [ ] NVIDIA GPU with latest drivers (for best performance)

## Installation Steps

### Step 1: Install FFmpeg

Open PowerShell or Command Prompt and run:

```cmd
winget install ffmpeg
```

Or download manually from [ffmpeg.org](https://ffmpeg.org/download.html)

### Step 2: Launch the Application

**Simply double-click `run.bat`**

That's it! The script will:
- Create a virtual environment
- Install all dependencies (first time only, ~5 min)
- Download AI models (~100 MB, first time only)
- Start the web server
- Open your browser automatically

### Step 3: Process Your First Video

1. **Drag & drop** your video file onto the upload area
2. Wait for upload to complete
3. Click **"Start Denoising"**
4. Watch the progress bar
5. Click **"Download Result"** when done

## First Time Setup

The first run will take 5-10 minutes because it needs to:
- Install PyTorch with CUDA support (~2GB)
- Install other Python packages (~500MB)
- Download RVRT model weights (~100MB)

**Subsequent runs start instantly!**

## Testing the Installation

If you want to verify everything is working before processing videos:

```cmd
python test_setup.py
```

This will check:
- ✓ Python version
- ✓ All dependencies
- ✓ CUDA availability
- ✓ FFmpeg installation
- ✓ Models downloaded

## Manual Start (Alternative)

If you prefer command line:

```cmd
# First time setup
python setup.py

# Start application
python main.py
```

Then open browser to: http://localhost:8000

## Recommended Settings

### For DJI Drone Videos (1080p/4K)
- **Clip Size**: 2-3
- **Tiling**: Enable for 4K
- **Tile Size**: 512x512

### For Action Cameras
- **Clip Size**: 3-4
- **Tiling**: Enable for 4K+

### For Low-light Videos
- **Clip Size**: 2
- **Tiling**: As needed

## Performance Expectations

On RTX 4090:
- **1080p**: ~30-40 FPS processing = 1.5-2 min for 1 min video
- **4K**: ~10-15 FPS processing = 4-6 min for 1 min video

## Troubleshooting

### "Python not found"
- Make sure Python is installed
- Verify it's in PATH: `python --version`

### "FFmpeg not found"
- Install FFmpeg: `winget install ffmpeg`
- Restart Command Prompt

### "CUDA not available"
- Update GPU drivers from [NVIDIA](https://www.nvidia.com/download/index.aspx)
- Application will work on CPU (slower)

### Application won't start
1. Open Command Prompt
2. Navigate to project folder
3. Run: `python test_setup.py`
4. Fix any errors shown

## Next Steps

Once everything works:

1. **Read the README**: Full documentation in `README.md`
2. **Experiment with settings**: Find what works best for your videos
3. **Check INSTALL.md**: Detailed installation guide
4. **Optimize performance**: Review performance tips in README

## Getting Help

- Check `README.md` for detailed documentation
- Check `INSTALL.md` for installation help
- Run `python test_setup.py` to diagnose issues
- Check console output for error messages

## Keyboard Shortcuts (in terminal)

- `Ctrl+C` - Stop the server
- `Ctrl+Break` - Force quit

## Tips for Best Results

1. ✨ **Use original files** - Don't compress before processing
2. 🎬 **Shorter clips first** - Test with 10-15 sec clips
3. 💾 **Free up disk space** - Need 2-3x video size free
4. 🔌 **Keep laptop plugged in** - For consistent performance
5. 🚫 **Close other apps** - Free up GPU memory

## Video Format Support

Supported formats:
- ✓ MP4 (most common)
- ✓ MOV (Apple, DJI)
- ✓ AVI
- ✓ MKV
- ✓ FLV
- ✓ WEBM

Output format: Same as input with `_denoise` suffix

## What to Expect

### Video Quality
- Reduced noise in dark areas
- Cleaner footage overall
- Preserved details and sharpness
- No resolution change

### Metadata Preserved
- Same framerate
- Same resolution
- Same audio track
- Same bitrate

## Need More Help?

Full documentation available:
- `README.md` - Complete user guide
- `INSTALL.md` - Detailed installation
- `test_setup.py` - System diagnostics

---

**Ready to enhance your videos!** 🎥✨

*Average setup time: 5-10 minutes (first time)*  
*Average processing: 1-2 minutes per minute of 1080p video on RTX 4090*
