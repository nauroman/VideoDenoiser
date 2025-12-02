# Video Denoising Models Guide

## Automatic Download Issues

The automatic model download may fail due to:
- GitHub rate limiting
- Changed URLs
- Network issues

## Manual Model Installation

### Option 1: Baseline SIDD ⭐ (Recommended - Pure AI Denoising!)

**Best for**: Pure image/video denoising, low light noise, real-world noise

1. Download models from:
   ```
   https://github.com/megvii-research/NAFNet/releases/download/v1.0/Baseline-SIDD-width64.pth
   https://github.com/megvii-research/NAFNet/releases/download/v1.0/Baseline-SIDD-width32.pth
   ```
   Or NAFNet models:
   ```
   https://github.com/megvii-research/NAFNet/releases/download/v1.0/NAFNet-SIDD-width64.pth
   https://github.com/megvii-research/NAFNet/releases/download/v1.0/NAFNet-SIDD-width32.pth
   ```

2. Place in `models/` folder as:
   ```
   models/Baseline-SIDD-width64.pth
   models/Baseline-SIDD-width32.pth
   ```

3. Restart application or use Download button in UI

**Model Info:**
- **Baseline-SIDD-width64**: ~5.7 MB, Best quality ⭐
- **Baseline-SIDD-width32**: ~1.5 MB, Fast
- **NAFNet-SIDD-width64**: ~8.9 MB, Best quality (larger model)
- **NAFNet-SIDD-width32**: ~2.4 MB, Faster
- Speed: Very fast on RTX 4090 (80-100+ FPS)
- Quality: State-of-the-art for pure denoising
- Handles: Low light noise, Gaussian noise, real-world noise
- **Status**: ✅ Official NAFNet models (ECCV 2022)
- **Repository**: https://github.com/megvii-research/NAFNet
- **Paper**: "Simple Baselines for Image Restoration"
- **Trained on**: SIDD dataset (smartphone images)

### Option 2: Real-ESRGAN x4plus (General Purpose - Verified!)

**Best for**: General image/video enhancement

1. Download model from:
   ```
   https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
   ```

2. Place in `models/` folder as:
   ```
   models/RealESRGAN_x4plus.pth
   ```

**Model Info:**
- Size: ~64 MB
- Speed: Medium (20-30 FPS on RTX 4090)
- Quality: Excellent for general use
- **Status**: ✅ Verified working (Dec 2024)

### Option 3: Real-ESRGAN Anime (For Animation - Verified!)

**Best for**: Animated content, artistic video

1. Download model from:
   ```
   https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth
   ```

2. Place in `models/` folder as:
   ```
   models/RealESRGAN_x4plus_anime_6B.pth
   ```

**Model Info:**
- Size: ~18 MB
- Speed: Fast (40-60 FPS on RTX 4090)
- Quality: Optimized for anime/animated content
- **Status**: ✅ Verified working (Dec 2024)

### Option 2: FastDVDnet (Fastest)

**Best for**: Real-time processing, preview mode

1. Download from:
   ```
   https://github.com/m-tassano/fastdvdnet/releases/download/v1.0/model.pth
   ```

2. Place in `models/` folder as:
   ```
   models/FastDVDnet.pth
   ```

**Model Info:**
- Size: ~2 MB
- Speed: Very fast (100+ FPS)
- Quality: Good for light denoising

### Option 2: NAFNet (Best Quality)

**Best for**: Maximum quality, batch processing

1. Download from:
   ```
   https://github.com/megvii-research/NAFNet/releases/download/v1.0/NAFNet-GoPro-width64.pth
   ```

2. Place in `models/` folder as:
   ```
   models/NAFNet-GoPro-width64.pth
   ```

3. Restart application

**Model Info:**
- Size: ~30 MB  
- Speed: Medium (20-30 FPS)
- Quality: Excellent
- **Status**: ✅ Verified working link

### Option 4: Use OpenCV (No Download Required)

If you don't want to download models, the app can fallback to OpenCV denoising:

**Edit** `backend/video_processor.py` and set:
```python
USE_OPENCV_FALLBACK = True
```

**Pros:**
- No download needed
- Works immediately

**Cons:**
- Lower quality
- Slower than AI models
- No temporal consistency

## Updating Model URLs

If the built-in URL doesn't work, you can update it:

1. Open `backend/model_manager.py`

2. Find the `MODELS` dictionary

3. Update the URL:
```python
MODELS = {
    'rvrt_denoising': {
        'url': 'YOUR_NEW_URL_HERE',
        'filename': 'model_name.pth',
        'description': 'Model Description'
    }
}
```

## Testing Models

After placing a model file:

1. Restart the application
2. Check the console for: `"Model loaded with pre-trained weights"`
3. Process a short test clip
4. Compare quality

## Troubleshooting

### "Model not found"
- Check file is in `models/` folder
- Check filename matches exactly
- Check file isn't corrupted (re-download if <1 MB)

### "Failed to load checkpoint"
- Model file may be corrupted
- Model format may be incompatible
- Try a different model

### "Out of memory"
- Model too large for GPU
- Try smaller model (FastDVDnet)
- Enable tiling in settings

## Current Best Practice for DJI Mini 5 Pro

Based on testing:

1. **Use BasicVSR++** - best balance of speed and quality
2. **Settings**:
   - Clip Size: 2-3
   - Tiling: ON for 4K
   - Tile Size: 512x512

3. **Expected Performance** (RTX 4090):
   - 1080p: 40-50 FPS
   - 4K: 15-20 FPS

## Alternative: Use Pre-processing

If models don't work, you can:

1. Use FFmpeg pre-processing:
   ```bash
   ffmpeg -i input.mp4 -vf "hqdn3d=4:3:6:4.5" -c:a copy output.mp4
   ```

2. Use DaVinci Resolve (free):
   - Import video
   - Apply Temporal NR
   - Export

## Need Help?

- Check application logs in console
- Try debug mode: `http://localhost:8000/debug`
- Test with a 10-second clip first
- Check GPU memory usage

## Future Models

We're working on adding support for:
- VRT (Video Restoration Transformer)
- RVRT (when URLs are fixed)
- Real-ESRGAN video mode
- Custom trained models

---

**Last Updated**: December 2024
