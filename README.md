# Video Denoiser 🎥✨

AI-powered video denoising application using NAFNet (Simple Baselines for Image Restoration). Designed for local processing on RTX GPUs, optimized for drone footage and general video denoising.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![CUDA](https://img.shields.io/badge/CUDA-11.8-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## ⚠️ Important: Model Download

The automatic model download may fail due to changed URLs on GitHub. If you get a download error:

1. **See [MODELS.md](MODELS.md)** for manual download instructions
2. The app will still run but output will be identical to input (no denoising)
3. Download and place a compatible model in `models/` folder
4. Recommended: **Baseline-SIDD-width64** (~5.7 MB, best quality)

## Features ✨

- 🚀 **State-of-the-art NAFNet** - ECCV 2022 AI model for video denoising
- 💻 **Local Processing** - Runs entirely on your PC, no cloud required
- 🎯 **GPU Optimized** - Leverages CUDA for blazing-fast processing
- 🎨 **Modern Web UI** - Beautiful interface with drag & drop support
- 🌓 **Dark/Light Theme** - Automatic theme switching based on preference
- 📊 **Real-time Progress** - Live updates on processing status
- 🔊 **Audio Preservation** - Maintains original audio with same bitrate
- ⚙️ **Metadata Preservation** - Keeps all video metadata intact
- 📁 **All Formats Supported** - Works with any video format (MP4, MOV, AVI, etc.)
- 🎮 **User-friendly** - One-click launch, no technical knowledge required
- ⚡ **Multi-pass Denoising** - Configurable strength (1-3 passes)
- 🎯 **Temporal Enhancement** - Optional frame averaging for smoother results

## Requirements 📋

### System Requirements
- **OS**: Windows 10/11
- **GPU**: NVIDIA GPU with CUDA support (RTX recommended)
- **RAM**: 8GB+ recommended
- **Storage**: 2GB+ free space

### Software Requirements
- **Python**: 3.8 or higher ([Download](https://www.python.org/downloads/))
- **FFmpeg**: Required for video processing ([Download](https://ffmpeg.org/download.html))
- **CUDA**: 11.8+ (usually comes with GPU drivers)

## Quick Start 🚀

### Option 1: One-Click Launch (Easiest)

1. **Double-click** `run.bat`
2. Wait for automatic setup (first time only)
3. Browser will open automatically
4. Start denoising videos!

### Option 2: Manual Setup

1. **Install dependencies**:
   ```bash
   python setup.py
   ```

2. **Start the application**:
   ```bash
   python main.py
   ```

3. **Open browser**:
   Navigate to `http://localhost:8000`

## Usage Guide 📖

### Basic Workflow

1. **Upload Video**
   - Drag & drop your video file onto the upload area
   - Or click to browse and select a file
   - Supports all video formats (MP4, MOV, AVI, MKV, etc.)

2. **Configure Settings** (Optional)
   - **AI Model**: Select downloaded pretrained model
   - **Denoising Strength**: Choose 1-3 passes (more = stronger)
   - **Max Quality**: Enable temporal enhancement for smoother results
   - **Clip Size**: Number of frames to process together (2-5 recommended)

3. **Start Processing**
   - Click "Start Denoising"
   - Monitor real-time progress
   - Processing time depends on video length and resolution

4. **Download Result**
   - Click "Download Result" when complete
   - Output file has `_denoise` suffix before extension
   - Original audio and metadata preserved

### Settings Explained ⚙️

#### AI Model
- **What it does**: Select which NAFNet model to use
- **Baseline-SIDD-width64**: Best quality (recommended)
- **NAFNet-SIDD-width64**: Best quality, slightly larger
- **NAFNet-SIDD-width32**: Faster, smaller model

#### Denoising Strength
- **1 pass**: Light denoising, fastest
- **2 passes**: Medium denoising (recommended)
- **3 passes**: Strong denoising, best quality but slower

#### Max Quality (Temporal Enhancement)
- **What it does**: Averages neighboring frames for smoother results
- **When to use**: For videos with camera shake or motion
- **Trade-off**: Slower but smoother output

#### Clip Size
- **What it does**: Number of consecutive frames processed together
- **Recommended**: 2-5 frames
- **Lower values**: Faster but may miss temporal patterns
- **Higher values**: Better quality but slower

## Project Structure 📁

```
VideoDenoiser/
├── backend/                 # Backend logic
│   ├── app.py              # FastAPI server
│   ├── video_processor.py  # Video processing with NAFNet
│   ├── model_manager.py    # Model download & management
│   └── nafnet_model.py     # NAFNet implementation
├── frontend/               # Web UI
│   └── index.html         # React-based interface
├── models/                # Pre-trained weights (auto-downloaded)
├── uploads/               # Temporary upload storage
├── outputs/               # Processed videos
├── requirements.txt       # Python dependencies
├── setup.py              # Setup script
├── main.py               # Application entry point
├── run.bat               # One-click launcher
└── README.md             # This file
```

## Technical Details 🔧

### NAFNet Model
- **Architecture**: Nonlinear Activation Free Network (ECCV 2022)
- **Paper**: "Simple Baselines for Image Restoration"
- **Purpose**: Pure AI denoising without upscaling
- **Input**: Video frames (any resolution)
- **Output**: Denoised video (same resolution)
- **Pre-trained**: On SIDD dataset (real-world smartphone noise)
- **Models**: Baseline, NAFNet width32, NAFNet width64

### Video Processing Pipeline
1. **Extract audio** from original video
2. **Process frames** with NAFNet in clips
3. **Apply multi-pass denoising** (1-3 passes)
4. **Optional temporal averaging** for smoother results
5. **Reconstruct video** with same codec/bitrate
6. **Merge audio** back with processed video
7. **Preserve metadata** (framerate, resolution, format)

### Performance Benchmarks (RTX 4090)

| Resolution | FPS (Processing) | Time (1 min video) | Passes |
|------------|------------------|-------------------|--------|
| 1080p      | 80-100 FPS       | ~40 sec           | 1      |
| 1080p      | 40-50 FPS        | ~1.5 min          | 2      |
| 4K         | 20-30 FPS        | ~2-3 min          | 1      |
| 4K         | 10-15 FPS        | ~4-6 min          | 2      |

*Note: Actual performance varies based on video complexity and settings*

## Troubleshooting 🔧

### Common Issues

#### "CUDA not available"
- **Solution**: Install NVIDIA drivers and CUDA toolkit
- Check: `nvidia-smi` in command prompt
- Download: [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)

#### "FFmpeg not found"
- **Solution**: Install FFmpeg and add to PATH
- Windows: `winget install ffmpeg`
- Or download from [ffmpeg.org](https://ffmpeg.org/download.html)

#### "Out of memory" error
- **Solution**: Enable tiling in settings
- Reduce clip size to 2
- Close other GPU-intensive applications

#### Slow processing
- **Check**: GPU is being used (not CPU)
- Verify CUDA is available
- Update GPU drivers
- Reduce video resolution if possible

#### Audio/video sync issues
- **Cause**: Usually from source video
- Try re-encoding source video first
- Use constant framerate (CFR) videos

### Getting Help

If you encounter issues:
1. Check this README's troubleshooting section
2. Verify all requirements are met
3. Check application logs in console
4. Create an issue on GitHub with:
   - Error message
   - System specifications
   - Video details (format, resolution, codec)

## Advanced Configuration 🛠️

### Custom Model Paths
Edit `backend/model_manager.py` to use custom model weights:
```python
MODELS = {
    'custom_model': {
        'url': 'your_custom_model_url',
        'filename': 'your_model.pth',
        'description': 'Custom NAFNet model'
    }
}
```

### API Usage
The application provides REST API endpoints:

- `POST /api/upload` - Upload video
- `POST /api/process/{job_id}` - Start processing
- `GET /api/status/{job_id}` - Get progress
- `GET /api/download/{job_id}` - Download result

### Command Line Usage
```python
from backend import VideoProcessor

processor = VideoProcessor(device='cuda')
processor.process_video(
    input_path='input.mp4',
    output_path='output_denoise.mp4',
    clip_size=2
)
```

## Best Practices 📝

### For DJI Drone Footage
- Use clip size of 2-3 for best results
- Enable tiling for 4K footage
- Process in original format (don't transcode first)

### For Low-light Videos
- Use smaller clip sizes (2-3)
- May take longer but produces better results
- Consider using tiling for longer videos

### For Action Videos
- Use clip size of 4-5 for better temporal consistency
- Expect longer processing times
- Tiling recommended for 4K+

## FAQ ❓

**Q: Does this work with CPU only?**
A: Yes, but 10-20x slower. GPU is highly recommended.

**Q: Can I process multiple videos at once?**
A: Currently one at a time. Batch processing planned for future release.

**Q: Will this improve compressed YouTube videos?**
A: Yes! NAFNet is good at handling compression artifacts and real-world noise.

**Q: Does it change the video length/framerate?**
A: No, all timing information is preserved exactly.

**Q: Can I use this commercially?**
A: Check NAFNet license. This wrapper is MIT licensed.

## Performance Tips 💡

1. **Close other applications** using GPU (games, 3D software)
2. **Use tiling** for high-resolution videos
3. **Process shorter clips** if memory limited
4. **Update GPU drivers** for best performance
5. **Use SSD** for video storage (faster I/O)

## Roadmap 🗺️

- [ ] Batch processing support
- [ ] Video preview comparison (before/after)
- [ ] Custom denoising strength slider
- [ ] Support for image sequences
- [ ] GPU memory optimization
- [ ] Multi-GPU support
- [ ] Video trimming/cropping in UI
- [ ] Preset profiles for different scenarios

## Contributing 🤝

Contributions are welcome! Please feel free to submit pull requests.

### Areas for Contribution
- Additional denoising models
- Performance optimizations
- UI/UX improvements
- Documentation
- Testing

## Acknowledgments 🙏

- **NAFNet**: [megvii-research/NAFNet](https://github.com/megvii-research/NAFNet)
- **PyTorch**: Deep learning framework
- **FFmpeg**: Video processing
- **FastAPI**: Web framework
- **React**: UI framework
- **Tailwind CSS**: Styling

## License 📄

This project is licensed under the MIT License. See LICENSE file for details.

Note: Pre-trained models may have their own licenses. Please check the NAFNet repository for model licensing.

## Support 💖

If you find this project helpful:
- ⭐ Star this repository
- 🐛 Report bugs
- 💡 Suggest features
- 📖 Improve documentation

---

**Made with ❤️ for drone enthusiasts and video creators**

*Tested on Windows 11 with RTX 4090 • Powered by NAFNet (ECCV 2022)*
