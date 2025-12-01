"""
Video Processor
Handles video denoising with RVRT and FFmpeg integration
"""

import os
import cv2
import torch
import numpy as np
import ffmpeg
import json
import logging
from pathlib import Path
from typing import Tuple, Optional, Callable
from datetime import datetime
import time

try:
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet
    REALESRGAN_AVAILABLE = True
except ImportError:
    REALESRGAN_AVAILABLE = False
    RealESRGANer = None
    RRDBNet = None

from .rvrt_model import create_rvrt_model
from .model_manager import ModelManager

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Processes videos with RVRT denoising while preserving all metadata"""
    
    def __init__(self, device: str = 'cuda'):
        """
        Initialize video processor
        
        Args:
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")
        
        self.model = None
        self.model_manager = ModelManager()
        self.model_has_weights = False  # Track if model has actual weights
        self.preferred_model = None  # User-selected model
        
    def load_model(self):
        """Load NAFNet model with pre-trained weights"""
        if self.model is None:
            logger.info("Loading NAFNet model...")
            
            # Try to find any downloaded NAFNet model
            models = self.model_manager.list_available_models()
            available_model = None
            
            # Check if user selected a specific model
            if self.preferred_model and self.preferred_model in models and models[self.preferred_model]['downloaded']:
                available_model = self.preferred_model
                logger.info(f"Using user-selected model: {available_model}")
            else:
                # Prefer Baseline SIDD width64 (best quality)
                for model_name in ['baseline_sidd_width64', 'nafnet_sidd_width64', 'nafnet_sidd_width32']:
                    if model_name in models and models[model_name]['downloaded']:
                        available_model = model_name
                        logger.info(f"Found downloaded NAFNet model: {model_name}")
                        break
            
            if available_model:
                model_path = self.model_manager.get_model_path(available_model)
                if model_path:
                    try:
                        # Load NAFNet model
                        from .nafnet_model import create_nafnet_model
                        self.model = create_nafnet_model(str(model_path), self.device)
                        self.model_has_weights = True
                        self.model_name = available_model
                        logger.info(f"NAFNet loaded with pre-trained weights: {available_model}")
                    except Exception as e:
                        logger.error(f"Failed to load NAFNet model: {e}")
                        logger.exception(e)
                        self._set_passthrough_mode()
                else:
                    self._set_passthrough_mode()
            else:
                self._set_passthrough_mode()
    
    def _set_passthrough_mode(self):
        """Set model to passthrough mode"""
        self.model = None
        self.model_has_weights = False
        logger.warning("No pre-trained model available")
        logger.warning("Video will be copied without denoising (passthrough mode)")
        logger.warning("Download a model from the UI to enable denoising")
    
    def get_video_info(self, video_path: str) -> dict:
        """
        Extract video metadata using FFprobe
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video metadata
        """
        try:
            probe = ffmpeg.probe(video_path)
            video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
            audio_info = next((s for s in probe['streams'] if s['codec_type'] == 'audio'), None)
            
            info = {
                'width': int(video_info['width']),
                'height': int(video_info['height']),
                'fps': eval(video_info['r_frame_rate']),  # e.g., "30/1" -> 30.0
                'duration': float(video_info.get('duration', 0)),
                'nb_frames': int(video_info.get('nb_frames', 0)),
                'codec': video_info['codec_name'],
                'bitrate': int(probe['format'].get('bit_rate', 0)),
                'format': probe['format']['format_name'],
                'has_audio': audio_info is not None,
            }
            
            if audio_info:
                info['audio_codec'] = audio_info['codec_name']
                info['audio_bitrate'] = int(audio_info.get('bit_rate', 128000))
                info['audio_sample_rate'] = int(audio_info.get('sample_rate', 48000))
                info['audio_channels'] = int(audio_info.get('channels', 2))
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get video info: {e}")
            raise
    
    def process_video(
        self,
        input_path: str,
        output_path: str,
        clip_size: int = 2,
        tile_size: Optional[Tuple[int, int]] = None,
        num_passes: int = 2,
        use_temporal: bool = True,
        progress_callback: Optional[Callable] = None
    ) -> str:
        """
        Process video with NAFNet denoising
        
        Args:
            input_path: Input video path
            output_path: Output video path
            clip_size: Number of frames to process together
            tile_size: Process video in tiles (width, height) for memory efficiency
            num_passes: Number of NAFNet passes (1-3)
            use_temporal: Enable temporal averaging for better quality
            progress_callback: Callback function for progress updates
            
        Returns:
            Path to output video
        """
        start_time = time.time()
        
        # Store settings for _denoise_clip
        self.num_passes = num_passes
        self.use_temporal = use_temporal
        
        # Load model if not loaded
        self.load_model()
        
        # Get video info
        logger.info(f"Processing video: {input_path}")
        video_info = self.get_video_info(input_path)
        logger.info(f"Video info: {video_info['width']}x{video_info['height']}, "
                   f"{video_info['fps']} fps, {video_info['duration']}s")
        
        # Create temporary files
        temp_dir = Path(output_path).parent / 'temp'
        temp_dir.mkdir(exist_ok=True)
        
        temp_video_path = temp_dir / f'temp_video_{datetime.now().strftime("%Y%m%d_%H%M%S")}.mp4'
        temp_audio_path = temp_dir / f'temp_audio_{datetime.now().strftime("%Y%m%d_%H%M%S")}.aac'
        
        try:
            # Extract audio if present
            if video_info['has_audio']:
                logger.info("Extracting audio...")
                self._extract_audio(input_path, str(temp_audio_path), video_info)
            
            # Process video frames
            logger.info("Processing frames with RVRT...")
            self._process_frames(
                input_path,
                str(temp_video_path),
                video_info,
                clip_size,
                tile_size,
                progress_callback
            )
            
            # Merge video and audio
            if video_info['has_audio']:
                logger.info("Merging video and audio...")
                self._merge_audio_video(
                    str(temp_video_path),
                    str(temp_audio_path),
                    output_path,
                    video_info
                )
            else:
                # Just copy the processed video
                os.rename(str(temp_video_path), output_path)
            
            elapsed_time = time.time() - start_time
            logger.info(f"Processing completed in {elapsed_time:.2f}s")
            
            if progress_callback:
                progress_callback(100, f"Completed in {elapsed_time:.1f}s")
            
            return output_path
            
        finally:
            # Cleanup temp files
            if temp_video_path.exists():
                temp_video_path.unlink()
            if temp_audio_path.exists():
                temp_audio_path.unlink()
    
    def _extract_audio(self, video_path: str, audio_path: str, video_info: dict):
        """Extract audio from video preserving bitrate"""
        try:
            audio_bitrate = video_info.get('audio_bitrate', 128000) // 1000  # Convert to kbps
            
            (
                ffmpeg
                .input(video_path)
                .output(audio_path, acodec='aac', audio_bitrate=f'{audio_bitrate}k')
                .overwrite_output()
                .run(quiet=True)
            )
        except Exception as e:
            logger.warning(f"Failed to extract audio: {e}")
    
    def _process_frames(
        self,
        input_path: str,
        output_path: str,
        video_info: dict,
        clip_size: int,
        tile_size: Optional[Tuple[int, int]],
        progress_callback: Optional[Callable]
    ):
        """Process video frames with RVRT"""
        cap = cv2.VideoCapture(input_path)
        
        width = video_info['width']
        height = video_info['height']
        fps = video_info['fps']
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            total_frames = video_info.get('nb_frames', 0)
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_buffer = []
        processed_frames = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # OpenCV reads in BGR, keep as is
                frame_buffer.append(frame)
                
                # Process when buffer is full or at the end
                if len(frame_buffer) >= clip_size or not ret:
                    if len(frame_buffer) > 0:
                        # Process clip
                        processed_clip = self._denoise_clip(frame_buffer, tile_size)
                        
                        # Write frames
                        for proc_frame in processed_clip:
                            # Already in BGR format
                            out.write(proc_frame)
                            processed_frames += 1
                            
                            # Update progress
                            if progress_callback and total_frames > 0:
                                progress = int((processed_frames / total_frames) * 90)  # 0-90%
                                elapsed = time.time()
                                progress_callback(
                                    progress,
                                    f"Processing: {processed_frames}/{total_frames} frames"
                                )
                        
                        frame_buffer = []
        
        finally:
            cap.release()
            out.release()
    
    def _denoise_clip(
        self,
        frames: list,
        tile_size: Optional[Tuple[int, int]] = None
    ) -> list:
        """
        Denoise a clip of frames using RVRT
        
        Args:
            frames: List of frames (H, W, C) in BGR format (from OpenCV)
            tile_size: Optional tile size for memory-efficient processing
            
        Returns:
            List of denoised frames in BGR format (for OpenCV)
        """
        # If no model or no weights, just return original frames (passthrough)
        if self.model is None or not self.model_has_weights:
            logger.debug("Passthrough mode - returning original frames")
            return frames
        
        # NAFNet AI Model Processing with Temporal Enhancement
        # Multiple passes + temporal averaging for stronger denoising
        logger.info(f"Processing {len(frames)} frames with NAFNet AI model")
        
        # Get configuration from processor settings
        num_passes = getattr(self, 'num_passes', 2)
        use_temporal = getattr(self, 'use_temporal', True)
        temporal_window = 2  # Frames to average (fixed at 2 for best quality/speed balance)
        
        logger.info(f"Denoising strength: {num_passes} passes, temporal={use_temporal}, window={temporal_window}")
        
        denoised_frames = frames
        
        try:
            with torch.no_grad():
                # Apply NAFNet multiple times for stronger denoising
                for pass_num in range(num_passes):
                    logger.info(f"NAFNet pass {pass_num + 1}/{num_passes}")
                    
                    # Step 1: Temporal averaging (if enabled)
                    if use_temporal and pass_num == 0:
                        logger.info("Applying temporal averaging...")
                        temporal_frames = []
                        for i in range(len(denoised_frames)):
                            # Average with neighboring frames
                            window_frames = []
                            for offset in range(-temporal_window, temporal_window + 1):
                                idx = max(0, min(len(denoised_frames) - 1, i + offset))
                                weight = 1.0 / (1.0 + abs(offset) * 0.3)
                                window_frames.append((denoised_frames[idx].astype(np.float32), weight))
                            
                            # Weighted average
                            total_weight = sum(w for _, w in window_frames)
                            averaged = sum(f * w for f, w in window_frames) / total_weight
                            temporal_frames.append(averaged.astype(np.uint8))
                        
                        denoised_frames = temporal_frames
                        logger.info("Temporal averaging complete")
                    
                    # Step 2: Apply NAFNet to each frame
                    pass_output = []
                    for i, frame in enumerate(denoised_frames):
                        # Convert BGR to RGB
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # Convert to tensor [C, H, W]
                        frame_tensor = torch.from_numpy(frame_rgb).float() / 255.0
                        frame_tensor = frame_tensor.permute(2, 0, 1)  # [C, H, W]
                        
                        # Add batch dimension [B=1, C, H, W]
                        frame_tensor = frame_tensor.unsqueeze(0).to(self.device)
                        
                        # NAFNet forward pass
                        output_tensor = self.model(frame_tensor)
                        
                        # Convert back to numpy [H, W, C]
                        output_tensor = output_tensor.squeeze(0)  # [C, H, W]
                        output_tensor = output_tensor.permute(1, 2, 0)  # [H, W, C]
                        output_np = (output_tensor.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
                        
                        # Convert RGB back to BGR
                        frame_bgr = cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)
                        pass_output.append(frame_bgr)
                        
                        # Log progress
                        if (i + 1) % 10 == 0 or (i + 1) == len(denoised_frames):
                            logger.info(f"Pass {pass_num + 1}: denoised {i + 1}/{len(denoised_frames)} frames")
                    
                    denoised_frames = pass_output
                
                logger.info(f"NAFNet processing complete: {num_passes} passes applied")
                return denoised_frames
                
        except Exception as e:
            logger.error(f"NAFNet processing failed: {e}")
            logger.exception(e)
            logger.warning("Returning original frames")
            return frames
    
    def _denoise_with_tiles(
        self,
        frames: torch.Tensor,
        tile_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Process video in tiles for memory efficiency"""
        B, T, C, H, W = frames.shape
        tile_h, tile_w = tile_size
        
        # Calculate number of tiles
        n_tiles_h = (H + tile_h - 1) // tile_h
        n_tiles_w = (W + tile_w - 1) // tile_w
        
        output = torch.zeros_like(frames)
        
        for i in range(n_tiles_h):
            for j in range(n_tiles_w):
                # Calculate tile boundaries
                y1 = i * tile_h
                y2 = min((i + 1) * tile_h, H)
                x1 = j * tile_w
                x2 = min((j + 1) * tile_w, W)
                
                # Extract tile
                tile = frames[:, :, :, y1:y2, x1:x2]
                
                # Process tile
                with torch.no_grad():
                    denoised_tile = self.model(tile)
                
                # Place back
                output[:, :, :, y1:y2, x1:x2] = denoised_tile
        
        return output
    
    def _merge_audio_video(
        self,
        video_path: str,
        audio_path: str,
        output_path: str,
        video_info: dict
    ):
        """Merge processed video with original audio"""
        try:
            bitrate = video_info.get('bitrate', 5000000) // 1000  # Convert to kbps
            
            video_stream = ffmpeg.input(video_path)
            audio_stream = ffmpeg.input(audio_path)
            
            (
                ffmpeg
                .output(
                    video_stream,
                    audio_stream,
                    output_path,
                    vcodec='libx264',
                    acodec='aac',
                    video_bitrate=f'{bitrate}k',
                    strict='experimental',
                    **{'pix_fmt': 'yuv420p'}  # Ensure correct pixel format
                )
                .overwrite_output()
                .run(quiet=True)
            )
        except Exception as e:
            logger.error(f"Failed to merge audio and video: {e}")
            raise
    
    @staticmethod
    def add_denoise_suffix(filename: str) -> str:
        """
        Add '_denoise' suffix before file extension
        
        Args:
            filename: Original filename
            
        Returns:
            Filename with _denoise suffix
        """
        path = Path(filename)
        return str(path.parent / f"{path.stem}_denoise{path.suffix}")
