"""
FastAPI Backend for Video Denoiser
Provides REST API for video upload, processing, and download
"""

import os
import asyncio
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime
import uuid
import json

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn

from .video_processor import VideoProcessor
from .model_manager import ModelManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Video Denoiser API", version="1.0.0")

# CORS middleware for web UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Global video processor
video_processor = VideoProcessor()

# Job tracking
jobs = {}


class JobStatus(BaseModel):
    """Job status model"""
    job_id: str
    status: str  # 'pending', 'processing', 'completed', 'error'
    progress: int
    message: str
    output_file: Optional[str] = None
    error: Optional[str] = None
    created_at: str
    updated_at: str


class ProcessRequest(BaseModel):
    """Video processing request model"""
    clip_size: int = Field(default=2, alias='clipSize')
    use_tiling: bool = Field(default=False, alias='useTiling')
    tile_width: Optional[int] = Field(default=512, alias='tileWidth')
    tile_height: Optional[int] = Field(default=512, alias='tileHeight')
    num_passes: int = Field(default=2, alias='numPasses')  # Number of NAFNet passes (1-3)
    use_temporal: bool = Field(default=True, alias='useTemporal')  # Max Quality mode
    model_name: Optional[str] = Field(default=None, alias='modelName')  # Selected model
    
    class Config:
        allow_population_by_field_name = True  # Allow both camelCase and snake_case


@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("Starting Video Denoiser API...")
    
    # Check model availability
    model_manager = ModelManager()
    models_status = model_manager.list_available_models()
    logger.info(f"Models status: {models_status}")


@app.get("/")
async def root():
    """Serve main HTML page"""
    html_file = Path("frontend") / "index.html"
    if html_file.exists():
        return FileResponse(html_file)
    else:
        return {
            "name": "Video Denoiser API",
            "version": "1.0.0",
            "status": "running",
            "error": "Frontend not found"
        }


@app.get("/debug")
async def debug_page():
    """Serve debug HTML page"""
    html_file = Path("frontend") / "debug.html"
    if html_file.exists():
        return FileResponse(html_file)
    else:
        return {"error": "Debug page not found"}


@app.get("/api")
async def api_root():
    """API root endpoint"""
    return {
        "name": "Video Denoiser API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    import torch
    return {
        "status": "healthy",
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    }


@app.get("/api/models")
async def list_models():
    """List available models and their status"""
    model_manager = ModelManager()
    return model_manager.list_available_models()


@app.post("/api/models/download/{model_name}")
async def download_model(model_name: str, background_tasks: BackgroundTasks):
    """
    Download a specific model
    
    Args:
        model_name: Name of the model to download
    """
    model_manager = ModelManager()
    
    # Check if model exists
    models = model_manager.list_available_models()
    if model_name not in models:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    
    # Check if already downloaded
    if models[model_name]['downloaded']:
        return {
            "status": "already_downloaded",
            "message": f"Model {model_name} is already downloaded"
        }
    
    # Start download in background
    background_tasks.add_task(download_model_task, model_name)
    
    return {
        "status": "downloading",
        "message": f"Started downloading {model_name}",
        "model_info": models[model_name]
    }


async def download_model_task(model_name: str):
    """Background task for model download"""
    try:
        logger.info(f"Starting download of model: {model_name}")
        model_manager = ModelManager()
        success = model_manager.download_model_by_name(model_name)
        
        if success:
            logger.info(f"Successfully downloaded model: {model_name}")
        else:
            logger.error(f"Failed to download model: {model_name}")
            
    except Exception as e:
        logger.error(f"Error downloading model {model_name}: {e}")


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """
    Upload video file
    
    Returns job_id for tracking
    """
    try:
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Save uploaded file
        file_ext = Path(file.filename).suffix
        upload_path = UPLOAD_DIR / f"{job_id}{file_ext}"
        
        with open(upload_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Get video info
        try:
            video_info = video_processor.get_video_info(str(upload_path))
        except Exception as e:
            upload_path.unlink()
            raise HTTPException(status_code=400, detail=f"Invalid video file: {str(e)}")
        
        # Create job
        jobs[job_id] = {
            "job_id": job_id,
            "status": "uploaded",
            "progress": 0,
            "message": "Video uploaded successfully",
            "input_file": str(upload_path),
            "output_file": None,
            "error": None,
            "cancelled": False,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "video_info": video_info,
            "original_filename": file.filename
        }
        
        logger.info(f"Video uploaded: {job_id} - {file.filename}")
        
        return {"job_id": job_id, "video_info": video_info}
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/process/{job_id}")
async def process_video(
    job_id: str,
    request: ProcessRequest,
    background_tasks: BackgroundTasks
):
    """
    Start video processing
    
    Processing happens in background
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    if job["status"] == "processing":
        raise HTTPException(status_code=400, detail="Job already processing")
    
    if job["status"] == "completed":
        raise HTTPException(status_code=400, detail="Job already completed")
    
    # Log received parameters
    logger.info(f"Processing request for job {job_id}: "
                f"model={request.model_name}, num_passes={request.num_passes}, "
                f"use_temporal={request.use_temporal}, clip_size={request.clip_size}")
    
    # Update job status
    job["status"] = "processing"
    job["progress"] = 0
    job["message"] = "Starting processing..."
    job["updated_at"] = datetime.now().isoformat()
    
    # Start processing in background
    background_tasks.add_task(
        process_video_task,
        job_id,
        request.clip_size,
        (request.tile_width, request.tile_height) if request.use_tiling else None,
        request.num_passes,
        request.use_temporal,
        request.model_name
    )
    
    return {"job_id": job_id, "status": "processing"}


async def process_video_task(
    job_id: str,
    clip_size: int,
    tile_size: Optional[tuple],
    num_passes: int,
    use_temporal: bool,
    model_name: Optional[str]
):
    """Background task for video processing"""
    job = jobs[job_id]
    
    try:
        input_path = job["input_file"]
        original_filename = job["original_filename"]
        
        # Generate output filename with _denoise suffix
        output_filename = VideoProcessor.add_denoise_suffix(original_filename)
        output_path = OUTPUT_DIR / output_filename
        
        # Progress callback
        def update_progress(progress: int, message: str):
            # Check if cancelled
            if job.get("cancelled", False):
                raise Exception("Processing cancelled by user")
            job["progress"] = progress
            job["message"] = message
            job["updated_at"] = datetime.now().isoformat()
            logger.info(f"Job {job_id}: {progress}% - {message}")
        
        # Process video
        update_progress(5, "Loading model...")
        
        # Set preferred model if specified
        if model_name:
            video_processor.preferred_model = model_name
            logger.info(f"Using preferred model: {model_name}")
        
        result_path = video_processor.process_video(
            input_path=input_path,
            output_path=str(output_path),
            clip_size=clip_size,
            tile_size=tile_size,
            num_passes=num_passes,
            use_temporal=use_temporal,
            progress_callback=update_progress
        )
        
        # Update job status
        job["status"] = "completed"
        job["progress"] = 100
        job["message"] = "Processing completed successfully"
        job["output_file"] = str(output_path)
        job["updated_at"] = datetime.now().isoformat()
        
        logger.info(f"Job {job_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        job["status"] = "error"
        job["error"] = str(e)
        job["message"] = f"Processing failed: {str(e)}"
        job["updated_at"] = datetime.now().isoformat()


@app.post("/api/cancel/{job_id}")
async def cancel_job(job_id: str):
    """Cancel processing job"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    if job["status"] != "processing":
        raise HTTPException(status_code=400, detail="Job is not processing")
    
    # Set cancellation flag
    job["cancelled"] = True
    job["status"] = "cancelled"
    job["message"] = "Processing cancelled by user"
    job["updated_at"] = datetime.now().isoformat()
    
    logger.info(f"Job {job_id} cancelled by user")
    
    return {"job_id": job_id, "status": "cancelled"}


@app.get("/api/status/{job_id}")
async def get_job_status(job_id: str):
    """Get job status and progress"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "progress": job["progress"],
        "message": job["message"],
        "output_file": Path(job["output_file"]).name if job["output_file"] else None,
        "error": job.get("error"),
        "created_at": job["created_at"],
        "updated_at": job["updated_at"],
        "video_info": job.get("video_info")
    }


@app.get("/api/download/{job_id}")
async def download_result(job_id: str):
    """Download processed video"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed")
    
    output_file = job["output_file"]
    if not output_file or not Path(output_file).exists():
        raise HTTPException(status_code=404, detail="Output file not found")
    
    return FileResponse(
        output_file,
        media_type="video/mp4",
        filename=Path(output_file).name
    )


@app.delete("/api/job/{job_id}")
async def delete_job(job_id: str):
    """Delete job and cleanup files"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    # Delete files
    if job["input_file"] and Path(job["input_file"]).exists():
        Path(job["input_file"]).unlink()
    
    if job["output_file"] and Path(job["output_file"]).exists():
        Path(job["output_file"]).unlink()
    
    # Remove job
    del jobs[job_id]
    
    return {"message": "Job deleted successfully"}


@app.get("/api/jobs")
async def list_jobs():
    """List all jobs"""
    return [
        {
            "job_id": job["job_id"],
            "status": job["status"],
            "progress": job["progress"],
            "original_filename": job.get("original_filename"),
            "created_at": job["created_at"],
        }
        for job in jobs.values()
    ]


# Static files are served directly by routes above
# No need to mount since we're serving index.html from root endpoint


def start_server(host: str = "127.0.0.1", port: int = 8000):
    """Start the FastAPI server"""
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    start_server()
