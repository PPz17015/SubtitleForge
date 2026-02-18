import logging
import os
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from .job_manager import job_manager
from .models import HealthResponse, JobCreate, JobResponse, JobStatus, SettingsResponse
from .websocket import manager

logger = logging.getLogger(__name__)

router = APIRouter()


def verify_api_key(x_api_key: Optional[str] = None):
    expected_key = os.getenv("API_KEY")
    if not expected_key:
        # No API key configured on server, allow all requests
        return x_api_key or ""
    if not x_api_key or x_api_key != expected_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return x_api_key


@router.get("/health", response_model=HealthResponse)
async def health_check():
    gpu_available = False
    try:
        import torch
        gpu_available = torch.cuda.is_available()
    except ImportError:
        pass

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        gpu_available=gpu_available
    )


@router.get("/settings", response_model=SettingsResponse)
async def get_settings():
    return SettingsResponse(
        api_keys_configured=bool(os.getenv("GEMINI_API_KEY")),
        default_source_lang="ja",
        default_target_lang="vi",
        max_file_size_mb=20480
    )


@router.post("/subtitle", response_model=JobResponse)
async def create_subtitle_job(
    request: JobCreate,
    api_key: str = Depends(verify_api_key)
):
    job = job_manager.create_job(request)

    import asyncio
    asyncio.create_task(job_manager.process_job(job.job_id, request))

    return job


@router.get("/subtitle", response_model=list[JobResponse])
async def list_jobs():
    return job_manager.list_jobs()


@router.get("/subtitle/{job_id}", response_model=JobResponse)
async def get_job_status(job_id: str):
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.delete("/subtitle/{job_id}")
async def delete_job(job_id: str, api_key: str = Depends(verify_api_key)):
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status == JobStatus.PROCESSING:
        raise HTTPException(status_code=400, detail="Cannot delete running job")

    del job_manager.jobs[job_id]
    return {"message": "Job deleted"}


@router.get("/subtitle/{job_id}/download")
async def download_result(job_id: str, output_format: str = "srt"):
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status != JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Job not completed")

    if not job.result_path:
        raise HTTPException(status_code=404, detail="No result file")

    result_path = Path(job.result_path)
    if not result_path.exists():
        raise HTTPException(status_code=404, detail="Result file not found")

    media_type = {
        "srt": "application/x-subrip",
        "vtt": "text/vtt",
        "ass": "text/x-ass"
    }.get(output_format, "application/octet-stream")

    return FileResponse(
        result_path,
        media_type=media_type,
        filename=f"{result_path.stem}.{output_format}"
    )


@router.post("/upload")
async def upload_video(file: UploadFile = File(...), api_key: str = Depends(verify_api_key)):
    # Validate file has a name
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    # Validate file type
    allowed_extensions = {'.mp4', '.mkv', '.avi', '.mov', '.webm', '.wmv'}
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {allowed_extensions}"
        )

    # Validate filename (prevent path traversal)
    safe_filename = Path(file.filename).name

    # Limit file size (20GB)
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)

    file_path = upload_dir / safe_filename

    # Write with size limit check
    max_size = 20 * (1024 ** 3)  # 20GB
    written = 0

    with open(file_path, "wb") as buffer:
        while chunk := await file.read(8192):
            written += len(chunk)
            if written > max_size:
                file_path.unlink(missing_ok=True)
                raise HTTPException(status_code=400, detail="File too large (max 20GB)")
            buffer.write(chunk)

    return {"filename": str(file_path), "size": file_path.stat().st_size}


@router.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    await manager.connect(websocket, job_id)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket, job_id)
