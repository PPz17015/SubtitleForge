from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ProcessingOptions(BaseModel):
    context_aware: bool = True
    speaker_diarization: bool = True
    quality_check: bool = True
    whisper_model: str = "small"
    use_gpu: bool = True


class SubtitleRequest(BaseModel):
    video_url: Optional[str] = None
    source_language: str = "ja"
    target_language: str = "vi"
    options: ProcessingOptions = Field(default_factory=ProcessingOptions)


class JobCreate(BaseModel):
    video_path: Optional[str] = None
    source_language: str = "ja"
    target_language: str = "vi"
    options: ProcessingOptions = Field(default_factory=ProcessingOptions)
    gemini_api_key: Optional[str] = None
    video_context: Optional[str] = None


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    created_at: datetime
    progress: int = 0
    message: str = ""
    result_path: Optional[str] = None
    error: Optional[str] = None


class JobProgress(BaseModel):
    job_id: str
    progress: int
    message: str
    stage: str


class HealthResponse(BaseModel):
    status: str
    version: str
    gpu_available: bool


class SettingsResponse(BaseModel):
    api_keys_configured: bool
    default_source_lang: str
    default_target_lang: str
    max_file_size_mb: int
