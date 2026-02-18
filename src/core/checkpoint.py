"""
Checkpoint Manager for SubtitleForge Pro.

Saves and restores processing state between runs, allowing resume
from the last completed stage instead of restarting from scratch.

Checkpoint Stages:
    1. audio_extracted  — Audio path + duration saved
    2. transcribed      — Full TranscriptionResult saved
    3. translated       — All segment translations saved
    4. quality_checked  — QA results saved
    5. completed        — Final, checkpoint auto-deleted

Translation sub-checkpoints:
    Saves after every N translated segments to allow mid-translation resume.
"""
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# Processing stages in order
STAGES = [
    "audio_extracted",
    "transcribed",
    "translated",
    "quality_checked",
    "completed",
]

# Stage display names for UI
STAGE_LABELS = {
    "audio_extracted": "Trích xuất audio",
    "transcribed": "Chuyển speech → text",
    "translated": "Dịch subtitle",
    "quality_checked": "Kiểm tra chất lượng",
    "completed": "Hoàn tất",
}


@dataclass
class CheckpointData:
    """Serializable checkpoint state."""

    # Metadata
    video_path: str = ""
    video_name: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0

    # Current completed stage
    completed_stage: str = ""

    # Stage 1: Audio extraction
    audio_path: Optional[str] = None
    audio_duration: float = 0.0

    # Stage 2: Transcription
    transcription: Optional[dict[str, Any]] = None

    # Stage 3: Translation (batch-level sub-checkpoints)
    translations: list[Optional[str]] = field(default_factory=list)
    translated_count: int = 0

    # Stage 4: Quality check
    quality_results: Optional[list[dict[str, Any]]] = None

    # Settings used (for validation on resume)
    settings_hash: str = ""


class CheckpointManager:
    """
    Manages checkpoint files for subtitle processing pipeline.

    Usage:
        manager = CheckpointManager(video_path)

        # Check for existing checkpoint
        if manager.has_checkpoint():
            data = manager.load()
            # Resume from data.completed_stage

        # Save after each stage
        manager.save_stage("audio_extracted", {"audio_path": ..., "duration": ...})
        manager.save_stage("transcribed", {"transcription": result.to_dict()})

        # Save translation progress (batch-level)
        manager.save_translation_progress(translations_so_far, total_count)

        # Clean up on completion
        manager.cleanup()
    """

    CHECKPOINT_SUFFIX = ".checkpoint.json"
    TRANSLATION_BATCH_SIZE = 50  # Save every N translations

    def __init__(self, video_path: Path):
        self.video_path = Path(video_path)
        self.checkpoint_path = self._get_checkpoint_path()

    def _get_checkpoint_path(self) -> Path:
        """Generate checkpoint file path next to the video file."""
        return self.video_path.parent / f"{self.video_path.stem}{self.CHECKPOINT_SUFFIX}"

    def _compute_settings_hash(self, settings: Any) -> str:
        """
        Compute a lightweight hash of relevant settings.

        Only includes settings that affect the output.
        If settings change, checkpoint is invalidated.
        """
        import hashlib

        relevant = {
            "source_language": getattr(settings, "source_language", ""),
            "target_language": getattr(settings, "target_language", ""),
            "whisper_model": getattr(settings, "whisper_model", ""),
            "use_context_aware": getattr(settings, "use_context_aware", True),
            "quality_check": getattr(settings, "quality_check", True),
        }
        raw = json.dumps(relevant, sort_keys=True)
        return hashlib.md5(raw.encode()).hexdigest()[:12]

    def has_checkpoint(self) -> bool:
        """Check if a checkpoint file exists for this video."""
        return self.checkpoint_path.exists()

    def load(self) -> Optional[CheckpointData]:
        """
        Load checkpoint data from file.

        Returns:
            CheckpointData if valid, None if file is corrupt/invalid.
        """
        if not self.checkpoint_path.exists():
            return None

        try:
            with open(self.checkpoint_path, encoding="utf-8") as f:
                raw = json.load(f)

            data = CheckpointData(
                video_path=raw.get("video_path", ""),
                video_name=raw.get("video_name", ""),
                created_at=raw.get("created_at", 0.0),
                updated_at=raw.get("updated_at", 0.0),
                completed_stage=raw.get("completed_stage", ""),
                audio_path=raw.get("audio_path"),
                audio_duration=raw.get("audio_duration", 0.0),
                transcription=raw.get("transcription"),
                translations=raw.get("translations", []),
                translated_count=raw.get("translated_count", 0),
                quality_results=raw.get("quality_results"),
                settings_hash=raw.get("settings_hash", ""),
            )

            # Validate video path matches
            if data.video_path != str(self.video_path):
                logger.warning("Checkpoint video path mismatch, ignoring")
                return None

            logger.info(
                f"Checkpoint loaded: stage={data.completed_stage}, "
                f"translations={data.translated_count}/{len(data.translations)}"
            )
            return data

        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Corrupt checkpoint file: {e}")
            return None

    def _save(self, data: CheckpointData) -> None:
        """Write checkpoint data to file atomically."""
        data.updated_at = time.time()
        data.video_path = str(self.video_path)
        data.video_name = self.video_path.name

        # Write to temp file first, then rename (atomic on most OS)
        temp_path = self.checkpoint_path.with_suffix(".tmp")
        try:
            raw = {
                "video_path": data.video_path,
                "video_name": data.video_name,
                "created_at": data.created_at,
                "updated_at": data.updated_at,
                "completed_stage": data.completed_stage,
                "audio_path": data.audio_path,
                "audio_duration": data.audio_duration,
                "transcription": data.transcription,
                "translations": data.translations,
                "translated_count": data.translated_count,
                "quality_results": data.quality_results,
                "settings_hash": data.settings_hash,
            }

            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(raw, f, ensure_ascii=False, indent=2)

            # Atomic rename
            temp_path.replace(self.checkpoint_path)
            logger.debug(f"Checkpoint saved: stage={data.completed_stage}")

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            # Clean up temp file
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)

    def create(self, settings: Any) -> CheckpointData:
        """Create a new checkpoint for this video."""
        data = CheckpointData(
            video_path=str(self.video_path),
            video_name=self.video_path.name,
            created_at=time.time(),
            updated_at=time.time(),
            settings_hash=self._compute_settings_hash(settings),
        )
        self._save(data)
        return data

    def save_audio_extracted(
        self, data: CheckpointData, audio_path: str, duration: float
    ) -> None:
        """Save checkpoint after audio extraction."""
        data.completed_stage = "audio_extracted"
        data.audio_path = audio_path
        data.audio_duration = duration
        self._save(data)
        logger.info("Checkpoint: audio_extracted ✓")

    def save_transcribed(
        self, data: CheckpointData, transcription_dict: dict[str, Any]
    ) -> None:
        """Save checkpoint after transcription."""
        data.completed_stage = "transcribed"
        data.transcription = transcription_dict
        # Pre-allocate translation slots
        segment_count = len(transcription_dict.get("segments", []))
        data.translations = [None] * segment_count
        data.translated_count = 0
        self._save(data)
        logger.info("Checkpoint: transcribed ✓")

    def save_translation_progress(
        self, data: CheckpointData, translations: list[Optional[str]], count: int
    ) -> None:
        """
        Save partial translation progress (batch-level checkpoint).

        Called after every TRANSLATION_BATCH_SIZE translations.
        """
        data.translations = translations
        data.translated_count = count
        # Don't update completed_stage yet — translation is still in progress
        self._save(data)
        logger.debug(f"Checkpoint: translation progress {count}/{len(translations)}")

    def save_translated(self, data: CheckpointData, translations: list[str]) -> None:
        """Save checkpoint after all translations complete."""
        data.completed_stage = "translated"
        data.translations = translations
        data.translated_count = len(translations)
        self._save(data)
        logger.info("Checkpoint: translated ✓")

    def save_quality_checked(
        self, data: CheckpointData, quality_results: list[dict[str, Any]]
    ) -> None:
        """Save checkpoint after quality check."""
        data.completed_stage = "quality_checked"
        data.quality_results = quality_results
        self._save(data)
        logger.info("Checkpoint: quality_checked ✓")

    def cleanup(self) -> None:
        """Delete checkpoint file after successful completion."""
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()
            logger.info("Checkpoint cleaned up (processing complete)")

    def is_settings_compatible(self, data: CheckpointData, settings: Any) -> bool:
        """Check if saved checkpoint is compatible with current settings."""
        current_hash = self._compute_settings_hash(settings)
        return data.settings_hash == current_hash

    def get_next_stage(self, data: CheckpointData) -> Optional[str]:
        """
        Get the next stage to execute after the completed stage.

        Returns:
            Next stage name, or None if all stages are complete.
        """
        if not data.completed_stage:
            return STAGES[0]

        try:
            idx = STAGES.index(data.completed_stage)
            if idx + 1 < len(STAGES):
                return STAGES[idx + 1]
            return None
        except ValueError:
            return STAGES[0]

    def get_resume_info(self, data: CheckpointData) -> str:
        """Get human-readable resume information."""
        stage_label = STAGE_LABELS.get(data.completed_stage, data.completed_stage)
        next_stage = self.get_next_stage(data)
        next_label = STAGE_LABELS.get(next_stage, next_stage) if next_stage else "N/A"

        # Check for partial translation
        partial_info = ""
        if (
            data.completed_stage == "transcribed"
            and data.translated_count > 0
            and len(data.translations) > 0
        ):
            partial_info = f"\n• Đã dịch: {data.translated_count}/{len(data.translations)} segments"

        import datetime

        updated = datetime.datetime.fromtimestamp(data.updated_at)
        time_str = updated.strftime("%H:%M:%S %d/%m/%Y")

        info = (
            f"• Video: {data.video_name}\n"
            f"• Đã hoàn thành: {stage_label}\n"
            f"• Tiếp tục từ: {next_label}"
            f"{partial_info}\n"
            f"• Lần cuối: {time_str}"
        )

        if data.transcription:
            seg_count = len(data.transcription.get("segments", []))
            info += f"\n• Segments: {seg_count}"

        return info
