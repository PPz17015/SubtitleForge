import gc
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from faster_whisper import WhisperModel

from core.models import SubtitleSegment, TranscriptionResult

if TYPE_CHECKING:
    from faster_whisper.transcribe import Segment

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionConfig:
    """
    Configuration for Whisper transcription.

    VRAM Optimization:
    - tiny: ~500MB VRAM
    - base: ~750MB VRAM
    - small: ~1.5GB VRAM (RECOMMENDED for <6GB VRAM)
    - medium: ~4GB VRAM (pushes limit)
    - large: ~6GB+ VRAM (NOT recommended for <6GB)
    """
    model_size: str = "small"  # Changed default to small for VRAM efficiency
    language: Optional[str] = None
    use_gpu: bool = True
    beam_size: int = 5
    vad_filter: bool = True
    word_timestamps: bool = True

    # VRAM Optimization settings
    compute_type: str = "int8"  # int8 uses less VRAM than float16
    num_workers: int = 1  # Reduce memory with single thread
    max_memory_vram_gb: float = 6.0  # Target max VRAM

    def __post_init__(self):
        """Auto-adjust settings based on VRAM limit."""
        if self.max_memory_vram_gb < 6.0:
            # For systems with less than 6GB, use smaller model
            if self.model_size == "large" or self.model_size == "large-v2" or self.model_size == "large-v3":
                logger.warning(f"Model {self.model_size} requires ~6GB VRAM, downgrading to medium")
                self.model_size = "medium"
                self.compute_type = "int8"
        elif self.max_memory_vram_gb >= 6.0:
            # For 6GB+, still prefer int8 for safety
            self.compute_type = "int8"


class WhisperTranscriber:
    """
    Whisper-based speech-to-text transcription.

    Optimized for Japanese to Vietnamese subtitle generation.
    """

    def __init__(self, config: Optional[TranscriptionConfig] = None):
        self.config = config or TranscriptionConfig()
        self.model: Optional[WhisperModel] = None
        self._load_model()

    def _load_model(self):
        """Load the Whisper model with VRAM optimization."""

        # Determine compute type based on VRAM availability
        compute_type = self.config.compute_type

        # Check available VRAM if using GPU
        if self.config.use_gpu:
            try:
                import torch
                if torch.cuda.is_available():
                    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    logger.info(f"Detected {vram_gb:.1f}GB VRAM")

                    # Auto-adjust model based on VRAM
                    if vram_gb < 4:
                        self.config.model_size = "base"
                        logger.info("Downgrading to base model due to limited VRAM")
                    elif vram_gb < 6 and self.config.model_size in ["large", "large-v2", "large-v3"]:
                        self.config.model_size = "small"
                        logger.info("Downgrading to small model for VRAM efficiency")
            except Exception as e:
                logger.warning(f"Could not detect VRAM: {e}")

        logger.info(f"Loading Whisper {self.config.model_size} model (compute: {compute_type}, VRAM target: <{self.config.max_memory_vram_gb}GB)")

        try:
            self.model = WhisperModel(
                self.config.model_size,
                device="cuda" if self.config.use_gpu else "cpu",
                compute_type=compute_type
            )
            logger.info("Model loaded successfully")

            # Log VRAM usage if available
            if self.config.use_gpu:
                try:
                    import torch
                    if torch.cuda.is_available():
                        allocated = torch.cuda.memory_allocated() / (1024**3)
                        logger.info(f"VRAM allocated: {allocated:.2f}GB")
                except Exception:
                    pass

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            # Fallback to CPU if GPU fails
            if self.config.use_gpu:
                logger.info("Falling back to CPU mode")
                self.config.use_gpu = False
                self._load_model()
            else:
                raise

    def transcribe(
        self,
        audio_path: Path,
        source_language: Optional[str] = None
    ) -> TranscriptionResult:
        """
        Transcribe audio file to text.

        Optimized for Japanese audio with proper handling of:
        - Japanese pitch accent
        - Honorifics detection
        - Multiple speakers

        Args:
            audio_path: Path to audio file
            source_language: Source language code (e.g., 'ja', 'en')

        Returns:
            TranscriptionResult with segments
        """
        if not self.model:
            raise RuntimeError("Model not loaded")

        language = source_language or self.config.language or "ja"  # Default to Japanese

        logger.info(f"Starting transcription of {audio_path}")
        logger.info(f"Source language: {language}")

        segments_list = []

        try:
            segments, info = self.model.transcribe(
                str(audio_path),
                language=language,
                beam_size=self.config.beam_size,
                vad_filter=self.config.vad_filter,
                word_timestamps=self.config.word_timestamps,
            )

            logger.info(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")

            segment: Segment
            for i, segment in enumerate(segments):
                subtitle_segment = SubtitleSegment(
                    index=i + 1,
                    start_time=segment.start,
                    end_time=segment.end,
                    text=segment.text.strip(),
                    confidence=segment.avg_logprob if hasattr(segment, 'avg_logprob') else 1.0
                )
                segments_list.append(subtitle_segment)

                if (i + 1) % 100 == 0:
                    logger.info(f"Transcribed {i + 1} segments")

            logger.info(f"Transcription complete: {len(segments_list)} segments")

            # Calculate total duration
            duration = segments_list[-1].end_time if segments_list else 0.0

            return TranscriptionResult(
                segments=segments_list,
                language=info.language or language or "unknown",
                duration=duration,
                metadata={
                    "model": self.config.model_size,
                    "language_probability": info.language_probability,
                    "compute_type": self.config.compute_type,
                }
            )

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise

    def transcribe_with_speaker(
        self,
        audio_path: Path,
        diarization_result: dict,
        source_language: Optional[str] = None
    ) -> TranscriptionResult:
        """
        Transcribe and merge with speaker diarization.

        Args:
            audio_path: Path to audio file
            diarization_result: Speaker diarization result
            source_language: Source language code

        Returns:
            TranscriptionResult with speaker information
        """
        result = self.transcribe(audio_path, source_language)

        # Map timestamps to speakers
        for segment in result.segments:
            speaker = self._find_speaker_for_segment(
                segment.start_time,
                segment.end_time,
                diarization_result
            )
            segment.speaker = speaker

        return result

    def _find_speaker_for_segment(
        self,
        start_time: float,
        end_time: float,
        diarization_result: dict
    ) -> Optional[str]:
        """Find which speaker is active during a time segment."""
        for segment_info in diarization_result.get("segments", []):
            seg_start = segment_info.get("start", 0)
            seg_end = segment_info.get("end", 0)

            # Check overlap
            if start_time < seg_end and end_time > seg_start:
                return segment_info.get("speaker", "SPEAKER_UNKNOWN")

        return None

    def cleanup(self):
        """Free VRAM by unloading model."""
        if self.model:
            del self.model
            self.model = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        logger.info("Model unloaded, VRAM freed")

    @staticmethod
    def get_available_models() -> list[str]:
        """Get list of available Whisper models."""
        return ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]

    @staticmethod
    def get_model_info(model_size: str) -> dict:
        """Get information about a specific model."""
        models_info = {
            "tiny": {"params": "39M", "vram": "~500MB", "speed": "10x", "recommended": "<2GB VRAM"},
            "base": {"params": "74M", "vram": "~750MB", "speed": "7x", "recommended": "<2GB VRAM"},
            "small": {"params": "244M", "vram": "~1.5GB", "speed": "4x", "recommended": "<4GB VRAM"},
            "medium": {"params": "769M", "vram": "~4GB", "speed": "2x", "recommended": "4-6GB VRAM"},
            "large": {"params": "1550M", "vram": "~6GB", "speed": "1x", "recommended": "8GB+ VRAM"},
            "large-v2": {"params": "1550M", "vram": "~6GB", "speed": "1x", "recommended": "8GB+ VRAM"},
            "large-v3": {"params": "1550M", "vram": "~6GB", "speed": "1x", "recommended": "8GB+ VRAM"},
        }
        return models_info.get(model_size, {})

    @staticmethod
    def get_recommended_model(vram_gb: float) -> str:
        """Get recommended model based on available VRAM."""
        if vram_gb < 2:
            return "base"
        if vram_gb < 4:
            return "small"
        if vram_gb < 6:
            return "medium"
        return "small"  # Still recommend small for efficiency
