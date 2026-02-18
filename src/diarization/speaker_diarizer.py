import gc
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class DiarizationConfig:
    """
    Configuration for speaker diarization.

    VRAM Optimization:
    - Uses efficient embedding models
    - Processes in chunks for long audio
    """
    min_speakers: int = 1
    max_speakers: int = 10
    embedding_model: str = "pyannote/embedding"
    segmentation_model: str = "pyannote/segmentation-3.0"

    # VRAM optimization
    max_memory_vram_gb: float = 6.0
    use_gpu: bool = True
    chunk_duration: float = 300.0  # Process in 5-minute chunks


class SpeakerDiarizer:
    """
    Speaker diarization using PyAnnote.

    Optimized for Japanese video content with multiple speakers.
    """

    def __init__(self, config: Optional[DiarizationConfig] = None):
        self.config = config or DiarizationConfig()
        self.pipeline = None
        self._load_pipeline()

    def _load_pipeline(self):
        """Load the speaker diarization pipeline with VRAM optimization."""

        # Check GPU availability and VRAM
        gpu_available = False
        vram_gb = 0

        try:
            import torch
            if torch.cuda.is_available():
                gpu_available = True
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                logger.info(f"Detected {vram_gb:.1f}GB VRAM for diarization")
        except Exception as e:
            logger.warning(f"Could not detect GPU: {e}")

        # Adjust based on available VRAM
        if not gpu_available or vram_gb < 3:
            self.config.use_gpu = False
            logger.info("Using CPU for diarization (insufficient VRAM)")

        try:
            from pyannote.audio import Pipeline

            logger.info("Loading speaker diarization pipeline...")

            # Read HuggingFace token from environment
            hf_token = os.environ.get("HF_TOKEN", os.environ.get("HF_AUTH_TOKEN"))

            if not hf_token:
                logger.warning("HF_TOKEN not set - speaker diarization may fail. Set HF_TOKEN environment variable.")

            try:
                self.pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=hf_token
                )
            except Exception as e:
                logger.warning(f"Could not load pyannote pipeline: {e}")
                self.pipeline = None
                return

            # Move to GPU if available and VRAM sufficient
            if self.config.use_gpu and gpu_available and vram_gb >= 3:
                try:
                    self.pipeline = self.pipeline.to(torch.device("cuda"))
                    logger.info("Diarization pipeline running on GPU")
                except Exception as e:
                    logger.warning(f"Could not use GPU for diarization: {e}")
                    self.config.use_gpu = False
            else:
                logger.info("Diarization pipeline running on CPU")

            # Log VRAM usage
            if self.config.use_gpu:
                try:
                    allocated = torch.cuda.memory_allocated() / (1024**3)
                    logger.info(f"Diarization VRAM allocated: {allocated:.2f}GB")
                except Exception:
                    pass

            logger.info("Diarization pipeline loaded successfully")

        except ImportError as e:
            logger.error(f"PyAnnote not available: {e}")
            logger.info("Speaker diarization will be skipped")
            self.pipeline = None
        except Exception as e:
            logger.error(f"Failed to load diarization pipeline: {e}")
            self.pipeline = None

    def diarize(self, audio_path: Path) -> dict[str, Any]:
        """
        Perform speaker diarization on audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            Dictionary with diarization results
        """
        if not self.pipeline:
            logger.warning("Diarization pipeline not loaded, returning empty result")
            return {
                "segments": [],
                "speakers": [],
                "num_speakers": 0,
                "error": "Pipeline not available"
            }

        logger.info(f"Starting speaker diarization for {audio_path}")

        try:
            # Run diarization
            diarization = self.pipeline(
                str(audio_path),
                min_speakers=self.config.min_speakers,
                max_speakers=self.config.max_speakers
            )

            # Convert to dictionary format
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append({
                    "start": turn.start,
                    "end": turn.end,
                    "speaker": speaker
                })

            # Get unique speakers
            speakers = list({seg["speaker"] for seg in segments})
            speakers.sort()

            result = {
                "segments": segments,
                "speakers": speakers,
                "num_speakers": len(speakers)
            }

            logger.info(f"Diarization complete: {len(speakers)} speakers identified")

            return result

        except Exception as e:
            logger.error(f"Diarization failed: {e}")
            return {
                "segments": [],
                "speakers": [],
                "num_speakers": 0,
                "error": str(e)
            }

    def get_speaker_segments(
        self,
        diarization_result: dict[str, Any],
        speaker: str
    ) -> list[dict[str, float]]:
        """Get all segments for a specific speaker."""
        return [
            {"start": seg["start"], "end": seg["end"]}
            for seg in diarization_result.get("segments", [])
            if seg["speaker"] == speaker
        ]

    def merge_close_segments(
        self,
        diarization_result: dict[str, Any],
        gap_threshold: float = 0.5
    ) -> dict[str, Any]:
        """Merge segments from the same speaker that are close together."""
        segments = diarization_result.get("segments", [])

        if not segments:
            return diarization_result

        merged = []
        current = None

        for seg in segments:
            if current is None:
                current = seg.copy()
            elif (seg["speaker"] == current["speaker"] and
                  seg["start"] - current["end"] < gap_threshold):
                current["end"] = seg["end"]
            else:
                merged.append(current)
                current = seg.copy()

        if current:
            merged.append(current)

        # Recalculate speakers
        speakers = list({s["speaker"] for s in merged})
        speakers.sort()

        return {
            "segments": merged,
            "speakers": speakers,
            "num_speakers": len(speakers)
        }

    def cleanup(self):
        """Free VRAM by unloading pipeline."""
        if self.pipeline:
            del self.pipeline
            self.pipeline = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        logger.info("Diarization pipeline unloaded, VRAM freed")


class SimpleDiarizer:
    """Simple speaker diarization fallback without pyannote."""

    def __init__(self):
        self.speaker_count = 0

    def diarize(self, audio_path: Path) -> dict[str, Any]:
        """
        Simple diarization - assigns segments to alternating speakers.
        This is a fallback when pyannote is not available.
        """
        logger.warning("Using simple diarization (pyannote not available)")

        return {
            "segments": [],
            "speakers": [],
            "num_speakers": 0,
            "error": "Simple diarization not available - install pyannote.audio"
        }
