import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class ProjectSettings:
    source_language: str = "ja"
    target_language: str = "vi"
    whisper_model: str = "small"
    use_gpu: bool = True
    gemini_api_key: Optional[str] = None
    use_context_aware: bool = True
    quality_check: bool = True
    video_context: Optional[str] = None

    # Speaker settings
    max_speakers: int = 10
    min_speakers: int = 1

    # Subtitle settings
    max_chars_per_line: int = 42
    max_lines: int = 2
    min_duration: float = 1.0
    max_duration: float = 7.0
    gap_between_subtitles: float = 0.1

    # Output settings
    output_formats: list[str] = field(default_factory=lambda: ["srt"])
    output_encoding: str = "utf-8"

    # Context settings
    speaker_relationships: dict[str, str] = field(default_factory=dict)

    # Runtime attributes (not saved)
    audio_path: Optional[str] = None
    transcription: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_language": self.source_language,
            "target_language": self.target_language,
            "whisper_model": self.whisper_model,
            "use_gpu": self.use_gpu,
            "gemini_api_key": "***" if self.gemini_api_key else None,
            "use_context_aware": self.use_context_aware,
            "max_speakers": self.max_speakers,
            "min_speakers": self.min_speakers,
            "max_chars_per_line": self.max_chars_per_line,
            "max_lines": self.max_lines,
            "min_duration": self.min_duration,
            "max_duration": self.max_duration,
            "gap_between_subtitles": self.gap_between_subtitles,
            "output_formats": self.output_formats,
            "output_encoding": self.output_encoding,
            "speaker_relationships": self.speaker_relationships,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProjectSettings":
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})

    @classmethod
    def from_yaml(cls, path: Path) -> "ProjectSettings":
        with open(path, encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    def save_yaml(self, path: Path):
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, allow_unicode=True)


@dataclass
class SubtitleSegment:
    index: int
    start_time: float
    end_time: float
    text: str
    speaker: Optional[str] = None
    translation: Optional[str] = None
    context: Optional[dict[str, Any]] = None
    confidence: float = 1.0

    def duration(self) -> float:
        return self.end_time - self.start_time

    def to_srt_format(self) -> str:
        return (
            f"{self.index}\n"
            f"{self._format_timestamp(self.start_time)} --> {self._format_timestamp(self.end_time)}\n"
            f"{self.text}\n"
        )

    def to_translated_srt_format(self) -> str:
        text = self.translation if self.translation else self.text
        speaker_label = f"[{self.speaker}] " if self.speaker else ""
        return (
            f"{self.index}\n"
            f"{self._format_timestamp(self.start_time)} --> {self._format_timestamp(self.end_time)}\n"
            f"{speaker_label}{text}\n"
        )

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "text": self.text,
            "speaker": self.speaker,
            "translation": self.translation,
            "context": self.context,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SubtitleSegment":
        return cls(**data)


@dataclass
class TranscriptionResult:
    segments: list[SubtitleSegment]
    language: str
    duration: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "segments": [s.to_dict() for s in self.segments],
            "language": self.language,
            "duration": self.duration,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TranscriptionResult":
        return cls(
            segments=[SubtitleSegment.from_dict(s) for s in data["segments"]],
            language=data["language"],
            duration=data["duration"],
            metadata=data.get("metadata", {}),
        )


@dataclass
class SpeakerInfo:
    speaker_id: str
    label: str
    relationship: Optional[str] = None
    segment_count: int = 0
    total_duration: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "speaker_id": self.speaker_id,
            "label": self.label,
            "relationship": self.relationship,
            "segment_count": self.segment_count,
            "total_duration": self.total_duration,
        }


class Project:
    def __init__(self, video_path: Path, settings: Optional[ProjectSettings] = None):
        self.video_path = video_path
        self.settings = settings or ProjectSettings()
        self.transcription: Optional[TranscriptionResult] = None
        self.speakers: dict[str, SpeakerInfo] = {}
        self.project_path: Optional[Path] = None

        # Create audio path
        self.audio_path = video_path.with_suffix('.wav')

    def save(self, path: Optional[Path] = None):
        if path is None:
            path = self.project_path or self.video_path.with_suffix('.sfproj')

        self.project_path = path

        data = {
            "video_path": str(self.video_path),
            "settings": self.settings.to_dict(),
            "transcription": self.transcription.to_dict() if self.transcription else None,
            "speakers": {k: v.to_dict() for k, v in self.speakers.items()},
        }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> "Project":
        with open(path, encoding='utf-8') as f:
            data = json.load(f)

        project = cls(
            video_path=Path(data["video_path"]),
            settings=ProjectSettings.from_dict(data.get("settings", {})),
        )
        project.project_path = path

        if data.get("transcription"):
            project.transcription = TranscriptionResult.from_dict(data["transcription"])

        project.speakers = {
            k: SpeakerInfo(**v) for k, v in data.get("speakers", {}).items()
        }

        return project
