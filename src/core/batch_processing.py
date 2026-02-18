import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class BatchItem:
    """Single item in batch processing."""
    video_path: Path
    output_dir: Optional[Path] = None
    status: str = "pending"  # pending, processing, completed, failed
    error: Optional[str] = None
    result: Optional[Any] = None

    def __post_init__(self):
        if self.output_dir is None:
            self.output_dir = self.video_path.parent


@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    max_concurrent: int = 2  # Max videos to process simultaneously
    stop_on_error: bool = False
    output_base_dir: Optional[Path] = None
    preserve_structure: bool = True  # Maintain folder structure

    # Processing settings (passed to each job)
    source_lang: str = "ja"
    target_lang: str = "vi"
    whisper_model: str = "small"
    use_gpu: bool = True
    gemini_api_key: Optional[str] = None
    context_aware: bool = True
    video_context: Optional[str] = None


class BatchProcessor:
    """
    Batch processor for multiple videos.

    Handles:
    - Multiple video processing
    - Progress tracking
    - Error handling
    - Parallel processing
    """

    def __init__(self, config: Optional[BatchConfig] = None):
        self.config = config or BatchConfig()
        self.items: list[BatchItem] = []
        self._lock = threading.Lock()
        self._progress_callback: Optional[Callable] = None

    def add_video(self, video_path: Path, output_dir: Optional[Path] = None):
        """Add a video to the batch."""
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        item = BatchItem(
            video_path=video_path,
            output_dir=output_dir or self.config.output_base_dir or video_path.parent
        )
        self.items.append(item)
        logger.info(f"Added to batch: {video_path.name}")

    def add_videos_from_folder(self, folder: Path, pattern: str = "*.mp4"):
        """Add all videos from a folder."""
        videos = list(folder.rglob(pattern)) if self.config.preserve_structure else list(folder.glob(pattern))

        for video in videos:
            output_dir = None
            if self.config.preserve_structure:
                rel_path = video.relative_to(folder)
                output_dir = self.config.output_base_dir / rel_path.parent if self.config.output_base_dir else video.parent

            self.add_video(video, output_dir)

        logger.info(f"Added {len(videos)} videos from {folder}")

    def set_progress_callback(self, callback: Callable[[int, int, str], None]):
        """Set callback for progress updates."""
        self._progress_callback = callback

    def _process_single(self, item: BatchItem) -> BatchItem:
        """Process a single video."""
        item.status = "processing"

        try:
            # Import here to avoid circular imports
            from core.audio_extractor import AudioExtractor
            from core.models import Project, ProjectSettings
            from core.subtitle_generator import SubtitleGenerator, SubtitleOptimizer
            from transcription.whisper_transcriber import TranscriptionConfig, WhisperTranscriber
            from translation.translator import (
                ContextAnalyzer,
                TranslationConfig,
                TranslationEngine,
                TranslationQualityChecker,
            )

            logger.info(f"Processing: {item.video_path.name}")

            # Create settings
            settings = ProjectSettings(
                source_language=self.config.source_lang,
                target_language=self.config.target_lang,
                whisper_model=self.config.whisper_model,
                use_gpu=self.config.use_gpu,
                gemini_api_key=self.config.gemini_api_key,
                use_context_aware=self.config.context_aware
            )

            project = Project(item.video_path, settings)

            # Step 1: Extract audio
            extractor = AudioExtractor()
            audio_path, duration = extractor.extract_audio(item.video_path)
            project.audio_path = audio_path

            # Step 2: Transcription
            transcriber = WhisperTranscriber(
                TranscriptionConfig(
                    model_size=self.config.whisper_model,
                    language=self.config.source_lang,
                    use_gpu=self.config.use_gpu
                )
            )
            transcription = transcriber.transcribe(audio_path, self.config.source_lang)
            project.transcription = transcription

            # Step 3: Translation (if configured)
            if self.config.target_lang != self.config.source_lang and self.config.gemini_api_key:
                translator = TranslationEngine(
                    TranslationConfig(
                        target_language=self.config.target_lang,
                        use_gemini=True,
                        gemini_api_key=self.config.gemini_api_key,
                        use_context_aware=self.config.context_aware
                    )
                )

                context_analyzer = ContextAnalyzer()
                context = context_analyzer.analyze_conversation([
                    {"text": s.text, "speaker": s.speaker}
                    for s in transcription.segments
                ])

                for i, segment in enumerate(transcription.segments):
                    segment_context = context_analyzer.get_context_for_segment(
                        i, transcription.segments, context
                    )
                    if self.config.video_context:
                        segment_context["video_context"] = self.config.video_context

                    translation = translator.translate(
                        segment.text,
                        self.config.source_lang,
                        segment_context
                    )
                    segment.translation = translation

                # Quality check
                if self.config.gemini_api_key:
                    quality_checker = TranslationQualityChecker(gemini_api_key=self.config.gemini_api_key)
                    check_segments = [
                        {"original": seg.text, "translation": seg.translation, "context": {"speaker": seg.speaker}}
                        for seg in transcription.segments
                    ]
                    quality_checker.batch_check(
                        check_segments,
                        self.config.source_lang,
                        self.config.target_lang,
                        self.config.video_context
                    )

                # Cleanup
                transcriber.cleanup()

            # Step 4: Optimize and export
            optimizer = SubtitleOptimizer(settings)
            optimized_segments = optimizer.optimize_segments(transcription.segments)

            for i, seg in enumerate(optimized_segments):
                seg.index = i + 1

            generator = SubtitleGenerator(settings)
            base_name = item.video_path.stem

            # Create output directory
            item.output_dir.mkdir(parents=True, exist_ok=True)

            generator.save_all_formats(
                optimized_segments,
                item.output_dir,
                base_name,
                include_translation=True,
                include_speaker=True
            )

            item.status = "completed"
            item.result = {
                "segments": len(optimized_segments),
                "duration": duration,
                "output_dir": str(item.output_dir)
            }

            logger.info(f"Completed: {item.video_path.name}")

        except Exception as e:
            item.status = "failed"
            item.error = str(e)
            logger.error(f"Failed: {item.video_path.name} - {e}")

        return item

    def process(self, progress_callback: Optional[Callable] = None) -> list[BatchItem]:
        """
        Process all videos in batch.

        Args:
            progress_callback: Optional callback(completed, total, status_message)

        Returns:
            List of BatchItem results
        """
        if not self.items:
            logger.warning("No items in batch")
            return []

        total = len(self.items)
        completed = 0

        logger.info(f"Starting batch processing: {total} videos")

        # Use thread pool for parallel processing
        with ThreadPoolExecutor(max_workers=self.config.max_concurrent) as executor:
            futures = {executor.submit(self._process_single, item): item for item in self.items}

            for future in as_completed(futures):
                item = future.result()
                completed += 1

                # Update progress
                status_msg = f"{item.status}: {item.video_path.name}"
                if progress_callback:
                    progress_callback(completed, total, status_msg)
                elif self._progress_callback:
                    self._progress_callback(completed, total, status_msg)

                # Check if should stop on error
                if self.config.stop_on_error and item.status == "failed":
                    logger.warning("Stopping batch due to error")
                    break

        # Summary
        success = sum(1 for i in self.items if i.status == "completed")
        failed = sum(1 for i in self.items if i.status == "failed")

        logger.info(f"Batch complete: {success} succeeded, {failed} failed")

        return self.items

    def get_results(self) -> dict[str, Any]:
        """Get batch processing results summary."""
        return {
            "total": len(self.items),
            "completed": sum(1 for i in self.items if i.status == "completed"),
            "failed": sum(1 for i in self.items if i.status == "failed"),
            "pending": sum(1 for i in self.items if i.status == "pending"),
            "processing": sum(1 for i in self.items if i.status == "processing"),
            "items": [
                {
                    "video": str(i.video_path),
                    "status": i.status,
                    "error": i.error,
                    "result": i.result
                }
                for i in self.items
            ]
        }


class GlossaryManager:
    """
    Manage translation glossaries.

    Features:
    - Create/load/save glossaries
    - Add/remove terms
    - Import/export formats
    - Auto-apply during translation
    """

    def __init__(self):
        self.terms: dict[str, dict[str, str]] = {}  # source -> {target, context}

    def add_term(
        self,
        source: str,
        target: str,
        context: Optional[str] = None,
        notes: Optional[str] = None
    ):
        """Add a term to the glossary."""
        self.terms[source.lower()] = {
            "source": source,
            "target": target,
            "context": context or "general",
            "notes": notes or ""
        }

    def remove_term(self, source: str):
        """Remove a term from glossary."""
        if source.lower() in self.terms:
            del self.terms[source.lower()]

    def get_translation(self, source: str) -> Optional[str]:
        """Get translation for a term."""
        return self.terms.get(source.lower(), {}).get("target")

    def has_term(self, source: str) -> bool:
        """Check if term exists in glossary."""
        return source.lower() in self.terms

    def get_all_terms(self) -> list[dict[str, str]]:
        """Get all glossary terms."""
        return list(self.terms.values())

    def save_to_file(self, path: Path):
        """Save glossary to JSON file."""
        import json

        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                "version": "1.0",
                "terms": self.terms
            }, f, indent=2, ensure_ascii=False)

        logger.info(f"Glossary saved to {path}")

    @classmethod
    def load_from_file(cls, path: Path) -> "GlossaryManager":
        """Load glossary from JSON file."""
        import json

        with open(path, encoding='utf-8') as f:
            data = json.load(f)

        manager = cls()
        manager.terms = data.get("terms", {})

        logger.info(f"Glossary loaded from {path}")
        return manager

    def export_csv(self, path: Path):
        """Export glossary to CSV."""
        import csv

        with open(path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Source", "Target", "Context", "Notes"])

            for term in self.terms.values():
                writer.writerow([
                    term["source"],
                    term["target"],
                    term["context"],
                    term["notes"]
                ])

        logger.info(f"Glossary exported to {path}")

    @classmethod
    def import_csv(cls, path: Path) -> "GlossaryManager":
        """Import glossary from CSV."""
        import csv

        manager = cls()

        with open(path, encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("Source") and row.get("Target"):
                    manager.add_term(
                        row["Source"],
                        row["Target"],
                        row.get("Context"),
                        row.get("Notes")
                    )

        logger.info(f"Glossary imported from {path}")
        return manager


class QualityAssurance:
    """
    Quality assurance tools for subtitles.

    Features:
    - Timing validation
    - Format validation
    - Translation consistency
    - Error reporting
    """

    def __init__(self):
        self.issues: list[dict[str, Any]] = []

    def validate_timing(
        self,
        segments: list[Any],
        min_duration: float = 1.0,
        max_duration: float = 7.0,
        min_gap: float = 0.1
    ) -> list[dict[str, Any]]:
        """Validate subtitle timing."""
        issues = []

        for i, segment in enumerate(segments):
            duration = segment.end_time - segment.start_time

            # Check minimum duration
            if duration < min_duration:
                issues.append({
                    "type": "timing",
                    "severity": "error",
                    "segment": i + 1,
                    "message": f"Duration {duration:.2f}s is below minimum {min_duration}s"
                })

            # Check maximum duration
            if duration > max_duration:
                issues.append({
                    "type": "timing",
                    "severity": "warning",
                    "segment": i + 1,
                    "message": f"Duration {duration:.2f}s exceeds maximum {max_duration}s"
                })

            # Check gap with next segment
            if i < len(segments) - 1:
                next_segment = segments[i + 1]
                gap = next_segment.start_time - segment.end_time

                if gap < min_gap:
                    issues.append({
                        "type": "timing",
                        "severity": "warning",
                        "segment": i + 1,
                        "message": f"Gap {gap:.2f}s is below minimum {min_gap}s"
                    })

        return issues

    def validate_text(
        self,
        segments: list[Any],
        max_chars_per_line: int = 42,
        max_lines: int = 2
    ) -> list[dict[str, Any]]:
        """Validate subtitle text."""
        issues = []

        for i, segment in enumerate(segments):
            text = segment.translation or segment.text

            # Check line length
            lines = text.split('\n')
            for j, line in enumerate(lines):
                if len(line) > max_chars_per_line:
                    issues.append({
                        "type": "format",
                        "severity": "warning",
                        "segment": i + 1,
                        "line": j + 1,
                        "message": f"Line {j+1} has {len(line)} chars, exceeds {max_chars_per_line}"
                    })

            # Check number of lines
            if len(lines) > max_lines:
                issues.append({
                    "type": "format",
                    "severity": "warning",
                    "segment": i + 1,
                    "message": f"Has {len(lines)} lines, exceeds {max_lines}"
                })

        return issues

    def validate_consistency(
        self,
        segments: list[Any],
        translation_memory: Optional[dict[str, str]] = None
    ) -> list[dict[str, Any]]:
        """Validate translation consistency."""
        issues = []

        # Check for repeated source with different translations
        source_translations: dict[str, list[str]] = {}

        for _i, segment in enumerate(segments):
            source = segment.text.lower().strip()
            translation = (segment.translation or "").lower().strip()

            if source not in source_translations:
                source_translations[source] = []

            if translation and translation not in source_translations[source]:
                source_translations[source].append(translation)

        # Find inconsistencies
        for source, translations in source_translations.items():
            if len(translations) > 1:
                issues.append({
                    "type": "consistency",
                    "severity": "warning",
                    "message": f"Source '{source[:30]}...' has {len(translations)} different translations",
                    "translations": translations
                })

        return issues

    def run_all_checks(
        self,
        segments: list[Any],
        min_duration: float = 1.0,
        max_duration: float = 7.0,
        min_gap: float = 0.1,
        max_chars: int = 42,
        max_lines: int = 2
    ) -> dict[str, Any]:
        """Run all QA checks."""
        self.issues = []

        # Timing checks
        self.issues.extend(self.validate_timing(segments, min_duration, max_duration, min_gap))

        # Text checks
        self.issues.extend(self.validate_text(segments, max_chars, max_lines))

        # Consistency checks
        self.issues.extend(self.validate_consistency(segments))

        return {
            "total_issues": len(self.issues),
            "errors": len([i for i in self.issues if i["severity"] == "error"]),
            "warnings": len([i for i in self.issues if i["severity"] == "warning"]),
            "issues": self.issues
        }
