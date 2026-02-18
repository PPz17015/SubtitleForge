import asyncio
import logging
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from api.models import JobCreate, JobResponse, JobStatus
from api.websocket import manager

logger = logging.getLogger(__name__)


class JobManager:
    def __init__(self):
        self.jobs: dict[str, JobResponse] = {}
        self.processing_lock = threading.Lock()

    def _extract_audio(self, video_path: Path):
        from core.audio_extractor import AudioExtractor
        extractor = AudioExtractor()
        return extractor.extract_audio(video_path)

    def create_job(self, request: JobCreate) -> JobResponse:
        job_id = str(uuid.uuid4())[:8]

        job = JobResponse(
            job_id=job_id,
            status=JobStatus.PENDING,
            created_at=datetime.now(),
            progress=0,
            message="Job created"
        )

        self.jobs[job_id] = job
        logger.info(f"Created job: {job_id}")

        return job

    def get_job(self, job_id: str) -> Optional[JobResponse]:
        return self.jobs.get(job_id)

    def list_jobs(self) -> list[JobResponse]:
        return list(self.jobs.values())

    async def update_progress(self, job_id: str, progress: int, message: str, stage: str = ""):
        if job_id in self.jobs:
            self.jobs[job_id].progress = progress
            self.jobs[job_id].message = message
            self.jobs[job_id].status = JobStatus.PROCESSING
            await manager.send_progress(job_id, progress, message, stage)

    async def complete_job(self, job_id: str, success: bool, message: str, result_path: Optional[str] = None, error: Optional[str] = None):
        if job_id in self.jobs:
            self.jobs[job_id].status = JobStatus.COMPLETED if success else JobStatus.FAILED
            self.jobs[job_id].progress = 100
            self.jobs[job_id].message = message
            self.jobs[job_id].result_path = result_path
            self.jobs[job_id].error = error
            await manager.send_completion(job_id, success, message, result_path)
            logger.info(f"Job {job_id} completed: {message}")

    async def process_job(self, job_id: str, request: JobCreate):
        job = self.get_job(job_id)
        if not job:
            return

        try:
            job.status = JobStatus.PROCESSING
            video_path = request.video_path

            if not video_path:
                await self.complete_job(job_id, False, "No video path provided", error="No video path")
                return

            video_path = Path(video_path)
            if not video_path.exists():
                await self.complete_job(job_id, False, "Video file not found", error="File not found")
                return

            await self.update_progress(job_id, 5, "Extracting audio...", "audio_extraction")

            # Run heavy processing in thread pool to avoid blocking
            audio_path, duration = await asyncio.to_thread(
                self._extract_audio, video_path
            )

            await self.update_progress(job_id, 20, "Transcribing audio...", "transcription")

            from transcription.whisper_transcriber import TranscriptionConfig, WhisperTranscriber
            transcriber = WhisperTranscriber(
                TranscriptionConfig(
                    model_size=request.options.whisper_model,
                    language=request.source_language,
                    use_gpu=request.options.use_gpu
                )
            )

            # Run transcription in thread
            transcription = await asyncio.to_thread(
                transcriber.transcribe, audio_path, request.source_language
            )

            await self.update_progress(job_id, 40, f"Transcribed {len(transcription.segments)} segments", "transcription")

            if request.target_language != request.source_language and request.gemini_api_key:
                await self.update_progress(job_id, 45, "Translating subtitles...", "translation")

                from translation.translator import ContextAnalyzer, TranslationConfig, TranslationEngine

                translator = TranslationEngine(
                    TranslationConfig(
                        target_language=request.target_language,
                        use_gemini=True,
                        gemini_api_key=request.gemini_api_key,
                        use_context_aware=request.options.context_aware
                    )
                )

                context_analyzer = ContextAnalyzer()
                context = context_analyzer.analyze_conversation([
                    {"text": s.text, "speaker": s.speaker}
                    for s in transcription.segments
                ])

                # Build character descriptions and scene contexts
                char_desc = context_analyzer.build_character_descriptions(
                    [{"text": s.text, "speaker": s.speaker} for s in transcription.segments],
                    context
                )
                if char_desc:
                    context["character_descriptions"] = char_desc

                scenes = context_analyzer.build_scene_contexts(transcription.segments)
                if scenes:
                    context["scenes"] = scenes

                # Prepare batch processing
                batch_segments = [
                    {"text": s.text, "speaker": s.speaker}
                    for s in transcription.segments
                ]

                batch_contexts = []
                for i in range(len(transcription.segments)):
                    segment_context = context_analyzer.get_context_for_segment(
                        i, transcription.segments, context
                    )
                    batch_contexts.append(segment_context)

                # Translate in thread using batch
                def do_translate():
                    if request.options.quality_check and request.gemini_api_key:
                        from translation.translator import TranslationQualityChecker
                        quality_checker = TranslationQualityChecker(gemini_api_key=request.gemini_api_key)
                        return translator.translate_batch_with_self_check(
                            batch_segments, request.source_language, batch_contexts,
                            video_context=request.video_context,
                            quality_checker=quality_checker
                        )
                    return translator.translate_batch(
                        batch_segments, request.source_language, batch_contexts
                    )

                translations = await asyncio.to_thread(do_translate)

                for seg, trans in zip(transcription.segments, translations):
                    seg.translation = trans

                await self.update_progress(job_id, 75, f"Translated {len(transcription.segments)} segments", "translation")

            # Cleanup transcriber VRAM (always, not just after translation)
            transcriber.cleanup()

            await self.update_progress(job_id, 80, "Generating subtitle files...", "output")

            from core.models import ProjectSettings
            from core.subtitle_generator import SubtitleGenerator, SubtitleOptimizer

            settings = ProjectSettings(
                source_language=request.source_language,
                target_language=request.target_language,
                whisper_model=request.options.whisper_model,
                use_gpu=request.options.use_gpu,
                gemini_api_key=request.gemini_api_key,
                use_context_aware=request.options.context_aware,
                output_formats=["srt", "vtt"]
            )

            optimizer = SubtitleOptimizer(settings)
            optimized_segments = optimizer.optimize_segments(transcription.segments)

            for i, seg in enumerate(optimized_segments):
                seg.index = i + 1

            output_dir = video_path.parent
            generator = SubtitleGenerator(settings)
            base_name = video_path.stem

            generator.save_all_formats(
                optimized_segments,
                output_dir,
                base_name,
                include_translation=True,
                include_speaker=True
            )

            # Cleanup temporary WAV file
            if audio_path and Path(audio_path).exists() and Path(audio_path).suffix.lower() == '.wav':
                try:
                    Path(audio_path).unlink()
                    logger.info(f"Cleaned up temporary audio file: {audio_path}")
                except Exception as cleanup_err:
                    logger.warning(f"Could not delete temp audio file: {cleanup_err}")

            result_path = str(output_dir / f"{base_name}.srt")
            await self.complete_job(job_id, True, "Processing complete!", result_path)

        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}", exc_info=True)
            await self.complete_job(job_id, False, str(e), error=str(e))


job_manager = JobManager()
