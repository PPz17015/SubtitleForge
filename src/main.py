#!/usr/bin/env python3
"""
SubtitleForge Pro - Professional Subtitle Generation and Translation

Offline subtitle generation with speaker diarization and context-aware translation.
Supports Japanese to Vietnamese translation with proper context understanding.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

# Fix import path - add src to path
_src_path = Path(__file__).parent.parent
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

# Import config to setup HF cache before any ML library loads

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from core.audio_extractor import AudioExtractor
from core.models import Project, ProjectSettings
from core.subtitle_generator import SubtitleGenerator, SubtitleOptimizer

# Try to import optional dependencies
try:
    from transcription.whisper_transcriber import TranscriptionConfig, WhisperTranscriber
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

try:
    from diarization.speaker_diarizer import SpeakerDiarizer
    DIARIZATION_AVAILABLE = True
except ImportError:
    DIARIZATION_AVAILABLE = False

try:
    from translation.translator import (  # noqa: F401
        ContextAnalyzer,
        TranslationConfig,
        TranslationEngine,
        TranslationQualityChecker,
    )
    TRANSLATION_AVAILABLE = True
except ImportError:
    TRANSLATION_AVAILABLE = False


console = Console()
logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """SubtitleForge Pro - Professional Subtitle Generation and Translation"""


@cli.command()
@click.argument('video_path', type=click.Path(exists=True, path_type=Path))
@click.option('--output', '-o', 'output_dir', type=click.Path(path_type=Path),
              help='Output directory for subtitles')
@click.option('--source', '-s', 'source_lang', default='ja',
              help='Source language code (default: ja for Japanese)')
@click.option('--target', '-t', 'target_lang', default='vi',
              help='Target language code (default: vi for Vietnamese)')
@click.option('--model', '-m', 'whisper_model', default='small',
              help='Whisper model size (tiny, base, small, medium, large)')
@click.option('--no-gpu', is_flag=True, help='Disable GPU acceleration')
@click.option('--gemini-key', 'gemini_api_key',
              help='Gemini API key for enhanced translation')
@click.option('--context-aware/--no-context', default=True,
              help='Enable context-aware translation')
@click.option('--video-context', '-vc', 'video_context', default=None,
              help='Video context description (e.g., "family dinner scene", "office meeting") for better translation')
@click.option('--speaker-relationship', '-sr', 'speaker_relationships', multiple=True, default=[],
              help='Speaker relationships (format: "SpeakerName:relationship" e.g., "Mother:mother-son")')
@click.option('--quality-check/--no-quality-check', default=True,
              help='Enable translation quality checking')
@click.option('--formats', '-f', multiple=True, default=['srt'],
              help='Output formats (srt, vtt, ass)')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def generate(
    video_path: Path,
    output_dir: Optional[Path],
    source_lang: str,
    target_lang: str,
    whisper_model: str,
    no_gpu: bool,
    gemini_api_key: Optional[str],
    context_aware: bool,
    video_context: Optional[str],
    speaker_relationships: tuple,
    quality_check: bool,
    formats: tuple,
    verbose: bool
):
    """Generate subtitles from video file."""

    setup_logging(verbose)

    # Check dependencies
    if not WHISPER_AVAILABLE:
        console.print("[red]Error: faster-whisper not installed. Run: pip install faster-whisper[/red]")
        sys.exit(1)

    # Set output directory
    if output_dir is None:
        output_dir = video_path.parent

    # Parse speaker relationships
    speaker_rels = {}
    for rel in speaker_relationships:
        if ":" in rel:
            speaker, relationship = rel.split(":", 1)
            speaker_rels[speaker] = relationship

    console.print(Panel.fit(
        f"[bold cyan]SubtitleForge Pro[/bold cyan]\n"
        f"Processing: {video_path.name}\n"
        f"Source: {source_lang} → Target: {target_lang}\n"
        f"Context: {video_context or 'Not provided'}",
        border_style="cyan"
    ))

    # Create settings
    settings = ProjectSettings(
        source_language=source_lang,
        target_language=target_lang,
        whisper_model=whisper_model,
        use_gpu=not no_gpu,
        gemini_api_key=gemini_api_key,
        use_context_aware=context_aware,
        output_formats=list(formats),
        speaker_relationships=speaker_rels
    )

    # Initialize project
    project = Project(video_path, settings)
    audio_path = None

    # Step 1: Extract audio
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console
    ) as progress:

        task = progress.add_task("[cyan]Extracting audio...", total=None)

        try:
            extractor = AudioExtractor()
            audio_path, duration = extractor.extract_audio(video_path)
            project.audio_path = audio_path
            progress.update(task, completed=True, description=f"[green]Audio extracted: {duration:.1f}s")
        except Exception as e:
            console.print(f"[red]Audio extraction failed: {e}[/red]")
            sys.exit(1)

        # Cleanup function for WAV file
        def cleanup_audio():
            if audio_path and audio_path.exists() and audio_path.suffix.lower() == '.wav':
                try:
                    audio_path.unlink()
                    console.print(f"[green]Cleaned up temp audio: {audio_path.name}[/green]")
                except Exception as cleanup_err:
                    console.print(f"[yellow]Warning: Could not delete temp audio file: {cleanup_err}[/yellow]")

        # Step 2: Speaker Diarization
        if DIARIZATION_AVAILABLE:
            progress.add_task("[cyan]Performing speaker diarization...", total=None)
            try:
                diarizer = SpeakerDiarizer()
                diarization_result = diarizer.diarize(audio_path)

                # Store speaker info
                for speaker in diarization_result.get("speakers", []):
                    project.speakers[speaker] = speaker

                console.print(f"[green]Detected {len(project.speakers)} speakers[/green]")
                progress.update(task, completed=True)
            except Exception as e:
                console.print(f"[yellow]Speaker diarization failed: {e}[/yellow]")
                diarization_result = {"segments": [], "speakers": []}
        else:
            console.print("[yellow]Speaker diarization not available (pyannote.audio required)[/yellow]")
            diarization_result = {"segments": [], "speakers": []}

        # Step 3: Transcription
        progress.add_task("[cyan]Transcribing audio...", total=None)
        try:
            transcriber = WhisperTranscriber(
                TranscriptionConfig(
                    model_size=whisper_model,
                    language=source_lang,
                    use_gpu=not no_gpu
                )
            )

            if diarization_result.get("speakers"):
                transcription = transcriber.transcribe_with_speaker(
                    audio_path, diarization_result, source_lang
                )
            else:
                transcription = transcriber.transcribe(audio_path, source_lang)

            project.transcription = transcription
            progress.update(task, completed=True,
                          description=f"[green]Transcribed: {len(transcription.segments)} segments")

        except Exception as e:
            console.print(f"[red]Transcription failed: {e}[/red]")
            sys.exit(1)

        # Step 4: Translation
        if target_lang != source_lang:
            progress.add_task("[cyan]Translating subtitles...", total=None)

            if TRANSLATION_AVAILABLE and gemini_api_key:
                try:
                    translator = TranslationEngine(
                        TranslationConfig(
                            target_language=target_lang,
                            use_gemini=True,
                            gemini_api_key=gemini_api_key,
                            use_context_aware=context_aware
                        )
                    )

                    # Analyze context
                    context_analyzer = ContextAnalyzer()

                    # Add user-provided speaker relationships to context
                    if speaker_rels:
                        for seg in transcription.segments:
                            if seg.speaker in speaker_rels:
                                seg.context = seg.context or {}
                                seg.context["relationship"] = speaker_rels[seg.speaker]

                    context = context_analyzer.analyze_conversation([
                        {"text": s.text, "speaker": s.speaker}
                        for s in transcription.segments
                    ])

                    # Apply user-provided relationships to context
                    if speaker_rels:
                        context["relationships"].update(speaker_rels)

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

                    # Translate segments using batch translation
                    batch_segments = [
                        {"text": s.text, "speaker": s.speaker}
                        for s in transcription.segments
                    ]

                    batch_contexts = []
                    for i in range(len(transcription.segments)):
                        segment_context = context_analyzer.get_context_for_segment(
                            i, transcription.segments, context
                        )
                        if video_context:
                            segment_context["video_context"] = video_context
                        batch_contexts.append(segment_context)

                    if quality_check and gemini_api_key:
                        from translation.translator import TranslationQualityChecker
                        quality_checker = TranslationQualityChecker(gemini_api_key=gemini_api_key)

                        translations = translator.translate_batch_with_self_check(
                            batch_segments, source_lang, batch_contexts,
                            video_context=video_context,
                            quality_checker=quality_checker
                        )
                        # Skip separate quality check step since it's built-in
                        # But we might still want to show final quality status?
                        # For now, let's keep the separate quality check as a final verification
                    else:
                        translations = translator.translate_batch(
                            batch_segments, source_lang, batch_contexts
                        )

                    for seg, trans in zip(transcription.segments, translations):
                        seg.translation = trans

                    progress.update(task, completed=True,
                                  description="[green]Translation complete[/green]")

                    # Step 4b: Quality Check
                    if quality_check and gemini_api_key:
                        progress.add_task("[cyan]Checking translation quality...", total=None)

                        try:
                            quality_checker = TranslationQualityChecker(gemini_api_key=gemini_api_key)

                            # Prepare segments for quality check
                            check_segments = [
                                {
                                    "original": seg.text,
                                    "translation": seg.translation,
                                    "context": {"speaker": seg.speaker}
                                }
                                for seg in transcription.segments
                            ]

                            # Run batch quality check
                            quality_results = quality_checker.batch_check(
                                check_segments,
                                source_language=source_lang,
                                target_language=target_lang,
                                video_context=video_context
                            )

                            # Count issues
                            issues_found = 0
                            for qr in quality_results:
                                if qr.get("needs_recheck") or not qr.get("is_good", True):
                                    issues_found += 1

                            if issues_found > 0:
                                console.print(f"[yellow]⚠ Found {issues_found} segments with potential issues[/yellow]")
                            else:
                                console.print("[green]✓ All translations passed quality check[/green]")

                            progress.update(task, completed=True)

                        except Exception as e:
                            console.print(f"[yellow]Quality check failed: {e}[/yellow]")
                            progress.update(task, completed=True)
                    else:
                        console.print("[yellow]Quality check skipped (disabled or no API key)[/yellow]")

                except Exception as e:
                    console.print(f"[yellow]Translation failed: {e}[/yellow]")
            else:
                console.print("[yellow]Skipping translation (no Gemini API key)[/yellow]")

        # Step 5: Optimize subtitles
        progress.add_task("[cyan]Optimizing subtitles...", total=None)
        optimizer = SubtitleOptimizer(settings)
        optimized_segments = optimizer.optimize_segments(transcription.segments)

        # Re-index segments
        for i, seg in enumerate(optimized_segments):
            seg.index = i + 1

        progress.update(task, completed=True)

        # Step 6: Generate output files
        progress.add_task("[cyan]Saving subtitle files...", total=None)
        generator = SubtitleGenerator(settings)

        base_name = video_path.stem
        generator.save_all_formats(
            optimized_segments,
            output_dir,
            base_name,
            include_translation=True,
            include_speaker=True
        )

        progress.update(task, completed=True)

    # Display summary
    console.print("\n")

    table = Table(title="Generation Summary", show_header=True, header_style="bold cyan")
    table.add_column("Item", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Video", video_path.name)
    table.add_row("Duration", f"{duration:.1f} seconds")
    table.add_row("Source Language", source_lang)
    table.add_row("Target Language", target_lang)
    table.add_row("Segments", str(len(optimized_segments)))
    table.add_row("Speakers", str(len(project.speakers)))
    table.add_row("Output Directory", str(output_dir))

    console.print(table)

    console.print("\n[bold green]✓ Subtitle generation complete![/bold green]")
    console.print(f"Output files: {output_dir}/{base_name}.*")

    # Cleanup temp WAV file
    cleanup_audio()

    # Save project
    project.save(video_path.with_suffix('.sfproj'))
    console.print(f"Project saved: {video_path.with_suffix('.sfproj')}")


@cli.command()
@click.argument('project_path', type=click.Path(exists=True, path_type=Path))
@click.option('--output', '-o', 'output_dir', type=click.Path(path_type=Path),
              help='Output directory for subtitles')
@click.option('--target', '-t', 'target_lang', default='vi',
              help='Target language code')
@click.option('--gemini-key', 'gemini_api_key', help='Gemini API key')
@click.option('--formats', '-f', multiple=True, default=['srt'],
              help='Output formats')
def translate(
    project_path: Path,
    output_dir: Optional[Path],
    target_lang: str,
    gemini_api_key: Optional[str],
    formats: tuple
):
    """Translate existing project subtitles."""

    setup_logging()

    if not TRANSLATION_AVAILABLE:
        console.print("[red]Error: Translation dependencies not installed[/red]")
        sys.exit(1)

    # Load project
    console.print(f"Loading project: {project_path}")
    project = Project.load(project_path)

    if not project.transcription:
        console.print("[red]No transcription found in project[/red]")
        sys.exit(1)

    # Set output directory
    if output_dir is None:
        output_dir = project_path.parent

    # Translate
    if gemini_api_key:
        translator = TranslationEngine(
            TranslationConfig(
                target_language=target_lang,
                use_gemini=True,
                gemini_api_key=gemini_api_key
            )
        )

        context_analyzer = ContextAnalyzer()
        context = context_analyzer.analyze_conversation([
            {"text": s.text, "speaker": s.speaker}
            for s in project.transcription.segments
        ])

        batch_segments = [
            {"text": s.text, "speaker": s.speaker}
            for s in project.transcription.segments
        ]

        batch_contexts = []
        for i in range(len(project.transcription.segments)):
            segment_context = context_analyzer.get_context_for_segment(
                i, project.transcription.segments, context
            )
            batch_contexts.append(segment_context)

        console.print(f"Translating {len(batch_segments)} segments in batches...")
        translations = translator.translate_batch(
            batch_segments,
            project.settings.source_language,
            batch_contexts
        )

        for seg, trans in zip(project.transcription.segments, translations):
            seg.translation = trans

        console.print("Translation complete!")

    # Save output
    settings = project.settings
    settings.output_formats = list(formats)
    settings.target_language = target_lang

    generator = SubtitleGenerator(settings)
    base_name = project_path.stem.replace('.sfproj', '')

    generator.save_all_formats(
        project.transcription.segments,
        output_dir,
        base_name,
        include_translation=True,
        include_speaker=True
    )

    console.print(f"[green]Output saved to {output_dir}[/green]")


@cli.command()
def info():
    """Show system information and available models."""

    console.print(Panel.fit(
        "[bold cyan]SubtitleForge Pro - System Information[/bold cyan]",
        border_style="cyan"
    ))

    table = Table(title="Dependency Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")

    table.add_row("Whisper (Transcription)", "✓ Available" if WHISPER_AVAILABLE else "✗ Not installed")
    table.add_row("PyAnnote (Diarization)", "✓ Available" if DIARIZATION_AVAILABLE else "✗ Not installed")
    table.add_row("Gemini (Translation)", "✓ Available" if TRANSLATION_AVAILABLE else "✗ Not installed")

    console.print(table)

    if WHISPER_AVAILABLE:
        console.print("\n[bold]Available Whisper models:[/bold]")
        for model in WhisperTranscriber.get_available_models():
            info = WhisperTranscriber.get_model_info(model)
            console.print(f"  {model}: {info.get('params', 'N/A')} params, {info.get('vram', 'N/A')} VRAM")


@cli.command()
@click.argument('video_path', type=click.Path(exists=True, path_type=Path))
def probe(video_path: Path):
    """Probe video file information."""

    extractor = AudioExtractor()
    info = extractor.get_video_info(video_path)

    if info:
        console.print(Panel.fit(
            f"[bold cyan]Video Information: {video_path.name}[/bold cyan]",
            border_style="cyan"
        ))

        if 'format' in info:
            fmt = info['format']
            table = Table(show_header=False)
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("Format", fmt.get('format_name', 'N/A'))
            table.add_row("Duration", f"{float(fmt.get('duration', 0)):.1f}s")
            table.add_row("Size", f"{int(fmt.get('size', 0)) / 1024 / 1024:.1f} MB")

            console.print(table)
    else:
        console.print("[yellow]Could not read video information[/yellow]")


@cli.command()
@click.argument('text', type=str)
@click.option('--source', '-s', 'source_lang', default='ja', help='Source language')
@click.option('--target', '-t', 'target_lang', default='vi', help='Target language')
@click.option('--gemini-key', 'gemini_api_key', required=True, help='Gemini API key')
@click.option('--speaker', help='Speaker name/label')
@click.option('--relationship', help='Relationship context (e.g., "mother-son")')
def translate_text(
    text: str,
    source_lang: str,
    target_lang: str,
    gemini_api_key: str,
    speaker: Optional[str],
    relationship: Optional[str]
):
    """Translate a single text string."""

    if not TRANSLATION_AVAILABLE:
        console.print("[red]Error: Translation dependencies not installed[/red]")
        sys.exit(1)

    translator = TranslationEngine(
        TranslationConfig(
            target_language=target_lang,
            use_gemini=True,
            gemini_api_key=gemini_api_key,
            use_context_aware=True
        )
    )

    context = {}
    if speaker:
        context["speaker"] = speaker
    if relationship:
        context["relationship"] = relationship

    result = translator.translate(text, source_lang, context if context else None)

    console.print(f"\n[bold]Original ({source_lang}):[/bold] {text}")
    console.print(f"[bold]Translation ({target_lang}):[/bold] {result}")

    if relationship:
        console.print(f"\n[dim]Context: {relationship}[/dim]")


@cli.command()
@click.argument('folder', type=click.Path(exists=True, path_type=Path))
@click.option('--output', '-o', 'output_dir', type=click.Path(path_type=Path),
              help='Output directory for subtitles')
@click.option('--pattern', '-p', default='*.mp4',
              help='File pattern to match (default: *.mp4)')
@click.option('--source', '-s', 'source_lang', default='ja',
              help='Source language code')
@click.option('--target', '-t', 'target_lang', default='vi',
              help='Target language code')
@click.option('--model', '-m', 'whisper_model', default='small',
              help='Whisper model size')
@click.option('--no-gpu', is_flag=True, help='Disable GPU')
@click.option('--gemini-key', 'gemini_api_key', help='Gemini API key')
@click.option('--concurrent', '-c', default=2, help='Max concurrent videos')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def batch(
    folder: Path,
    output_dir: Optional[Path],
    pattern: str,
    source_lang: str,
    target_lang: str,
    whisper_model: str,
    no_gpu: bool,
    gemini_api_key: Optional[str],
    concurrent: int,
    verbose: bool
):
    """Process multiple videos in a folder."""

    setup_logging(verbose)

    if not WHISPER_AVAILABLE:
        console.print("[red]Error: faster-whisper not installed[/red]")
        sys.exit(1)

    console.print(Panel.fit(
        f"[bold cyan]Batch Processing[/bold cyan]\n"
        f"Folder: {folder.name}\n"
        f"Pattern: {pattern}",
        border_style="cyan"
    ))

    # Import batch processor
    from core.batch_processing import BatchConfig, BatchProcessor

    config = BatchConfig(
        max_concurrent=concurrent,
        output_base_dir=output_dir,
        source_lang=source_lang,
        target_lang=target_lang,
        whisper_model=whisper_model,
        use_gpu=not no_gpu,
        gemini_api_key=gemini_api_key
    )

    processor = BatchProcessor(config)
    processor.add_videos_from_folder(folder, pattern)

    if not processor.items:
        console.print("[yellow]No videos found[/yellow]")
        return

    console.print(f"[cyan]Found {len(processor.items)} videos to process[/cyan]")

    # Process with progress
    def progress(completed, total, status):
        console.print(f"[{completed}/{total}] {status}")

    processor.process(progress)

    # Summary
    summary = processor.get_results()

    table = Table(title="Batch Results", show_header=True)
    table.add_column("Status", style="cyan")
    table.add_column("Count", style="green")

    table.add_row("Total", str(summary["total"]))
    table.add_row("Completed", str(summary["completed"]))
    table.add_row("Failed", str(summary["failed"]))

    console.print(table)


@cli.command()
@click.argument('project_path', type=click.Path(exists=True, path_type=Path))
@click.option('--min-duration', default=1.0, help='Minimum subtitle duration (seconds)')
@click.option('--max-duration', default=7.0, help='Maximum subtitle duration (seconds)')
@click.option('--max-chars', default=42, help='Maximum characters per line')
def qa(project_path: Path, min_duration: float, max_duration: float, max_chars: int):
    """Run quality assurance checks on a project."""

    from core.batch_processing import QualityAssurance

    console.print(f"[cyan]Running QA on: {project_path.name}[/cyan]")

    # Load project
    project = Project.load(project_path)

    if not project.transcription:
        console.print("[red]No transcription found in project[/red]")
        return

    # Run QA
    qa = QualityAssurance()
    results = qa.run_all_checks(
        project.transcription.segments,
        min_duration=min_duration,
        max_duration=max_duration,
        max_chars=max_chars
    )

    # Display results
    console.print("\n[bold]QA Results:[/bold]")
    console.print(f"Total Issues: {results['total_issues']}")
    console.print(f"[red]Errors: {results['errors']}[/red]")
    console.print(f"[yellow]Warnings: {results['warnings']}[/yellow]")

    if results["issues"]:
        console.print("\n[bold]Issues:[/bold]")
        for issue in results["issues"][:20]:  # Show first 20
            severity_icon = "❌" if issue["severity"] == "error" else "⚠️"
            console.print(f"  {severity_icon} [{issue['type']}] {issue['message']}")

        if len(results["issues"]) > 20:
            console.print(f"\n[dim]... and {len(results['issues']) - 20} more issues[/dim]")


@cli.command()
@click.argument('action', type=str)
@click.option('--glossary', '-g', 'glossary_path', type=click.Path(),
              help='Glossary file path')
@click.option('--source', '-s', help='Source term')
@click.option('--target', '-t', help='Target translation')
@click.option('--context', '-c', help='Context (general, family, professional, etc.)')
def glossary(action: str, glossary_path: Optional[str], source: Optional[str], target: Optional[str], context: Optional[str]):
    """Manage translation glossaries."""

    from core.batch_processing import GlossaryManager

    manager = GlossaryManager()

    # Convert string path to Path object
    gpath = Path(glossary_path) if glossary_path else None

    # Load existing if provided
    if gpath and gpath.exists() and action != 'import':
        manager = GlossaryManager.load_from_file(gpath)

    if action == 'add':
        if not source or not target:
            console.print("[red]Error: --source and --target required[/red]")
            return

        manager.add_term(source, target, context)

        if gpath:
            manager.save_to_file(gpath)
            console.print(f"[green]Added and saved: {source} → {target}[/green]")
        else:
            console.print(f"[green]Added: {source} → {target}[/green]")

    elif action == 'remove':
        if not source:
            console.print("[red]Error: --source required[/red]")
            return

        manager.remove_term(source)

        if gpath:
            manager.save_to_file(gpath)
            console.print(f"[green]Removed: {source}[/green]")

    elif action == 'list':
        terms = manager.get_all_terms()

        if not terms:
            console.print("[yellow]Glossary is empty[/yellow]")
            return

        table = Table(title="Glossary Terms")
        table.add_column("Source", style="cyan")
        table.add_column("Target", style="green")
        table.add_column("Context", style="yellow")

        for term in terms:
            table.add_row(term["source"], term["target"], term.get("context", ""))

        console.print(table)

    elif action == 'export':
        if not gpath:
            console.print("[red]Error: --glossary required[/red]")
            return

        manager.export_csv(gpath)
        console.print(f"[green]Exported to: {gpath}[/green]")

    elif action == 'import':
        if not gpath:
            console.print("[red]Error: --glossary required[/red]")
            return

        manager = GlossaryManager.import_csv(gpath)
        console.print(f"[green]Imported {len(manager.get_all_terms())} terms[/green]")


@cli.command()
@click.option('--host', default='0.0.0.0', help='Host to bind to')
@click.option('--port', '-p', type=int, default=8080, help='Port to bind to')
@click.option('--reload', is_flag=True, help='Enable auto-reload')
@click.option('--api-key', 'api_key', help='API key for authentication')
def server(host: str, port: int, reload: bool, api_key: Optional[str]):
    """Start the REST API server."""

    console.print(Panel.fit(
        f"[bold cyan]SubtitleForge Pro API Server[/bold cyan]\n"
        f"Starting on http://{host}:{port}\n"
        f"API Docs: http://{host}:{port}/docs",
        border_style="cyan"
    ))

    from api.server import run_server

    if api_key:
        import os
        os.environ["API_KEY"] = api_key
        console.print("[green]API key configured[/green]")

    run_server(host, port, reload)


if __name__ == '__main__':
    cli()
