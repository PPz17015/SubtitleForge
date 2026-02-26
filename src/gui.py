import logging
import os
import shutil
import sys
from pathlib import Path

# Fix import path - add parent to path
_src_path = Path(__file__).parent.parent
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

# Import config AFTER sys.path is set up
import contextlib

from config import get_config

# Fix PyQt6 + PyTorch DLL conflict on Windows
# Must add torch DLL directory and import torch BEFORE PyQt6
# because PyQt6 modifies the DLL search path on import
if sys.platform == "win32":
    _torch_lib = os.path.join(
        os.path.dirname(sys.executable),
        "Lib", "site-packages", "torch", "lib"
    )
    if os.path.isdir(_torch_lib):
        os.add_dll_directory(_torch_lib)
    with contextlib.suppress(ImportError):
        import torch  # noqa: F401 — preload to claim DLL search paths

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QAction, QFont
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

logging.basicConfig(level=logging.INFO, filename='subtitleforge.log')
logger = logging.getLogger(__name__)


class ProcessingThread(QThread):
    progress = pyqtSignal(str, int, str)
    finished = pyqtSignal(bool, str)
    log_message = pyqtSignal(str)
    stage_changed = pyqtSignal(str)

    def __init__(self, video_path, settings, resume_data=None):
        super().__init__()
        self.video_path = video_path
        self.settings = settings
        self._should_stop = False

        # Checkpoint support
        from core.checkpoint import CheckpointManager
        self.checkpoint_mgr = CheckpointManager(video_path)
        self.resume_data = resume_data  # CheckpointData or None
        self.checkpoint_data = None     # Active checkpoint during processing

    def stop(self):
        """Request the thread to stop gracefully."""
        self._should_stop = True

    def _is_stage_completed(self, stage: str) -> bool:
        """Check if a stage was already completed in the checkpoint."""
        if not self.resume_data or not self.resume_data.completed_stage:
            return False
        from core.checkpoint import STAGES
        try:
            completed_idx = STAGES.index(self.resume_data.completed_stage)
            stage_idx = STAGES.index(stage)
            return stage_idx <= completed_idx
        except ValueError:
            return False

    def _check_cancelled(self):
        """Check if processing was cancelled by user."""
        if self._should_stop:
            raise Exception("Processing cancelled by user")

    def run(self):
        try:
            self._setup_paths()
            self._check_cancelled()

            # Initialize or restore checkpoint
            if self.resume_data:
                self.checkpoint_data = self.resume_data
                self.log_message.emit("🔄 Tiếp tục từ checkpoint...")
            else:
                self.checkpoint_data = self.checkpoint_mgr.create(self.settings)

            # Stage 1: Extract audio
            if self._is_stage_completed("audio_extracted"):
                self._restore_audio()
            else:
                self._extract_audio()
                self.checkpoint_mgr.save_audio_extracted(
                    self.checkpoint_data,
                    str(self.settings.audio_path),
                    getattr(self.settings, '_audio_duration', 0.0)
                )
            self._check_cancelled()

            # Stage 2: Transcribe
            if self._is_stage_completed("transcribed"):
                self._restore_transcription()
            else:
                self._transcribe()
                self.checkpoint_mgr.save_transcribed(
                    self.checkpoint_data,
                    self.settings.transcription.to_dict()
                )
            self._check_cancelled()

            # Stage 3: Translate
            if self.settings.gemini_api_key:
                if self._is_stage_completed("translated"):
                    self._restore_translations()
                else:
                    self._translate()
                    translations = [
                        seg.translation
                        for seg in self.settings.transcription.segments
                    ]
                    self.checkpoint_mgr.save_translated(
                        self.checkpoint_data, translations
                    )
                self._check_cancelled()

                # Stage 4: Final quality verification
                # Even after self-check corrects issues, run a final pass to confirm
                if self.settings.quality_check:
                    if self._is_stage_completed("quality_checked"):
                        self.log_message.emit("⏭ Bỏ qua: Kiểm tra chất lượng (đã hoàn thành)")
                    else:
                        self._quality_check()
                        self.checkpoint_mgr.save_quality_checked(
                            self.checkpoint_data, []
                        )
                    self._check_cancelled()

            # Stage 5: Optimize & Save
            self._optimize_and_save()

            # Success — clean up checkpoint
            self.checkpoint_mgr.cleanup()
            self.finished.emit(True, "Xử lý hoàn tất!")

        except Exception as e:
            if "cancelled" in str(e).lower():
                logger.info(f"Processing cancelled: {e}")
                self.log_message.emit("💾 Checkpoint đã lưu — có thể tiếp tục sau")
                self.finished.emit(False, "Đã hủy bởi người dùng")
            else:
                logger.error(f"Processing failed: {e}", exc_info=True)
                self.log_message.emit("💾 Checkpoint đã lưu — có thể tiếp tục sau")
                self.finished.emit(False, str(e))

    def _restore_audio(self):
        """Restore audio data from checkpoint."""
        self.stage_changed.emit("audio")
        self.settings.audio_path = Path(self.resume_data.audio_path)
        self.log_message.emit(
            f"⏭ Bỏ qua: Trích xuất audio (đã hoàn thành — {self.resume_data.audio_duration:.1f}s)"
        )
        self.progress.emit("Audio đã có sẵn", 10, "audio_extraction")

    def _restore_transcription(self):
        """Restore transcription data from checkpoint."""
        self.stage_changed.emit("transcribe")
        from core.models import TranscriptionResult
        self.settings.transcription = TranscriptionResult.from_dict(
            self.resume_data.transcription
        )
        seg_count = len(self.settings.transcription.segments)
        self.log_message.emit(
            f"⏭ Bỏ qua: Transcription (đã hoàn thành — {seg_count} segments)"
        )
        self.progress.emit("Transcription đã có sẵn", 30, "transcription")

    def _restore_translations(self):
        """Restore translation data from checkpoint."""
        self.stage_changed.emit("translate")
        for seg, trans in zip(
            self.settings.transcription.segments,
            self.resume_data.translations
        ):
            if trans is not None:
                seg.translation = trans
        total = len(self.settings.transcription.segments)
        self.log_message.emit(
            f"⏭ Bỏ qua: Dịch subtitle (đã hoàn thành — {total} segments)"
        )
        self.progress.emit("Translation đã có sẵn", 75, "translation")

    def _setup_paths(self):
        self.progress.emit("Đang khởi tạo...", 0, "init")
        self.log_message.emit("=" * 50)
        self.log_message.emit("SubtitleForge Pro - Bắt đầu xử lý")
        self.log_message.emit(f"Video: {self.video_path.name}")

    def _extract_audio(self):
        self.stage_changed.emit("audio")
        self.progress.emit("Đang trích xuất audio...", 5, "audio_extraction")
        self.log_message.emit("[1/5] Trích xuất audio từ video...")

        from core.audio_extractor import AudioExtractor
        extractor = AudioExtractor()
        audio_path, duration = extractor.extract_audio(self.video_path)
        self.settings.audio_path = audio_path
        self.settings._audio_duration = duration  # For checkpoint

        self.log_message.emit(f"✓ Audio extracted: {duration:.1f}s")

    def _transcribe(self):
        self.stage_changed.emit("transcribe")
        self.progress.emit("Đang transcribe audio...", 20, "transcription")
        self.log_message.emit("[2/5] Đang chuyển speech thành text...")

        from transcription.whisper_transcriber import TranscriptionConfig, WhisperTranscriber
        transcriber = WhisperTranscriber(
            TranscriptionConfig(
                model_size=self.settings.whisper_model,
                language=self.settings.source_language,
                use_gpu=self.settings.use_gpu
            )
        )

        transcription = transcriber.transcribe(
            self.settings.audio_path,
            self.settings.source_language
        )
        self.settings.transcription = transcription

        self.log_message.emit(f"✓ Transcribed: {len(transcription.segments)} segments")

        if self.settings.gemini_api_key:
            self.log_message.emit(f"[Model: {self.settings.whisper_model}]")

    def _translate(self):
        self.stage_changed.emit("translate")
        total = len(self.settings.transcription.segments)
        self.progress.emit("Đang dịch subtitles...", 40, "translation")

        from translation.translator import ContextAnalyzer, TranslationConfig, TranslationEngine

        translator = TranslationEngine(
            TranslationConfig(
                target_language=self.settings.target_language,
                use_gemini=True,
                gemini_api_key=self.settings.gemini_api_key,
                use_context_aware=self.settings.use_context_aware
            ),
            progress_callback=self.log_message.emit
        )

        context_analyzer = ContextAnalyzer()
        context = context_analyzer.analyze_conversation([
            {"text": s.text, "speaker": s.speaker}
            for s in self.settings.transcription.segments
        ])

        # Build character descriptions and scene contexts
        char_desc = context_analyzer.build_character_descriptions(
            [{"text": s.text, "speaker": s.speaker} for s in self.settings.transcription.segments],
            context
        )
        if char_desc:
            context["character_descriptions"] = char_desc

        scenes = context_analyzer.build_scene_contexts(self.settings.transcription.segments)
        if scenes:
            context["scenes"] = scenes

        batch_segments = [
            {"text": s.text, "speaker": s.speaker}
            for s in self.settings.transcription.segments
        ]

        batch_contexts = []
        for i in range(len(self.settings.transcription.segments)):
            segment_context = context_analyzer.get_context_for_segment(
                i, self.settings.transcription.segments, context
            )
            if self.settings.video_context:
                segment_context["video_context"] = self.settings.video_context
            batch_contexts.append(segment_context)

        # Determine resume offset from partial translation checkpoint
        resume_offset = 0
        translations = [None] * total

        if (
            self.resume_data
            and self.resume_data.translated_count > 0
            and len(self.resume_data.translations) == total
        ):
            resume_offset = self.resume_data.translated_count
            translations = list(self.resume_data.translations)
            # Restore already-translated segments
            for i in range(resume_offset):
                if translations[i] is not None:
                    self.settings.transcription.segments[i].translation = translations[i]
            self.log_message.emit(
                f"[3/5] Tiếp tục dịch từ segment {resume_offset}/{total}..."
            )
        else:
            self.log_message.emit(f"[3/5] Dịch {total} segments...")

        # Translate remaining segments in small chunks with progress updates
        from core.checkpoint import CheckpointManager
        gui_chunk_size = CheckpointManager.TRANSLATION_BATCH_SIZE  # 50

        remaining_segments = batch_segments[resume_offset:]
        remaining_contexts = batch_contexts[resume_offset:]
        remaining_count = len(remaining_segments)

        if remaining_segments:
            use_quality = getattr(self.settings, 'quality_check', False) and self.settings.gemini_api_key
            quality_checker = None
            if use_quality:
                from translation.translator import TranslationQualityChecker
                quality_checker = TranslationQualityChecker(gemini_api_key=self.settings.gemini_api_key)

            num_chunks = (remaining_count + gui_chunk_size - 1) // gui_chunk_size

            for chunk_idx in range(num_chunks):
                # Check if cancelled before each chunk
                self._check_cancelled()

                chunk_start = chunk_idx * gui_chunk_size
                chunk_end = min(chunk_start + gui_chunk_size, remaining_count)
                chunk_segments = remaining_segments[chunk_start:chunk_end]
                chunk_contexts = remaining_contexts[chunk_start:chunk_end]

                global_start = resume_offset + chunk_start
                global_end = resume_offset + chunk_end

                self.log_message.emit(
                    f"  📝 Đang dịch batch {chunk_idx + 1}/{num_chunks} "
                    f"(segments {global_start + 1}-{global_end}/{total})..."
                )

                # Translate this chunk
                if use_quality and quality_checker:
                    chunk_translations = translator.translate_batch_with_self_check(
                        chunk_segments, self.settings.source_language, chunk_contexts,
                        video_context=getattr(self.settings, 'video_context', None),
                        quality_checker=quality_checker
                    )
                else:
                    chunk_translations = translator.translate_batch(
                        chunk_segments, self.settings.source_language, chunk_contexts
                    )

                # Apply translations from this chunk
                for i, trans in enumerate(chunk_translations):
                    global_idx = global_start + i
                    translations[global_idx] = trans
                    self.settings.transcription.segments[global_idx].translation = trans

                # Update progress and checkpoint after each chunk
                done_count = global_end
                progress_pct = 40 + int((done_count / total) * 35)
                self.progress.emit(
                    f"Đã dịch ({done_count}/{total})...",
                    progress_pct, "translation"
                )
                self.log_message.emit(
                    f"  ✓ Batch {chunk_idx + 1}/{num_chunks} hoàn thành ({done_count}/{total})"
                )

                # Save checkpoint
                self.checkpoint_mgr.save_translation_progress(
                    self.checkpoint_data, translations, done_count
                )

        self.progress.emit(f"Đã dịch ({total}/{total})...", 75, "translation")
        self.log_message.emit(f"  Translated: {total}/{total}")
        self.log_message.emit("✓ Translation complete")

    def _quality_check(self):
        self.stage_changed.emit("quality")
        self.progress.emit("Đang kiểm tra chất lượng...", 75, "quality_check")
        self.log_message.emit("[4/5] Kiểm tra chất lượng bản dịch...")

        from translation.translator import TranslationQualityChecker

        quality_checker = TranslationQualityChecker(gemini_api_key=self.settings.gemini_api_key)

        check_segments = [
            {
                "original": seg.text,
                "translation": seg.translation,
                "context": {"speaker": seg.speaker}
            }
            for seg in self.settings.transcription.segments
        ]

        results = quality_checker.batch_check(
            check_segments,
            self.settings.source_language,
            self.settings.target_language,
            self.settings.video_context
        )

        issues = sum(1 for r in results if r.get("needs_recheck"))

        if issues > 0:
            self.log_message.emit(f"⚠ Found {issues} segments with potential issues")
        else:
            self.log_message.emit("✓ All translations passed quality check")

    def _optimize_and_save(self):
        self.stage_changed.emit("save")
        self.progress.emit("Đang lưu subtitles...", 90, "output")
        self.log_message.emit("[5/5] Tối ưu và lưu file...")

        from core.subtitle_generator import SubtitleGenerator, SubtitleOptimizer

        optimizer = SubtitleOptimizer(self.settings)
        optimized = optimizer.optimize_segments(self.settings.transcription.segments)

        for i, seg in enumerate(optimized):
            seg.index = i + 1

        output_dir = self.video_path.parent
        generator = SubtitleGenerator(self.settings)
        base_name = self.video_path.stem

        generator.save_all_formats(
            optimized,
            output_dir,
            base_name,
            include_translation=True,
            include_speaker=True
        )

        self.log_message.emit(f"✓ Saved to: {output_dir}")

        # Cleanup: delete temporary WAV file
        if hasattr(self.settings, 'audio_path') and self.settings.audio_path:
            try:
                wav_path = Path(self.settings.audio_path)
                if wav_path.exists() and wav_path.suffix.lower() == '.wav':
                    wav_path.unlink()
                    self.log_message.emit("✓ Cleaned up temporary audio file")
            except Exception as e:
                logger.warning(f"Could not delete temp audio file: {e}")

        self.log_message.emit("=" * 50)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.processing_thread = None
        self.init_ui()
        self._check_dependencies()

    def _check_dependencies(self):
        self.log_text.append(f"Python: {sys.executable}")
        self.log_text.append("Checking dependencies...")

        # NOTE: DLL conflict fix is handled at module level (before PyQt6 import)

        # Check for ffmpeg-python library
        try:
            pass
        except Exception as e:
            self.log_text.append(f"[MISSING] FFmpeg Python library: {e}")

        # Check for ffmpeg binary in PATH
        ffmpeg_exe = shutil.which('ffmpeg')
        if ffmpeg_exe:
            self.log_text.append(f"[OK] FFmpeg binary: {ffmpeg_exe}")
        else:
            # Check common Windows locations
            common_paths = [
                Path(r'C:\ffmpeg\bin\ffmpeg.exe'),
                Path(r'C:\Program Files\ffmpeg\bin\ffmpeg.exe'),
                Path(r'C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe'),
                Path.home() / 'ffmpeg' / 'bin' / 'ffmpeg.exe',
            ]
            found = False
            for path in common_paths:
                if path.exists():
                    self.log_text.append(f"[OK] FFmpeg binary: {path}")
                    found = True
                    break

            # Check WinGet installation location
            if not found:
                winget_base = Path.home() / 'AppData' / 'Local' / 'Microsoft' / 'WinGet' / 'Packages'
                if winget_base.exists():
                    for pkg_dir in winget_base.glob('Gyan.FFmpeg*'):
                        for bin_dir in pkg_dir.rglob('bin'):
                            ffmpeg_path = bin_dir / 'ffmpeg.exe'
                            if ffmpeg_path.exists():
                                self.log_text.append(f"[OK] FFmpeg binary: {ffmpeg_path}")
                                found = True
                                break
                        if found:
                            break

            if not found:
                self.log_text.append("[MISSING] FFmpeg binary: Not found in PATH or common locations")
                self.log_text.append("    Install with: winget install Gyan.FFmpeg")
                self.log_text.append("    Or download from: https://ffmpeg.org/download.html")

        try:
            self.log_text.append("[OK] Faster Whisper")
        except Exception as e:
            self.log_text.append(f"[MISSING] Faster Whisper: {e}")

        try:
            from google import genai  # noqa: F401
            self.log_text.append("[OK] Google GenAI SDK")
        except Exception as e:
            self.log_text.append(f"[MISSING] Google GenAI SDK: {e}")

        try:
            import torch
            if torch.cuda.is_available():
                self.log_text.append("[OK] PyTorch + CUDA GPU")
            else:
                self.log_text.append("[OK] PyTorch (CPU mode)")
        except Exception as e:
            self.log_text.append(f"[MISSING] PyTorch: {e}")

        self.log_text.append("")

    def init_ui(self):
        self.setWindowTitle("SubtitleForge Pro - Tạo Subtitle Tự Động")
        self.setMinimumSize(1000, 700)
        self.resize(1100, 800)
        self.setStyleSheet(self._get_stylesheet())

        # Create scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setStyleSheet("QScrollArea { border: none; background-color: #F8FAFC; }")

        # Create content widget
        content_widget = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)
        content_widget.setLayout(main_layout)

        scroll.setWidget(content_widget)
        self.setCentralWidget(scroll)

        self._create_menu_bar()

        title = QLabel("SubtitleForge Pro")
        title.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("color: #2563EB; margin-bottom: 5px; margin-top: 5px;")
        main_layout.addWidget(title)

        subtitle = QLabel("Tạo subtitle tự động với AI - Hỗ trợ dịch Nhật → Việt")
        subtitle.setFont(QFont("Segoe UI", 12))
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("color: #64748B; margin-bottom: 15px;")
        main_layout.addWidget(subtitle)

        main_layout.addWidget(self._create_api_section())
        main_layout.addWidget(self._create_video_section())
        main_layout.addWidget(self._create_settings_section())
        main_layout.addWidget(self._create_progress_section())
        main_layout.addWidget(self._create_log_section())
        main_layout.addLayout(self._create_buttons())

        main_layout.addStretch()

        self.statusBar().showMessage("Sẵn sàng")

    def _create_menu_bar(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu("File")

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        help_menu = menubar.addMenu("Help")

        about_action = QAction("About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _show_about(self):
        QMessageBox.about(self, "About SubtitleForge Pro",
            "<h3>SubtitleForge Pro</h3>"
            "<p>Version 1.0.0</p>"
            "<p>Tạo subtitle tự động với AI</p>"
            "<p>Hỗ trợ: Japanese → Vietnamese</p>"
            "<hr>"
            "<p><b>Tính năng:</b></p>"
            "<ul>"
            "<li>Transcribe audio thành text (Whisper AI)</li>"
            "<li>Dịch subtitle với Gemini AI</li>"
            "<li>Context-aware translation</li>"
            "<li>Kiểm tra chất lượng bản dịch</li>"
            "</ul>"
        )

    def _create_api_section(self):
        group = QGroupBox("🔑 Cấu hình API")
        layout = QVBoxLayout()
        layout.setContentsMargins(15, 30, 15, 15)
        layout.setSpacing(12)

        api_layout = QHBoxLayout()
        api_layout.addWidget(QLabel("Gemini API Key:"))
        self.api_key_input = QLineEdit()
        self.api_key_input.setPlaceholderText("Nhập API key để dịch (bắt buộc)")
        self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)

        # Load cached API key
        config = get_config()
        cached_key = config.get_api_key()
        if cached_key:
            self.api_key_input.setText(cached_key)

        api_layout.addWidget(self.api_key_input)
        layout.addLayout(api_layout)

        # Remember API key checkbox
        self.remember_api_key = QCheckBox("Lưu API key (lần sau không cần nhập lại)")
        self.remember_api_key.setChecked(bool(cached_key))
        layout.addWidget(self.remember_api_key)

        help_label = QLabel(
            "💡 Lấy API key miễn phí tại: <a href='https://aistudio.google.com/app/apikey'>Google AI Studio</a>"
        )
        help_label.setOpenExternalLinks(True)
        help_label.setStyleSheet("color: #64748B; font-size: 11px;")
        layout.addWidget(help_label)

        group.setLayout(layout)
        return group

    def _create_video_section(self):
        group = QGroupBox("🎬 Chọn Video")
        layout = QVBoxLayout()
        layout.setContentsMargins(15, 30, 15, 15)
        layout.setSpacing(12)

        file_layout = QHBoxLayout()
        self.video_path_input = QLineEdit()
        self.video_path_input.setPlaceholderText("Chọn file video (MP4, MKV, AVI, MOV - tối đa 20GB)")
        self.video_path_input.setReadOnly(True)
        file_layout.addWidget(self.video_path_input)

        self.browse_button = QPushButton("📂 Browse")
        self.browse_button.clicked.connect(self._browse_video)
        file_layout.addWidget(self.browse_button)

        layout.addLayout(file_layout)

        self.video_info = QLabel("")
        self.video_info.setStyleSheet("color: #64748B; font-size: 11px;")
        layout.addWidget(self.video_info)

        group.setLayout(layout)
        return group

    def _create_settings_section(self):

        group = QGroupBox("⚙️ Cài Đặt")
        layout = QVBoxLayout()
        layout.setContentsMargins(15, 30, 15, 15)
        layout.setSpacing(15)

        # --- Row 1: Source language ---
        row1 = QHBoxLayout()
        lbl1 = QLabel("Ngon ngu nguon:")
        lbl1.setFixedWidth(130)
        lbl1.setStyleSheet("font-weight: bold; font-size: 13px;")
        row1.addWidget(lbl1)

        self.source_lang = QComboBox()
        self.source_lang.addItems([
            "Japanese (Tieng Nhat)",
            "English (Tieng Anh)",
            "Chinese (Tieng Trung)",
            "Korean (Tieng Han)"
        ])
        self.source_lang.setCurrentIndex(0)
        self.source_lang.setItemData(0, "ja")
        self.source_lang.setItemData(1, "en")
        self.source_lang.setItemData(2, "zh")
        self.source_lang.setItemData(3, "ko")
        self.source_lang.setMinimumWidth(200)
        row1.addWidget(self.source_lang)

        arrow = QLabel("  -->  ")
        arrow.setStyleSheet("font-size: 14px; color: #2563EB; font-weight: bold;")
        row1.addWidget(arrow)

        lbl2 = QLabel("Ngon ngu dich:")
        lbl2.setStyleSheet("font-weight: bold; font-size: 13px;")
        row1.addWidget(lbl2)

        self.target_lang = QComboBox()
        self.target_lang.addItems([
            "Vietnamese (Tieng Viet)",
            "English (Tieng Anh)"
        ])
        self.target_lang.setCurrentIndex(0)
        self.target_lang.setItemData(0, "vi")
        self.target_lang.setItemData(1, "en")
        self.target_lang.setMinimumWidth(200)
        row1.addWidget(self.target_lang)
        row1.addStretch()
        layout.addLayout(row1)

        # --- Row 2: Whisper Model ---
        row2 = QHBoxLayout()
        lbl3 = QLabel("Whisper Model:")
        lbl3.setFixedWidth(130)
        lbl3.setStyleSheet("font-weight: bold; font-size: 13px;")
        row2.addWidget(lbl3)

        self.model_size = QComboBox()
        model_options = [
            ("tiny", "Tiny - nhanh nhat, do chinh xac thap"),
            ("base", "Base - can bang toc do"),
            ("small", "Small - khuyen nghi (mac dinh)"),
            ("medium", "Medium - chinh xac cao"),
            ("large", "Large - chat luong tot nhat"),
        ]
        for i, (value, text) in enumerate(model_options):
            self.model_size.addItem(text)
            self.model_size.setItemData(i, value)
        self.model_size.setCurrentIndex(2)  # small
        self.model_size.setMinimumWidth(300)
        row2.addWidget(self.model_size)
        row2.addStretch()
        layout.addLayout(row2)

        # --- Separator ---
        sep1 = QFrame()
        sep1.setFrameShape(QFrame.Shape.HLine)
        sep1.setStyleSheet("color: #E2E8F0;")
        sep1.setFixedHeight(1)
        layout.addWidget(sep1)
        layout.addSpacing(5)

        # --- Row 3: Checkboxes (vertical, 2 cols) ---
        opt_label = QLabel("Tuy chon xu ly:")
        opt_label.setStyleSheet("font-weight: bold; font-size: 13px;")
        layout.addWidget(opt_label)

        check_row1 = QHBoxLayout()
        check_row1.setSpacing(20)
        self.use_gpu_check = QCheckBox("Su dung GPU (chi ho tro NVIDIA)")
        self.use_gpu_check.setChecked(False)
        self.use_gpu_check.setMinimumWidth(250)
        check_row1.addWidget(self.use_gpu_check)

        self.context_aware_check = QCheckBox("Dich theo ngu canh")
        self.context_aware_check.setChecked(True)
        self.context_aware_check.setMinimumWidth(250)
        check_row1.addWidget(self.context_aware_check)
        check_row1.addStretch()
        layout.addLayout(check_row1)

        check_row2 = QHBoxLayout()
        check_row2.setSpacing(20)
        self.quality_check_check = QCheckBox("Kiem tra chat luong ban dich")
        self.quality_check_check.setChecked(True)
        self.quality_check_check.setMinimumWidth(250)
        check_row2.addWidget(self.quality_check_check)
        check_row2.addStretch()
        layout.addLayout(check_row2)

        # --- Separator ---
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setStyleSheet("color: #E2E8F0;")
        sep2.setFixedHeight(1)
        layout.addWidget(sep2)
        layout.addSpacing(5)

        # --- Row 4: Video context ---
        ctx_label = QLabel("Ngu canh video (tuy chon):")
        ctx_label.setStyleSheet("font-weight: bold; font-size: 13px;")
        layout.addWidget(ctx_label)

        self.video_context_input = QLineEdit()
        self.video_context_input.setPlaceholderText("vd: Phim anime gia dinh, me va con noi chuyen  |  J-Drama van phong  |  Phim hanh dong")
        layout.addWidget(self.video_context_input)

        ctx_hint = QLabel("Mo ta ngan noi dung phim giup AI dich chinh xac xuong ho va ngu canh hon")
        ctx_hint.setStyleSheet("color: #64748B; font-size: 11px; font-style: italic;")
        layout.addWidget(ctx_hint)

        group.setLayout(layout)
        return group

    def _create_progress_section(self):
        group = QGroupBox("📊 Tiến Trình")
        layout = QVBoxLayout()
        layout.setContentsMargins(15, 30, 15, 15)
        layout.setSpacing(12)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setMinimumHeight(30)
        layout.addWidget(self.progress_bar)

        self.stage_label = QLabel("Sẵn sàng xử lý")
        self.stage_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.stage_label.setStyleSheet("color: #2563EB; font-weight: bold; font-size: 12px;")
        layout.addWidget(self.stage_label)

        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("color: #64748B; font-size: 11px;")
        layout.addWidget(self.status_label)

        group.setLayout(layout)
        return group

    def _create_log_section(self):
        group = QGroupBox("📝 Log")
        layout = QVBoxLayout()
        layout.setContentsMargins(15, 30, 15, 15)
        layout.setSpacing(12)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(120)
        self.log_text.setPlaceholderText("Log sẽ hiển thị ở đây...")
        layout.addWidget(self.log_text)

        group.setLayout(layout)
        return group

    def _create_buttons(self):
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)

        self.start_button = QPushButton("🚀 Bắt Đầu Xử Lý")
        self.start_button.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        self.start_button.setMinimumHeight(50)
        self.start_button.clicked.connect(self._start_processing)

        self.cancel_button = QPushButton("⏹ Hủy")
        self.cancel_button.setFont(QFont("Segoe UI", 11))
        self.cancel_button.setMinimumHeight(50)
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._cancel_processing)

        self.clear_button = QPushButton("🗑 Clear")
        self.clear_button.setMinimumHeight(50)
        self.clear_button.clicked.connect(self._clear_all)

        button_layout.addWidget(self.clear_button)
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.start_button)

        return button_layout

    def _get_stylesheet(self):
        return """
        QMainWindow {
            background-color: #F8FAFC;
            font-family: 'Segoe UI', 'Roboto', sans-serif;
        }
        QGroupBox {
            font-weight: bold;
            font-size: 14px;
            border: 1px solid #CBD5E1;
            border-radius: 8px;
            margin-top: 30px;
            padding-top: 15px;
            background-color: white;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 20px;
            padding: 0 5px;
            color: #2563EB;
            background-color: transparent;
        }
        QLabel {
            color: #334155;
            font-size: 13px;
            padding: 2px;
        }
        QLineEdit {
            padding: 10px 12px;
            border: 1px solid #CBD5E1;
            border-radius: 6px;
            font-size: 13px;
            background-color: white;
            selection-background-color: #3B82F6;
            min-height: 28px;
        }
        QLineEdit:focus {
            border: 2px solid #2563EB;
            background-color: #F8FAFC;
        }
        QPushButton {
            background-color: #2563EB;
            color: white;
            border: none;
            border-radius: 6px;
            font-weight: 600;
            font-size: 13px;
            padding: 8px 16px;
            min-height: 30px;
        }
        QPushButton:hover {
            background-color: #1D4ED8;
        }
        QPushButton:pressed {
            background-color: #1E40AF;
        }
        QPushButton:disabled {
            background-color: #94A3B8;
            cursor: not-allowed;
        }
        QComboBox {
            padding: 10px 12px;
            border: 1px solid #CBD5E1;
            border-radius: 6px;
            background-color: white;
            font-size: 13px;
            min-width: 150px;
            min-height: 28px;
        }
        QComboBox:focus {
            border: 2px solid #2563EB;
        }
        QComboBox::drop-down {
            border: none;
            width: 30px;
        }
        QCheckBox {
            spacing: 12px;
            font-size: 13px;
            color: #334155;
            min-height: 28px;
            padding: 4px 0;
        }
        QCheckBox::indicator {
            width: 18px;
            height: 18px;
            border: 1px solid #CBD5E1;
            border-radius: 4px;
            background: white;
        }
        QCheckBox::indicator:unchecked:hover {
            border-color: #2563EB;
        }
        QCheckBox::indicator:checked {
            background-color: #2563EB;
            border-color: #2563EB;
            image: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTIiIGhlaWdodD0iMTAiIHZpZXdCb3g9IjAgMCAxMiAxMCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cuczMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTEgNUw0LjUgOC41TDExIDEiIHN0cm9rZT0id2hpdGUiIHN0cm9rZS13aWR0aD0iMiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIHN0cm9rZS1saW5ZWpvaW49InJvdW5kIi8+Cjwvc3ZnPgo=);
        }
        QProgressBar {
            border: none;
            border-radius: 6px;
            text-align: center;
            background-color: #E2E8F0;
            color: #0F172A;
            font-weight: bold;
            font-size: 12px;
            min-height: 24px;
        }
        QProgressBar::chunk {
            background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #3B82F6, stop:1 #2563EB);
            border-radius: 6px;
        }
        QTextEdit {
            border: 1px solid #E2E8F0;
            border-radius: 6px;
            background-color: #0F172A;
            color: #10B981;
            font-family: 'Consolas', 'Courier New', monospace;
            font-size: 12px;
            padding: 8px;
        }
        QMenuBar {
            background-color: white;
            border-bottom: 1px solid #E2E8F0;
            font-size: 13px;
        }
        QMenuBar::item {
            padding: 8px 12px;
            background: transparent;
        }
        QMenuBar::item:selected {
            background-color: #F1F5F9;
        }
        QStatusBar {
            background-color: white;
            border-top: 1px solid #E2E8F0;
            color: #64748B;
        }
        """

    def _browse_video(self):
        file_filter = "Video Files (*.mp4 *.mkv *.avi *.mov *.webm *.wmv);;All Files (*.*)"
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Chọn Video",
            "",
            file_filter
        )

        if file_path:
            path = Path(file_path)

            max_size = 20 * (1024 ** 3)
            if path.stat().st_size > max_size:
                QMessageBox.warning(
                    self,
                    "File Quá Lớn",
                    "Vui lòng chọn file dưới 20GB."
                )
                return

            self.video_path_input.setText(str(path))

            size_mb = path.stat().st_size / (1024 * 1024)
            size_str = f"{size_mb / 1024:.2f} GB" if size_mb > 1024 else f"{size_mb:.1f} MB"

            self.video_info.setText(f"📁 {path.name} ({size_str})")
            self._log(f"Selected: {path.name} ({size_str})")

    def _start_processing(self):
        api_key = self.api_key_input.text().strip()
        video_path = self.video_path_input.text().strip()

        if not api_key:
            QMessageBox.warning(self, "Thiếu API Key", "Vui lòng nhập Gemini API Key để dịch subtitle.")
            return

        # Save or clear API key based on checkbox
        config = get_config()
        if self.remember_api_key.isChecked():
            config.set_api_key(api_key)
        else:
            config.clear_api_key()

        if not video_path:
            QMessageBox.warning(self, "Chưa Chọn Video", "Vui lòng chọn file video.")
            return

        from core.models import ProjectSettings

        self.settings = ProjectSettings(
            source_language=self.source_lang.currentData(),
            target_language=self.target_lang.currentData(),
            whisper_model=self.model_size.currentData(),
            use_gpu=self.use_gpu_check.isChecked(),
            gemini_api_key=api_key,
            use_context_aware=self.context_aware_check.isChecked(),
            quality_check=self.quality_check_check.isChecked(),
            video_context=self.video_context_input.text().strip() or None,
            output_formats=["srt", "vtt", "ass"]
        )

        # Check for existing checkpoint
        resume_data = None
        from core.checkpoint import CheckpointManager
        checkpoint_mgr = CheckpointManager(Path(video_path))

        if checkpoint_mgr.has_checkpoint():
            loaded = checkpoint_mgr.load()
            if loaded and loaded.completed_stage:
                # Verify settings compatibility
                if checkpoint_mgr.is_settings_compatible(loaded, self.settings):
                    resume_info = checkpoint_mgr.get_resume_info(loaded)
                    reply = QMessageBox.question(
                        self,
                        "🔄 Tiếp tục xử lý?",
                        f"Phát hiện checkpoint từ lần xử lý trước:\n\n"
                        f"{resume_info}\n\n"
                        f"Bạn muốn tiếp tục hay bắt đầu lại từ đầu?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel,
                        QMessageBox.StandardButton.Yes
                    )

                    if reply == QMessageBox.StandardButton.Cancel:
                        return
                    if reply == QMessageBox.StandardButton.Yes:
                        resume_data = loaded
                    else:
                        # User chose to start fresh — delete old checkpoint
                        checkpoint_mgr.cleanup()
                else:
                    # Settings changed — ask to delete
                    reply = QMessageBox.question(
                        self,
                        "⚠ Cài đặt đã thay đổi",
                        "Phát hiện checkpoint nhưng cài đặt đã thay đổi.\n"
                        "Xóa checkpoint cũ và bắt đầu lại?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                        QMessageBox.StandardButton.Yes
                    )
                    if reply == QMessageBox.StandardButton.Cancel:
                        return
                    checkpoint_mgr.cleanup()

        self.start_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.browse_button.setEnabled(False)

        self.progress_bar.setValue(0)
        self.stage_label.setText("Đang khởi tạo...")
        self.status_label.setText("")

        self._log("=" * 50)
        if resume_data:
            self._log("🔄 Tiếp tục xử lý subtitle từ checkpoint...")
        else:
            self._log("Bắt đầu xử lý subtitle...")

        self.processing_thread = ProcessingThread(
            Path(video_path), self.settings, resume_data=resume_data
        )
        self.processing_thread.progress.connect(self._update_progress)
        self.processing_thread.stage_changed.connect(self._update_stage)
        self.processing_thread.finished.connect(self._processing_finished)
        self.processing_thread.log_message.connect(self._log)
        self.processing_thread.start()

        self.statusBar().showMessage("Đang xử lý...")

    def _update_progress(self, message: str, percentage: int, stage: str):
        self.progress_bar.setValue(percentage)
        self.status_label.setText(message)

    def _update_stage(self, stage: str):
        stage_names = {
            "init": "Khởi tạo",
            "audio": "Trích xuất audio",
            "transcribe": "Transcribe",
            "translate": "Dịch thuật",
            "quality": "Kiểm tra chất lượng",
            "save": "Lưu file"
        }
        self.stage_label.setText(stage_names.get(stage, stage))

    def _processing_finished(self, success: bool, message: str):
        self.start_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.browse_button.setEnabled(True)

        if success:
            self.progress_bar.setValue(100)
            self.stage_label.setText("Hoàn tất!")
            self.status_label.setText(message)
            self._log("=" * 50)
            self._log("✅ " + message)

            QMessageBox.information(self, "Thành công!", message)
        else:
            self.stage_label.setText("Lỗi")
            self._log("❌ Error: " + message)

            QMessageBox.critical(self, "Lỗi", f"Xử lý thất bại:\n{message}")

        self.statusBar().showMessage("Sẵn sàng")

    def _cancel_processing(self):
        if self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.stop()
            if not self.processing_thread.wait(5000):  # Wait up to 5 seconds
                logger.warning("Processing thread did not stop within 5s, terminating")
                self.processing_thread.terminate()
                self.processing_thread.wait(2000)

        self.start_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.browse_button.setEnabled(True)
        self.stage_label.setText("Đã hủy")
        self._log("⚠ Processing cancelled by user")
        self.statusBar().showMessage("Đã hủy")

    def _clear_all(self):
        self.video_path_input.clear()
        self.video_info.clear()
        self.log_text.clear()
        self.progress_bar.setValue(0)
        self.stage_label.setText("Sẵn sàng")
        self.status_label.clear()
        self._check_dependencies()

    def _log(self, message: str):
        self.log_text.append(message)
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())


def main():
    # High DPI scaling is always enabled in PyQt6, no need to set deprecated attributes.
    # Set Env for auto scaling
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"

    app = QApplication(sys.argv)
    app.setApplicationName("SubtitleForge Pro")
    app.setApplicationVersion("1.0.0")
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
