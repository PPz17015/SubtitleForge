@echo off
echo ====================================
echo Building SubtitleForge Pro .exe
echo ====================================

REM Set PYTHONPATH
set PYTHONPATH=%~dp0src

REM Build with PyInstaller
pyinstaller --name SubtitleForgePro ^
    --onedir ^
    --windowed ^
    --add-data "src;src" ^
    --hidden-import=PyQt6 ^
    --hidden-import=PyQt6.QtCore ^
    --hidden-import=PyQt6.QtGui ^
    --hidden-import=PyQt6.QtWidgets ^
    --hidden-import=core ^
    --hidden-import=core.models ^
    --hidden-import=core.audio_extractor ^
    --hidden-import=core.subtitle_generator ^
    --hidden-import=transcription ^
    --hidden-import=transcription.whisper_transcriber ^
    --hidden-import=translation ^
    --hidden-import=translation.translator ^
    --hidden-import=diarization ^
    --hidden-import=click ^
    --hidden-import=rich ^
    --hidden-import=rich.progress ^
    --hidden-import=ffmpeg ^
    --hidden-import=yaml ^
    --hidden-import=numpy ^
    --hidden-import=pandas ^
    src/gui.py

echo.
echo ====================================
echo Build complete!
echo Output: dist\SubtitleForgePro\SubtitleForgePro.exe
echo ====================================
pause
