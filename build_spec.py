# -*- mode: python ; coding: utf-8 -*-
import sys
from pathlib import Path

block_cipher = None

# Add source directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

a = Analysis(
    ['src/gui.py'],
    pathex=[],
    binaries=[],
    datas=[
        # Include required data files
    ],
    hiddenimports=[
        # Core modules
        'core',
        'core.models',
        'core.audio_extractor',
        'core.subtitle_generator',
        'core.batch_processing',
        # Transcription
        'transcription',
        'transcription.whisper_transcriber',
        # Diarization
        'diarization',
        'diarization.speaker_diarizer',
        # Translation
        'translation',
        'translation.translator',
        # API
        'api',
        'api.models',
        'api.routes',
        'api.server',
        'api.job_manager',
        'api.websocket',
        'api.plugin_system',
        # Dependencies
        'PyQt6',
        'PyQt6.QtCore',
        'PyQt6.QtGui',
        'PyQt6.QtWidgets',
        'click',
        'rich',
        'rich.console',
        'rich.progress',
        'ffmpeg',
        'yaml',
        'numpy',
        'pandas',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'test',
        'tests',
        '.pytest_cache',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='SubtitleForgePro',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # No console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='assets/icon.ico' if Path('assets/icon.ico').exists() else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='SubtitleForgePro',
)

# Optional: Create single file executable
# exe = EXE(pyz, a.scripts, a.binaries, a.zipfiles, a.datas, [], name='SubtitleForgePro', console=False)
