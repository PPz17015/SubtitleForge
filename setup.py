from setuptools import setup, find_packages

setup(
    name="subtitleforge-pro",
    version="1.0.0",
    description="Professional offline subtitle generation with speaker diarization and context-aware translation",
    author="SubtitleForge Team",
    packages=find_packages(),
    install_requires=[
        "ffmpeg-python>=0.2.0",
        "faster-whisper>=1.0.0",
        "pyannote.audio>=3.0.0",
        "google-genai>=1.0.0",
        "tqdm>=4.66.0",
        "pyyaml>=6.0",
        "python-dotenv>=1.0.0",
        "click>=8.1.0",
        "rich>=13.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "srt>=3.5.0",
        "pysrt>=1.1.10",
        "webvtt-py>=0.5.1",
        "pyqt6>=6.5.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pyinstaller>=6.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "subtitleforge=src.main:cli",
        ],
    },
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: End Users/Desktop",
        "Topic :: Multimedia :: Video :: Subtitles",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
