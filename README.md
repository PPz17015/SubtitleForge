# SubtitleForge Pro

Professional-grade subtitle generation and translation application powered by **Faster Whisper** and **Google Gemini AI**. Supports speaker diarization, context-aware translation, quality self-check, and checkpoint-based resume.

## ✨ Features

- **🎙️ Transcription**: Faster Whisper (CTranslate2) — CPU & GPU supported
- **🌐 Translation**: Google Gemini 2.5 Flash with adaptive rate limiting
- **🔍 Quality Self-Check**: Automatic quality review & correction loop
- **🔗 Context Coherence**: Cross-segment consistency check (pronouns, terms)
- **💾 Checkpoint Resume**: Pause anytime — resume from where you left off
- **🎯 Speaker Diarization**: Auto-detect speakers (via PyAnnote)
- **📝 Multiple Formats**: SRT, VTT, ASS output
- **🖥️ Desktop GUI**: PyQt6-based desktop app with real-time progress

## 🏗️ Architecture

```
src/
├── core/
│   ├── models.py               # Data models (Project, SubtitleSegment)
│   ├── audio_extractor.py      # FFmpeg audio extraction
│   ├── subtitle_generator.py   # SRT/VTT/ASS generation
│   ├── checkpoint.py           # Checkpoint save/load/resume
│   └── batch_processing.py     # Batch job management
├── transcription/
│   └── whisper_transcriber.py  # Faster Whisper transcription
├── diarization/
│   └── speaker_diarizer.py     # Speaker diarization
├── translation/
│   └── translator.py           # Gemini translation + quality check
├── api/
│   ├── server.py               # REST API server
│   ├── routes.py               # API endpoints
│   ├── models.py               # API request/response models
│   ├── job_manager.py          # Job queue management
│   ├── plugin_system.py        # Plugin loader
│   └── websocket.py            # WebSocket support
├── ui/                         # UI components
├── gui.py                      # PyQt6 desktop GUI
├── main.py                     # CLI entry point
└── config.py                   # App configuration
plugins/
└── json_output.py              # JSON output plugin
tests/                          # 160+ unit tests
```

## 🚀 Installation

### Prerequisites

1. **Python 3.9+**
2. **FFmpeg**: Required for audio extraction
   ```bash
   # Windows (winget)
   winget install Gyan.FFmpeg

   # macOS
   brew install ffmpeg

   # Linux
   sudo apt install ffmpeg
   ```

### Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configure Gemini API

Get API key from [Google AI Studio](https://aistudio.google.com/app/apikey):

```bash
export GEMINI_API_KEY=your_api_key_here
```

## 📖 Usage

### Desktop GUI (Recommended)

```bash
python src/gui.py
```

The GUI provides:
- Video file selection
- Real-time progress with detailed status per batch
- Checkpoint resume dialog
- Quality check toggle
- All settings in one place

### CLI

```bash
# Basic: Japanese → Vietnamese
python -m src.main generate video.mp4 --source ja --target vi --gemini-key YOUR_KEY

# Use medium model for better accuracy
python -m src.main generate video.mp4 --model medium

# Multiple output formats
python -m src.main generate video.mp4 -f srt -f vtt -f ass

# Check system info
python -m src.main info
```

### CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--source, -s` | `ja` | Source language code |
| `--target, -t` | `vi` | Target language code |
| `--model, -m` | `small` | Whisper model (tiny/base/small/medium/large) |
| `--output, -o` | same dir | Output directory |
| `--gemini-key` | env var | Gemini API key |
| `--no-gpu` | false | Force CPU mode |
| `--context-aware` | true | Context-aware translation |
| `--formats, -f` | `srt` | Output formats |

## ⚡ Performance

### Whisper Models

| Model | Parameters | VRAM | Relative Speed | Recommended For |
|-------|-----------|------|----------------|-----------------|
| tiny | 39M | ~1GB | 10x | Quick draft |
| base | 74M | ~1GB | 7x | Basic use |
| **small** | 244M | ~2GB | 4x | **Default — good balance** |
| medium | 769M | ~5GB | 2x | Better accuracy |
| large-v3 | 1550M | ~10GB | 1x | Best accuracy |

### Translation Speed

- **Batch size**: 40 segments per API call
- **Quality check**: 50 segments per check
- **Rate limiting**: Adaptive (1s base, auto-throttle on 429)
- **Checkpoint**: Auto-save every 50 translated segments

**Benchmarks** (1266 segments, Gemini 2.5 Flash):

| Quality Check | API Calls | Estimated Time |
|--------------|-----------|----------------|
| ✅ On | ~104 | ~25-30 min |
| ❌ Off | ~33 | ~8-10 min |

### GPU vs CPU

| Hardware | Whisper medium (3h video) |
|----------|--------------------------|
| GTX 1650+ (4GB VRAM) | ~15-20 min |
| i5-11400 (12 threads) | ~25-40 min |
| i5-9400F (6 threads) | ~40-60 min |

> Translation (Gemini API) does not use GPU — only internet.

## 🔄 Checkpoint System

Processing automatically saves progress after each stage:

1. ✅ Audio extraction
2. ✅ Transcription
3. ✅ Translation (saved every 50 segments)
4. ✅ Quality check
5. ✅ Output generation

If interrupted, the app detects the checkpoint on next run and offers to resume.

## 🌐 Supported Languages

| Code | Language |
|------|----------|
| `ja` | Japanese (日本語) |
| `vi` | Vietnamese (Tiếng Việt) |
| `en` | English |
| `zh` | Chinese (中文) |
| `ko` | Korean (한국어) |

**Japanese → Vietnamese** has special optimizations:
- Pronoun handling (母-子, 先生-生徒)
- Honorific mapping (-san, -sensei, -chan → Vietnamese equivalents)
- Formality adjustment based on context

## 🧪 Testing

```bash
# Run all tests (160+)
python -m pytest tests/ -q

# Run with coverage
python -m pytest tests/ --cov=src

# Lint
python -m ruff check src/ tests/
```

## 🔧 Tech Stack

| Component | Technology |
|-----------|-----------|
| Transcription | [Faster Whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2) |
| Translation | [Google GenAI SDK](https://github.com/googleapis/python-genai) (Gemini 2.5 Flash) |
| Diarization | [PyAnnote](https://github.com/pyannote/pyannote-audio) |
| Audio | FFmpeg |
| GUI | PyQt6 |
| Testing | pytest |
| Linting | Ruff |

## 📄 License

MIT License

## 🆘 Troubleshooting

### FFmpeg not found
```bash
ffmpeg -version  # Verify FFmpeg is in PATH
```

### CUDA/GPU issues
```bash
python -m src.main generate video.mp4 --no-gpu  # Force CPU mode
```

### Rate limiting (429 errors)
The app has adaptive rate limiting built-in. If you see excessive 429 errors:
- Reduce translation batch size in settings
- Check your [Gemini API quota](https://aistudio.google.com/)

### Segments missing translation
- Try a larger Whisper model (`medium` or `large-v3`) for better transcription
- Check if audio quality is too low
