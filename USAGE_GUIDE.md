# SubtitleForge Pro — Usage Guide

Automatic subtitle generation from video using AI. Simply select a video file → the system processes it automatically → notifies you when complete.

---

## 1. Installation

```bash
pip install -r requirements.txt
```

> **Requirements:** Python 3.10+, FFmpeg installed and available in PATH, NVIDIA GPU (optional, for acceleration).

---

## 2. Launch the Application

```bash
python src/gui.py
```

The **SubtitleForge Pro** window will appear.

---

## 3. Step-by-Step Usage

### Step 1: Enter Gemini API Key

Enter your API key in the **"Gemini API Key"** field under **🔑 API Configuration**.

> 💡 Get a free API key at: [Google AI Studio](https://aistudio.google.com/app/apikey)

### Step 2: Select Video File

Click **📂 Browse** → select your video file (MP4, MKV, AVI, MOV, up to 20GB).

File information (name, size) will be displayed below.

### Step 3: Configuration (Optional)

The **⚙️ Settings** section has the following options. Defaults are already optimized — you can skip this if no changes are needed:

| Option | Default | Description |
|--------|---------|-------------|
| Source Language | Japanese | Original language of the video |
| Target Language | Vietnamese | Language to translate into |
| Whisper Model | Small | Speech recognition model (Medium for higher quality) |
| Use GPU | ✅ On | Speed up processing with NVIDIA GPU |
| Context-Aware Translation | ✅ On | AI analyzes characters and scenes for more accurate translation |
| Quality Check | ✅ On | Automatically verify and fix translation errors |
| Video Context | (empty) | Describe video content for better translation |

**Video context suggestions:**
- `Family anime, mother and son conversation`
- `Office J-Drama, coworkers at the same company`
- `Action movie, police investigation team`

### Step 4: Start Processing

Click **🚀 Start Processing**. The system will automatically run through 5 stages:

```
1. Audio Extraction    → Extract audio from video
2. Transcription       → Speech-to-text recognition (Whisper AI)
3. Translation         → Translate subtitles to target language (Gemini AI)
4. Quality Check       → Auto-fix mistranslations, pronoun errors
5. Save Files          → Export subtitle files (SRT + VTT + ASS)
```

Monitor progress via:
- **Progress bar** — completion percentage
- **Status** — current stage
- **Log** — detailed step-by-step output

### Step 5: Get Results

When complete, the app will show **"Processing complete!"**.

Subtitle files are saved in the same directory as the video:

```
📁 Video directory/
├── movie.srt    ← SubRip (most common)
├── movie.vtt    ← WebVTT (for web)
└── movie.ass    ← ASS (with styling)
```

---

## 4. Choosing a Whisper Model

| Model | VRAM | Speed | Quality | When to Use |
|-------|------|-------|---------|-------------|
| Tiny | ~1GB | Very fast | Low | Quick testing |
| Small | ~2GB | Moderate | Good | Default, sufficient for most use |
| Medium | ~5GB | Slow | Very good | Complex Japanese audio |
| Large | ~10GB | Very slow | Excellent | Highest quality |

---

## 5. Quality Check System (Automatic)

When **"Quality Check"** is enabled, the system automatically:

1. **Character Analysis** — identifies speakers and determines tone
2. **Scene Detection** — splits video into scenes based on silence gaps
3. **Smart Translation** — prompts include character + scene information
4. **Batch Quality Check** — verifies 50 segments per API call (fast, cost-effective)
5. **Coherence Check** — ensures consistent pronouns and names throughout
6. **Auto-Correction** — re-translates erroneous segments with specific feedback
7. **Final Verification** — second quality pass to confirm all fixes are correct

---

## 6. Translation Performance

| Setting | Value |
|---------|-------|
| Batch size | 50 segments per API call |
| Rate limiting | Adaptive (0.15s base, auto-throttle on 429) |
| Quality check | 50 segments per check batch |
| Checkpoint | Auto-save every 50 translated segments |

---

## 7. Troubleshooting

| Issue | Solution |
|-------|----------|
| FFmpeg not found | Install FFmpeg and add to PATH |
| GPU / CUDA error | Uncheck "Use GPU", run on CPU |
| Slow translation | Switch Whisper Model to Small |
| API key error | Verify key at [Google AI Studio](https://aistudio.google.com/app/apikey) |
| File too large | Limit is 20GB, recommended under 10GB |
| Want to cancel mid-process | Click **⏹ Cancel** button |
