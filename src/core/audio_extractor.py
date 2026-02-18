import logging
import shutil
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class AudioExtractor:
    def __init__(self):
        self.supported_formats = ['.mp4', '.mkv', '.avi', '.mov', '.webm', '.wmv', '.flv']
        self.ffmpeg_path = self._find_ffmpeg()

    def _find_ffmpeg(self) -> str:
        """Find ffmpeg executable in PATH or common locations."""
        ffmpeg_exe = shutil.which('ffmpeg')
        if ffmpeg_exe:
            return ffmpeg_exe

        # Check common Windows locations
        common_paths = [
            Path(r'C:\ffmpeg\bin\ffmpeg.exe'),
            Path(r'C:\Program Files\ffmpeg\bin\ffmpeg.exe'),
            Path(r'C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe'),
            Path.home() / 'ffmpeg' / 'bin' / 'ffmpeg.exe',
        ]

        # Check WinGet installation location
        winget_base = Path.home() / 'AppData' / 'Local' / 'Microsoft' / 'WinGet' / 'Packages'
        if winget_base.exists():
            for pkg_dir in winget_base.glob('Gyan.FFmpeg*'):
                for bin_dir in pkg_dir.rglob('bin'):
                    ffmpeg_path = bin_dir / 'ffmpeg.exe'
                    if ffmpeg_path.exists():
                        return str(ffmpeg_path)

        for path in common_paths:
            if path.exists():
                return str(path)

        # Return 'ffmpeg' and let it fail with a clear error
        return 'ffmpeg'

    def _find_ffprobe(self) -> str:
        """Find ffprobe executable in PATH or common locations."""
        ffprobe_exe = shutil.which('ffprobe')
        if ffprobe_exe:
            return ffprobe_exe

        # Check common Windows locations
        common_paths = [
            Path(r'C:\ffmpeg\bin\ffprobe.exe'),
            Path(r'C:\Program Files\ffmpeg\bin\ffprobe.exe'),
            Path(r'C:\Program Files (x86)\ffmpeg\bin\ffprobe.exe'),
            Path.home() / 'ffmpeg' / 'bin' / 'ffprobe.exe',
        ]

        # Check WinGet installation location
        winget_base = Path.home() / 'AppData' / 'Local' / 'Microsoft' / 'WinGet' / 'Packages'
        if winget_base.exists():
            for pkg_dir in winget_base.glob('Gyan.FFmpeg*'):
                for bin_dir in pkg_dir.rglob('bin'):
                    ffprobe_path = bin_dir / 'ffprobe.exe'
                    if ffprobe_path.exists():
                        return str(ffprobe_path)

        for path in common_paths:
            if path.exists():
                return str(path)

        return 'ffprobe'

    def extract_audio(
        self,
        video_path: Path,
        output_path: Optional[Path] = None,
        sample_rate: int = 16000,
        channels: int = 1,
        codec: str = 'pcm_s16le'
    ) -> tuple[Path, float]:
        """
        Extract audio from video file.

        Args:
            video_path: Path to video file
            output_path: Output audio file path (optional)
            sample_rate: Target sample rate (default: 16000 for Whisper)
            channels: Number of audio channels (default: 1 for mono)
            codec: Audio codec to use

        Returns:
            Tuple of (output_path, duration_seconds)
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Check if ffmpeg is available
        if self.ffmpeg_path == 'ffmpeg' and not shutil.which('ffmpeg'):
            raise FileNotFoundError(
                "FFmpeg not found. Please install FFmpeg and add it to your PATH.\n"
                "Download from: https://ffmpeg.org/download.html\n"
                "For Windows: Extract to C:\\ffmpeg and add C:\\ffmpeg\\bin to PATH"
            )

        if output_path is None:
            output_path = video_path.with_suffix('.wav')

        # Build ffmpeg command
        cmd = [
            self.ffmpeg_path,
            '-i', str(video_path),
            '-vn',  # No video
            '-acodec', codec,
            '-ar', str(sample_rate),
            '-ac', str(channels),
            '-y',  # Overwrite output
            str(output_path)
        ]

        logger.info(f"Extracting audio from {video_path}")

        try:
            subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )

            # Get duration
            duration = self._get_duration(output_path)
            logger.info(f"Audio extracted successfully: {output_path}, duration: {duration}s")

            return output_path, duration

        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg error: {e.stderr}")
            raise RuntimeError(f"Failed to extract audio: {e.stderr}") from e

    def _get_duration(self, audio_path: Path) -> float:
        """Get audio duration in seconds using ffprobe."""
        cmd = [
            self._find_ffprobe(),
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            str(audio_path)
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return float(result.stdout.strip())
        except Exception as e:
            logger.warning(f"Could not get duration: {e}")
            return 0.0

    def is_valid_video(self, path: Path) -> bool:
        """Check if file is a valid video file."""
        if not path.exists():
            return False
        return path.suffix.lower() in self.supported_formats

    def get_video_info(self, video_path: Path) -> dict:
        """Get video file information."""
        cmd = [
            self._find_ffprobe(),
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            str(video_path)
        ]

        try:
            import json
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return json.loads(result.stdout)
        except Exception as e:
            logger.warning(f"Could not get video info: {e}")
            return {}
