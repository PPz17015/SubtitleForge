import logging
from pathlib import Path
from typing import Optional

from .models import ProjectSettings, SubtitleSegment

logger = logging.getLogger(__name__)


class SubtitleGenerator:
    """Generate subtitle files in various formats."""

    def __init__(self, settings: Optional[ProjectSettings] = None):
        self.settings = settings or ProjectSettings()

    def generate_srt(
        self,
        segments: list[SubtitleSegment],
        include_translation: bool = True,
        include_speaker: bool = True
    ) -> str:
        """
        Generate SRT subtitle file content.

        Args:
            segments: List of subtitle segments
            include_translation: Whether to include translation
            include_speaker: Whether to include speaker labels

        Returns:
            SRT file content as string
        """
        output_lines = []

        for segment in segments:
            # Determine text to use
            text = segment.translation if include_translation and segment.translation else segment.text

            # Add speaker label if requested
            if include_speaker and segment.speaker:
                text = f"[{segment.speaker}] {text}"

            # Format subtitle
            output_lines.append(str(segment.index))
            output_lines.append(f"{self._format_timestamp(segment.start_time)} --> {self._format_timestamp(segment.end_time)}")
            output_lines.append(text)
            output_lines.append("")  # Empty line between subtitles

        return "\n".join(output_lines)

    def generate_vtt(
        self,
        segments: list[SubtitleSegment],
        include_translation: bool = True,
        include_speaker: bool = True
    ) -> str:
        """Generate WebVTT subtitle file content."""
        output_lines = ["WEBVTT", ""]

        for segment in segments:
            # Determine text to use
            text = segment.translation if include_translation and segment.translation else segment.text

            # Add speaker label if requested
            if include_speaker and segment.speaker:
                text = f"<v {segment.speaker}>{text}"

            # Format subtitle
            output_lines.append(f"{self._format_timestamp_vtt(segment.start_time)} --> {self._format_timestamp_vtt(segment.end_time)}")
            output_lines.append(text)
            output_lines.append("")

        return "\n".join(output_lines)

    def generate_ass(
        self,
        segments: list[SubtitleSegment],
        include_translation: bool = True,
        include_speaker: bool = True
    ) -> str:
        """Generate ASS/SSA subtitle file content."""
        # Header
        header = """[Script Info]
Title: SubtitleForge Pro Export
ScriptType: v4.00+
Collisions: Normal
PlayDepth: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

        output_lines = [header]

        for segment in segments:
            # Determine text to use
            text = segment.translation if include_translation and segment.translation else segment.text

            # Escape special characters FIRST (before adding ASS tags)
            text = text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")

            # Add speaker label if requested (after escaping)
            if include_speaker and segment.speaker:
                text = f"{{\\an8}}[{segment.speaker}]\\N{text}"

            # Format subtitle
            start = self._format_timestamp_ass(segment.start_time)
            end = self._format_timestamp_ass(segment.end_time)

            output_lines.append(
                f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}"
            )

        return "\n".join(output_lines)

    def generate_text(
        self,
        segments: list[SubtitleSegment],
        include_translation: bool = True,
        include_speaker: bool = True,
        include_timestamps: bool = False
    ) -> str:
        """Generate plain text transcript."""
        output_lines = []

        for segment in segments:
            # Determine text to use
            text = segment.translation if include_translation and segment.translation else segment.text

            # Add speaker label if requested
            if include_speaker and segment.speaker:
                text = f"[{segment.speaker}] {text}"

            if include_timestamps:
                text = f"[{self._format_timestamp(segment.start_time)}] {text}"

            output_lines.append(text)

        return "\n".join(output_lines)

    def save_srt(
        self,
        segments: list[SubtitleSegment],
        output_path: Path,
        include_translation: bool = True,
        include_speaker: bool = True
    ):
        """Save SRT file to disk."""
        content = self.generate_srt(segments, include_translation, include_speaker)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)

        logger.info(f"SRT saved to {output_path}")

    def save_vtt(
        self,
        segments: list[SubtitleSegment],
        output_path: Path,
        include_translation: bool = True,
        include_speaker: bool = True
    ):
        """Save VTT file to disk."""
        content = self.generate_vtt(segments, include_translation, include_speaker)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)

        logger.info(f"VTT saved to {output_path}")

    def save_ass(
        self,
        segments: list[SubtitleSegment],
        output_path: Path,
        include_translation: bool = True,
        include_speaker: bool = True
    ):
        """Save ASS file to disk."""
        content = self.generate_ass(segments, include_translation, include_speaker)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)

        logger.info(f"ASS saved to {output_path}")

    def save_all_formats(
        self,
        segments: list[SubtitleSegment],
        output_dir: Path,
        base_name: str,
        include_translation: bool = True,
        include_speaker: bool = True
    ):
        """Save all configured output formats."""
        output_dir.mkdir(parents=True, exist_ok=True)

        formats = self.settings.output_formats

        if "srt" in formats:
            self.save_srt(
                segments,
                output_dir / f"{base_name}.srt",
                include_translation,
                include_speaker
            )

        if "vtt" in formats:
            self.save_vtt(
                segments,
                output_dir / f"{base_name}.vtt",
                include_translation,
                include_speaker
            )

        if "ass" in formats:
            self.save_ass(
                segments,
                output_dir / f"{base_name}.ass",
                include_translation,
                include_speaker
            )

        # Also save plain text transcript
        text_path = output_dir / f"{base_name}.txt"
        content = self.generate_text(segments, include_translation, include_speaker, False)
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(content)

        logger.info(f"All formats saved to {output_dir}")

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        """Format timestamp for SRT (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    @staticmethod
    def _format_timestamp_vtt(seconds: float) -> str:
        """Format timestamp for VTT (HH:MM:SS.mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

    @staticmethod
    def _format_timestamp_ass(seconds: float) -> str:
        """Format timestamp for ASS (H:MM:SS.cc)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        centisecs = int((seconds % 1) * 100)
        return f"{hours}:{minutes:02d}:{secs:02d}.{centisecs:02d}"


class SubtitleOptimizer:
    """Optimize subtitle timing and formatting."""

    def __init__(self, settings: Optional[ProjectSettings] = None):
        self.settings = settings or ProjectSettings()

    def optimize_segments(self, segments: list[SubtitleSegment]) -> list[SubtitleSegment]:
        """Optimize subtitle segments for better readability."""

        # Apply timing constraints
        optimized = []

        for segment in segments:
            # Enforce minimum duration
            if segment.duration() < self.settings.min_duration:
                segment.end_time = segment.start_time + self.settings.min_duration

            # Enforce maximum duration
            if segment.duration() > self.settings.max_duration:
                segment.end_time = segment.start_time + self.settings.max_duration

            # Split long segments if needed
            text_length = len(segment.text)
            max_chars = self.settings.max_chars_per_line * self.settings.max_lines

            if text_length > max_chars:
                split_point = self._find_split_point(segment.text, max_chars // 2)

                if split_point > 0:
                    ratio = split_point / text_length

                    trans_split = None
                    if segment.translation:
                        trans_len = len(segment.translation)
                        raw_trans_pos = int(trans_len * ratio)
                        trans_split = self._find_split_point(segment.translation, raw_trans_pos)

                    first_part = SubtitleSegment(
                        index=len(optimized) + 1,
                        start_time=segment.start_time,
                        end_time=segment.start_time + (segment.duration() * split_point / text_length),
                        text=segment.text[:split_point],
                        speaker=segment.speaker,
                        translation=segment.translation[:trans_split] if segment.translation and trans_split else None
                    )
                    optimized.append(first_part)

                    second_part = SubtitleSegment(
                        index=len(optimized) + 1,
                        start_time=first_part.end_time + self.settings.gap_between_subtitles,
                        end_time=segment.end_time,
                        text=segment.text[split_point:],
                        speaker=segment.speaker,
                        translation=segment.translation[trans_split:] if segment.translation and trans_split else None
                    )
                    optimized.append(second_part)

                    continue

            optimized.append(segment)

        # Re-index segments
        for i, seg in enumerate(optimized):
            seg.index = i + 1

        # Enforce gap between subtitles
        optimized = self._enforce_gaps(optimized)

        return optimized

    def _find_split_point(self, text: str, target_pos: int) -> int:
        """Find a good point to split text."""
        # Try to split at sentence boundary
        for char in [".", "!", "?", "。", "！", "？"]:
            pos = text.rfind(char, 0, target_pos)
            if pos > target_pos // 2:
                return pos + 1

        # Try to split at comma or pause
        for char in [",", "、", "，", ";", ":", ";"]:
            pos = text.rfind(char, 0, target_pos)
            if pos > target_pos // 3:
                return pos + 1

        # Fall back to splitting at space
        pos = text.rfind(" ", 0, target_pos)
        if pos > 0:
            return pos

        return target_pos

    def _enforce_gaps(self, segments: list[SubtitleSegment]) -> list[SubtitleSegment]:
        """Ensure minimum gap between subtitles."""
        if not segments:
            return segments

        min_gap = self.settings.gap_between_subtitles

        for i in range(1, len(segments)):
            prev_end = segments[i - 1].end_time
            curr_start = segments[i].start_time

            if curr_start - prev_end < min_gap:
                # Adjust current segment start
                segments[i].start_time = prev_end + min_gap

                # Adjust duration to maintain end time
                duration = segments[i].end_time - curr_start
                segments[i].end_time = segments[i].start_time + duration

        return segments
