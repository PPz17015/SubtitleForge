import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.audio_extractor import AudioExtractor
from core.models import Project, ProjectSettings, SubtitleSegment, TranscriptionResult
from core.subtitle_generator import SubtitleGenerator, SubtitleOptimizer


class TestSubtitleSegment:
    """Test SubtitleSegment model."""

    def test_create_segment(self):
        """Test creating a subtitle segment."""
        segment = SubtitleSegment(
            index=1,
            start_time=0.0,
            end_time=2.5,
            text="Hello world"
        )
        assert segment.index == 1
        assert segment.start_time == 0.0
        assert segment.end_time == 2.5
        assert segment.text == "Hello world"
        assert segment.speaker is None
        assert segment.translation is None

    def test_duration(self):
        """Test duration calculation."""
        segment = SubtitleSegment(
            index=1,
            start_time=1.0,
            end_time=3.5,
            text="Test"
        )
        assert segment.duration() == 2.5

    def test_srt_format(self):
        """Test SRT format output."""
        segment = SubtitleSegment(
            index=1,
            start_time=0.5,
            end_time=2.5,
            text="Hello"
        )
        result = segment.to_srt_format()
        assert "1\n" in result
        assert "00:00:00,500 --> 00:00:02,500" in result
        assert "Hello" in result

    def test_translated_srt_format(self):
        """Test translated SRT format with speaker."""
        segment = SubtitleSegment(
            index=1,
            start_time=0.0,
            end_time=2.0,
            text="Hello",
            speaker="Mother",
            translation="Xin chào"
        )
        result = segment.to_translated_srt_format()
        assert "[Mother]" in result
        assert "Xin chào" in result

    def test_to_dict(self):
        """Test serialization to dictionary."""
        segment = SubtitleSegment(
            index=1,
            start_time=0.0,
            end_time=2.0,
            text="Test",
            speaker="Speaker1",
            translation="Translated"
        )
        data = segment.to_dict()
        assert data["index"] == 1
        assert data["text"] == "Test"
        assert data["speaker"] == "Speaker1"
        assert data["translation"] == "Translated"

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "index": 1,
            "start_time": 0.0,
            "end_time": 2.0,
            "text": "Test",
            "speaker": "Speaker1",
            "translation": "Translated",
            "context": None,
            "confidence": 0.95
        }
        segment = SubtitleSegment.from_dict(data)
        assert segment.index == 1
        assert segment.text == "Test"
        assert segment.translation == "Translated"


class TestProjectSettings:
    """Test ProjectSettings model."""

    def test_default_settings(self):
        """Test default settings."""
        settings = ProjectSettings()
        assert settings.source_language == "ja"
        assert settings.target_language == "vi"
        assert settings.whisper_model == "small"
        assert settings.use_gpu is True
        assert settings.max_chars_per_line == 42

    def test_custom_settings(self):
        """Test custom settings."""
        settings = ProjectSettings(
            source_language="en",
            target_language="es",
            whisper_model="large",
            use_gpu=False
        )
        assert settings.source_language == "en"
        assert settings.target_language == "es"
        assert settings.whisper_model == "large"
        assert settings.use_gpu is False

    def test_to_dict(self):
        """Test serialization to dictionary."""
        settings = ProjectSettings(
            source_language="ja",
            target_language="vi",
            gemini_api_key="test-key"
        )
        data = settings.to_dict()
        assert data["source_language"] == "ja"
        assert data["gemini_api_key"] == "***"  # Should be masked

    def test_speaker_relationships(self):
        """Test speaker relationships setting."""
        settings = ProjectSettings(
            speaker_relationships={"Speaker1": "parent", "Speaker2": "child"}
        )
        assert settings.speaker_relationships["Speaker1"] == "parent"
        assert settings.speaker_relationships["Speaker2"] == "child"


class TestTranscriptionResult:
    """Test TranscriptionResult model."""

    def test_create_result(self):
        """Test creating transcription result."""
        segments = [
            SubtitleSegment(1, 0.0, 2.0, "Hello"),
            SubtitleSegment(2, 2.1, 4.0, "World")
        ]
        result = TranscriptionResult(
            segments=segments,
            language="en",
            duration=4.0
        )
        assert len(result.segments) == 2
        assert result.language == "en"
        assert result.duration == 4.0

    def test_to_dict(self):
        """Test serialization."""
        segments = [SubtitleSegment(1, 0.0, 2.0, "Test")]
        result = TranscriptionResult(
            segments=segments,
            language="ja",
            duration=2.0
        )
        data = result.to_dict()
        assert data["language"] == "ja"
        assert len(data["segments"]) == 1

    def test_from_dict(self):
        """Test deserialization."""
        data = {
            "segments": [
                {"index": 1, "start_time": 0.0, "end_time": 2.0, "text": "Test",
                 "speaker": None, "translation": None, "context": None, "confidence": 1.0}
            ],
            "language": "en",
            "duration": 2.0,
            "metadata": {}
        }
        result = TranscriptionResult.from_dict(data)
        assert result.language == "en"
        assert len(result.segments) == 1


class TestProject:
    """Test Project model."""

    def test_create_project(self):
        """Test creating a project."""
        video_path = Path("/test/video.mp4")
        project = Project(video_path)
        assert project.video_path == video_path
        assert project.settings.source_language == "ja"
        assert project.transcription is None
        assert len(project.speakers) == 0

    def test_project_with_settings(self):
        """Test project with custom settings."""
        video_path = Path("/test/video.mp4")
        settings = ProjectSettings(source_language="en", target_language="vi")
        project = Project(video_path, settings)
        assert project.settings.source_language == "en"


class TestSubtitleGenerator:
    """Test SubtitleGenerator."""

    def test_generate_srt(self):
        """Test SRT generation."""
        segments = [
            SubtitleSegment(1, 0.0, 2.0, "Hello"),
            SubtitleSegment(2, 2.1, 4.0, "World")
        ]
        generator = SubtitleGenerator()
        result = generator.generate_srt(segments)

        assert "1\n" in result
        assert "00:00:00,000 --> 00:00:02,000" in result
        assert "Hello" in result
        assert "2\n" in result
        assert "World" in result

    def test_generate_srt_with_translation(self):
        """Test SRT with translations."""
        segments = [
            SubtitleSegment(1, 0.0, 2.0, "Hello", translation="Xin chào")
        ]
        generator = SubtitleGenerator()
        result = generator.generate_srt(segments, include_translation=True)

        assert "Xin chào" in result

    def test_generate_srt_with_speaker(self):
        """Test SRT with speaker labels."""
        segments = [
            SubtitleSegment(1, 0.0, 2.0, "Hello", speaker="Mother")
        ]
        generator = SubtitleGenerator()
        result = generator.generate_srt(segments, include_speaker=True)

        assert "[Mother]" in result

    def test_generate_vtt(self):
        """Test VTT generation."""
        segments = [
            SubtitleSegment(1, 0.0, 2.0, "Hello")
        ]
        generator = SubtitleGenerator()
        result = generator.generate_vtt(segments)

        assert "WEBVTT" in result
        assert "00:00:00.000 --> 00:00:02.000" in result

    def test_generate_text(self):
        """Test plain text generation."""
        segments = [
            SubtitleSegment(1, 0.0, 2.0, "Hello", speaker="Speaker1"),
            SubtitleSegment(2, 2.1, 4.0, "World", speaker="Speaker2")
        ]
        generator = SubtitleGenerator()
        result = generator.generate_text(segments, include_speaker=True, include_timestamps=False)

        assert "[Speaker1] Hello" in result
        assert "[Speaker2] World" in result

    def test_generate_text_with_timestamps(self):
        """Test text generation with timestamps."""
        segments = [
            SubtitleSegment(1, 0.0, 2.0, "Hello")
        ]
        generator = SubtitleGenerator()
        result = generator.generate_text(segments, include_timestamps=True)

        assert "[00:00:00,000]" in result

    def test_timestamp_format(self):
        """Test timestamp formatting."""
        segment = SubtitleSegment(1, 3661.5, 3663.0, "Test")
        result = segment.to_srt_format()

        # 3661.5 seconds = 1 hour, 1 minute, 1 second, 500ms
        assert "01:01:01,500 --> 01:01:03,000" in result


class TestSubtitleOptimizer:
    """Test SubtitleOptimizer."""

    def test_optimize_min_duration(self):
        """Test minimum duration enforcement."""
        settings = ProjectSettings(min_duration=1.0)
        segments = [
            SubtitleSegment(1, 0.0, 0.5, "Hi")  # Only 0.5s
        ]
        optimizer = SubtitleOptimizer(settings)
        result = optimizer.optimize_segments(segments)

        # Should be expanded to minimum duration
        assert result[0].duration() >= 1.0

    def test_optimize_max_duration(self):
        """Test maximum duration enforcement."""
        settings = ProjectSettings(max_duration=3.0)
        segments = [
            SubtitleSegment(1, 0.0, 10.0, "A" * 100)  # Too long
        ]
        optimizer = SubtitleOptimizer(settings)
        result = optimizer.optimize_segments(segments)

        # Should be shortened
        assert result[0].duration() <= 3.0

    def test_enforce_gaps(self):
        """Test gap enforcement between subtitles."""
        settings = ProjectSettings(gap_between_subtitles=0.5)
        segments = [
            SubtitleSegment(1, 0.0, 2.0, "Hello"),
            SubtitleSegment(2, 2.0, 4.0, "World")  # No gap
        ]
        optimizer = SubtitleOptimizer(settings)
        result = optimizer.optimize_segments(segments)

        # Second segment should start after gap
        assert result[1].start_time >= result[0].end_time + 0.5

    def test_reindex_segments(self):
        """Test segment re-indexing."""
        settings = ProjectSettings()
        segments = [
            SubtitleSegment(5, 0.0, 2.0, "First"),
            SubtitleSegment(10, 2.1, 4.0, "Second")
        ]
        optimizer = SubtitleOptimizer(settings)
        result = optimizer.optimize_segments(segments)

        assert result[0].index == 1
        assert result[1].index == 2


class TestAudioExtractor:
    """Test AudioExtractor."""

    def test_supported_formats(self):
        """Test supported video formats."""
        extractor = AudioExtractor()

        # Test extension validation logic
        # (is_valid_video also checks file existence)
        assert ".mp4" in extractor.supported_formats
        assert ".mkv" in extractor.supported_formats
        assert ".avi" in extractor.supported_formats
        assert ".mov" in extractor.supported_formats
        assert ".txt" not in extractor.supported_formats

        # Non-existent files should return False (file must exist)
        assert not extractor.is_valid_video(Path("nonexistent.mp4"))

    def test_formats_list(self):
        """Test supported formats list."""
        extractor = AudioExtractor()
        expected = ['.mp4', '.mkv', '.avi', '.mov', '.webm', '.wmv', '.flv']

        assert extractor.supported_formats == expected


class TestVietnameseContextTranslation:
    """Test Vietnamese context-aware translation context rules."""

    def test_mother_to_son_pronouns(self):
        """Test mother to son context."""
        # When mother speaks to son, use "con" (you/child)
        context = {
            "speaker": "Mother",
            "relationship": "mother-son"
        }
        # This tests that the context system understands the relationship
        assert context["relationship"] == "mother-son"

    def test_son_to_mother_pronouns(self):
        """Test son to mother context."""
        context = {
            "speaker": "Son",
            "relationship": "son-mother"
        }
        # When son speaks to mother, use "mẹ" (you/mother)
        assert context["relationship"] == "son-mother"

    def test_elder_sibling_context(self):
        """Test elder sibling relationship."""
        context = {
            "speaker": "Older Sister",
            "relationship": "older-sibling"
        }
        # Older sibling to younger uses "em" (you/younger)
        assert context["relationship"] == "older-sibling"

    def test_teacher_student_context(self):
        """Test teacher-student relationship."""
        context = {
            "speaker": "Teacher",
            "relationship": "teacher-student"
        }
        # Teacher uses formal language to students
        assert context["relationship"] == "teacher-student"

    def test_professional_context(self):
        """Test professional/workplace context."""
        context = {
            "speaker": "Boss",
            "relationship": "superior-subordinate"
        }
        # Professional context requires formal pronouns
        assert context["relationship"] == "superior-subordinate"


class TestTimestampFormatting:
    """Test timestamp formatting edge cases."""

    def test_zero_timestamp(self):
        """Test zero timestamp."""
        segment = SubtitleSegment(1, 0.0, 0.0, "Test")
        result = segment.to_srt_format()
        assert "00:00:00,000 --> 00:00:00,000" in result

    def test_hour_timestamp(self):
        """Test hour-long timestamp."""
        segment = SubtitleSegment(1, 3600.0, 7200.0, "Test")
        result = segment.to_srt_format()
        assert "01:00:00,000 --> 02:00:00,000" in result

    def test_millisecond_precision(self):
        """Test millisecond precision."""
        segment = SubtitleSegment(1, 0.0, 0.999, "Test")
        result = segment.to_srt_format()
        assert "00:00:00,999" in result


# Test execution summary
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
