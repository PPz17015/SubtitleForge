"""
Unit tests for the Checkpoint Manager.

Coverage:
- Create, save, load, and cleanup checkpoints
- Stage-level checkpoint progression
- Batch-level translation sub-checkpoints
- Settings compatibility validation
- Corrupt file handling
- Resume info generation
"""
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.checkpoint import (
    STAGE_LABELS,
    STAGES,
    CheckpointManager,
)


@pytest.fixture
def temp_video(tmp_path):
    """Create a fake video file for testing."""
    video = tmp_path / "test_video.mp4"
    video.write_text("fake video content")
    return video


@pytest.fixture
def manager(temp_video):
    """Create a CheckpointManager for the temp video."""
    return CheckpointManager(temp_video)


@pytest.fixture
def mock_settings():
    """Create mock project settings."""
    settings = MagicMock()
    settings.source_language = "ja"
    settings.target_language = "vi"
    settings.whisper_model = "small"
    settings.use_context_aware = True
    settings.quality_check = True
    return settings


class TestCheckpointCreation:
    """Test creating and initializing checkpoints."""

    def test_checkpoint_path_next_to_video(self, manager, temp_video):
        """Checkpoint file should be created next to the video."""
        expected = temp_video.parent / "test_video.checkpoint.json"
        assert manager.checkpoint_path == expected

    def test_no_checkpoint_initially(self, manager):
        """No checkpoint should exist before creating one."""
        assert not manager.has_checkpoint()
        assert manager.load() is None

    def test_create_checkpoint(self, manager, mock_settings):
        """Creating a checkpoint should write a valid file."""
        data = manager.create(mock_settings)

        assert manager.has_checkpoint()
        assert data.video_name == "test_video.mp4"
        assert data.created_at > 0
        assert data.settings_hash != ""
        assert data.completed_stage == ""

    def test_create_with_settings_hash(self, manager, mock_settings):
        """Settings hash should be deterministic."""
        data1 = manager.create(mock_settings)

        # Same settings = same hash
        data2 = manager.create(mock_settings)
        assert data1.settings_hash == data2.settings_hash

        # Different settings = different hash
        mock_settings.target_language = "en"
        data3 = manager.create(mock_settings)
        assert data1.settings_hash != data3.settings_hash


class TestStageSaving:
    """Test saving checkpoint at each stage."""

    def test_save_audio_extracted(self, manager, mock_settings):
        """Save checkpoint after audio extraction."""
        data = manager.create(mock_settings)
        manager.save_audio_extracted(data, "/tmp/audio.wav", 120.5)

        loaded = manager.load()
        assert loaded is not None
        assert loaded.completed_stage == "audio_extracted"
        assert loaded.audio_path == "/tmp/audio.wav"
        assert loaded.audio_duration == 120.5

    def test_save_transcribed(self, manager, mock_settings):
        """Save checkpoint after transcription."""
        data = manager.create(mock_settings)
        manager.save_audio_extracted(data, "/tmp/audio.wav", 120.5)

        transcription_dict = {
            "segments": [
                {"index": 1, "start_time": 0.0, "end_time": 2.0, "text": "Hello"},
                {"index": 2, "start_time": 2.0, "end_time": 4.0, "text": "World"},
            ],
            "language": "ja",
            "duration": 120.5,
            "metadata": {},
        }
        manager.save_transcribed(data, transcription_dict)

        loaded = manager.load()
        assert loaded is not None
        assert loaded.completed_stage == "transcribed"
        assert len(loaded.transcription["segments"]) == 2
        # Pre-allocated translation slots
        assert len(loaded.translations) == 2
        assert loaded.translated_count == 0

    def test_save_translated(self, manager, mock_settings):
        """Save checkpoint after translation."""
        data = manager.create(mock_settings)
        manager.save_translated(data, ["Xin chào", "Thế giới"])

        loaded = manager.load()
        assert loaded is not None
        assert loaded.completed_stage == "translated"
        assert loaded.translations == ["Xin chào", "Thế giới"]
        assert loaded.translated_count == 2

    def test_save_quality_checked(self, manager, mock_settings):
        """Save checkpoint after quality check."""
        data = manager.create(mock_settings)
        manager.save_quality_checked(data, [{"ok": True}])

        loaded = manager.load()
        assert loaded is not None
        assert loaded.completed_stage == "quality_checked"

    def test_stage_progression(self, manager, mock_settings):
        """Stages should progress correctly."""
        data = manager.create(mock_settings)

        assert data.completed_stage == ""
        manager.save_audio_extracted(data, "/tmp/a.wav", 10.0)
        assert data.completed_stage == "audio_extracted"

        transcription = {"segments": [{"index": 1, "start_time": 0, "end_time": 1, "text": "hi"}], "language": "ja", "duration": 10}
        manager.save_transcribed(data, transcription)
        assert data.completed_stage == "transcribed"

        manager.save_translated(data, ["xin chào"])
        assert data.completed_stage == "translated"


class TestTranslationSubCheckpoints:
    """Test batch-level translation checkpoints."""

    def test_save_partial_translation(self, manager, mock_settings):
        """Save partial translation progress."""
        data = manager.create(mock_settings)

        # Simulate 5 segments, 2 translated
        translations = ["Xin chào", "Thế giới", None, None, None]
        manager.save_translation_progress(data, translations, 2)

        loaded = manager.load()
        assert loaded is not None
        # Stage should NOT be "translated" yet (still in progress)
        assert loaded.completed_stage == ""
        assert loaded.translated_count == 2
        assert loaded.translations == ["Xin chào", "Thế giới", None, None, None]

    def test_resume_from_partial_translation(self, manager, mock_settings):
        """Verify partial translation data is preserved for resume."""
        data = manager.create(mock_settings)
        transcription = {
            "segments": [
                {"index": i, "start_time": i, "end_time": i + 1, "text": f"seg{i}"}
                for i in range(100)
            ],
            "language": "ja",
            "duration": 100,
        }
        manager.save_transcribed(data, transcription)

        # Simulate translating first 50
        translations = [f"trans_{i}" for i in range(50)] + [None] * 50
        manager.save_translation_progress(data, translations, 50)

        loaded = manager.load()
        assert loaded.translated_count == 50
        assert loaded.translations[49] == "trans_49"
        assert loaded.translations[50] is None

    def test_incremental_progress(self, manager, mock_settings):
        """Translation progress should increment correctly."""
        data = manager.create(mock_settings)

        total = 10
        translations = [None] * total

        # First batch: 3 translations
        for i in range(3):
            translations[i] = f"trans_{i}"
        manager.save_translation_progress(data, translations, 3)

        loaded = manager.load()
        assert loaded.translated_count == 3

        # Second batch: 3 more translations
        for i in range(3, 6):
            translations[i] = f"trans_{i}"
        manager.save_translation_progress(data, translations, 6)

        loaded = manager.load()
        assert loaded.translated_count == 6
        assert loaded.translations[5] == "trans_5"
        assert loaded.translations[6] is None


class TestResumeLogic:
    """Test stage skipping and resume logic."""

    def test_get_next_stage_from_empty(self, manager, mock_settings):
        """First stage should be audio_extracted."""
        data = manager.create(mock_settings)
        assert manager.get_next_stage(data) == "audio_extracted"

    def test_get_next_stage_progression(self, manager, mock_settings):
        """Next stage should follow the correct order."""
        data = manager.create(mock_settings)

        data.completed_stage = "audio_extracted"
        assert manager.get_next_stage(data) == "transcribed"

        data.completed_stage = "transcribed"
        assert manager.get_next_stage(data) == "translated"

        data.completed_stage = "translated"
        assert manager.get_next_stage(data) == "quality_checked"

        data.completed_stage = "quality_checked"
        assert manager.get_next_stage(data) == "completed"

    def test_get_next_stage_completed(self, manager, mock_settings):
        """No next stage after completed."""
        data = manager.create(mock_settings)
        data.completed_stage = "completed"
        assert manager.get_next_stage(data) is None

    def test_settings_compatible(self, manager, mock_settings):
        """Same settings should be compatible."""
        data = manager.create(mock_settings)
        assert manager.is_settings_compatible(data, mock_settings)

    def test_settings_incompatible(self, manager, mock_settings):
        """Changed settings should be incompatible."""
        data = manager.create(mock_settings)
        mock_settings.whisper_model = "large"
        assert not manager.is_settings_compatible(data, mock_settings)


class TestCleanup:
    """Test checkpoint cleanup."""

    def test_cleanup_deletes_file(self, manager, mock_settings):
        """Cleanup should delete the checkpoint file."""
        manager.create(mock_settings)
        assert manager.has_checkpoint()

        manager.cleanup()
        assert not manager.has_checkpoint()

    def test_cleanup_nonexistent_is_safe(self, manager):
        """Cleanup on nonexistent file should not raise."""
        manager.cleanup()  # Should not raise


class TestErrorHandling:
    """Test error handling for corrupt/invalid checkpoints."""

    def test_corrupt_json(self, manager):
        """Corrupt JSON file should return None on load."""
        manager.checkpoint_path.write_text("not valid json {{{")
        assert manager.load() is None

    def test_wrong_video_path(self, manager, mock_settings, tmp_path):
        """Checkpoint for different video should return None."""
        data = manager.create(mock_settings)
        manager.save_audio_extracted(data, "/tmp/a.wav", 10.0)

        # Create manager for different video
        other_video = tmp_path / "other.mp4"
        other_video.write_text("other")
        other_manager = CheckpointManager(other_video)

        # Copy checkpoint file
        import shutil
        shutil.copy(manager.checkpoint_path, other_manager.checkpoint_path)

        # Should reject because video_path doesn't match
        loaded = other_manager.load()
        assert loaded is None

    def test_empty_file(self, manager):
        """Empty file should return None on load."""
        manager.checkpoint_path.write_text("")
        assert manager.load() is None


class TestResumeInfo:
    """Test human-readable resume info generation."""

    def test_resume_info_basic(self, manager, mock_settings):
        """Resume info should contain video name and stage."""
        data = manager.create(mock_settings)
        data.completed_stage = "transcribed"
        data.updated_at = time.time()

        info = manager.get_resume_info(data)
        assert "test_video.mp4" in info
        assert "Chuyển speech → text" in info  # transcribed stage label
        assert "Dịch subtitle" in info  # next stage label

    def test_resume_info_with_partial_translation(self, manager, mock_settings):
        """Resume info should show partial translation progress."""
        data = manager.create(mock_settings)
        data.completed_stage = "transcribed"
        data.translations = [None] * 100
        data.translated_count = 42
        data.updated_at = time.time()

        info = manager.get_resume_info(data)
        assert "42/100" in info

    def test_resume_info_with_segments(self, manager, mock_settings):
        """Resume info should show segment count."""
        data = manager.create(mock_settings)
        data.completed_stage = "transcribed"
        data.transcription = {"segments": [{"text": f"seg{i}"} for i in range(50)]}
        data.updated_at = time.time()

        info = manager.get_resume_info(data)
        assert "50" in info


class TestStageConstants:
    """Test stage constants are well-defined."""

    def test_all_stages_have_labels(self):
        """All stages should have display labels."""
        for stage in STAGES:
            assert stage in STAGE_LABELS, f"Missing label for stage: {stage}"

    def test_stage_order(self):
        """Stages should be in the correct order."""
        expected = [
            "audio_extracted",
            "transcribed",
            "translated",
            "quality_checked",
            "completed",
        ]
        assert expected == STAGES


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
