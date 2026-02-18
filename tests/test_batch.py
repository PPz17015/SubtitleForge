import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.batch_processing import BatchConfig, BatchItem, BatchProcessor, GlossaryManager, QualityAssurance
from core.models import SubtitleSegment


class TestBatchConfig:
    """Test BatchConfig."""

    def test_default_config(self):
        """Test default batch config."""
        config = BatchConfig()

        assert config.max_concurrent == 2
        assert config.source_lang == "ja"
        assert config.target_lang == "vi"
        assert config.whisper_model == "small"

    def test_custom_config(self):
        """Test custom batch config."""
        config = BatchConfig(
            max_concurrent=4,
            source_lang="en",
            target_lang="vi",
            use_gpu=False
        )

        assert config.max_concurrent == 4
        assert config.source_lang == "en"
        assert config.use_gpu is False


class TestBatchItem:
    """Test BatchItem."""

    def test_create_item(self):
        """Test creating batch item."""
        item = BatchItem(
            video_path=Path("test.mp4"),
            output_dir=Path("./output")
        )

        assert item.video_path == Path("test.mp4")
        assert item.output_dir == Path("./output")
        assert item.status == "pending"

    def test_default_output_dir(self):
        """Test default output directory."""
        item = BatchItem(video_path=Path("/videos/test.mp4"))

        assert item.output_dir == Path("/videos")


class TestGlossaryManager:
    """Test GlossaryManager."""

    def test_add_term(self):
        """Test adding glossary term."""
        manager = GlossaryManager()
        manager.add_term("Hello", "Xin chào", "greeting")

        assert manager.has_term("Hello")
        assert manager.get_translation("Hello") == "Xin chào"

    def test_remove_term(self):
        """Test removing glossary term."""
        manager = GlossaryManager()
        manager.add_term("Hello", "Xin chào")
        manager.remove_term("Hello")

        assert not manager.has_term("Hello")

    def test_case_insensitive(self):
        """Test case-insensitive lookup."""
        manager = GlossaryManager()
        manager.add_term("Hello", "Xin chào")

        assert manager.has_term("hello")
        assert manager.has_term("HELLO")

    def test_get_all_terms(self):
        """Test getting all terms."""
        manager = GlossaryManager()
        manager.add_term("Hello", "Xin chào")
        manager.add_term("Goodbye", "Tạm biệt")

        terms = manager.get_all_terms()

        assert len(terms) == 2


class TestQualityAssurance:
    """Test QualityAssurance."""

    def test_validate_timing_valid(self):
        """Test timing validation with valid segments."""
        qa = QualityAssurance()
        segments = [
            SubtitleSegment(1, 0.0, 2.0, "Hello"),
            SubtitleSegment(2, 2.5, 4.5, "World")
        ]

        issues = qa.validate_timing(segments, min_duration=1.0, min_gap=0.1)

        # Should have no issues with proper gap
        assert len(issues) == 0

    def test_validate_timing_short(self):
        """Test detecting short duration."""
        qa = QualityAssurance()
        segments = [
            SubtitleSegment(1, 0.0, 0.5, "Hi")  # Too short
        ]

        issues = qa.validate_timing(segments, min_duration=1.0)

        assert len(issues) == 1
        assert "below minimum" in issues[0]["message"]

    def test_validate_text_long_line(self):
        """Test detecting long lines."""
        qa = QualityAssurance()
        segments = [
            SubtitleSegment(1, 0.0, 2.0, "A" * 50)  # Too long
        ]

        issues = qa.validate_text(segments, max_chars_per_line=42)

        assert len(issues) >= 1

    def test_validate_consistency(self):
        """Test translation consistency check."""
        qa = QualityAssurance()

        segments = [
            SubtitleSegment(1, 0.0, 2.0, "Hello", translation="Xin chào"),
            SubtitleSegment(2, 2.0, 4.0, "Hello", translation="Chào bạn")  # Different translation
        ]

        issues = qa.validate_consistency(segments)

        assert len(issues) == 1
        assert issues[0]["type"] == "consistency"

    def test_run_all_checks(self):
        """Test running all QA checks."""
        qa = QualityAssurance()
        segments = [
            SubtitleSegment(1, 0.0, 2.0, "Hello", translation="Xin chào"),
            SubtitleSegment(2, 2.5, 4.0, "World", translation="Thế giới")
        ]

        results = qa.run_all_checks(segments)

        assert "total_issues" in results
        assert "errors" in results
        assert "warnings" in results
        assert "issues" in results


class TestBatchProcessor:
    """Test BatchProcessor."""

    def test_create_processor(self):
        """Test creating batch processor."""
        processor = BatchProcessor()

        assert processor.config.max_concurrent == 2
        assert len(processor.items) == 0

    def test_add_video_not_found(self):
        """Test adding non-existent video."""
        processor = BatchProcessor()

        with pytest.raises(FileNotFoundError):
            processor.add_video(Path("nonexistent.mp4"))

    def test_get_results_empty(self):
        """Test getting results for empty batch."""
        processor = BatchProcessor()

        results = processor.get_results()

        assert results["total"] == 0
        assert results["completed"] == 0


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
