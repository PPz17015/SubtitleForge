"""
Tests for JSON Output Plugin.

Coverage:
- Correct attribute names (start_time/end_time, not start/end)
- JSON structure and encoding
- Edge cases: empty segments, missing optional attributes
"""
import json
import sys
import tempfile
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.models import SubtitleSegment


class TestJSONOutputPlugin:
    """Test JSON output plugin functionality."""

    @pytest.fixture
    def plugin(self):
        """Load the JSON output plugin."""
        # Import directly — plugins/ is not a package, so we load manually
        import importlib.util
        plugin_path = Path(__file__).parent.parent / "plugins" / "json_output.py"
        spec = importlib.util.spec_from_file_location("json_output", plugin_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.register()

    @pytest.fixture
    def sample_segments(self):
        """Create sample subtitle segments for testing."""
        return [
            SubtitleSegment(
                index=1,
                start_time=0.0,
                end_time=2.5,
                text="こんにちは",
                speaker="Mother",
                translation="Xin chào con"
            ),
            SubtitleSegment(
                index=2,
                start_time=2.8,
                end_time=5.0,
                text="お母さん",
                speaker="Son",
                translation="Mẹ ơi"
            ),
            SubtitleSegment(
                index=3,
                start_time=5.5,
                end_time=8.0,
                text="元気ですか",
                translation="Con có khỏe không?"
            ),
        ]

    def test_plugin_metadata(self, plugin):
        """Test plugin name, version, and description."""
        assert plugin.name == "json_output"
        assert plugin.version == "1.0.0"
        assert "JSON" in plugin.description

    def test_write_creates_valid_json(self, plugin, sample_segments):
        """Test that write() creates a valid JSON file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            output_path = Path(f.name)

        try:
            result = plugin.write(sample_segments, output_path)
            assert result is True

            # Read and parse the JSON
            with open(output_path, encoding='utf-8') as f:
                data = json.load(f)

            assert isinstance(data, list)
            assert len(data) == 3
        finally:
            output_path.unlink(missing_ok=True)

    def test_correct_attribute_names(self, plugin, sample_segments):
        """
        CRITICAL: Verify that start_time/end_time (not start/end)
        are used from SubtitleSegment.

        This test catches the bug that was found in the original code
        where seg.start/seg.end were used instead of seg.start_time/seg.end_time.
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            output_path = Path(f.name)

        try:
            plugin.write(sample_segments, output_path)

            with open(output_path, encoding='utf-8') as f:
                data = json.load(f)

            # Verify first segment timing
            assert data[0]["start"] == 0.0
            assert data[0]["end"] == 2.5

            # Verify second segment timing
            assert data[1]["start"] == 2.8
            assert data[1]["end"] == 5.0

            # Verify third segment timing
            assert data[2]["start"] == 5.5
            assert data[2]["end"] == 8.0
        finally:
            output_path.unlink(missing_ok=True)

    def test_segment_fields(self, plugin, sample_segments):
        """Test that all expected fields are present in each segment."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            output_path = Path(f.name)

        try:
            plugin.write(sample_segments, output_path)

            with open(output_path, encoding='utf-8') as f:
                data = json.load(f)

            expected_fields = {"index", "start", "end", "text", "translation", "speaker"}

            for item in data:
                assert set(item.keys()) == expected_fields, (
                    f"Expected fields {expected_fields}, got {set(item.keys())}"
                )
        finally:
            output_path.unlink(missing_ok=True)

    def test_unicode_content(self, plugin, sample_segments):
        """Test that Unicode characters (Japanese, Vietnamese) are preserved."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            output_path = Path(f.name)

        try:
            plugin.write(sample_segments, output_path)

            with open(output_path, encoding='utf-8') as f:
                data = json.load(f)

            # Japanese characters preserved
            assert data[0]["text"] == "こんにちは"
            assert data[1]["text"] == "お母さん"

            # Vietnamese diacritics preserved
            assert data[0]["translation"] == "Xin chào con"
            assert data[2]["translation"] == "Con có khỏe không?"
        finally:
            output_path.unlink(missing_ok=True)

    def test_speaker_data(self, plugin, sample_segments):
        """Test speaker information is correctly written."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            output_path = Path(f.name)

        try:
            plugin.write(sample_segments, output_path)

            with open(output_path, encoding='utf-8') as f:
                data = json.load(f)

            assert data[0]["speaker"] == "Mother"
            assert data[1]["speaker"] == "Son"
            # Segment 3 has no speaker → getattr returns None (attribute exists but is None)
            assert data[2]["speaker"] is None or data[2]["speaker"] == ""
        finally:
            output_path.unlink(missing_ok=True)

    def test_missing_optional_attributes(self, plugin):
        """Test segments without translation or speaker."""
        segments = [
            SubtitleSegment(
                index=1,
                start_time=0.0,
                end_time=2.0,
                text="Plain text only"
            )
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            output_path = Path(f.name)

        try:
            plugin.write(segments, output_path)

            with open(output_path, encoding='utf-8') as f:
                data = json.load(f)

            # translation and speaker are None (SubtitleSegment defaults)
            assert data[0]["translation"] is None or data[0]["translation"] == ""
            assert data[0]["speaker"] is None or data[0]["speaker"] == ""
        finally:
            output_path.unlink(missing_ok=True)

    def test_empty_segments_list(self, plugin):
        """Test writing an empty list of segments."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            output_path = Path(f.name)

        try:
            result = plugin.write([], output_path)
            assert result is True

            with open(output_path, encoding='utf-8') as f:
                data = json.load(f)

            assert data == []
        finally:
            output_path.unlink(missing_ok=True)

    def test_json_indentation(self, plugin, sample_segments):
        """Test that the JSON output is pretty-printed with indentation."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            output_path = Path(f.name)

        try:
            plugin.write(sample_segments, output_path)

            with open(output_path, encoding='utf-8') as f:
                raw_content = f.read()

            # Pretty-printed JSON should have newlines and leading spaces
            assert "\n" in raw_content
            assert "  " in raw_content  # indent=2
        finally:
            output_path.unlink(missing_ok=True)

    def test_index_order(self, plugin, sample_segments):
        """Test that segment indices are preserved in output."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            output_path = Path(f.name)

        try:
            plugin.write(sample_segments, output_path)

            with open(output_path, encoding='utf-8') as f:
                data = json.load(f)

            indices = [item["index"] for item in data]
            assert indices == [1, 2, 3]
        finally:
            output_path.unlink(missing_ok=True)


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
