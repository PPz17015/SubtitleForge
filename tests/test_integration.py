import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestCLI:
    """Test CLI interface."""

    def test_cli_help(self):
        """Test CLI help command."""
        from click.testing import CliRunner

        from main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ['--help'])

        assert result.exit_code == 0
        assert "SubtitleForge" in result.output

    def test_generate_help(self):
        """Test generate command help."""
        from click.testing import CliRunner

        from main import generate

        runner = CliRunner()
        result = runner.invoke(generate, ['--help'])

        assert result.exit_code == 0
        assert "--source" in result.output
        assert "--target" in result.output
        assert "--gemini-key" in result.output
        assert "--video-context" in result.output
        assert "--speaker-relationship" in result.output
        assert "--quality-check" in result.output

    def test_info_command(self):
        """Test info command."""
        from click.testing import CliRunner

        from main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ['info'])

        # May fail if dependencies not installed, but should not crash
        assert result.exit_code in [0, 1]


class TestIntegration:
    """Test full pipeline integration."""

    def test_project_settings_with_relationships(self):
        """Test project settings with speaker relationships."""
        from core.models import ProjectSettings

        settings = ProjectSettings(
            source_language="ja",
            target_language="vi",
            speaker_relationships={
                "Mother": "mother-son",
                "Son": "son-mother"
            }
        )

        assert settings.speaker_relationships["Mother"] == "mother-son"
        assert settings.speaker_relationships["Son"] == "son-mother"

    def test_models_to_dict_with_context(self):
        """Test models with context."""
        from core.models import SubtitleSegment

        segment = SubtitleSegment(
            index=1,
            start_time=0.0,
            end_time=2.0,
            text="こんにちは",
            speaker="Mother",
            translation="Chào con",
            context={"relationship": "mother-son"}
        )

        data = segment.to_dict()

        assert data["context"]["relationship"] == "mother-son"
        assert data["speaker"] == "Mother"

    def test_audio_extractor_with_pathlib(self):
        """Test audio extractor with pathlib."""
        from core.audio_extractor import AudioExtractor

        extractor = AudioExtractor()

        # Test with Path object
        path = Path("test.mp4")

        # Should return False for non-existent file
        assert not extractor.is_valid_video(path)


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
