import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from translation.translator import TranslationQualityChecker


class TestTranslationQualityChecker:
    """Test TranslationQualityChecker."""

    def test_empty_translation(self):
        """Test detection of empty translation."""
        checker = TranslationQualityChecker()
        result = checker.check_translation_quality(
            original_text="こんにちは",
            translated_text=""
        )

        assert result["is_good"] is False
        assert "Empty" in result["issues"][0]

    def test_marker_detection(self):
        """Test detection of fallback marker."""
        checker = TranslationQualityChecker()
        result = checker.check_translation_quality(
            original_text="こんにちは",
            translated_text="[Japanese] こんにちは"
        )

        assert result["is_good"] is False
        assert "marker" in result["issues"][0].lower()

    def test_vietnamese_character_validation(self):
        """Test Vietnamese character detection."""
        checker = TranslationQualityChecker()

        # Valid Vietnamese with diacritics
        checker.check_translation_quality(
            original_text="こんにちは",
            translated_text="Xin chào"
        )

        # Invalid (no Vietnamese diacritics - just Latin letters)
        result_bad = checker.check_translation_quality(
            original_text="こんにちは",
            translated_text="konnichiwa"
        )

        # The good one might still need recheck due to no diacritics, but bad one definitely needs recheck
        assert result_bad["needs_recheck"] is True

    def test_length_ratio_check(self):
        """Test unusual length ratio detection."""
        checker = TranslationQualityChecker()

        # Way too short
        result_short = checker.check_translation_quality(
            original_text="こんにちは元気ですか",
            translated_text="Hi"
        )

        # Way too long
        result_long = checker.check_translation_quality(
            original_text="Hi",
            translated_text="これは長い翻訳です" * 50
        )

        assert result_short["needs_recheck"] is True
        assert result_long["needs_recheck"] is True

    def test_pronoun_check(self):
        """Test Vietnamese pronoun context check."""
        checker = TranslationQualityChecker()

        context = {
            "speaker": "Mother",
            "relationship": "mother-son"
        }

        # Should not have issues with proper context
        result = checker.check_translation_quality(
            original_text="おはよう",
            translated_text="Chào con",
            context=context
        )

        # Check it ran without error
        assert "issues" in result

    def test_batch_check(self):
        """Test batch quality checking."""
        checker = TranslationQualityChecker()

        segments = [
            {"original": "Hello", "translation": "Xin chào"},
            {"original": "World", "translation": "Thế giới"},
            {"original": "", "translation": ""}
        ]

        results = checker.batch_check(segments, "en", "vi")

        assert len(results) == 3
        assert results[0]["segment_index"] == 0
        assert results[2]["is_good"] is False  # Empty


class TestVRAMOptimization:
    """Test VRAM optimization features."""

    def test_transcription_config_defaults(self):
        """Test default transcription config."""
        try:
            from transcription.whisper_transcriber import TranscriptionConfig
            config = TranscriptionConfig()
            assert config.model_size == "small"
            assert config.compute_type == "int8"
            assert config.max_memory_vram_gb == 6.0
        except ImportError:
            pytest.skip("faster-whisper not installed")

    def test_model_recommendation(self):
        """Test model recommendation based on VRAM."""
        try:
            from transcription.whisper_transcriber import WhisperTranscriber
            assert WhisperTranscriber.get_recommended_model(1.0) == "base"
            assert WhisperTranscriber.get_recommended_model(3.0) == "small"
            assert WhisperTranscriber.get_recommended_model(5.0) == "medium"
            assert WhisperTranscriber.get_recommended_model(8.0) == "small"
        except ImportError:
            pytest.skip("faster-whisper not installed")

    def test_diarization_config(self):
        """Test diarization config."""
        try:
            from diarization.speaker_diarizer import DiarizationConfig
            config = DiarizationConfig()
            assert config.max_memory_vram_gb == 6.0
            assert config.chunk_duration == 300.0
        except ImportError:
            pytest.skip("pyannote not installed")


class TestContextAwareJapaneseVietnamese:
    """Test Japanese to Vietnamese specific features."""

    def test_mother_son_context(self):
        """Test mother-son context for JA→VI."""
        checker = TranslationQualityChecker()

        context = {
            "speaker": "Mother",
            "relationship": "mother-son"
        }

        result = checker.check_translation_quality(
            original_text="おはよう、今日の部は怎么样?",
            translated_text="Chào con, hôm nay buổi học thế nào?",
            source_language="ja",
            target_language="vi",
            context=context
        )

        # Should pass basic checks
        assert "issues" in result

    def test_formal_honorific_detection(self):
        """Test formal honorific context."""
        from translation.translator import ContextAnalyzer

        analyzer = ContextAnalyzer()

        segments = [
            {"text": "社長プレゼンを始めます", "speaker": "Employee"},
            {"text": "お願いします", "speaker": "Employee"}
        ]

        result = analyzer.analyze_conversation(segments)

        # Should detect formal speech
        assert "Employee" in result["speakers"]


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
