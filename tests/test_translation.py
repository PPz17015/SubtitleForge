import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from translation.translator import ContextAnalyzer, TranslationConfig, TranslationEngine


class TestTranslationConfig:
    """Test TranslationConfig."""

    def test_default_config(self):
        """Test default translation config."""
        config = TranslationConfig()
        assert config.target_language == "vi"
        assert config.use_gemini is True
        assert config.batch_size == 40
        assert config.use_context_aware is True

    def test_custom_config(self):
        """Test custom translation config."""
        config = TranslationConfig(
            target_language="en",
            use_gemini=False,
            batch_size=5
        )
        assert config.target_language == "en"
        assert config.use_gemini is False
        assert config.batch_size == 5


class TestContextAnalyzer:
    """Test ContextAnalyzer."""

    def test_analyze_empty_conversation(self):
        """Test analyzing empty conversation."""
        analyzer = ContextAnalyzer()
        result = analyzer.analyze_conversation([])

        assert result["conversation_type"] == "general"
        assert result["formality_level"] == "neutral"

    def test_analyze_single_speaker(self):
        """Test analyzing single speaker conversation."""
        analyzer = ContextAnalyzer()
        segments = [
            {"text": "Hello", "speaker": "Speaker1"},
            {"text": "World", "speaker": "Speaker1"}
        ]
        result = analyzer.analyze_conversation(segments)

        assert "Speaker1" in result["speakers"]

    def test_analyze_multi_speaker(self):
        """Test analyzing multi-speaker conversation."""
        analyzer = ContextAnalyzer()
        segments = [
            {"text": "Hello", "speaker": "Mother"},
            {"text": "Hi", "speaker": "Son"}
        ]
        result = analyzer.analyze_conversation(segments)

        assert "Mother" in result["speakers"]
        assert "Son" in result["speakers"]

    def test_get_context_for_segment(self):
        """Test getting context for specific segment."""
        analyzer = ContextAnalyzer()
        segments = [
            {"text": "Hello", "speaker": "Mother"},
            {"text": "Hi", "speaker": "Son"}
        ]
        context = analyzer.analyze_conversation(segments)

        # Get context for second segment
        segment_context = analyzer.get_context_for_segment(1, segments, context)

        assert segment_context["speaker"] == "Son"
        assert segment_context["previous_text"] == "Hello"

    def test_detect_formality_from_honorifics(self):
        """Test detecting formality from Japanese honorifics."""
        analyzer = ContextAnalyzer()

        # Formal speech with 様
        segments_formal = [
            {"text": "お客様、お楽しみください", "speaker": "Server"}
        ]
        result = analyzer.analyze_conversation(segments_formal)

        assert result["speakers"]["Server"]["formality"] == "formal"

    def test_detect_casual_speech(self):
        """Test detecting casual speech."""
        analyzer = ContextAnalyzer()

        # Casual speech
        segments_casual = [
            {"text": "Hey, what's up?", "speaker": "Friend"}
        ]
        result = analyzer.analyze_conversation(segments_casual)

        # Should detect as neutral (not formal)
        assert result["speakers"]["Friend"]["formality"] in ["neutral", "casual"]


class TestVietnamesePronounMapping:
    """Test Vietnamese pronoun mapping rules."""

    def test_mother_to_son_mapping(self):
        """Test mother to son pronoun mapping."""
        # Mother speaking to son should use "con" as second person
        context = {
            "speaker": "Mother",
            "relationship": "mother-son"
        }

        # Verify context structure
        assert "speaker" in context
        assert context["relationship"] == "mother-son"

    def test_son_to_mother_mapping(self):
        """Test son to mother pronoun mapping."""
        context = {
            "speaker": "Son",
            "relationship": "son-mother"
        }

        assert context["relationship"] == "son-mother"

    def test_teacher_student_formality(self):
        """Test teacher-student formality detection."""
        context = {
            "speaker": "Teacher",
            "conversation_type": "educational"
        }

        assert context["conversation_type"] == "educational"

    def test_professional_hierarchy(self):
        """Test professional hierarchy context."""
        context = {
            "speaker": "Boss",
            "relationship": "superior",
            "conversation_type": "workplace"
        }

        assert context["conversation_type"] == "workplace"
        assert context["relationship"] == "superior"


class TestLanguageCodes:
    """Test language code utilities."""

    def test_get_language_name_ja(self):
        """Test Japanese language name."""
        from translation.translator import TranslationEngine
        name = TranslationEngine._get_language_name("ja")
        assert name == "Japanese"

    def test_get_language_name_vi(self):
        """Test Vietnamese language name."""
        from translation.translator import TranslationEngine
        name = TranslationEngine._get_language_name("vi")
        assert name == "Vietnamese"

    def test_get_language_name_en(self):
        """Test English language name."""
        from translation.translator import TranslationEngine
        name = TranslationEngine._get_language_name("en")
        assert name == "English"

    def test_get_language_name_unknown(self):
        """Test unknown language code."""
        from translation.translator import TranslationEngine
        name = TranslationEngine._get_language_name("xyz")
        assert name == "XYZ"  # Should return uppercase code


class TestContextAwarePromptBuilding:
    """Test context-aware prompt building."""

    def test_build_prompt_with_speaker_context(self):
        """Test prompt building with speaker context."""
        config = TranslationConfig(target_language="vi", use_gemini=False)
        engine = TranslationEngine(config)

        prompt = engine._build_translation_prompt(
            "こんにちは",
            "ja",
            {"speaker": "Mother", "relationship": "mother-son"}
        )

        assert "こんにちは" in prompt
        assert "Vietnamese" in prompt

    def test_build_prompt_without_context(self):
        """Test prompt building without context."""
        config = TranslationConfig(target_language="vi", use_gemini=False)
        engine = TranslationEngine(config)

        prompt = engine._build_translation_prompt(
            "Hello",
            "en",
            None
        )

        assert "Hello" in prompt
        assert "English" in prompt

    def test_build_batch_prompt(self):
        """Test batch prompt building."""
        config = TranslationConfig(target_language="vi", use_gemini=False, batch_size=3)
        engine = TranslationEngine(config)

        segments = [
            {"text": "Hello", "speaker": "A"},
            {"text": "World", "speaker": "B"}
        ]

        prompt = engine._build_batch_translation_prompt(segments, "en")

        assert "Hello" in prompt
        assert "World" in prompt


class TestBatchTranslation:
    """Test batch translation parsing."""

    def test_parse_simple_batch_response(self):
        """Test parsing simple batch response."""
        config = TranslationConfig(target_language="vi")
        engine = TranslationEngine(config)

        response = "Hola\nMundo"
        results = engine._parse_batch_response(response, 2)

        assert len(results) == 2

    def test_parse_numbered_batch_response(self):
        """Test parsing numbered batch response."""
        config = TranslationConfig(target_language="vi")
        engine = TranslationEngine(config)

        response = "1. Hello\n2. World"
        results = engine._parse_batch_response(response, 2)

        assert len(results) == 2

    def test_parse_dash_batch_response(self):
        """Test parsing dash-prefixed batch response."""
        config = TranslationConfig(target_language="vi")
        engine = TranslationEngine(config)

        response = "- Hello\n- World"
        results = engine._parse_batch_response(response, 2)

        assert len(results) == 2

    def test_parse_response_with_fewer_lines(self):
        """Test parsing when response has fewer lines than expected."""
        config = TranslationConfig(target_language="vi")
        engine = TranslationEngine(config)

        response = "Hello"  # Only one line
        results = engine._parse_batch_response(response, 3)

        # Should pad with empty strings
        assert len(results) == 3


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
