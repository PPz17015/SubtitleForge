import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class TranslationConfig:
    target_language: str = "vi"
    use_gemini: bool = True
    gemini_api_key: Optional[str] = None
    use_context_aware: bool = True
    batch_size: int = 50


class TranslationEngine:
    def __init__(self, config: Optional[TranslationConfig] = None, progress_callback=None):
        self.config = config or TranslationConfig()
        self.gemini_client = None
        self._max_retries = 5
        self._base_interval = 0.15  # Base rate limit: ~400 RPM (Tier 1 allows 300)
        self._min_interval = 0.15   # Current interval (adaptive)
        self._max_interval = 4.0    # Max backoff interval
        self._last_call_time = 0
        self._consecutive_429s = 0  # Track consecutive rate limits
        self._progress_callback = progress_callback


        if self.config.use_gemini and self.config.gemini_api_key:
            self._init_gemini()

    def _notify(self, message: str):
        """Send status update to GUI via callback."""
        if self._progress_callback:
            from contextlib import suppress
            with suppress(Exception):
                self._progress_callback(message)

    def _init_gemini(self):
        """Initialize Gemini API client."""
        try:
            from google import genai

            self.gemini_client = genai.Client(api_key=self.config.gemini_api_key)
            self._model_name = 'gemini-2.5-flash'
            logger.info("Gemini API initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize Gemini: {e}")
            self.gemini_client = None

    def _call_gemini_with_retry(self, prompt: str) -> str:
        """Call Gemini API with retry logic and adaptive rate limiting."""
        for attempt in range(self._max_retries):
            # Adaptive rate limiting
            elapsed = time.time() - self._last_call_time
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)

            try:
                self._last_call_time = time.time()
                response = self.gemini_client.models.generate_content(
                    model=self._model_name,
                    contents=prompt
                )

                # Success — quickly recover interval
                self._consecutive_429s = 0
                if self._min_interval > self._base_interval:
                    self._min_interval = max(
                        self._base_interval,
                        self._min_interval * 0.5  # Recover 50% per success
                    )

                return response.text.strip()
            except Exception as e:
                is_rate_limit = "429" in str(e) or "quota" in str(e).lower()

                if is_rate_limit:
                    self._consecutive_429s += 1
                    # Adaptive: increase interval to avoid more 429s
                    self._min_interval = min(
                        self._max_interval,
                        self._min_interval * 1.5
                    )

                if attempt < self._max_retries - 1:
                    wait = (2 ** attempt) * (4 if is_rate_limit else 2)
                    logger.warning(
                        f"Gemini call failed (attempt {attempt+1}/{self._max_retries}), "
                        f"retry in {wait}s (interval={self._min_interval:.1f}s): {e}"
                    )
                    time.sleep(wait)
                else:
                    logger.error(f"Gemini call failed after {self._max_retries} attempts: {e}")
                    raise
        return ""

    def translate(
        self,
        text: str,
        source_language: str = "ja",
        context: Optional[dict[str, Any]] = None
    ) -> str:
        """
        Translate a single text segment.

        Args:
            text: Text to translate
            source_language: Source language code
            context: Context information for context-aware translation

        Returns:
            Translated text
        """
        if self.config.use_gemini and self.gemini_client:
            return self._translate_with_gemini(text, source_language, context)
        return self._translate_fallback(text, source_language)

    def translate_batch(
        self,
        segments: list[dict[str, Any]],
        source_language: str = "ja",
        contexts: Optional[list[dict[str, Any]]] = None
    ) -> list[str]:
        """
        Translate multiple segments.

        Args:
            segments: List of segment dictionaries with 'text' and optional 'speaker'
            source_language: Source language code
            contexts: Optional list of context dictionaries for each segment

        Returns:
            List of translated texts
        """
        if self.config.use_gemini and self.gemini_client:
            return self._translate_batch_with_gemini(segments, source_language, contexts)
        return [self._translate_fallback(s["text"], source_language) for s in segments]

    def _translate_with_gemini(
        self,
        text: str,
        source_language: str,
        context: Optional[dict[str, Any]] = None
    ) -> str:
        """Translate using Gemini API with context awareness."""
        try:
            prompt = self._build_translation_prompt(
                text, source_language, context
            )

            result_text = self._call_gemini_with_retry(prompt)
            return result_text

        except Exception as e:
            logger.error(f"Gemini translation failed: {e}")
            return self._translate_fallback(text, source_language)

    def _translate_batch_with_gemini(
        self,
        segments: list[dict[str, Any]],
        source_language: str,
        contexts: Optional[list[dict[str, Any]]] = None
    ) -> list[str]:
        """Translate multiple segments using Gemini."""
        translations = []
        use_context = contexts is not None and self.config.use_context_aware
        total_batches = (len(segments) + self.config.batch_size - 1) // self.config.batch_size

        for batch_idx, i in enumerate(range(0, len(segments), self.config.batch_size)):
            batch = segments[i:i + self.config.batch_size]
            batch_contexts = None
            if use_context:
                batch_contexts = contexts[i:i + self.config.batch_size]

            self._notify(
                f"    🔄 Gửi API ({batch_idx+1}/{total_batches}): "
                f"{len(batch)} segments..."
            )

            try:
                if use_context and batch_contexts:
                    prompt = self._build_batch_translation_prompt_with_context(batch, batch_contexts, source_language)
                else:
                    prompt = self._build_batch_translation_prompt(batch, source_language)
                response_text = self._call_gemini_with_retry(prompt)

                results = self._parse_batch_response(response_text, len(batch))
                translations.extend(results)
                self._notify(
                    f"    ✅ API ({batch_idx+1}/{total_batches}): "
                    f"nhận {len(results)} bản dịch"
                )

            except Exception as e:
                logger.error(f"Batch translation failed: {e}")
                self._notify(f"    ⚠️ API ({batch_idx+1}/{total_batches}): lỗi, dùng fallback")
                for seg in batch:
                    translations.append(self._translate_fallback(
                        seg.get("text", ""), source_language
                    ))

        return translations

    def translate_batch_with_self_check(
        self,
        segments: list[dict[str, Any]],
        source_language: str = "ja",
        contexts: Optional[list[dict[str, Any]]] = None,
        video_context: Optional[str] = None,
        max_rounds: int = 2,
        quality_checker: Optional['TranslationQualityChecker'] = None
    ) -> list[str]:
        """
        Translate batch with self-check loop.

        Args:
            segments: List of segment dicts
            source_language: Source language code
            contexts: Optional list of contexts
            video_context: Optional video context description
            max_rounds: Maximum number of correction rounds
            quality_checker: Instance of TranslationQualityChecker

        Returns:
            List of translated texts
        """
        if not quality_checker:
            logger.warning("No quality checker provided for self-check loop")
            return self.translate_batch(segments, source_language, contexts)

        # Round 1: Initial translation
        logger.info("Self-check Round 1: Initial translation")
        self._notify("  🔤 Round 1: Dịch lần đầu...")
        translations = self.translate_batch(segments, source_language, contexts)
        self._notify(f"  ✅ Round 1: Hoàn thành — {len(translations)} bản dịch")

        # Rounds 2+: Quality check → re-translate bad segments
        for round_num in range(1, max_rounds):
            # Batch quality check (20 seg/call instead of 1/call)
            check_inputs = [
                {
                    "original": seg.get("text", ""),
                    "translation": trans,
                    "context": ctx
                }
                for seg, trans, ctx in zip(segments, translations, contexts or [{} for _ in segments])
            ]

            self._notify(f"  🔍 Round {round_num+1}: Kiểm tra chất lượng {len(check_inputs)} segments...")
            quality_results = quality_checker.batch_check(
                check_inputs,
                source_language=source_language,
                target_language=self.config.target_language,
                video_context=video_context
            )

            # Identify segments needing correction
            bad_indices = [
                i for i, res in enumerate(quality_results)
                if res.get("needs_recheck") or not res.get("is_good", True)
            ]

            if not bad_indices:
                logger.info(f"Self-check passed after round {round_num}")
                self._notify("  ✅ Quality check passed — không cần sửa")
                break

            logger.info(f"Self-check Round {round_num+1}: Correcting {len(bad_indices)} segments")
            self._notify(f"  🔧 Sửa {len(bad_indices)} segments chất lượng thấp...")

            # Re-translate bad segments in batches (not one-by-one)
            retranslate_batch_size = self.config.batch_size
            for batch_start in range(0, len(bad_indices), retranslate_batch_size):
                batch_indices = bad_indices[batch_start:batch_start + retranslate_batch_size]
                batch_segs = []
                batch_ctxs = []

                for idx in batch_indices:
                    feedback = quality_results[idx].get("gemini_check", {}).get("suggested_fix", "")
                    issue_msg = "; ".join(quality_results[idx].get("issues", []))

                    batch_segs.append(segments[idx])
                    ctx = (contexts[idx] if contexts else {}).copy()
                    ctx["previous_translation"] = translations[idx]
                    ctx["feedback"] = feedback or issue_msg
                    batch_ctxs.append(ctx)

                try:
                    new_translations = self.translate_batch(
                        batch_segs, source_language, batch_ctxs
                    )
                    for idx, new_trans in zip(batch_indices, new_translations):
                        if new_trans and new_trans != translations[idx]:
                            translations[idx] = new_trans
                except Exception as e:
                    logger.warning(f"Batch re-translation failed: {e}")

        # Final pass: Context coherence check
        logger.info("Running context coherence check...")
        self._notify("  🔗 Kiểm tra tính nhất quán ngữ cảnh...")
        coherence_issues = quality_checker.check_context_coherence(
            segments, translations,
            source_language=source_language,
            target_language=self.config.target_language,
            video_context=video_context
        )

        if coherence_issues:
            logger.info(f"Coherence check found {len(coherence_issues)} issues, correcting...")
            self._notify(f"  🔧 Sửa {len(coherence_issues)} lỗi nhất quán...")

            # Batch re-translate coherence issues
            coherence_indices = []
            coherence_ctxs = []
            coherence_segs = []

            for issue in coherence_issues:
                try:
                    raw_seg = issue.get("segment", 0)
                    # Gemini may return a list instead of int
                    if isinstance(raw_seg, list):
                        raw_seg = raw_seg[0] if raw_seg else 0
                    seg_idx = int(raw_seg) - 1
                except (TypeError, ValueError, IndexError):
                    continue

                group_start = issue.get("group_start", 0)
                actual_idx = group_start + seg_idx if seg_idx < 50 else seg_idx

                if 0 <= actual_idx < len(translations):
                    suggested = issue.get("suggested", "")
                    reason = issue.get("reason", "")

                    if suggested and reason:
                        coherence_indices.append(actual_idx)
                        coherence_segs.append(segments[actual_idx])
                        ctx = (contexts[actual_idx] if contexts else {}).copy()
                        ctx["previous_translation"] = translations[actual_idx]
                        ctx["feedback"] = f"Coherence issue: {reason}. Suggested pronoun: {suggested}"
                        coherence_ctxs.append(ctx)

            if coherence_indices:
                try:
                    new_translations = self.translate_batch(
                        coherence_segs, source_language, coherence_ctxs
                    )
                    for idx, new_trans in zip(coherence_indices, new_translations):
                        if new_trans and new_trans != translations[idx]:
                            translations[idx] = new_trans
                except Exception as e:
                    logger.warning(f"Coherence re-translation failed: {e}")
        else:
            logger.info("Context coherence check passed!")

        return translations

    def _translate_with_limit_retry(self, text, source_lang, context):
        """Helper to translate single segment with error handling."""
        try:
            return self.translate(text, source_lang, context)
        except Exception as e:
            logger.error(f"Re-translation failed: {e}")
            return None

    def _build_translation_prompt(
        self,
        text: str,
        source_language: str,
        context: Optional[dict[str, Any]] = None
    ) -> str:
        """Build a translation prompt with context."""

        source_lang_name = self._get_language_name(source_language)
        target_lang_name = self._get_language_name(self.config.target_language)

        prompt = f"""You are a professional translator specializing in {source_lang_name} to {target_lang_name} translation.

Your task is to translate the following text from {source_lang_name} to {target_lang_name}.

"""

        # Add context information if available
        if context:
            # Character descriptions
            if context.get("character_descriptions"):
                prompt += f"=== CHARACTERS ===\n{context['character_descriptions']}\n\n"

            # Scene context
            if context.get("scene_context"):
                prompt += f"=== CURRENT SCENE ===\n{context['scene_context']}\n\n"

            # Video context
            if context.get("video_context"):
                prompt += f"=== VIDEO CONTEXT ===\n{context['video_context']}\n\n"

            if "speaker" in context:
                prompt += f"Current speaker: {context['speaker']}\n"

            if "relationship" in context:
                prompt += f"Relationship between speakers: {context['relationship']}\n"

            if "conversation_type" in context:
                prompt += f"Conversation type: {context['conversation_type']}\n"

            if "previous_text" in context:
                prompt += f"Previous dialogue: {context['previous_text']}\n"

            # Feedback from quality check (for re-translation)
            if context.get("feedback"):
                prompt += f"\n⚠️ CORRECTION NOTE: Previous translation had issues: {context['feedback']}\n"
                prompt += f"Previous translation: {context.get('previous_translation', '')}\n"
                prompt += "Please fix the issues noted above.\n"

        prompt += f"""
IMPORTANT - Vietnamese Translation Context Rules:
- If translating from Japanese to Vietnamese:
  - When a mother speaks to her son, use "con" (you/child) and appropriate mother terms
  - When a son speaks to his mother, use "mẹ" (you/mother) and "con" (I/child)
  - Use appropriate Vietnamese pronouns based on the relationship and age difference
  - Pay attention to honorifics and social hierarchy in Japanese and map to Vietnamese equivalents
  - Keep character names consistent throughout
  - Match the emotional tone of the original

Text to translate:
{text}

Translation:"""

        return prompt

    def _build_batch_translation_prompt(
        self,
        segments: list[dict[str, Any]],
        source_language: str
    ) -> str:
        """Build a batch translation prompt."""

        source_lang_name = self._get_language_name(source_language)
        target_lang_name = self._get_language_name(self.config.target_language)

        prompt = f"""Translate the following {source_lang_name} text to {target_lang_name}.
For each line, provide only the translation without any numbering or extra text.

"""

        for i, seg in enumerate(segments):
            text = seg.get("text", "")
            speaker = seg.get("speaker", "")

            line = f"{i+1}. {text}"
            if speaker:
                line += f" [{speaker}]"

            prompt += line + "\n"

        prompt += "\nTranslations (one per line):"

        return prompt

    def _build_batch_translation_prompt_with_context(
        self,
        segments: list[dict[str, Any]],
        contexts: list[dict[str, Any]],
        source_language: str
    ) -> str:
        """Build a batch translation prompt with context."""

        source_lang_name = self._get_language_name(source_language)
        target_lang_name = self._get_language_name(self.config.target_language)

        # Collect shared context from first context entry
        shared_context = contexts[0] if contexts else {}

        prompt = f"""You are a professional translator specializing in {source_lang_name} to {target_lang_name} translation.
"""

        # Add character descriptions (shared across batch)
        char_desc = shared_context.get("character_descriptions", "")
        if char_desc:
            prompt += f"\n=== CHARACTERS ===\n{char_desc}\n"

        # Add scene context
        scene_ctx = shared_context.get("scene_context", "")
        if scene_ctx:
            prompt += f"\n=== CURRENT SCENE ===\n{scene_ctx}\n"

        # Add video context
        video_ctx = shared_context.get("video_context", "")
        if video_ctx:
            prompt += f"\n=== VIDEO CONTEXT ===\n{video_ctx}\n"

        prompt += f"""
IMPORTANT - Vietnamese Translation Context Rules:
- If translating from Japanese to Vietnamese:
  - Use appropriate Vietnamese pronouns based on the relationship and age difference
  - Pay attention to honorifics and social hierarchy in Japanese
  - Keep character names consistent throughout
  - Match the emotional tone of the original

Translate the following {source_lang_name} dialogue to {target_lang_name}.
For each line, provide ONLY the translation, one per line, without numbering.

"""

        for i, (seg, ctx) in enumerate(zip(segments, contexts)):
            text = seg.get("text", "")
            speaker = seg.get("speaker", "")

            line = f"{i+1}. "
            if speaker:
                line += f"[{speaker}] "
            line += text

            # Add per-segment relationship hints
            if ctx and ctx.get("relationship"):
                line += f" (relationship: {ctx['relationship']})"

            prompt += line + "\n"

        prompt += "\nTranslations (one per line):"

        return prompt

    def _parse_batch_response(self, response: str, expected_count: int) -> list[str]:
        """Parse batch translation response."""
        lines = response.strip().split('\n')

        results = []
        import re

        for line in lines:
            line = line.strip()
            # Remove numbering using regex (e.g., "1. ", "10. ", "- ")
            line = re.sub(r'^(\d+\.|-)\s*', '', line)
            results.append(line)

        # Ensure we have the right number of results
        while len(results) < expected_count:
            results.append("")

        return results[:expected_count]

    def _translate_fallback(self, text: str, source_language: str) -> str:
        """Fallback translation - returns original text with marker."""
        logger.warning("Using fallback translation (no API configured)")
        return f"[{self._get_language_name(source_language)}] {text}"

    @staticmethod
    def _get_language_name(code: str) -> str:
        """Get language name from code."""
        languages = {
            "ja": "Japanese",
            "vi": "Vietnamese",
            "en": "English",
            "zh": "Chinese",
            "ko": "Korean",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "ru": "Russian",
            "pt": "Portuguese",
            "ar": "Arabic",
            "hi": "Hindi",
            "th": "Thai",
            "id": "Indonesian",
            "ms": "Malay",
            "tl": "Tagalog",
            "my": "Burmese",
            "km": "Khmer",
            "lo": "Lao",
        }
        return languages.get(code, code.upper())


class ContextAnalyzer:
    """Analyze conversation context for better translation."""

    def __init__(self):
        self.conversation_history = []
        self.speaker_profiles = {}

    def analyze_conversation(
        self,
        segments: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Analyze conversation to determine context.

        Args:
            segments: List of segments with text and speaker info

        Returns:
            Dictionary with context information
        """
        context = {
            "speakers": {},
            "relationships": {},
            "conversation_type": "general",
            "formality_level": "neutral"
        }

        # Analyze speakers
        speakers = {seg.get("speaker", "unknown") for seg in segments}

        for speaker in speakers:
            speaker_segments = [s for s in segments if s.get("speaker") == speaker]
            context["speakers"][speaker] = self._analyze_speaker(speaker_segments)

        # Detect relationships based on language patterns
        context["relationships"] = self._detect_relationships(segments)

        return context

    def _analyze_speaker(self, segments: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyze individual speaker characteristics."""
        if not segments:
            return {"formality": "neutral"}

        # Simple heuristics
        text = " ".join(s.get("text", "") for s in segments)

        # Check for honorifics (Japanese example)
        has_honorifics = any(h in text for h in ["様", "さん", "先生", "氏", "課長", "部長", "社長"])

        return {
            "formality": "formal" if has_honorifics else "neutral",
            "segment_count": len(segments)
        }

    def _detect_relationships(self, segments: list[dict[str, Any]]) -> dict[str, str]:
        """Detect relationships between speakers."""
        relationships = {}

        # This would need more sophisticated analysis
        # For now, return empty - relationships would be user-provided or detected via advanced NLP

        return relationships

    def get_context_for_segment(
        self,
        segment_index: int,
        segments: list[Any],
        context: dict[str, Any]
    ) -> dict[str, Any]:
        """Get context for a specific segment."""

        if segment_index > 0:
            previous = segments[segment_index - 1]
            previous_text = previous.text if hasattr(previous, 'text') else previous.get("text", "")
        else:
            previous_text = ""

        current = segments[segment_index]
        speaker = current.speaker if hasattr(current, 'speaker') else current.get("speaker", "unknown")

        # Build context
        segment_context = {
            "speaker": speaker,
            "previous_text": previous_text,
            "conversation_type": context.get("conversation_type", "general"),
            "relationship": context.get("relationships", {}).get(speaker)
        }

        # Add speaker profile
        speaker_profiles = context.get("speakers", {})
        if speaker in speaker_profiles:
            segment_context.update(speaker_profiles[speaker])

        # Add character descriptions (shared across segments)
        if "character_descriptions" in context:
            segment_context["character_descriptions"] = context["character_descriptions"]

        # Add scene context for this segment
        if "scenes" in context:
            scene = self._get_scene_for_segment(segment_index, context["scenes"])
            if scene:
                segment_context["scene_context"] = scene

        return segment_context

    def build_character_descriptions(
        self,
        segments: list[dict[str, Any]],
        context: dict[str, Any]
    ) -> str:
        """
        Build a text description of characters from analyzed speaker data.

        Example output:
            Speaker_00: Speaks 45 times, formal tone (uses honorifics)
            Speaker_01: Speaks 30 times, casual tone
        """
        speakers = context.get("speakers", {})
        if not speakers:
            return ""

        descriptions = []
        for name, profile in speakers.items():
            count = profile.get("segment_count", 0)
            formality = profile.get("formality", "neutral")

            desc = f"- {name}: {count} lines"
            if formality != "neutral":
                desc += f", {formality} tone"

            # Check relationships
            relationships = context.get("relationships", {})
            if name in relationships:
                desc += f", {relationships[name]}"

            descriptions.append(desc)

        return "\n".join(descriptions)

    def build_scene_contexts(
        self,
        segments: list[Any]
    ) -> list[dict[str, Any]]:
        """
        Detect scene boundaries and build scene context for each segment.

        Scene changes detected by:
        - Time gap > 5 seconds between segments
        - Speaker change after a gap
        """
        scenes = []
        current_scene_start = 0
        current_speakers = set()

        for i, seg in enumerate(segments):
            # Get timing
            start_time = seg.start_time if hasattr(seg, 'start_time') else seg.get("start_time", 0)
            speaker = seg.speaker if hasattr(seg, 'speaker') else seg.get("speaker", "unknown")
            # Normalize None speaker to prevent str.join() crash
            current_speakers.add(speaker or "unknown")

            is_scene_break = False
            if i > 0:
                prev = segments[i - 1]
                prev_end = prev.end_time if hasattr(prev, 'end_time') else prev.get("end_time", 0)
                gap = start_time - prev_end

                if gap > 5.0:  # 5 second gap = likely scene change
                    is_scene_break = True

            if is_scene_break:
                speaker_names = ', '.join(s for s in current_speakers if s)
                scenes.append({
                    "start_index": current_scene_start,
                    "end_index": i - 1,
                    "speakers": list(current_speakers),
                    "description": f"Scene with {speaker_names}"
                })
                current_scene_start = i
                current_speakers = {speaker or "unknown"}

        # Final scene
        speaker_names = ', '.join(s for s in current_speakers if s)
        scenes.append({
            "start_index": current_scene_start,
            "end_index": len(segments) - 1,
            "speakers": list(current_speakers),
            "description": f"Scene with {speaker_names}"
        })

        return scenes

    def _get_scene_for_segment(
        self,
        segment_index: int,
        scenes: list[dict[str, Any]]
    ) -> Optional[str]:
        """Find the scene context for a given segment index."""
        for scene in scenes:
            if scene["start_index"] <= segment_index <= scene["end_index"]:
                return scene.get("description", "")
        return None


class TranslationQualityChecker:
    """
    Quality checker for translation to ensure:
    1. Translation is related to video context
    2. No translation errors
    3. Proper Vietnamese pronouns based on context
    """

    def __init__(self, gemini_api_key: Optional[str] = None):
        self.gemini_client = None
        self.gemini_api_key = gemini_api_key
        self._max_retries = 3
        self._min_interval = 0.15  # Rate limit: match TranslationEngine
        self._last_call_time = 0
        if gemini_api_key:
            self._init_gemini()

    def _init_gemini(self):
        """Initialize Gemini API for quality checking."""
        try:
            from google import genai
            self.gemini_client = genai.Client(api_key=self.gemini_api_key)
            self._model_name = 'gemini-2.5-flash'
            logger.info("Quality checker Gemini initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize quality checker: {e}")

    def _call_gemini_with_retry(self, prompt: str) -> str:
        """Call Gemini API with retry logic and rate limiting for quality checking."""
        for attempt in range(self._max_retries):
            # Rate limiting
            elapsed = time.time() - self._last_call_time
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)

            try:
                self._last_call_time = time.time()
                response = self.gemini_client.models.generate_content(
                    model=self._model_name,
                    contents=prompt
                )
                return response.text.strip()
            except Exception as e:
                is_rate_limit = "429" in str(e) or "quota" in str(e).lower()

                if attempt < self._max_retries - 1:
                    wait = (2 ** attempt) * (4 if is_rate_limit else 2)
                    logger.warning(f"Gemini quality check failed (attempt {attempt+1}), retry in {wait}s: {e}")
                    time.sleep(wait)
                else:
                    logger.error(f"Gemini quality check failed after {self._max_retries} attempts: {e}")
                    raise
        return ""

    def check_translation_quality(
        self,
        original_text: str,
        translated_text: str,
        source_language: str = "ja",
        target_language: str = "vi",
        context: Optional[dict[str, Any]] = None,
        video_context: Optional[str] = None
    ) -> dict[str, Any]:
        """
        Check translation quality.

        Args:
            original_text: Original Japanese text
            translated_text: Translated Vietnamese text
            source_language: Source language code
            target_language: Target language code
            context: Speaker context (speaker, relationship)
            video_context: Brief description of video content for context validation

        Returns:
            Dictionary with quality assessment
        """
        result = {
            "is_good": True,
            "issues": [],
            "score": 1.0,
            "needs_recheck": False
        }

        # Check 1: Empty translation
        if not translated_text or translated_text.strip() == "":
            result["is_good"] = False
            result["issues"].append("Empty translation")
            result["score"] = 0.0
            return result

        # Check 2: Marker present (fallback translation)
        if "[" in translated_text and "]" in translated_text:
            result["is_good"] = False
            result["issues"].append("Translation marker detected - API may have failed")
            result["score"] = 0.1
            return result

        # Check 3: Vietnamese character validation
        vietnamese_chars = "ăâđêôơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ"
        has_vietnamese = any(c in vietnamese_chars for c in translated_text.lower())

        if not has_vietnamese:
            result["issues"].append("No Vietnamese characters detected")
            result["needs_recheck"] = True

        # Check 4: Length ratio (should be roughly similar)
        orig_len = len(original_text)
        trans_len = len(translated_text)
        ratio = trans_len / max(orig_len, 1)

        if ratio < 0.3 or ratio > 3.0:
            result["issues"].append(f"Unusual length ratio: {ratio:.2f}")
            result["needs_recheck"] = True

        # Check 5: Context-aware pronoun check (for JA→VI)
        if source_language == "ja" and target_language == "vi":
            pronoun_issues = self._check_vietnamese_pronouns(
                translated_text, context or {}
            )
            if pronoun_issues:
                result["issues"].extend(pronoun_issues)
                result["needs_recheck"] = True

        # Check 6: Use Gemini for deep context check if available
        if self.gemini_client and (video_context or result["needs_recheck"]):
            gemini_check = self._check_with_gemini(
                original_text, translated_text, source_language,
                target_language, context, video_context
            )
            result["gemini_check"] = gemini_check
            if not gemini_check.get("is_accurate", True):
                result["is_good"] = False
                result["issues"].append("Gemini detected potential issues")

        # Calculate final score
        if result["issues"]:
            result["score"] = max(0.0, 1.0 - (len(result["issues"]) * 0.2))

        return result

    def _check_vietnamese_pronouns(
        self,
        translation: str,
        context: dict[str, Any]
    ) -> list[str]:
        """Check if Vietnamese pronouns match the context."""
        issues = []

        relationship = context.get("relationship", "")
        speaker = context.get("speaker", "")

        # Check for context-specific pronoun issues
        if "mother-son" in relationship or "mother-child" in relationship:
            # Mother to son should use "con" (you/child)
            # Son to mother should use "mẹ" (you/mother)
            if "con" in translation.lower():
                # Check if used correctly
                pass  # This is correct
            if "mẹ" in translation.lower() and "son" in speaker.lower():
                # Son using "mẹ" to mother is correct
                pass

        # Check for unnatural pronoun combinations
        # "tôi" and "bạn" together is unusual in family context
        if "tôi" in translation.lower() and "bạn" in translation.lower():
            # This could be wrong in family context
            issues.append("Unusual pronoun combination (tôi + bạn)")

        return issues

    def _check_with_gemini(
        self,
        original: str,
        translation: str,
        source_lang: str,
        target_lang: str,
        context: Optional[dict],
        video_context: Optional[str]
    ) -> dict:
        """Use Gemini to verify translation accuracy."""
        try:
            prompt = f"""You are a translation quality checker.

Original text ({source_lang}): {original}
Translation ({target_lang}): {translation}

Context: {context or 'None'}
Video context: {video_context or 'Not provided'}

Check if:
1. The translation accurately reflects the original meaning
2. The translation is appropriate for the context (family, professional, etc.)
3. Vietnamese pronouns are used correctly if the target is Vietnamese

Respond in JSON format:
{{
    "is_accurate": true/false,
    "issues": ["issue1", "issue2"],
    "suggested_fix": "if issues found, suggest fix"
}}
"""
            result_text = self._call_gemini_with_retry(prompt)

            import json
            import re

            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())

        except Exception as e:
            logger.warning(f"Gemini quality check failed: {e}")

        return {"is_accurate": True, "issues": []}

    def batch_check(
        self,
        segments: list[dict[str, Any]],
        source_language: str = "ja",
        target_language: str = "vi",
        video_context: Optional[str] = None
    ) -> list[dict[str, Any]]:
        """
        Optimized batch quality check — checks 20 segments per API call.

        Args:
            segments: List of dicts with 'original', 'translation', 'context'
            source_language: Source language code
            target_language: Target language code
            video_context: Brief description of video content

        Returns:
            List of quality check results (one per segment)
        """
        results = []
        batch_size = 50

        for batch_start in range(0, len(segments), batch_size):
            batch = segments[batch_start:batch_start + batch_size]

            # Quick rule-based checks first (no API call)
            batch_results = []
            needs_gemini_check = False

            for i, seg in enumerate(batch):
                original = seg.get("original", "")
                translation = seg.get("translation", "")
                result = {
                    "is_good": True,
                    "issues": [],
                    "score": 1.0,
                    "needs_recheck": False,
                    "segment_index": batch_start + i
                }

                # Quick checks (no API)
                if not translation or translation.strip() == "":
                    result["is_good"] = False
                    result["issues"].append("Empty translation")
                    result["score"] = 0.0
                    result["needs_recheck"] = True
                elif "[" in translation and "]" in translation:
                    result["is_good"] = False
                    result["issues"].append("Translation marker detected")
                    result["score"] = 0.1
                    result["needs_recheck"] = True
                else:
                    # Length ratio check
                    ratio = len(translation) / max(len(original), 1)
                    if ratio < 0.3 or ratio > 3.0:
                        result["issues"].append(f"Unusual length ratio: {ratio:.2f}")
                        result["needs_recheck"] = True
                        needs_gemini_check = True

                    # Vietnamese character check
                    vietnamese_chars = "ăâđêôơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ"
                    if not any(c in vietnamese_chars for c in translation.lower()):
                        result["issues"].append("No Vietnamese characters")
                        result["needs_recheck"] = True
                        needs_gemini_check = True

                batch_results.append(result)

            # Gemini batch check (1 API call for 20 segments)
            if self.gemini_client and (needs_gemini_check or video_context):
                try:
                    gemini_results = self._batch_check_with_gemini(
                        batch, source_language, target_language, video_context
                    )

                    # Merge Gemini results into batch_results
                    for idx, gemini_result in enumerate(gemini_results):
                        if idx < len(batch_results):
                            if not gemini_result.get("is_accurate", True):
                                batch_results[idx]["is_good"] = False
                                batch_results[idx]["needs_recheck"] = True
                                batch_results[idx]["issues"].append("Gemini detected issues")
                            batch_results[idx]["gemini_check"] = gemini_result
                except Exception as e:
                    logger.warning(f"Gemini batch check failed: {e}")

            results.extend(batch_results)

        # Log summary
        needs_recheck = sum(1 for r in results if r.get("needs_recheck"))
        if needs_recheck:
            logger.info(f"Quality check: {needs_recheck}/{len(results)} segments need recheck")

        return results

    def _batch_check_with_gemini(
        self,
        segments: list[dict[str, Any]],
        source_lang: str,
        target_lang: str,
        video_context: Optional[str]
    ) -> list[dict]:
        """Check multiple translations at once with a single Gemini call."""
        try:
            pairs = ""
            for i, seg in enumerate(segments):
                original = seg.get("original", "")
                translation = seg.get("translation", "")
                speaker = seg.get("context", {}).get("speaker", "") if seg.get("context") else ""
                pairs += f"{i+1}. [{speaker}] {original} → {translation}\n"

            prompt = f"""You are a translation quality checker for {source_lang} to {target_lang} translations.

Video context: {video_context or 'Not provided'}

Review these translation pairs and identify any issues:
{pairs}

For EACH pair, check:
1. Translation accuracy (meaning preserved?)
2. Vietnamese pronouns appropriate for context?
3. Names/terms consistent?

Respond in JSON format - an array with one object per pair:
[
    {{"index": 1, "is_accurate": true, "issues": [], "suggested_fix": ""}},
    {{"index": 2, "is_accurate": false, "issues": ["wrong pronoun"], "suggested_fix": "use 'anh' instead of 'tôi'"}}
]

Only include entries that have issues. If all are correct, return an empty array [].
"""
            result_text = self._call_gemini_with_retry(prompt)

            import json
            import re

            # Parse JSON array
            json_match = re.search(r'\[.*\]', result_text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())

                # Build results for all segments
                results = [{"is_accurate": True, "issues": []} for _ in segments]

                for item in parsed:
                    idx = item.get("index", 0) - 1
                    if 0 <= idx < len(results):
                        results[idx] = item

                return results

        except Exception as e:
            logger.warning(f"Batch Gemini check failed: {e}")

        return [{"is_accurate": True, "issues": []} for _ in segments]

    def check_context_coherence(
        self,
        segments: list[dict[str, Any]],
        translations: list[str],
        source_language: str = "ja",
        target_language: str = "vi",
        video_context: Optional[str] = None,
        group_size: int = 50
    ) -> list[dict[str, Any]]:
        """
        Check translation coherence across consecutive segments.

        Detects:
        - Inconsistent pronoun usage for same character
        - Name/title inconsistencies
        - Tone shifts that don't match the original

        Returns list of issues with segment indices.
        """
        if not self.gemini_client:
            return []

        all_issues = []

        for group_start in range(0, len(segments), group_size):
            group_end = min(group_start + group_size, len(segments))
            group_segs = segments[group_start:group_end]
            group_trans = translations[group_start:group_end]

            try:
                dialogue = ""
                for i, (seg, trans) in enumerate(zip(group_segs, group_trans)):
                    speaker = seg.get("speaker", seg.get("context", {}).get("speaker", "?"))
                    original = seg.get("text", seg.get("original", ""))
                    dialogue += f"{group_start + i + 1}. [{speaker}] {original} → {trans}\n"

                prompt = f"""Review these consecutive subtitle translations for coherence:

Video context: {video_context or 'Not provided'}

{dialogue}

Check for:
1. Are pronouns consistent for each speaker throughout? (e.g., same character should always use the same pronoun pair)
2. Are names/titles translated consistently?
3. Is the tone appropriate and consistent for each speaker?
4. Any dialogue that seems disconnected from the flow?

Respond in JSON:
{{
    "coherence_score": 0-100,
    "issues": [
        {{"segment": 5, "type": "pronoun", "current": "tôi", "suggested": "anh", "reason": "Speaker_01 is older male"}}
    ]
}}

If everything looks coherent, return {{"coherence_score": 100, "issues": []}}
"""
                result_text = self._call_gemini_with_retry(prompt)

                import json
                import re

                json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group())
                    issues = parsed.get("issues", [])

                    for issue in issues:
                        issue["group_start"] = group_start
                        all_issues.append(issue)

                    score = parsed.get("coherence_score", 100)
                    if score < 80:
                        logger.info(f"Coherence score for segments {group_start}-{group_end}: {score}/100")

            except Exception as e:
                logger.warning(f"Context coherence check failed for group {group_start}: {e}")

        return all_issues

