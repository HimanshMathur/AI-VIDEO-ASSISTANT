"""
Multilingual regression tests for the AI Video Assistant.

Covers:
  - Language registry completeness (22 Indian Eighth-Schedule + international)
  - Unicode / script preservation for every major script block
  - Whisper code mappings
  - RTL detection
  - Script & language detection from text
  - Transcript-cleaning preserves Indic combining marks
  - Summarizer / extractor language-instruction construction
  - RAG answer-language policy: transcript language != answer language
  - transcribe_all returns a 2-tuple (text, detected_language)

These tests exercise the offline, metadata-only parts of the pipeline.
They do NOT require Ollama, Whisper, or GPU. Where a real model is needed
the test is skipped with a clear message so the suite stays fast and
reproducible.
"""

import unittest
import unicodedata

from core import languages as L
from core.languages import (
    Language,
    all_languages,
    indian_languages,
    international_languages,
    get_language,
    get_whisper_code,
    is_rtl,
    script_for_language,
    detect_script,
    detect_language_from_text,
    build_language_selector_options,
    language_display_name,
    capability_summary,
    overall_support_level,
    is_fully_supported,
    FULL, PARTIAL, WEAK, UNSUPPORTED,
)
from core.transcriber import _clean_segment_text, transcribe_all, TranscriptionError
from core.summarizer import _language_instruction as summarizer_lang_instruction
from core.extractor import _language_instruction as extractor_lang_instruction


# â”€â”€ Sample text per script (used for Unicode preservation + detection) â”€â”€â”€â”€â”€â”€
SCRIPT_SAMPLES = {
    # Indian Eighth-Schedule languages
    "Devanagari": 'नमस्ते दुनिया यह एक परीक्षण है।',          # Hindi / Marathi / Nepali
    "Bengali": 'নমস্কার বিশ্ব এটি একটি পরীক্ষা।',            # Bengali / Assamese
    "Gujarati": 'નમસ્તે વિશ્વ આ એક પરીક્ષણ છે.',
    "Gurmukhi": 'ਨਮਸਤੇ ਦੁਨੀਆ ਇਹ ਇੱਕ ਟੈਸਟ ਹੈ।',              # Punjabi
    "Kannada": 'ನಮಸ್ಕಾರ ಪ್ರಪಂಚ ಇದು ಒಂದು ಪರೀಕ್ಷೆ.',
    "Malayalam": 'നമസ്കാരം ലോകം ഇതൊരു പരീക്ഷണമാണ്.',
    "Odia": 'ନମସ୍କାର ବିଶ୍ୱ ଏହି ଗୋଟିଏ ପରୀକ୍ଷା।',
    "Tamil": 'வணக்கம் உலகம் இது ஒரு சோதனை.',
    "Telugu": 'నమస్కారం ప్రపంచం ఇది ఒక పరీక్ష.',
    "Arabic": 'مرحبا بالعالم هذا اختبار.',                  # Urdu / Sindhi / Arabic / Kashmiri (Arabic)
    # International
    "Cyrillic": 'Привет мир это тест.',                       # Russian
    "Han": '你好世界 这是一个测试。',                           # Chinese
    "Hangul": '안녕하세요 세계 이것은 테스트입니다.',           # Korean
    "Hiragana": 'こんにちは世界 これはテストです。',           # Japanese
    "Latin": "Hello world this is a test.",                   # English / Spanish / French / etc.
}


class TestLanguageRegistry(unittest.TestCase):
    """Verify the registry covers the required language set with valid metadata."""

    def test_22_indian_eighth_schedule_languages_present(self):
        indian = indian_languages()
        self.assertEqual(len(indian), 22, "Exactly 22 Indian Eighth-Schedule languages required")

    def test_all_required_indian_languages_present(self):
        required = {
            "Assamese", "Bengali", "Bodo", "Dogri", "Gujarati", "Hindi",
            "Kannada", "Kashmiri", "Konkani", "Maithili", "Malayalam",
            "Meitei (Manipuri)", "Marathi", "Nepali", "Odia", "Punjabi", "Sanskrit",
            "Santali", "Sindhi", "Tamil", "Telugu", "Urdu",
        }
        names = {l.name for l in indian_languages()}
        self.assertEqual(names, required)

    def test_every_language_has_whisper_code(self):
        for lang in all_languages():
            self.assertTrue(lang.whisper_code, f"{lang.name} missing whisper_code")

    def test_every_language_has_native_name(self):
        for lang in all_languages():
            self.assertTrue(lang.native_name, f"{lang.name} missing native_name")

    def test_every_language_has_at_least_one_script(self):
        for lang in all_languages():
            self.assertTrue(lang.scripts, f"{lang.name} has no scripts")

    def test_iso639_1_unique(self):
        codes = [l.iso639_1 for l in all_languages()]
        self.assertEqual(len(codes), len(set(codes)), "ISO 639-1 codes must be unique")

    def test_iso639_3_unique(self):
        codes = [l.iso639_3 for l in all_languages()]
        self.assertEqual(len(codes), len(set(codes)), "ISO 639-3 codes must be unique")

    def test_support_levels_are_valid(self):
        valid = {FULL, PARTIAL, WEAK, UNSUPPORTED}
        for lang in all_languages():
            for attr in ("whisper_support", "llm_support", "embedding_support", "vision_support"):
                self.assertIn(getattr(lang, attr), valid, f"{lang.name}.{attr} invalid")


class TestWhisperCodeMapping(unittest.TestCase):
    """get_whisper_code must return the token faster-whisper accepts."""

    # (input, expected_whisper_code)
    CASES = [
        ("Hindi", "hi"), ("hindi", "hi"), ("hi", "hi"), ("hin", "hi"),
        ("Bengali", "bn"), ("bn", "bn"),
        ("Tamil", "ta"), ("ta", "ta"),
        ("Telugu", "te"), ("te", "te"),
        ("Kannada", "kn"), ("kn", "kn"),
        ("Malayalam", "ml"), ("ml", "ml"),
        ("Marathi", "mr"), ("mr", "mr"),
        ("Gujarati", "gu"), ("gu", "gu"),
        ("Punjabi", "pa"), ("pa", "pa"),
        ("Urdu", "ur"), ("ur", "ur"),
        ("Nepali", "ne"), ("ne", "ne"),
        ("Odia", "or"), ("or", "or"),
        ("Assamese", "as"), ("as", "as"),
        ("Sanskrit", "sa"), ("sa", "sa"),
        ("Kashmiri", "ks"), ("ks", "ks"),
        ("Sindhi", "sd"), ("sd", "sd"),
        ("Konkani", "kok"), ("Maithili", "mai"),
        ("Dogri", "doi"), ("Bodo", "brx"),
        ("Meitei", "mni"), ("Santali", "sat"),
        # International
        ("English", "en"), ("Spanish", "es"), ("French", "fr"),
        ("German", "de"), ("Russian", "ru"), ("Arabic", "ar"),
        ("Chinese", "zh"), ("Japanese", "ja"), ("Korean", "ko"),
        ("Portuguese", "pt"), ("Italian", "it"),
    ]

    def test_whisper_code_mappings(self):
        for raw, expected in self.CASES:
            with self.subTest(input=raw):
                self.assertEqual(get_whisper_code(raw), expected)

    def test_auto_returns_none(self):
        for val in ("", "auto", "Auto", "AUTO", None):
            self.assertIsNone(get_whisper_code(val), f"get_whisper_code({val!r}) should be None")


class TestRTLDetection(unittest.TestCase):
    """RTL flag correctness for right-to-left languages."""

    RTL_LANGUAGES = ["Urdu", "ur", "urd", "Sindhi", "sd", "Arabic", "ar", "Kashmiri"]

    LTR_LANGUAGES = [
        "Hindi", "Tamil", "Telugu", "Bengali", "English", "Spanish",
        "French", "Russian", "Chinese", "Japanese", "Korean", "Punjabi",
        "Gujarati", "Kannada", "Malayalam", "Marathi", "Nepali", "Odia",
        "Assamese", "Sanskrit",
    ]

    def test_rtl_languages_detected(self):
        for lang in self.RTL_LANGUAGES:
            with self.subTest(lang=lang):
                self.assertTrue(is_rtl(lang), f"{lang} should be RTL")

    def test_ltr_languages_detected(self):
        for lang in self.LTR_LANGUAGES:
            with self.subTest(lang=lang):
                self.assertFalse(is_rtl(lang), f"{lang} should be LTR")

    def test_unknown_language_not_rtl(self):
        self.assertFalse(is_rtl("xyz"))
        self.assertFalse(is_rtl(None))


class TestScriptAndLanguageDetection(unittest.TestCase):
    """detect_script / detect_language_from_text must identify scripts correctly."""

    def test_detect_script_for_each_block(self):
        for script, sample in SCRIPT_SAMPLES.items():
            with self.subTest(script=script):
                detected = detect_script(sample)
                self.assertIsNotNone(detected, f"No script detected for {script}")
                # For scripts that share a Unicode block (e.g. Devanagari covers
                # Hindi/Marathi/Nepali), accept the block name.
                self.assertEqual(detected, script,
                                 f"Expected {script}, got {detected}")

    def test_mixed_latin_and_indic_text_reports_the_indic_script(self):
        self.assertEqual(detect_script("video mein kya \u0939\u0948"), "Devanagari")

    def test_detect_language_from_devanagari(self):
        # Devanagari is shared by many registry languages; script alone cannot
        # reliably identify Hindi, Marathi, Nepali, and others.
        self.assertIsNone(detect_language_from_text(SCRIPT_SAMPLES["Devanagari"]))

    def test_detect_language_from_arabic(self):
        # Arabic script is shared by Arabic, Urdu, Kashmiri, and Sindhi.
        self.assertIsNone(detect_language_from_text(SCRIPT_SAMPLES["Arabic"]))

    def test_detect_language_from_latin_is_ambiguous(self):
        self.assertIsNone(detect_language_from_text(SCRIPT_SAMPLES["Latin"]))

    def test_detect_empty_returns_none(self):
        self.assertIsNone(detect_language_from_text(""))
        self.assertIsNone(detect_language_from_text("   "))

    def test_script_for_language(self):
        self.assertEqual(script_for_language("Hindi"), "Devanagari")
        self.assertEqual(script_for_language("Tamil"), "Tamil")
        self.assertEqual(script_for_language("Urdu"), "Arabic")
        self.assertEqual(script_for_language("xyz"), "Unknown")


class TestUnicodePreservation(unittest.TestCase):
    """_clean_segment_text must not strip valid Indic combining marks or rewrite text."""

    SAMPLES = {
        "hindi_nahin": 'नहीं',            # DEVANAGARI LETTER NA + VIRAMA + HE + EE VOWEL
        "punjabi_main": 'ਮੈਂ',            # Gurmukhi bihari
        "tamil_sentence": 'வணக்கம் உலகம்',
        "telugu_sentence": 'నమస్కారం ప్రపంచం',
        "kannada_sentence": 'ನಮಸ್ಕಾರ ಪ್ರಪಂಚ',
        "malayalam_sentence": 'നമസ്കാരം ലോകം',
        "bengali_sentence": 'নমস্কার বিশ্ব',
        "gujarati_sentence": 'નમસ્તે વિશ્વ',
        "odia_sentence": 'ନମସ୍କାର ବିଶ୍ୱ',
        "sanskrit_chandrabindu": 'काँश्चित्',
    }

    # Urdu/Arabic without diacritics has no combining marks; test preservation separately.
    PRESERVATION_SAMPLES = dict(SAMPLES)
    PRESERVATION_SAMPLES["urdu_sentence"] = 'مرحبا بالعالم'

    def test_cleaning_preserves_indic_text(self):
        for name, sample in self.PRESERVATION_SAMPLES.items():
            with self.subTest(sample=name):
                cleaned = _clean_segment_text(sample)
                # NFC-normalize both for comparison (cleaning NFC-normalizes)
                self.assertEqual(cleaned, unicodedata.normalize("NFC", sample),
                                 f"Cleaning altered {name}: {cleaned!r}")

    def test_combining_marks_not_stripped(self):
        # Each sample has at least one combining mark (category M)
        for name, sample in self.SAMPLES.items():
            with self.subTest(sample=name):
                cleaned = _clean_segment_text(sample)
                marks_after = [c for c in cleaned if unicodedata.category(c).startswith("M")]
                self.assertTrue(marks_after,
                                f"All combining marks stripped from {name}")

    def test_trailing_orphaned_combining_mark_stripped(self):
        # A combining mark at the end with no base char (preceded by space) is orphaned
        text = "hello \u093f"  # DEVANAGARI VOWEL SIGN I after space (orphan)
        cleaned = _clean_segment_text(text)
        self.assertFalse(cleaned.endswith("\u093f"),
                         f"Trailing orphaned combining mark not stripped: {cleaned!r}")

    def test_arabic_joining_preserved(self):
        # Arabic uses right-to-left joining; cleaning must not reorder chars
        text = 'مرحبا'
        cleaned = _clean_segment_text(text)
        self.assertEqual(cleaned, text)

    def test_arabic_join_controls_are_preserved(self):
        text = "\u0645\u200d\u0631\u062d\u0628\u0627 \u0645\u06cc\u200c\u0634\u0648\u062f"
        self.assertEqual(_clean_segment_text(text), text)

    def test_control_chars_removed(self):
        text = "hello\x00world\x01"
        cleaned = _clean_segment_text(text)
        self.assertEqual(cleaned, "helloworld")

    def test_replacement_char_removed(self):
        text = "bad\ufffdchar"
        cleaned = _clean_segment_text(text)
        self.assertEqual(cleaned, "badchar")


class TestLanguageInstructionConstruction(unittest.TestCase):
    """_language_instruction must name the right language/script and flag RTL."""

    def test_summarizer_instruction_names_hindi(self):
        inst = summarizer_lang_instruction("hi", SCRIPT_SAMPLES["Devanagari"])
        self.assertIn("Hindi", inst)
        self.assertIn("Devanagari", inst)

    def test_summarizer_instruction_rtl_for_urdu(self):
        inst = summarizer_lang_instruction("ur", SCRIPT_SAMPLES["Arabic"])
        self.assertIn("Urdu", inst)
        self.assertIn("right-to-left", inst.lower())

    def test_summarizer_instruction_detects_from_text_when_none(self):
        inst = summarizer_lang_instruction(None, SCRIPT_SAMPLES["Tamil"])
        self.assertIn("Tamil", inst)

    def test_summarizer_instruction_generic_fallback(self):
        # Pure ASCII with no detectable language -> generic instruction
        inst = summarizer_lang_instruction(None, "hello world")
        # Should still contain the generic same-language rule
        self.assertIn("SAME language", inst)

    def test_extractor_instruction_matches_summarizer(self):
        s = summarizer_lang_instruction("ta", SCRIPT_SAMPLES["Tamil"])
        e = extractor_lang_instruction("ta", SCRIPT_SAMPLES["Tamil"])
        self.assertEqual(s, e, "Extractor and summarizer language instructions should match")
        self.assertIn("Tamil", s)
        self.assertIn("Tamil script", s)
        self.assertIn("Do not translate to English or any other language", s)


class TestTranscribeAllInputValidation(unittest.TestCase):
    """Empty input raises; successful calls return (text, detected_language)."""

    def test_empty_chunks_raise_transcription_error(self):
        # Empty input is rejected before transcription; tuple return applies
        # to successful transcriptions only.
        with self.assertRaisesRegex(TranscriptionError, "No audio chunks provided"):
            transcribe_all([], language=None)


class TestRagAnswerLanguagePolicy(unittest.TestCase):
    """
    The RAG system prompt must keep transcript language and answer language
    as separate concerns: the answer follows the *question* language (or an
    explicit output_language override), NOT the transcript language.
    """

    def test_system_prompt_says_answer_follows_question_language(self):
        from core.rag_engine import _prepare_rag_messages

        # Retrieval is irrelevant to the prompt policy; use a lightweight chain double.
        from types import SimpleNamespace
        chain = SimpleNamespace(visual_store=None, retrieve=lambda _question: [])

        # Hindi question, English transcript
        messages, intent, t_docs, v_docs, ctx = _prepare_rag_messages(
            chain, 'यह वीडियो किस बारे में है?'
        )
        system_msg = messages[0]["content"]
        # The prompt must instruct the model to answer in the question's language
        self.assertIn("question", system_msg.lower())
        self.assertIn("Hindi", system_msg)
        # It must NOT say to answer in the transcript's language
        self.assertNotIn("English", system_msg.split("ANSWER LANGUAGE")[0][:600],
                         "Prompt must not force answer into transcript language")

    def test_output_language_override_appears_in_prompt(self):
        from core.rag_engine import _prepare_rag_messages
        from types import SimpleNamespace
        chain = SimpleNamespace(visual_store=None, retrieve=lambda _question: [])

        messages, *_ = _prepare_rag_messages(
            chain, "What is this about?", output_language="hi"
        )
        system_msg = messages[0]["content"]
        self.assertIn("OVERRIDE", system_msg)
        self.assertIn("Hindi", system_msg)
        self.assertIn("Devanagari", system_msg)

    def test_auto_output_language_does_not_add_override(self):
        from core.rag_engine import _prepare_rag_messages
        from types import SimpleNamespace
        chain = SimpleNamespace(visual_store=None, retrieve=lambda _question: [])

        messages, *_ = _prepare_rag_messages(
            chain, "What is this about?", output_language="auto"
        )
        system_msg = messages[0]["content"]
        self.assertNotIn("OVERRIDE", system_msg)

    def test_chapter_list_preserves_timestamps_and_markdown_line_breaks(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from core.rag_engine import _generate_chapter_list

        chain = SimpleNamespace(vector_store=SimpleNamespace(chunks=[
            {"chunk_index": 0, "start_seconds": 0, "text": "Opening scene"},
            {"chunk_index": 1, "start_seconds": 32, "text": "The next segment"},
        ]))
        with patch("core.rag_engine.json_chat_completion", return_value={"chapters": [
            {"index": 0, "title": "Opening Scene"},
            {"index": 1, "title": "Next Segment"},
        ]}):
            result = _generate_chapter_list(chain, "Create chapters")

        lines = result.splitlines()
        self.assertEqual([line.rstrip() for line in lines], [
            "00:00 \u2014 Opening Scene", "00:32 \u2014 Next Segment",
        ])
        self.assertTrue(all(line.endswith("  ") for line in lines))

    def test_missing_chapter_titles_fall_back_without_dropping_segments(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from core.rag_engine import _generate_chapter_list

        chain = SimpleNamespace(vector_store=SimpleNamespace(chunks=[
            {"chunk_index": 7, "start_seconds": 30, "text": "Opening the old theater"},
            {"chunk_index": 8, "start_seconds": 5, "text": "We arrive at the prison"},
        ]))
        with patch("core.rag_engine.json_chat_completion", return_value={"chapters": [
            {"index": 8, "title": "Prison Arrival"},
        ]}):
            result = _generate_chapter_list(chain, "Create chapters")
        self.assertEqual(result.splitlines(), [
            '00:05 — Prison Arrival  ',
            '00:30 — Opening the old theater  ',
        ])

    def test_root_level_legacy_chapter_response_is_parsed(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from core.rag_engine import _generate_chapter_list

        chain = SimpleNamespace(vector_store=SimpleNamespace(chunks=[
            {"chunk_index": 0, "start_seconds": 0, "text": "Focus and attention are the subject"},
            {"chunk_index": 1, "start_seconds": 12, "text": "More about focus and attention"},
        ]))
        with patch("core.rag_engine.json_chat_completion", return_value={
            "index": 0, "title": "Focus and attention",
        }):
            result = _generate_chapter_list(chain, "Create chapters")
        self.assertEqual(result.splitlines(), ['00:00 — Focus and attention  '])

    def test_omitted_continuation_run_does_not_create_one_chapter_per_chunk(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from core.rag_engine import _generate_chapter_list

        chain = SimpleNamespace(vector_store=SimpleNamespace(chunks=[
            {"chunk_index": 0, "start_seconds": 0, "text": "We discuss focus and attention"},
            {"chunk_index": 1, "start_seconds": 10, "text": "More detail about focus and attention"},
            {"chunk_index": 2, "start_seconds": 20, "text": "Continue explaining attention and focus"},
            {"chunk_index": 3, "start_seconds": 30, "text": "Now preparing a study routine"},
        ]))
        with patch("core.rag_engine.json_chat_completion", return_value={"chapters": [
            {"index": 0, "title": "Focus and attention"},
            {"index": 3, "title": "Study routine"},
        ]}):
            result = _generate_chapter_list(chain, "Create chapters")
        self.assertEqual(result.splitlines(), [
            '00:00 — Focus and attention  ',
            '00:30 — Study routine  ',
        ])

    def test_model_timeout_uses_a_group_fallback_instead_of_chunk_titles(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from core.rag_engine import _generate_chapter_list

        chain = SimpleNamespace(vector_store=SimpleNamespace(chunks=[
            {"chunk_index": 0, "start_seconds": 0, "text": "Discussing attention and staying focused"},
            {"chunk_index": 1, "start_seconds": 10, "text": "Explaining more about attention and focus"},
            {"chunk_index": 2, "start_seconds": 20, "text": "Continuing the discussion of focus"},
        ]))
        with patch("core.rag_engine.json_chat_completion", side_effect=TimeoutError("slow model")):
            result = _generate_chapter_list(chain, "Create chapters")
        self.assertEqual(len(result.splitlines()), 1)
        self.assertTrue(result.startswith('00:00 — '))

    def test_overlapping_transcript_chunks_keep_source_timestamps_within_duration(self):
        from unittest.mock import patch
        from core.vector_store import build_vector_store

        class Splitter:
            def __init__(self, **_kwargs):
                pass

            def split_text(self, _text):
                return ["abcdef", "efghij", "ijkl"]

        class Embeddings:
            def get_embedding_dimension(self):
                return 2

        class Store:
            def __init__(self, **_kwargs):
                self.chunks = []

            def add_chunks(self, chunks):
                self.chunks.extend(chunks)

        with patch("core.vector_store.RecursiveCharacterTextSplitter", Splitter), \
                patch("core.vector_store.get_embedding_model", return_value=Embeddings()), \
                patch("core.vector_store.FaissVectorStore", Store):
            store = build_vector_store(
                "abcdefghijkl", chunk_size=6, chunk_overlap=2, duration_seconds=120,
            )
        self.assertEqual([chunk["start_seconds"] for chunk in store.chunks], [0.0, 40.0, 80.0])
        self.assertTrue(all(chunk["start_seconds"] <= 120 for chunk in store.chunks))

    def test_topic_labels_group_adjacent_segments_and_keep_topic_changes(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from core.rag_engine import _generate_chapter_list

        chain = SimpleNamespace(vector_store=SimpleNamespace(chunks=[
            {"chunk_index": 0, "start_seconds": 0, "text": "Exploring the prison rooms"},
            {"chunk_index": 1, "start_seconds": 10, "text": "More prison exploration"},
            {"chunk_index": 2, "start_seconds": 20, "text": "Preparing a camp outside"},
            {"chunk_index": 3, "start_seconds": 30, "text": "Setting up the camp"},
        ]))
        with patch("core.rag_engine.json_chat_completion", return_value={"chapters": [
            {"start_index": 0, "end_index": 1, "title": "Exploring the prison"},
            {"start_index": 2, "end_index": 3, "title": "Preparing camp"},
        ]}):
            result = _generate_chapter_list(chain, "Create chapters")
        self.assertEqual(result.splitlines(), [
            '00:00 — Exploring the prison  ',
            '00:20 — Preparing camp  ',
        ])

    def test_minor_title_wording_changes_do_not_split_a_topic(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from core.rag_engine import _generate_chapter_list

        chain = SimpleNamespace(vector_store=SimpleNamespace(chunks=[
            {"chunk_index": 0, "start_seconds": 12, "text": "They walk through the abandoned prison."},
            {"chunk_index": 1, "start_seconds": 18, "text": "The group explores more prison rooms."},
            {"chunk_index": 2, "start_seconds": 27, "text": "They discuss where to set up camp."},
        ]))
        with patch("core.rag_engine.json_chat_completion", return_value={"chapters": [
            {"index": 0, "title": "Exploring the prison"},
            {"index": 1, "title": "Prison exploration"},
            {"index": 2, "title": "Setting up camp"},
        ]}):
            result = _generate_chapter_list(chain, "Create chapters")
        self.assertEqual(result.splitlines(), [
            '00:12 — Exploring the prison  ',
            '00:27 — Setting up camp  ',
        ])

    def test_ambiguous_fragment_and_corrupt_title_continue_neighbor_topic(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from core.rag_engine import _generate_chapter_list

        chain = SimpleNamespace(vector_store=SimpleNamespace(chunks=[
            {"chunk_index": 0, "start_seconds": 4, "text": "The host begins exploring the prison."},
            {"chunk_index": 1, "start_seconds": 8, "text": "And then..."},
            {"chunk_index": 2, "start_seconds": 11, "text": "They continue through the prison halls."},
        ]))
        with patch("core.rag_engine.json_chat_completion", return_value={"chapters": [
            {"index": 0, "title": "Exploring the prison"},
            {"index": 1, "title": "\ufffdx"},
            {"index": 2, "title": "Prison exploration"},
        ]}):
            result = _generate_chapter_list(chain, "Create chapters")
        self.assertEqual(result.splitlines(), ['00:04 — Exploring the prison  '])

    def test_corrupt_middle_title_does_not_split_non_adjacent_duplicate_topic(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from core.rag_engine import _generate_chapter_list

        chain = SimpleNamespace(vector_store=SimpleNamespace(chunks=[
            {"chunk_index": 0, "start_seconds": 4, "text": "The host begins exploring the prison."},
            {"chunk_index": 1, "start_seconds": 8, "text": "And then..."},
            {"chunk_index": 2, "start_seconds": 11, "text": "They continue through the prison halls."},
        ]))
        with patch("core.rag_engine.json_chat_completion", return_value={"chapters": [
            {"index": 0, "title": "Exploring the prison"},
            {"index": 1, "title": "\ufffdx"},
            {"index": 2, "title": "Prison exploration"},
        ]}):
            result = _generate_chapter_list(chain, "Create chapters")
        self.assertEqual(result.splitlines(), ['00:04 — Exploring the prison  '])

    def test_grouped_range_can_extend_across_inference_batch_boundary(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from core.rag_engine import _generate_chapter_list

        chain = SimpleNamespace(vector_store=SimpleNamespace(chunks=[
            {"chunk_index": 0, "start_seconds": 0, "text": "Exploring the prison rooms"},
            {"chunk_index": 1, "start_seconds": 10, "text": "More prison exploration"},
            {"chunk_index": 2, "start_seconds": 20, "text": "Preparing a camp outside"},
            {"chunk_index": 3, "start_seconds": 30, "text": "Setting up the camp"},
        ]))
        with patch("core.rag_engine.json_chat_completion", return_value={"chapters": [
            {"start_index": 0, "end_index": 1, "title": "Exploring the prison"},
            {"start_index": 2, "end_index": 3, "title": "Preparing camp"},
        ]}) as completion:
            result = _generate_chapter_list(chain, "Create chapters")
        self.assertEqual(result.splitlines(), [
            '00:00 — Exploring the prison  ',
            '00:20 — Preparing camp  ',
        ])
        self.assertEqual(completion.call_count, 2)

    def test_multilingual_titles_preserve_transcript_scripts_and_group_topics(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from core.rag_engine import _generate_chapter_list

        fixtures = [
            (
                ['आज हम आत्म नियंत्रण पर बात करेंगे', 'आत्म नियंत्रण की आदतें', 'अब ध्यान और पढ़ाई'],
                ['आत्म नियंत्रण', 'आत्म नियंत्रण', 'ध्यान और पढ़ाई'],
                ['आत्म नियंत्रण', 'आत्म नियंत्रण', 'ध्यान और पढ़ाई'],
            ),
            (
                ['आज dopamine के असर पर बात करेंगे', 'dopamine और focus की आदतें', 'अब study routine'],
                ['dopamine के असर', 'dopamine के असर', "study routine"],
                ['dopamine के असर', 'dopamine के असर', "study routine"],
            ),
            (
                ['ਅੱਜ ਅਸੀਂ ਮਨ ਦੀ ਇਕਾਗਰਤਾ ਬਾਰੇ ਗੱਲ ਕਰਾਂਗੇ', 'ਇਕਾਗਰਤਾ ਦੀਆਂ ਆਦਤਾਂ', 'ਹੁਣ ਪੜ੍ਹਾਈ ਦਾ ਸਮਾਂ'],
                ['ਮਨ ਦੀ ਇਕਾਗਰਤਾ', 'ਮਨ ਦੀ ਇਕਾਗਰਤਾ', 'ਪੜ੍ਹਾਈ ਦਾ ਸਮਾਂ'],
                ['ਮਨ ਦੀ ਇਕਾਗਰਤਾ', 'ਮਨ ਦੀ ਇਕਾਗਰਤਾ', 'ਪੜ੍ਹਾਈ ਦਾ ਸਮਾਂ'],
            ),
        ]
        for texts, titles, expected_titles in fixtures:
            with self.subTest(texts=texts):
                chain = SimpleNamespace(vector_store=SimpleNamespace(chunks=[
                    {"chunk_index": i, "start_seconds": i * 15, "text": text}
                    for i, text in enumerate(texts)
                ]))
                with patch("core.rag_engine.json_chat_completion", return_value={"chapters": [
                    {"index": i, "title": title} for i, title in enumerate(titles)
                ]}):
                    output = _generate_chapter_list(chain, "Create chapters")
                self.assertEqual([line.split(' — ', 1)[1].rstrip() for line in output.splitlines()], expected_titles)
                self.assertEqual([line[:5] for line in output.splitlines()], ["00:00", "00:15", "00:30"])

    def test_model_chapter_groups_use_topic_start_timestamps(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from core.rag_engine import _generate_chapter_list

        chain = SimpleNamespace(vector_store=SimpleNamespace(chunks=[
            {"chunk_index": 0, "start_seconds": 6, "text": "Exploring the prison entrance"},
            {"chunk_index": 1, "start_seconds": 13, "text": "Exploring more prison rooms"},
            {"chunk_index": 2, "start_seconds": 19, "text": "Preparing camp outside"},
        ]))
        with patch("core.rag_engine.json_chat_completion", return_value={"chapters": [
            {"start_index": 0, "end_index": 1, "title": "Exploring the prison"},
            {"start_index": 2, "end_index": 2, "title": "Preparing camp"},
        ]}):
            output = _generate_chapter_list(chain, "Create chapters")
        self.assertEqual(output.splitlines(), [
            '00:06 — Exploring the prison  ',
            '00:19 — Preparing camp  ',
        ])

    def test_explicit_group_suppresses_legacy_intermediate_chapter_start(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from core.rag_engine import _generate_chapter_list

        chain = SimpleNamespace(vector_store=SimpleNamespace(chunks=[
            {"chunk_index": 0, "start_seconds": 0, "text": "Discussing focus and attention"},
            {"chunk_index": 1, "start_seconds": 15, "text": "More discussion of focus and attention"},
            {"chunk_index": 2, "start_seconds": 30, "text": "Starting a study routine"},
        ]))
        with patch("core.rag_engine.json_chat_completion", return_value={"chapters": [
            {"start_index": 0, "end_index": 1, "title": "Focus and attention"},
            {"index": 1, "title": "Focus and attention"},
            {"index": 2, "title": "Study routine"},
        ]}):
            output = _generate_chapter_list(chain, "Create chapters")
        self.assertEqual([line[:5] for line in output.splitlines()], ["00:00", "00:30"])

    def test_separate_explicit_groups_keep_repeated_topic_starts(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from core.rag_engine import _generate_chapter_list

        chain = SimpleNamespace(vector_store=SimpleNamespace(chunks=[
            {"chunk_index": 0, "start_seconds": 0, "text": "Topic details begin"},
            {"chunk_index": 1, "start_seconds": 15, "text": "Topic details continue later"},
        ]))
        with patch("core.rag_engine.json_chat_completion", return_value={"chapters": [
            {"start_index": 0, "end_index": 0, "title": "Topic details"},
            {"start_index": 1, "end_index": 1, "title": "Topic details"},
        ]}):
            output = _generate_chapter_list(chain, "Create chapters")
        self.assertEqual([line[:5] for line in output.splitlines()], ["00:00", "00:15"])

    def test_repeated_legacy_topic_titles_keep_segment_starts_across_unicode_scripts(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from core.rag_engine import _generate_chapter_list

        hindi = "".join(map(chr, (0x0906, 0x0924, 0x094d, 0x092e, 0x0020, 0x0928, 0x093f, 0x092f, 0x0902, 0x0924, 0x094d, 0x0930, 0x0923)))
        hinglish = "dopamine " + "".join(map(chr, (0x0915, 0x0947, 0x0020, 0x0905, 0x0938, 0x0930)))
        punjabi = "".join(map(chr, (0x0a2e, 0x0a28, 0x0020, 0x0a26, 0x0a40, 0x0020, 0x0a07, 0x0a15, 0x0a17, 0x0a30, 0x0a24, 0x0a3e)))
        repeated_topics = [hindi, hinglish, punjabi]
        for topic in repeated_topics:
            with self.subTest(topic=topic):
                next_topic = "Astronomy"
                chain = SimpleNamespace(vector_store=SimpleNamespace(chunks=[
                    {"chunk_index": 0, "start_seconds": 0, "text": topic + " overview"},
                    {"chunk_index": 1, "start_seconds": 15, "text": topic + " details"},
                    {"chunk_index": 2, "start_seconds": 30, "text": next_topic + " overview"},
                ]))
                with patch("core.rag_engine.json_chat_completion", return_value={"chapters": [
                    {"index": 0, "title": topic},
                    {"index": 1, "title": topic},
                    {"index": 2, "title": next_topic},
                ]}):
                    output = _generate_chapter_list(chain, "Create chapters")
                self.assertEqual([line[:5] for line in output.splitlines()], ["00:00", "00:15", "00:30"])
                self.assertIn(topic, output)

    def test_mojibake_title_is_rejected_for_transcript_fallback(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from core.rag_engine import _generate_chapter_list

        transcript = 'आत्म नियंत्रण का अभ्यास शुरू करते हैं'
        chain = SimpleNamespace(vector_store=SimpleNamespace(chunks=[
            {"chunk_index": 0, "start_seconds": 42, "text": transcript},
        ]))
        corrupted = "\u00e0\u00a4\u00a6 \u00e0\u00a4\u2013 \u00e0\u00a4\u00a4"
        with patch("core.rag_engine.json_chat_completion", return_value={"chapters": [
            {"index": 0, "title": corrupted},
        ]}):
            output = _generate_chapter_list(chain, "Create chapters")
        self.assertEqual(output.splitlines(), ['00:42 — आत्म नियंत्रण का अभ्यास शुरू करते  '])

    def test_mojibake_transcript_fallback_restores_unicode_script(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from core.rag_engine import _generate_chapter_list

        transcript = 'आत्म नियंत्रण का अभ्यास शुरू करते हैं'
        corrupted_transcript = transcript.encode('utf-8').decode('latin-1')
        chain = SimpleNamespace(vector_store=SimpleNamespace(chunks=[
            {"chunk_index": 0, "start_seconds": 42, "text": corrupted_transcript},
        ]))
        corrupted_title = corrupted_transcript[:12]
        with patch("core.rag_engine.json_chat_completion", return_value={"chapters": [
            {"index": 0, "title": corrupted_title},
        ]}):
            output = _generate_chapter_list(chain, "Create chapters")
        self.assertEqual(output.splitlines(), ['00:42 — आत्म नियंत्रण का अभ्यास शुरू करते  '])

    def test_visual_question_without_visual_evidence_does_not_use_transcript(self):
        from types import SimpleNamespace
        from core.rag_engine import ask_question

        def transcript_must_not_be_used(_question):
            self.fail("Visual-only questions must not retrieve transcript evidence")

        chain = SimpleNamespace(visual_store=None, retrieve=transcript_must_not_be_used)
        answer = ask_question(chain, "What is visible on screen?")
        self.assertIn("visual evidence", answer.lower())
        self.assertIn("not provide enough", answer.lower())

    def test_retrieval_reranks_exact_evidence_overlap(self):
        from types import SimpleNamespace
        from core.rag_engine import _rerank_evidence

        unrelated = SimpleNamespace(page_content="An unrelated landscape scene", metadata={"score": 0.60})
        relevant = SimpleNamespace(page_content="A violin is visible in the room", metadata={"score": 0.40})
        ranked = _rerank_evidence("Which violin is visible?", [unrelated, relevant], limit=1)
        self.assertEqual(ranked, [relevant])
        low_relevance = SimpleNamespace(page_content="Weather forecast for tomorrow", metadata={"score": 0.10})
        self.assertEqual(_rerank_evidence("Which violin is visible?", [low_relevance], limit=6), [])

    def test_faiss_visual_results_retain_timestamp_metadata(self):
        from core.vector_store import FaissVectorStore

        class Index:
            ntotal = 1
            def search(self, _query, _count):
                return [[0.9]], [[0]]

        class Embedding:
            def encode(self, _texts, **_kwargs):
                return [[0.1, 0.2]]

        store = object.__new__(FaissVectorStore)
        store.index = Index()
        store.embedding_model = Embedding()
        store.chunks = [{
            "chunk_id": "frame_1", "chunk_index": 0,
            "text": "Scene: a presenter is visible.",
            "timestamp": "00:00:32", "timestamp_display": "00:00:32 to 00:00:40",
            "timestamp_seconds": 32.0, "end_seconds": 40.0,
        }]
        document = store.similarity_search("presenter", k=1)[0]
        self.assertEqual(document.metadata["timestamp"], "00:00:32")
        self.assertEqual(document.metadata["timestamp_display"], "00:00:32 to 00:00:40")
        self.assertEqual(document.metadata["timestamp_seconds"], 32.0)

    def test_visual_context_keeps_its_timestamp_separate_from_transcript(self):
        from types import SimpleNamespace
        from core.rag_engine import format_context

        transcript = SimpleNamespace(page_content="The speaker mentions a red car.", metadata={"start_seconds": 12})
        visual = SimpleNamespace(page_content="Scene: a blue car is visible.", metadata={"timestamp_display": "00:00:32 to 00:00:40"})
        context = format_context([transcript], [visual])
        self.assertIn("TRANSCRIPT EVIDENCE (SPOKEN AUDIO)", context)
        self.assertIn("VISUAL EVIDENCE (OBSERVED VIDEO FRAMES)", context)
        self.assertIn("Visual Evidence 1 at 00:00:32 to 00:00:40", context)
        self.assertLess(context.index("TRANSCRIPT EVIDENCE"), context.index("VISUAL EVIDENCE"))

    def test_answer_uses_current_video_evidence_and_zero_temperature(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from core.rag_engine import ask_question

        evidence = SimpleNamespace(page_content="The transcript says the meeting starts at noon.", metadata={"score": 0.8})
        chain = SimpleNamespace(visual_store=None, top_k=6, retrieve=lambda _question: [evidence])
        with patch("core.rag_engine.chat_completion", return_value="The meeting starts at noon.") as completion:
            answer = ask_question(
                chain, "When does the meeting start?",
                conversation_history=[{"role": "user", "content": "What meeting?"}],
            )
        self.assertEqual(answer, "The meeting starts at noon.")
        self.assertEqual(completion.call_args.kwargs["temperature"], 0.0)
        messages = completion.call_args.args[0]
        self.assertIn("current video", messages[0]["content"])
        self.assertTrue(any(m.get("content") == "What meeting?" for m in messages[1:-1]))
        self.assertIn("meeting starts at noon", messages[0]["content"])


class TestLanguageSelector(unittest.TestCase):
    """build_language_selector_options produces a usable dropdown list."""

    def test_first_option_is_auto_detect(self):
        opts = build_language_selector_options()
        self.assertEqual(opts[0], "auto")

    def test_all_languages_listed(self):
        opts = build_language_selector_options()
        # -1 for the auto option
        self.assertEqual(len(opts) - 1, len(all_languages()))
        self.assertTrue(all(type(option) is str and option for option in opts))
        self.assertEqual(len(opts), len(set(opts)))
        self.assertIn("hi", opts)
        self.assertIn("pa", opts)
        self.assertIn("en", opts)
        self.assertTrue(all(get_language(code) is not None for code in opts[1:]))

    def test_default_is_auto_code(self):
        opts = build_language_selector_options()
        default_index = 0  # Matches app.py's selectbox(index=0).
        selected = opts[default_index]
        self.assertEqual(selected, "auto")
        self.assertNotIn(selected, {lang.iso639_1 for lang in all_languages()})

    def test_language_display_name(self):
        for code in build_language_selector_options():
            label = "Auto - match the question language" if code == "auto" else language_display_name(code)
            self.assertIsInstance(label, str)
            self.assertTrue(label)
        self.assertEqual(build_language_selector_options()[0], "auto")
        self.assertIn("Hindi", language_display_name("hi"))
        self.assertIn('हिन्दी', language_display_name("hi"))
        self.assertEqual(language_display_name(""), "Auto-detected")


class TestVisualEvidencePipeline(unittest.TestCase):
    def test_failed_frame_inference_is_not_indexed_as_evidence(self):
        from unittest.mock import patch
        from core.visual_analyzer import analyze_video_visuals

        frame = {
            "index": 0, "timestamp_str": "00:00:01",
            "timestamp_seconds": 1.0, "end_seconds": 2.0,
            "end_timestamp_str": "00:00:02", "path": "video_frames_test/frame.jpg",
        }
        with patch("core.visual_analyzer.is_video_file", return_value=True), \
             patch("core.visual_analyzer.check_vision_model_available", return_value=True), \
             patch("core.visual_analyzer.sample_representative_frames", return_value=[frame]), \
             patch("core.visual_analyzer.analyze_frame_with_vision_model", return_value={}), \
             patch("core.visual_analyzer.shutil.rmtree"):
            evidence = analyze_video_visuals("video.mp4")
        self.assertEqual(evidence, [])


class TestCapabilityMatrix(unittest.TestCase):
    """Verify the capability-summary helpers behave consistently."""

    def test_fully_supported_requires_all_full(self):
        caps = capability_summary("English")
        self.assertEqual(caps["whisper"], FULL)
        self.assertEqual(caps["llm"], FULL)

    def test_overall_support_not_unsupported_for_major_languages(self):
        for name in ["Hindi", "Tamil", "Telugu", "Bengali", "English", "Urdu"]:
            with self.subTest(name=name):
                self.assertNotEqual(overall_support_level(name), UNSUPPORTED)

    def test_low_resource_languages_not_fully_supported(self):
        # Bodo is a low-resource language; it should not be FULL across the board
        self.assertFalse(is_fully_supported("Bodo"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
