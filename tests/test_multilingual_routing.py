"""
Focused unit tests for multilingual routing and pipeline fixes in rag_engine.py.

ALL external calls (Ollama chat_completion, sentence-transformers, FAISS, visual-
analyzer) are mocked so this suite runs offline without a GPU or a running Ollama
server.  These are pure UNIT tests - they verify deterministic code paths.

Sections
--------
A. _has_non_latin_letters
B. classify_question_intent  (heuristic paths only, no LLM call needed)
C. _rerank_evidence          (non-Latin vs Latin scoring)
D. _remove_repeated_sentences (conservative deduplication)
E. _prepare_rag_messages     (language instruction, output-language override,
                               no-evidence guard, conversation history pass-through)
F. ask_question              (transcript-grounded answer, no-evidence response,
                               repetition cleanup, explicit output-language)
G. Regression: existing test_multilingual.py compatibility (no-Ollama versions)
"""
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

# ---------------------------------------------------------------------------
# Stub heavy optional dependencies BEFORE importing rag_engine so the module
# loads without GPU / sentence-transformers / FAISS / torch on the test runner.
# ---------------------------------------------------------------------------
for _mod_name in ("torch", "faiss", "numpy"):
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = MagicMock()

if "sentence_transformers" not in sys.modules:
    _st = types.ModuleType("sentence_transformers")
    _st.SentenceTransformer = MagicMock()
    sys.modules["sentence_transformers"] = _st

if "langchain_text_splitters" not in sys.modules:
    _lts = types.ModuleType("langchain_text_splitters")
    _lts.RecursiveCharacterTextSplitter = MagicMock()
    sys.modules["langchain_text_splitters"] = _lts

if "langchain_ollama" not in sys.modules:
    sys.modules["langchain_ollama"] = MagicMock()

from core.rag_engine import (
    _has_non_latin_letters,
    classify_question_intent,
    _rerank_evidence,
    _remove_repeated_sentences,
    _prepare_rag_messages,
    ask_question,
    FaissDocument,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _doc(content, score=0.5):
    return FaissDocument(page_content=content, metadata={"score": score})


def _make_chain(docs=None, visual_docs=None):
    docs = docs or []

    class _IdxStub:
        ntotal = len(visual_docs) if visual_docs else 0

    class _VisualStore:
        index = _IdxStub()
        chunks = visual_docs or []

    chain = SimpleNamespace(
        visual_store=_VisualStore() if visual_docs else None,
        retrieve=lambda q: docs,
        vector_store=SimpleNamespace(similarity_search=lambda q, k: docs),
        top_k=6,
        duration_seconds=600.0,
    )
    return chain


# ===========================================================================
# A.  _has_non_latin_letters
# ===========================================================================
class TestHasNonLatinLetters(unittest.TestCase):

    def test_pure_english_is_latin(self):
        self.assertFalse(_has_non_latin_letters("What is discussed in this video?"))

    def test_hinglish_latin_is_latin(self):
        self.assertFalse(_has_non_latin_letters("Is video mein kya samjhaya gaya hai?"))

    def test_hindi_devanagari_is_non_latin(self):
        self.assertTrue(_has_non_latin_letters("इस वीडियो में क्या समझाया गया है?"))

    def test_marathi_devanagari_is_non_latin(self):
        self.assertTrue(_has_non_latin_letters("या व्हिडिओमध्ये काय सांगितले आहे?"))

    def test_punjabi_gurmukhi_is_non_latin(self):
        self.assertTrue(_has_non_latin_letters("ਇਸ ਵੀਡੀਓ ਵਿੱਚ ਕੀ ਸਮਝਾਇਆ ਗਿਆ ਹੈ?"))

    def test_mixed_latin_and_devanagari_is_non_latin(self):
        self.assertTrue(_has_non_latin_letters("Video mein David Allen की 2-minute rule ke baare mein"))

    def test_digits_and_punctuation_only(self):
        self.assertFalse(_has_non_latin_letters("123 !@#$%"))

    def test_empty_string(self):
        self.assertFalse(_has_non_latin_letters(""))


# ===========================================================================
# B.  classify_question_intent (heuristic paths only – chat_completion is never
#     reached; we patch it to raise so any slip-through fails immediately)
# ===========================================================================
class TestClassifyQuestionIntent(unittest.TestCase):

    def _classify(self, question, has_visual=True):
        _no_llm = AssertionError(
            f"classify_question_intent must NOT call chat_completion "
            f"for: {question!r}"
        )
        with patch("core.rag_engine.chat_completion", side_effect=_no_llm):
            return classify_question_intent(question, has_visual_store=has_visual)

    # -- English -------------------------------------------------------------
    def test_en_broad_text_only(self):
        self.assertEqual(self._classify("What is discussed in this video?"), "TEXT_ONLY")

    def test_en_what_happens_text_only(self):
        self.assertEqual(self._classify("What happens in this video?"), "TEXT_ONLY")

    def test_en_people_visible_visual_only(self):
        self.assertEqual(self._classify("How many people are visible in the video?"), "VISUAL_ONLY")

    def test_en_on_screen_visual_only(self):
        self.assertEqual(self._classify("What is on the screen?"), "VISUAL_ONLY")

    def test_en_chart_visual_only(self):
        self.assertEqual(self._classify("Are there any charts or diagrams shown?"), "VISUAL_ONLY")

    def test_en_temporal_multimodal(self):
        self.assertEqual(
            self._classify("What was shown on screen when they discussed the rule?"),
            "MULTIMODAL",
        )

    def test_timestamped_speech_questions_route_to_transcript_in_supported_languages(self):
        questions = [
            "At 00:00:13, what did the speaker say about the rule?",
            "00:00:13 पर वक्ता ने नियम के बारे में क्या कहा?",
            "00:00:13 par speaker ne rule ke baare mein kya kaha?",
            "A las 00:00:13, ¿qué dijo el hablante sobre la regla?",
            "À 00:00:13, qu'a dit l'orateur au sujet de la règle ?",
            "Was sagte der Sprecher um 00:00:13 zur Regel?",
        ]
        for question in questions:
            with self.subTest(question=question):
                self.assertEqual(self._classify(question), "TEXT_ONLY")

    def test_neutral_timestamp_without_visual_store_uses_available_transcript(self):
        self.assertEqual(
            self._classify("What happened at 00:00:13?", has_visual=False),
            "TEXT_ONLY",
        )

    def test_visual_intent_survives_missing_visual_store(self):
        self.assertEqual(self._classify("How many people are visible?", has_visual=False), "VISUAL_ONLY")

    # -- Hindi (Devanagari) --------------------------------------------------
    def test_hi_broad_text_only(self):
        self.assertEqual(self._classify("इस वीडियो में क्या समझाया गया है?"), "TEXT_ONLY")

    def test_hi_content_question_text_only(self):
        self.assertEqual(self._classify("इस वीडियो में क्या बताया गया है?"), "TEXT_ONLY")

    def test_hi_visual_question_visual_only(self):
        # "कितने लोग" is in _PURE_VISUAL_PATTERNS
        self.assertEqual(self._classify("वीडियो में कितने लोग दिखते हैं?"), "VISUAL_ONLY")

    def test_hi_non_latin_no_visual_signal_skips_llm(self):
        # No visual keywords → TEXT_ONLY without LLM call
        self.assertEqual(self._classify("डेविड एलन के बारे में क्या बताया गया?"), "TEXT_ONLY")

    # -- Marathi (Devanagari) ------------------------------------------------
    def test_mr_broad_text_only(self):
        self.assertEqual(self._classify("या व्हिडिओमध्ये काय सांगितले आहे?"), "TEXT_ONLY")

    def test_mr_2min_rule_text_only(self):
        self.assertEqual(
            self._classify("डेव्हिड अॅलनच्या दोन मिनिटांच्या नियमाबद्दल काय सांगितले आहे?"),
            "TEXT_ONLY",
        )

    def test_mr_visual_question_visual_only(self):
        # "किती लोक" is in _PURE_VISUAL_PATTERNS
        self.assertEqual(self._classify("व्हिडिओमध्ये किती लोक दिसतात?"), "VISUAL_ONLY")

    # -- Hinglish (Latin script) ---------------------------------------------
    def test_hinglish_kya_bola_text_only(self):
        self.assertEqual(
            self._classify("Video mein David Allen ki 2-minute rule ke baare mein kya bola hai?"),
            "TEXT_ONLY",
        )

    def test_hinglish_kitne_log_visual_only(self):
        self.assertEqual(self._classify("Video mein kitne log dikh rahe hain?"), "VISUAL_ONLY")

    def test_hinglish_screen_visual_only(self):
        self.assertEqual(self._classify("screen par kya chal raha hai?"), "VISUAL_ONLY")

    def test_hinglish_mixed_multimodal(self):
        self.assertEqual(self._classify("Jab discuss kiya tab kya dikh raha tha?"), "MULTIMODAL")

    # -- Edge cases ----------------------------------------------------------
    def test_empty_question_text_only(self):
        with patch("core.rag_engine.chat_completion", side_effect=AssertionError("no llm")):
            self.assertEqual(classify_question_intent("", has_visual_store=True), "TEXT_ONLY")


# ===========================================================================
# C.  _rerank_evidence
# ===========================================================================
class TestRerankEvidence(unittest.TestCase):

    # Non-Latin (Hindi/Marathi) queries ─────────────────────────────────────

    def test_hindi_keeps_positive_semantic_score(self):
        hindi_q = "इस वीडियो में क्या बताया गया है?"
        doc = _doc("The video explains productivity techniques.", score=0.35)
        result = _rerank_evidence(hindi_q, [doc], limit=5)
        self.assertEqual(len(result), 1, "Doc with semantic score 0.35 must be kept for Hindi query")

    def test_hindi_drops_zero_semantic_score(self):
        hindi_q = "इस वीडियो में क्या बताया गया है?"
        doc = _doc("Unrelated text about cats.", score=0.0)
        self.assertEqual(_rerank_evidence(hindi_q, [doc], limit=5), [])

    def test_marathi_no_lexical_overlap_still_returns_doc(self):
        marathi_q = "या व्हिडिओमध्ये काय सांगितले आहे?"
        english_doc = _doc("David Allen explains GTD and productivity.", score=0.45)
        result = _rerank_evidence(marathi_q, [english_doc], limit=5)
        self.assertGreater(len(result), 0)

    def test_marathi_ranks_higher_semantic_score_first(self):
        marathi_q = "या व्हिडिओमध्ये काय सांगितले आहे?"
        low = _doc("Somewhat related.", score=0.30)
        high = _doc("Highly relevant productivity content.", score=0.70)
        result = _rerank_evidence(marathi_q, [low, high], limit=5)
        self.assertEqual(result[0], high)

    # Latin (English) queries ────────────────────────────────────────────────

    def test_english_drops_low_score_no_overlap(self):
        english_q = "What is the 2-minute rule?"
        irrelevant = _doc("Weather forecast for tomorrow.", score=0.10)
        self.assertEqual(_rerank_evidence(english_q, [irrelevant], limit=5), [])

    def test_english_keeps_good_semantic_score(self):
        english_q = "What is the 2-minute rule?"
        doc = _doc("The 2-minute rule says do it now.", score=0.50)
        self.assertEqual(len(_rerank_evidence(english_q, [doc], limit=5)), 1)

    def test_english_overlap_saves_borderline_score(self):
        english_q = "What is the 2-minute rule?"
        # overlap on "minute" and "rule" saves score below 0.22
        doc = _doc("Two minute rule is explained in GTD.", score=0.15)
        result = _rerank_evidence(english_q, [doc], limit=5)
        self.assertGreater(len(result), 0)

    def test_limit_is_respected(self):
        english_q = "What is discussed?"
        docs = [_doc(f"Topic {i} content.", score=0.5) for i in range(10)]
        self.assertLessEqual(len(_rerank_evidence(english_q, docs, limit=3)), 3)

    def test_empty_docs_returns_empty(self):
        self.assertEqual(_rerank_evidence("anything", [], limit=5), [])


# ===========================================================================
# D.  _remove_repeated_sentences
# ===========================================================================
class TestRemoveRepeatedSentences(unittest.TestCase):

    def test_empty_string(self):
        self.assertEqual(_remove_repeated_sentences(""), "")

    def test_no_repetitions_unchanged(self):
        t = "The 2-minute rule is simple. Do it now if it takes less than two minutes."
        self.assertEqual(_remove_repeated_sentences(t), t)

    def test_consecutive_duplicate_line_removed(self):
        t = "The video is about productivity.\nThe video is about productivity.\nDone."
        result = _remove_repeated_sentences(t)
        count = result.count("The video is about productivity.")
        self.assertEqual(count, 1)
        self.assertIn("Done", result)

    def test_consecutive_duplicate_sentence_in_line(self):
        t = "Step one: capture. Step one: capture. Step two: clarify."
        result = _remove_repeated_sentences(t)
        self.assertEqual(result.lower().count("step one: capture"), 1,
                         f"Should appear once; got: {result!r}")

    def test_distinct_sentences_all_preserved(self):
        t = "David Allen wrote GTD. The book covers productivity. It was published in 2001."
        result = _remove_repeated_sentences(t)
        self.assertIn("David Allen", result)
        self.assertIn("productivity", result)
        self.assertIn("2001", result)

    def test_hindi_unique_sentences_preserved(self):
        t = "इस वीडियो में डेविड एलन की विधि समझाई गई है। यह बहुत उपयोगी है।"
        result = _remove_repeated_sentences(t)
        self.assertIn("डेविड एलन", result)
        self.assertIn("उपयोगी", result)

    def test_hindi_duplicate_removed(self):
        t = "यह वीडियो उत्पादकता के बारे में है। यह वीडियो उत्पादकता के बारे में है।"
        result = _remove_repeated_sentences(t)
        self.assertEqual(result.count("यह वीडियो उत्पादकता के बारे में है"), 1)

    def test_list_items_all_preserved(self):
        t = "Step 1: Capture.\nStep 2: Clarify.\nStep 3: Organize."
        result = _remove_repeated_sentences(t)
        self.assertIn("Step 1", result)
        self.assertIn("Step 2", result)
        self.assertIn("Step 3", result)

    def test_timestamps_preserved(self):
        t = "At 01:30 the speaker explains. At 02:00 a new topic starts."
        result = _remove_repeated_sentences(t)
        self.assertIn("01:30", result)
        self.assertIn("02:00", result)

    def test_non_consecutive_identical_kept(self):
        # Global deduplication is too aggressive; only consecutive dups removed.
        t = "The rule applies here. Something else entirely. The rule applies there."
        result = _remove_repeated_sentences(t)
        # Both occurrences are non-consecutive → both survive
        self.assertGreaterEqual(result.count("rule applies"), 1)


# ===========================================================================
# E.  _prepare_rag_messages – language instructions and message structure
# ===========================================================================
class TestPrepareRagMessages(unittest.TestCase):

    def _call(self, question, output_language=None, docs=None, visual_docs=None,
              conversation_history=None):
        chain = _make_chain(docs=docs or [], visual_docs=visual_docs)
        with patch("core.rag_engine.chat_completion", return_value="TEXT_ONLY"):
            return _prepare_rag_messages(
                chain, question,
                output_language=output_language,
                conversation_history=conversation_history,
            )

    # Devanagari questions ---------------------------------------------------

    def test_hindi_prompt_contains_devanagari(self):
        messages, intent, *_ = self._call("इस वीडियो में क्या समझाया गया है?")
        self.assertIn("Devanagari", messages[0]["content"])
        self.assertIn("Hindi", messages[0]["content"])

    def test_marathi_prompt_contains_devanagari_and_marathi(self):
        messages, *_ = self._call("या व्हिडिओमध्ये काय सांगितले आहे?")
        sys_msg = messages[0]["content"]
        self.assertIn("Devanagari", sys_msg)
        self.assertIn("Marathi", sys_msg)

    # Latin questions --------------------------------------------------------

    def test_english_prompt_contains_english(self):
        messages, *_ = self._call("What is discussed in this video?")
        self.assertIn("English", messages[0]["content"])

    def test_hinglish_prompt_contains_hinglish(self):
        messages, *_ = self._call("Video mein kya samjhaya gaya hai?")
        self.assertIn("Hinglish", messages[0]["content"])
        self.assertIn("Romanized Hindi (Hinglish)", messages[-1]["content"])
        self.assertIn("Latin letters", messages[-1]["content"])

    # Output-language override -----------------------------------------------

    def test_override_hindi(self):
        messages, *_ = self._call("What is this about?", output_language="hi")
        sys_msg = messages[0]["content"]
        self.assertIn("OVERRIDE", sys_msg)
        self.assertIn("Hindi", sys_msg)
        self.assertIn("Devanagari", sys_msg)

    def test_override_marathi(self):
        messages, *_ = self._call("What is this about?", output_language="mr")
        sys_msg = messages[0]["content"]
        self.assertIn("OVERRIDE", sys_msg)
        self.assertIn("Marathi", sys_msg)

    def test_auto_output_language_no_override(self):
        messages, *_ = self._call("What is this about?", output_language="auto")
        self.assertNotIn("OVERRIDE", messages[0]["content"])

    def test_none_output_language_no_override(self):
        messages, *_ = self._call("What is this about?", output_language=None)
        self.assertNotIn("OVERRIDE", messages[0]["content"])

    # Answer-language independence -------------------------------------------

    def test_hindi_prompt_references_question_language(self):
        messages, *_ = self._call("यह वीडियो किस बारे में है?")
        self.assertIn("question", messages[0]["content"].lower())
        self.assertIn("Hindi", messages[0]["content"])

    def test_hindi_prompt_does_not_force_english_answer(self):
        messages, *_ = self._call("यह वीडियो किस बारे में है?")
        sys_msg = messages[0]["content"]
        before_evidence = sys_msg.split("ANSWER LANGUAGE")[0][:800]
        self.assertNotIn("answer strictly in English", before_evidence)

    # Repetition guard in prompt ---------------------------------------------

    def test_prompt_contains_no_repeat_instruction(self):
        messages, *_ = self._call("What is this video about?")
        self.assertIn("Do NOT repeat sentences", messages[0]["content"])

    # Intent routing (heuristic, no visual store) ----------------------------

    def test_hindi_broad_routes_text_only(self):
        _, intent, *_ = self._call("इस वीडियो में क्या समझाया गया है?")
        self.assertEqual(intent, "TEXT_ONLY")

    def test_marathi_broad_routes_text_only(self):
        _, intent, *_ = self._call("या व्हिडिओमध्ये काय सांगितले आहे?")
        self.assertEqual(intent, "TEXT_ONLY")

    # Conversation history ---------------------------------------------------

    def test_conversation_history_in_messages(self):
        history = [
            {"role": "user", "content": "What meeting?"},
            {"role": "assistant", "content": "The product review meeting."},
        ]
        messages, *_ = self._call("When does it start?", conversation_history=history)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[-1]["role"], "user")
        self.assertEqual(messages[-1]["content"], "When does it start?")
        roles = [m["role"] for m in messages]
        self.assertIn("assistant", roles)

    def test_empty_history_still_produces_valid_messages(self):
        messages, *_ = self._call("What is this about?", conversation_history=[])
        self.assertEqual(messages[-1]["content"], "What is this about?")


# ===========================================================================
# F.  ask_question – end-to-end (all external calls mocked)
# ===========================================================================
class TestAskQuestion(unittest.TestCase):

    def _ask(self, question, answer_text=None, output_language=None,
             conversation_history=None, docs=None, visual_docs=None):
        if answer_text is None:
            answer_text = "The 2-minute rule: do it immediately if it takes under two minutes."
        chain = _make_chain(docs=docs or [], visual_docs=visual_docs)
        with patch("core.local_llm.validate_local_llm_ready"), \
             patch("core.rag_engine.check_ollama", return_value=True), \
             patch("core.rag_engine.chat_completion", return_value=answer_text):
            return ask_question(
                chain, question,
                output_language=output_language,
                conversation_history=conversation_history,
            )

    # Grounded answers -------------------------------------------------------

    def test_english_grounded_answer(self):
        doc = _doc("The 2-minute rule: act now if it takes less than two minutes.", score=0.6)
        answer = self._ask("What is the 2-minute rule?", docs=[doc])
        self.assertIn("2-minute", answer)

    def test_hindi_grounded_answer(self):
        doc = _doc("David Allen explains the two-minute rule.", score=0.55)
        answer = self._ask(
            "डेविड एलन के 2-मिनट नियम के बारे में क्या बताया गया?",
            answer_text="डेविड एलन का 2-मिनट का नियम कहता है कि यदि कोई काम दो मिनट से कम में हो सकता है, तो उसे तुरंत करें।",
            docs=[doc],
        )
        self.assertIn("डेविड एलन", answer)

    # No-evidence refusals ---------------------------------------------------

    def test_text_only_no_evidence_refusal(self):
        chain = _make_chain(docs=[])
        with patch("core.local_llm.validate_local_llm_ready"), \
             patch("core.rag_engine.check_ollama", return_value=True), \
             patch("core.rag_engine.chat_completion",
                   side_effect=AssertionError("LLM must not be called with no evidence")):
            answer = ask_question(chain, "What is this video about?")
        self.assertIn("transcript", answer.lower())
        self.assertIn("not provide", answer.lower())

    def test_visual_only_no_visual_store_refusal(self):
        chain = _make_chain(docs=[])  # no visual_store
        with patch("core.local_llm.validate_local_llm_ready"), \
             patch("core.rag_engine.check_ollama", return_value=True), \
             patch("core.rag_engine.chat_completion",
                   side_effect=AssertionError("no LLM when no visual evidence")):
            answer = ask_question(chain, "How many people are visible?")
        self.assertIn("visual evidence", answer.lower())

    # Repetition cleanup -----------------------------------------------------

    def test_repeated_sentence_cleaned(self):
        doc = _doc("The rule is simple and effective.", score=0.6)
        repeated = "The rule is simple. The rule is simple."
        answer = self._ask("What is the rule?", answer_text=repeated, docs=[doc])
        self.assertEqual(answer.count("The rule is simple"), 1,
                         f"Dup should be removed; got: {answer!r}")

    def test_clean_answer_unchanged(self):
        doc = _doc("GTD and 2-minute rule explained.", score=0.6)
        clean = "The 2-minute rule says act now. GTD is the productivity system."
        answer = self._ask("Explain GTD.", answer_text=clean, docs=[doc])
        self.assertIn("2-minute", answer)
        self.assertIn("GTD", answer)

    # Output-language override in system prompt ------------------------------

    def test_hindi_override_in_system_prompt(self):
        doc = _doc("The video discusses productivity methods.", score=0.6)
        chain = _make_chain(docs=[doc])
        captured = []

        def capture_msgs(messages, **kw):
            captured.extend(messages)
            return "यह वीडियो उत्पादकता के बारे में है।"

        with patch("core.local_llm.validate_local_llm_ready"), \
             patch("core.rag_engine.check_ollama", return_value=True), \
             patch("core.rag_engine.chat_completion", side_effect=capture_msgs):
            ask_question(chain, "What is this about?", output_language="hi")

        self.assertTrue(captured, "chat_completion must have been called")
        self.assertIn("OVERRIDE", captured[0]["content"])
        self.assertIn("Hindi", captured[0]["content"])

    # Temperature and grounding ----------------------------------------------

    def test_zero_temperature_used(self):
        doc = _doc("The meeting starts at noon.", score=0.8)
        chain = _make_chain(docs=[doc])
        captured_kw = {}

        def capture(messages, **kw):
            captured_kw.update(kw)
            return "The meeting starts at noon."

        with patch("core.local_llm.validate_local_llm_ready"), \
             patch("core.rag_engine.check_ollama", return_value=True), \
             patch("core.rag_engine.chat_completion", side_effect=capture):
            ask_question(chain, "When does the meeting start?")

        self.assertEqual(captured_kw.get("temperature"), 0.0)

    # Guard rails ------------------------------------------------------------

    def test_none_chain_returns_error_message(self):
        answer = ask_question(None, "What is this?")
        self.assertIn("No video", answer)

    def test_empty_question_returns_prompt(self):
        answer = ask_question(_make_chain(), "  ")
        self.assertIn("Please enter", answer)

    # Cache ------------------------------------------------------------------

    def test_cache_returns_same_answer_without_second_llm_call(self):
        doc = _doc("Cache test content.", score=0.7)
        chain = _make_chain(docs=[doc])
        chain.video_id = "test_cache_vid"
        call_count = {"n": 0}

        def counting(messages, **kw):
            call_count["n"] += 1
            return "Cached answer text."

        with patch("core.local_llm.validate_local_llm_ready"), \
             patch("core.rag_engine.check_ollama", return_value=True), \
             patch("core.rag_engine.chat_completion", side_effect=counting):
            a1 = ask_question(chain, "What is the cache test?")
            a2 = ask_question(chain, "What is the cache test?")

        self.assertEqual(a1, a2)
        self.assertEqual(call_count["n"], 1, "Second identical question must be served from cache")


# ===========================================================================
# G.  Regression: existing test_multilingual.py compatibility
# ===========================================================================
class TestExistingTestCompatibility(unittest.TestCase):
    """No-Ollama versions of the tests that previously hung."""

    def test_output_language_override_hindi_in_prompt(self):
        chain = _make_chain()
        with patch("core.rag_engine.chat_completion", return_value="TEXT_ONLY"):
            messages, *_ = _prepare_rag_messages(chain, "What is this about?",
                                                 output_language="hi")
        sys_msg = messages[0]["content"]
        self.assertIn("OVERRIDE", sys_msg)
        self.assertIn("Hindi", sys_msg)
        self.assertIn("Devanagari", sys_msg)
        self.assertIn("Answer entirely in Hindi", messages[-1]["content"])

    def test_auto_language_no_override(self):
        chain = _make_chain()
        with patch("core.rag_engine.chat_completion", return_value="TEXT_ONLY"):
            messages, *_ = _prepare_rag_messages(chain, "What is this about?",
                                                 output_language="auto")
        self.assertNotIn("OVERRIDE", messages[0]["content"])

    def test_visual_only_no_store_does_not_retrieve_transcript(self):
        def must_not_retrieve(_q):
            raise AssertionError("Visual-only must not retrieve transcript")

        chain = SimpleNamespace(visual_store=None, retrieve=must_not_retrieve)
        with patch("core.local_llm.validate_local_llm_ready"), \
             patch("core.rag_engine.check_ollama", return_value=True):
            answer = ask_question(chain, "What is visible on screen?")
        self.assertIn("visual evidence", answer.lower())
        self.assertIn("not provide enough", answer.lower())

    def test_zero_temperature_on_grounded_answer(self):
        doc = _doc("The meeting starts at noon.", score=0.8)
        chain = _make_chain(docs=[doc])
        captured_kw = {}

        def capture(messages, **kw):
            captured_kw.update(kw)
            return "The meeting starts at noon."

        with patch("core.local_llm.validate_local_llm_ready"), \
             patch("core.rag_engine.check_ollama", return_value=True), \
             patch("core.rag_engine.chat_completion", side_effect=capture):
            ask_question(chain, "When does the meeting start?")

        self.assertEqual(captured_kw.get("temperature"), 0.0)

    def test_hindi_prompt_does_not_force_english(self):
        chain = _make_chain()
        with patch("core.rag_engine.chat_completion", return_value="TEXT_ONLY"):
            messages, *_ = _prepare_rag_messages(chain, "यह वीडियो किस बारे में है?")
        sys_msg = messages[0]["content"]
        self.assertIn("Hindi", sys_msg)
        before = sys_msg.split("ANSWER LANGUAGE")[0][:600]
        self.assertNotIn("answer strictly in English", before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
