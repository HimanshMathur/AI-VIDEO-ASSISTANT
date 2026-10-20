"""Focused tests for local cross-frame person association."""
import os
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _stub_unavailable_model_stack_dependencies():
    """Keep focused RAG tests runnable without importing torch/model packages."""
    try:
        import core.vector_store  # noqa: F401
    except ModuleNotFoundError as exc:
        if exc.name not in {"torch", "faiss", "sentence_transformers", "langchain_text_splitters"}:
            raise
        module = types.ModuleType("core.vector_store")

        class _Document:
            def __init__(self, page_content, metadata):
                self.page_content = page_content
                self.metadata = metadata

        class _VectorStore:
            pass

        module.FaissDocument = _Document
        module.FaissVectorStore = _VectorStore
        module.build_vector_store = lambda *args, **kwargs: None
        module.build_visual_vector_store = lambda *args, **kwargs: None
        sys.modules["core.vector_store"] = module

    try:
        import core.local_llm  # noqa: F401
    except ModuleNotFoundError as exc:
        if exc.name not in {"requests", "urllib3"}:
            raise
        module = types.ModuleType("core.local_llm")
        module.chat_completion = lambda *args, **kwargs: ""
        module.chat_completion_stream = lambda *args, **kwargs: iter(())
        module.json_chat_completion = lambda *args, **kwargs: None
        module.check_ollama = lambda: False
        module.OLLAMA_MODEL = "test-model"
        sys.modules["core.local_llm"] = module


_stub_unavailable_model_stack_dependencies()

from core.rag_engine import (  # noqa: E402
    FaissDocument,
    LocalRAGChain,
    _associate_people_across_frames,
    ask_question,
    _question_timestamp,
    _prepare_rag_messages,
    classify_question_intent,
    format_context,
)
from core.visual_analyzer import parse_vision_structured_output  # noqa: E402


def _frame(timestamp, people, count):
    return {"timestamp": timestamp, "timestamp_seconds": int(timestamp), "people": people, "people_count": count}


class PeopleCountParsingTests(unittest.TestCase):
    def test_count_parses_numeric_person_description(self):
        for description in (
            "1 (a person wearing a black t-shirt).",
            "1 (a man wearing a black t-shirt).",
            "1 (a woman wearing a blue top).",
        ):
            with self.subTest(description=description):
                parsed = parse_vision_structured_output(
                    f"PEOPLE: {description}\nSCENE: Indoor studio.",
                    "00:00:40",
                )
                self.assertEqual(parsed["people_count"], 1)

    def test_count_parses_multiple_people(self):
        parsed = parse_vision_structured_output(
            "PEOPLE: 2 people are visible.\nSCENE: Studio.",
            "00:00:40",
        )
        self.assertEqual(parsed["people_count"], 2)


class PeopleIdentityTests(unittest.TestCase):
    def _associate(self, result, frames):
        with patch("core.rag_engine.json_chat_completion", return_value=result) as model:
            summary = _associate_people_across_frames(frames)
        model.assert_called_once()
        return summary

    def test_same_person_reappearing_is_counted_once(self):
        summary = self._associate({
            "distinct_count": 1, "uncertain": False,
            "observations": [
                {"timestamp": "00:00:01", "person_id": "Person 1", "evidence": "same red shirt and glasses"},
                {"timestamp": "00:00:08", "person_id": "Person 1", "evidence": "same red shirt and glasses"},
            ], "reason": "distinctive appearance repeats",
        }, [_frame("1", "one person in a red shirt and glasses", 1), _frame("8", "one person in a red shirt and glasses", 1)])
        self.assertIn("1 distinct people", summary)

    def test_two_people_in_one_frame_keep_separate_ids(self):
        summary = self._associate({
            "distinct_count": 2, "uncertain": False,
            "observations": [
                {"timestamp": "00:00:01", "person_id": "Person 1", "evidence": "blue shirt"},
                {"timestamp": "00:00:01", "person_id": "Person 2", "evidence": "yellow shirt"},
            ], "reason": "distinct clothing",
        }, [_frame("1", "two people: one in blue and one in yellow", 2)])
        self.assertIn("2 distinct people", summary)

    def test_reappearances_across_combinations_reuse_ids(self):
        summary = self._associate({
            "distinct_count": 2, "uncertain": False,
            "observations": [
                {"timestamp": "00:00:01", "person_id": "Person 1", "evidence": "blue shirt"},
                {"timestamp": "00:00:01", "person_id": "Person 2", "evidence": "yellow shirt"},
                {"timestamp": "00:00:07", "person_id": "Person 2", "evidence": "yellow shirt"},
                {"timestamp": "00:00:12", "person_id": "Person 1", "evidence": "blue shirt"},
            ], "reason": "distinct clothing repeats",
        }, [_frame("1", "blue shirt and yellow shirt", 2), _frame("7", "yellow shirt", 1), _frame("12", "blue shirt", 1)])
        self.assertIn("2 distinct people", summary)

    def test_full_video_frame_association_uses_sufficient_json_output_budget(self):
        frames = [
            _frame(str(i), f"one person in a red shirt at frame {i}", 1)
            for i in range(20)
        ]
        with patch("core.rag_engine.json_chat_completion", return_value={
            "distinct_count": 1, "uncertain": False,
            "observations": [
                {"timestamp": str(i), "person_id": "Person 1", "evidence": "red shirt"}
                for i in range(20)
            ],
            "reason": "consistent appearance",
        }) as model:
            summary = _associate_people_across_frames(frames)
        self.assertIn("about 1 distinct people", summary)
        self.assertEqual(model.call_args.kwargs["max_tokens"], 1500)

    def test_ambiguous_descriptions_do_not_assert_exact_total(self):
        summary = self._associate({
            "distinct_count": None, "uncertain": True, "observations": [],
            "reason": "generic descriptions do not establish identity",
        }, [_frame("1", "two people", 2), _frame("8", "two people", 2)])
        self.assertIn("do not support a reliable video-wide", summary)
        self.assertNotIn("2 distinct people", summary)

    def test_partial_identity_labels_do_not_discard_supported_estimate(self):
        summary = self._associate({
            "distinct_count": 2, "uncertain": False,
            "observations": [
                {"timestamp": "00:00:01", "person_id": "Person 1", "evidence": "one of two people"},
            ], "reason": "two people are simultaneously visible in one frame",
        }, [_frame("1", "two people are visible", 2)])
        self.assertIn("about 2 distinct people", summary)


class RAGRegressionTests(unittest.TestCase):
    def test_general_content_questions_use_transcript_route_and_mixed_questions_use_both(self):
        self.assertEqual(classify_question_intent("Tell me what happens in the video."), "TEXT_ONLY")
        self.assertEqual(classify_question_intent("How can we learn English after watching the video?"), "TEXT_ONLY")
        self.assertEqual(classify_question_intent("What is being discussed and what is visible?"), "MULTIMODAL")
        self.assertEqual(classify_question_intent("What is visible at 5:00?"), "VISUAL_ONLY")
        self.assertEqual(classify_question_intent("What is happening visually at 5:06?"), "VISUAL_ONLY")
        self.assertEqual(classify_question_intent("What color is the kettle at 5:06?"), "VISUAL_ONLY")

    def test_timestamp_visual_attribute_uses_visual_only_route(self):
        self.assertEqual(
            classify_question_intent("What color shirt is the person wearing at 0:58?"),
            "VISUAL_ONLY",
        )
        self.assertEqual(
            classify_question_intent("At 5:00, what example is discussed and what is visible?"),
            "MULTIMODAL",
        )

    def _chain(self, transcript_docs=None, visual_docs=None):
        transcript_docs = transcript_docs or []
        visual_docs = visual_docs or []
        visual_store = SimpleNamespace(index=SimpleNamespace(ntotal=1), chunks=[])
        chain = MagicMock()
        chain.visual_store = visual_store
        chain.vector_store = SimpleNamespace(chunks=[])
        chain.video_id = None
        chain._answer_cache = {}
        chain.top_k = 4
        chain.duration_seconds = 300
        chain.retrieve.return_value = transcript_docs
        chain.retrieve_visual.return_value = visual_docs
        chain.retrieve_visual_by_timestamp.return_value = visual_docs
        for document in transcript_docs + visual_docs:
            document.metadata.setdefault("score", 0.9)
        return chain

    def _ask_people_count_with_mocked_association(self, chunks, association, answer):
        chain = self._chain()
        chain.visual_store.index.ntotal = len(chunks)
        chain.visual_store.chunks = chunks
        with patch("core.rag_engine.json_chat_completion", return_value=association) as associate, \
                patch("core.rag_engine.chat_completion", return_value=answer) as generate:
            actual = ask_question(chain, "how many persons are in the video")
        associate.assert_called_once()
        generate.assert_called_once()
        self.assertNotIn("visual evidence does not provide enough information", actual.casefold())
        self.assertIn("CROSS-FRAME PERSON IDENTITY ANALYSIS", generate.call_args.args[0][0]["content"])
        return actual

    def test_timestamp_question_without_visual_store_uses_available_transcript(self):
        transcript = FaissDocument("The speaker says the next step is to turn left.", {
            "chunk_id": "speech-1", "start_seconds": 122.0, "end_seconds": 130.0,
        })
        chain = self._chain(transcript_docs=[transcript])
        chain.visual_store = None
        with patch("core.rag_engine.chat_completion", return_value="The speaker says to turn left.") as model:
            answer = ask_question(chain, "what happen at 2:06 sec in the video ??")
        self.assertIn("turn left", answer)
        self.assertIn("speaker says", model.call_args.args[0][0]["content"])
        chain.retrieve.assert_called_once()
        chain.retrieve_visual_by_timestamp.assert_not_called()

    def test_timestamp_question_with_missing_visual_index_uses_transcript(self):
        transcript = FaissDocument("The speaker describes a turn at the intersection.", {
            "chunk_id": "speech-1", "start_seconds": 122.0, "end_seconds": 130.0,
        })
        chain = self._chain(transcript_docs=[transcript])
        chain.visual_store.index = None
        with patch("core.rag_engine.chat_completion", return_value="The speaker describes a turn.") as model:
            answer = ask_question(chain, "What happens at 2:06 in the video?")
        self.assertIn("turn", answer)
        self.assertIn("speaker describes", model.call_args.args[0][0]["content"])

    def test_empty_visual_index_is_handled_without_crashing(self):
        transcript = FaissDocument("The speaker describes a turn at the intersection.", {
            "chunk_id": "speech-1", "start_seconds": 122.0, "end_seconds": 130.0,
        })
        chain = self._chain(transcript_docs=[transcript])
        chain.visual_store.index.ntotal = 0
        with patch("core.rag_engine.chat_completion", return_value="The speaker describes a turn.") as model:
            answer = ask_question(chain, "What happens at 2:06 in the video?")
        self.assertIn("turn", answer)
        self.assertIn("speaker describes", model.call_args.args[0][0]["content"])

    def test_general_visual_question_uses_visual_evidence_with_no_transcript(self):
        visual = FaissDocument("Scene: a person waves beside a window.", {"timestamp": "00:00:05"})
        chain = self._chain(visual_docs=[visual])
        with patch("core.rag_engine.chat_completion", return_value="A person waves beside a window.") as model:
            answer = ask_question(chain, "What is happening on screen?")
        self.assertIn("waves", answer)
        self.assertIn("Scene: a person waves", model.call_args.args[0][0]["content"])

    def test_insufficient_answer_retries_with_full_retrieved_visual_description(self):
        visual = FaissDocument(
            "Scene: " + ("x" * 520) + "\nPeople: 1 (a man wearing a black t-shirt).",
            {"timestamp": "00:00:40"},
        )
        chain = self._chain(visual_docs=[visual])
        with patch(
            "core.rag_engine.chat_completion",
            side_effect=[
                "The video does not provide enough visual evidence to determine that.",
                "The man is wearing a black t-shirt.",
            ],
        ) as generate:
            answer = ask_question(chain, "What color shirt is the man wearing at 0:58?")
        self.assertEqual(answer, "The man is wearing a black t-shirt.")
        self.assertEqual(generate.call_count, 2)
        self.assertIn("People: 1 (a man wearing a black t-shirt)", generate.call_args_list[1].args[0][0]["content"])

    def test_broad_timestamp_visual_retry_describes_active_frame(self):
        visual = FaissDocument(
            "Scene: Indoor studio.\nPeople: 1 woman with gray hair.\n"
            "Actions: Holding a microphone.\nObjects: White kettle.",
            {
                "timestamp": "00:04:49", "timestamp_display": "00:04:49 to 00:05:13",
                "timestamp_seconds": 289.0, "end_seconds": 313.0,
                "start_time": 289.0, "end_time": 313.0,
            },
        )
        chain = self._chain(visual_docs=[visual])
        chain.duration_seconds = 412.85
        chain.retrieve_visual_by_timestamp.return_value = [visual]
        with patch(
            "core.rag_engine.chat_completion",
            side_effect=[
                "The video does not provide enough visual evidence to determine that.",
                "A woman is holding a microphone in an indoor studio.",
            ],
        ) as generate:
            answer = ask_question(chain, "What is visible at 5:00?")
        self.assertEqual(answer, "A woman is holding a microphone in an indoor studio.")
        self.assertEqual(generate.call_count, 2)
        retry_prompt = generate.call_args_list[1].args[0][0]["content"]
        self.assertIn("When the user broadly asks what is visible or happening", retry_prompt)
        self.assertIn("active from 04:49 to 05:13", retry_prompt)
        self.assertIn("Actions: Holding a microphone", retry_prompt)
        self.assertNotIn("TRANSCRIPT EVIDENCE", retry_prompt)

    def test_zero_fifty_eight_timestamp_retrieves_nearest_visual_frame_without_transcript(self):
        self.assertEqual(_question_timestamp("what happen at 0:58 sec in the video"), 58.0)
        visual = FaissDocument("Scene: a car turns at an intersection.", {
            "timestamp": "00:00:57", "timestamp_seconds": 57.0,
        })
        chain = self._chain(visual_docs=[visual])
        with patch("core.rag_engine.chat_completion", return_value="A car turns at an intersection.") as model:
            answer = ask_question(chain, "what happen at 0:58 sec in the video")
        self.assertIn("car turns", answer)
        self.assertIn("Scene: a car turns", model.call_args.args[0][0]["content"])
        chain.retrieve_visual_by_timestamp.assert_called_with(58.0, k=2)

    def test_timestamped_prompt_uses_aligned_frame_without_semantic_distractors(self):
        target = FaissDocument("The speaker wears a black shirt.", {"timestamp": "00:00:40", "chunk_id": "target"})
        unrelated = FaissDocument("A woman in a blue top appears.", {"timestamp": "00:05:13", "chunk_id": "elsewhere"})
        chain = self._chain(visual_docs=[target, unrelated])
        chain.retrieve_visual_by_timestamp.return_value = [target]
        _, _, _, visual_docs, _ = _prepare_rag_messages(chain, "What color shirt is shown at 0:58?")
        self.assertEqual([doc.metadata["timestamp"] for doc in visual_docs], ["00:00:40"])

    def test_visual_context_keeps_structured_fields_after_500_characters(self):
        description = "Scene: " + ("x" * 520) + "\nPeople: 1 (a man wearing a black t-shirt)."
        context = format_context([], [FaissDocument(description, {"timestamp": "00:00:40"})])
        self.assertIn("People: 1 (a man wearing a black t-shirt)", context)

    def test_timestamp_retrieval_prefers_active_frame_over_adjacent_frame(self):
        chain = LocalRAGChain(
            vector_store=SimpleNamespace(),
            visual_store=SimpleNamespace(chunks=[
                {"text": "Previous frame.", "timestamp": "00:00:01", "timestamp_seconds": 1, "start_time": 1, "end_time": 40},
                {"text": "Frame active at 0:58.", "timestamp": "00:00:40", "timestamp_seconds": 40, "start_time": 40, "end_time": 91},
            ]),
        )
        docs = chain.retrieve_visual_by_timestamp(58, k=2)
        self.assertEqual([doc.metadata["timestamp"] for doc in docs], ["00:00:40"])

    def test_timestamp_outside_video_duration_returns_uncertainty_without_retrieval(self):
        chain = self._chain()
        chain.duration_seconds = 300
        with patch("core.rag_engine.chat_completion") as generate:
            answer = ask_question(chain, "What is visible at 5:01?")
        self.assertIn("outside the video's duration", answer)
        chain.retrieve_visual_by_timestamp.assert_not_called()
        chain.retrieve_visual.assert_not_called()
        generate.assert_not_called()

    def test_people_count_full_path_deduplicates_one_person_across_frames(self):
        chunks = [
            {"timestamp": "00:00:01", "timestamp_seconds": 1, "people_count": 1,
             "text": "People: One person wearing a red shirt and glasses."},
            {"timestamp": "00:00:08", "timestamp_seconds": 8, "people_count": 1,
             "text": "People: One person wearing a red shirt and glasses."},
        ]
        result = self._ask_people_count_with_mocked_association(chunks, {
            "distinct_count": 1, "uncertain": False,
            "observations": [
                {"timestamp": "00:00:01", "person_id": "Person 1", "evidence": "red shirt and glasses"},
                {"timestamp": "00:00:08", "person_id": "Person 1", "evidence": "red shirt and glasses"},
            ], "reason": "appearance matches across frames",
        }, "There is one distinct person in the sampled frames.")
        self.assertIn("one distinct person", result)

    def test_people_count_full_path_counts_two_distinct_people_in_one_frame(self):
        chunks = [{
            "timestamp": "00:00:01", "timestamp_seconds": 1, "people_count": 2,
            "text": "People: Two people, one wearing blue and one wearing yellow.",
        }]
        result = self._ask_people_count_with_mocked_association(chunks, {
            "distinct_count": 2, "uncertain": False,
            "observations": [
                {"timestamp": "00:00:01", "person_id": "Person 1", "evidence": "blue clothing"},
                {"timestamp": "00:00:01", "person_id": "Person 2", "evidence": "yellow clothing"},
            ], "reason": "distinct people visible together",
        }, "Two distinct people are visible.")
        self.assertIn("Two distinct people", result)

    def test_people_count_full_path_reuses_both_people_on_reappearance(self):
        chunks = [
            {"timestamp": "00:00:01", "timestamp_seconds": 1, "people_count": 2,
             "text": "People: One person wears blue; another wears yellow."},
            {"timestamp": "00:00:07", "timestamp_seconds": 7, "people_count": 1,
             "text": "People: The person in yellow appears."},
            {"timestamp": "00:00:12", "timestamp_seconds": 12, "people_count": 1,
             "text": "People: The person in blue appears."},
        ]
        result = self._ask_people_count_with_mocked_association(chunks, {
            "distinct_count": 2, "uncertain": False,
            "observations": [
                {"timestamp": "00:00:01", "person_id": "Person 1", "evidence": "blue clothing"},
                {"timestamp": "00:00:01", "person_id": "Person 2", "evidence": "yellow clothing"},
                {"timestamp": "00:00:07", "person_id": "Person 2", "evidence": "yellow clothing"},
                {"timestamp": "00:00:12", "person_id": "Person 1", "evidence": "blue clothing"},
            ], "reason": "both clothing descriptions repeat",
        }, "Two distinct people appear across the sampled frames.")
        self.assertIn("Two distinct people", result)

    def test_timestamp_question_combines_visual_and_transcript_evidence(self):
        visual = FaissDocument("Scene: a car turns at an intersection.", {"timestamp": "00:02:04"})
        transcript = FaissDocument("The driver says they are turning left.", {
            "chunk_id": "speech-1", "start_seconds": 122.0, "end_seconds": 130.0,
        })
        chain = self._chain(transcript_docs=[transcript], visual_docs=[visual])
        with patch("core.rag_engine.chat_completion", return_value="The car turns while the driver says they are turning left.") as model:
            answer = ask_question(chain, "What happens at 2:06 in the video?")
        self.assertIn("car turns", answer)
        generation_context = model.call_args.args[0][0]["content"]
        self.assertIn("driver says", generation_context)
        self.assertIn("car turns", generation_context)
        chain.retrieve_visual_by_timestamp.assert_called_with(126.0, k=2)

    def test_empty_visual_index_and_no_transcript_returns_insufficient_response(self):
        chain = self._chain()
        chain.visual_store.index.ntotal = 0
        chain.retrieve.return_value = []
        with patch("core.rag_engine.chat_completion") as model:
            answer = ask_question(chain, "What happens at 2:06 in the video?")
        self.assertIn("do not provide enough information", answer)
        model.assert_not_called()

    def test_missing_visual_store_and_transcript_returns_honest_uncertainty(self):
        chain = self._chain()
        chain.visual_store = None
        chain.retrieve.return_value = []
        with patch("core.rag_engine.chat_completion") as model:
            answer = ask_question(chain, "how many persons are in the video")
        self.assertIn("not provide enough information", answer.casefold())
        model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
