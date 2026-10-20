"""
Regression tests for the three production fixes:
  Issue 1: YouTube player black screen (dedicated iframe player + stable key)
  Issue 2: Indian-language text corruption (Unicode cleaning + Punjabi RAG prompt)
  Issue 3: Active video disappears after browser refresh (query params + JSON cache)

Run with:
    python -m unittest tests.test_issues -v
or:
    python tests/test_issues.py
"""
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# Ensure project root is importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _install_streamlit_mock():
    """Install a lightweight streamlit mock so app.py can be imported outside
    of a running Streamlit server. Returns the mock session_state dict."""

    class _SessionState(dict):
        """Dict that also supports attribute access, like st.session_state."""
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError:
                raise AttributeError(name)
        def __setattr__(self, name, value):
            self[name] = value
        def __delattr__(self, name):
            try:
                del self[name]
            except KeyError:
                raise AttributeError(name)

    class _QueryParams(dict):
        pass

    session_state = _SessionState()
    query_params = _QueryParams()

    mock_st = MagicMock()
    mock_st.session_state = session_state
    mock_st.query_params = query_params
    # cache_data decorator: return the function unchanged
    mock_st.cache_data = lambda **kw: (lambda f: f)

    # Widgets must return falsy values so the module-level UI code does not
    # take branches that write files or call pipeline logic.
    mock_st.text_input.return_value = ""
    mock_st.file_uploader.return_value = None
    mock_st.radio.return_value = "YouTube URL"
    mock_st.button.return_value = False
    mock_st.chat_input.return_value = None
    # st.columns / st.tabs return sequences that are unpacked at module level.
    mock_st.columns = lambda spec: tuple(MagicMock() for _ in spec) if isinstance(spec, (list, tuple)) else (MagicMock(), MagicMock())
    mock_st.tabs = lambda labels: tuple(MagicMock() for _ in labels)

    # components.v1.html is called during import-time function definitions only
    # (it is invoked at runtime, not at import). MagicMock handles it.

    sys.modules["streamlit"] = mock_st
    sys.modules["streamlit.components"] = MagicMock()
    sys.modules["streamlit.components.v1"] = MagicMock()
    return session_state, query_params


# Install mock BEFORE importing app
_session_state, _query_params = _install_streamlit_mock()

# Import modules under test
from core.transcriber import _clean_segment_text  # noqa: E402
import app  # noqa: E402


class TestYouTubeVideoId(unittest.TestCase):
    """Issue 1: YouTube video ID extraction must handle all URL forms."""

    def test_standard_watch_url(self):
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        self.assertEqual(app._youtube_video_id(url), "dQw4w9WgXcQ")

    def test_youtu_be_short_url(self):
        url = "https://youtu.be/dQw4w9WgXcQ"
        self.assertEqual(app._youtube_video_id(url), "dQw4w9WgXcQ")

    def test_shorts_url(self):
        url = "https://www.youtube.com/shorts/dQw4w9WgXcQ"
        self.assertEqual(app._youtube_video_id(url), "dQw4w9WgXcQ")

    def test_embed_url(self):
        url = "https://www.youtube.com/embed/dQw4w9WgXcQ"
        self.assertEqual(app._youtube_video_id(url), "dQw4w9WgXcQ")

    def test_url_with_extra_query_params(self):
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=10s&ab_channel=Test"
        self.assertEqual(app._youtube_video_id(url), "dQw4w9WgXcQ")

    def test_non_youtube_url_returns_empty(self):
        self.assertEqual(app._youtube_video_id("https://example.com/video"), "")

    def test_invalid_id_length_returns_empty(self):
        # 10 chars instead of 11
        self.assertEqual(app._youtube_video_id("https://www.youtube.com/watch?v=short"), "")


class TestUnicodeCleaning(unittest.TestCase):
    """Issue 2: _clean_segment_text must not corrupt Indic scripts."""

    def test_hindi_nahin_preserved(self):
        # "नहीं" (no) ends in anusvara + vowel sign; must NOT be stripped
        text = "नहीं"
        self.assertEqual(_clean_segment_text(text), text)

    def test_punjabi_main_preserved(self):
        # "ਮੈਂ" (I) ends in bindi/tippi; must NOT be stripped
        text = "ਮੈਂ"
        self.assertEqual(_clean_segment_text(text), text)

    def test_hindi_sentence_preserved(self):
        text = "मैं जा रहा हूं"
        self.assertEqual(_clean_segment_text(text), text)

    def test_punjabi_sentence_preserved(self):
        text = "ਮੈਂ ਪੰਜਾਬੀ ਬੋਲਦਾ ਹਾਂ"
        self.assertEqual(_clean_segment_text(text), text)

    def test_hinglish_mixed_script_preserved(self):
        text = "video mein kya है"
        self.assertEqual(_clean_segment_text(text), text)

    def test_english_trailing_whitespace_stripped(self):
        self.assertEqual(_clean_segment_text("hello world  "), "hello world")

    def test_english_plain_unchanged(self):
        self.assertEqual(_clean_segment_text("the quick brown fox"), "the quick brown fox")

    def test_orphaned_mark_at_start_stripped(self):
        # A combining mark with no preceding base character
        self.assertEqual(_clean_segment_text("\u093e"), "")

    def test_orphaned_mark_after_space_stripped(self):
        self.assertEqual(_clean_segment_text("hello \u093e"), "hello")

    def test_empty_string(self):
        self.assertEqual(_clean_segment_text(""), "")


class TestRagPromptLanguages(unittest.TestCase):
    """Issue 2: RAG system prompt must explicitly list Punjabi/Gurmukhi."""

    def _get_system_prompt(self):
        """Build a minimal RAG chain and extract the system prompt."""
        from core.rag_engine import build_rag_chain, _prepare_rag_messages

        chain = build_rag_chain(
            "This is a test transcript with some content. It has multiple sentences.",
            duration_seconds=100.0,
        )
        messages, _intent, _td, _vd, _ctx = _prepare_rag_messages(chain, "What is this about?")
        return messages[0]["content"]

    def test_prompt_contains_punjabi_clause(self):
        prompt = self._get_system_prompt()
        self.assertIn("Punjabi", prompt)
        self.assertIn("Gurmukhi", prompt)

    def test_prompt_contains_hindi_clause(self):
        prompt = self._get_system_prompt()
        self.assertIn("Hindi", prompt)
        self.assertIn("Devanagari", prompt)

    def test_prompt_contains_hinglish_clause(self):
        prompt = self._get_system_prompt()
        self.assertIn("Hinglish", prompt)

    def test_prompt_contains_english_clause(self):
        prompt = self._get_system_prompt()
        self.assertIn("English", prompt)


class TestVideoStateCache(unittest.TestCase):
    """Issue 3: analysis results must persist to a local JSON cache and reload."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        # Patch CACHE_DIR to a temp directory
        self._orig_cache_dir = app.CACHE_DIR
        app.CACHE_DIR = self.tmpdir

    def tearDown(self):
        app.CACHE_DIR = self._orig_cache_dir
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_save_and_load_round_trip(self):
        video_id = "dQw4w9WgXcQ"
        result = {
            "title": "Test Video",
            "source_name": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "source_type": "YouTube URL",
            "source_path": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "duration_seconds": 120.0,
            "transcript": "नहीं मैं जा रहा हूं",
            "visual_evidence": [],
            "summary": "A test summary",
            "action_items": "Do things",
            "key_decisions": "Decided",
            "open_questions": "Why?",
            "rag_chain": object(),  # non-serializable, must be excluded
            "timing": {"total": 10.0},
        }
        app._save_cached_result(video_id, result)

        loaded = app._load_cached_result(video_id)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["title"], "Test Video")
        self.assertEqual(loaded["transcript"], "नहीं मैं जा रहा हूं")
        self.assertEqual(loaded["duration_seconds"], 120.0)
        # rag_chain must NOT be in the cached file
        self.assertNotIn("rag_chain", loaded)

    def test_load_nonexistent_returns_none(self):
        self.assertIsNone(app._load_cached_result("nonexistent_id_xyz"))

    def test_save_without_video_id_is_noop(self):
        # Should not raise
        app._save_cached_result("", {"transcript": "x"})

    def test_indic_text_round_trips_through_json(self):
        video_id = "indic_test_123"
        punjabi = "ਮੈਂ ਪੰਜਾਬੀ ਬੋਲਦਾ ਹਾਂ"
        hindi = "नहीं मैं जा रहा हूं"
        app._save_cached_result(video_id, {"transcript": hindi + " " + punjabi})
        loaded = app._load_cached_result(video_id)
        self.assertEqual(loaded["transcript"], hindi + " " + punjabi)


class TestCurrentVideoIsolation(unittest.TestCase):
    def test_replacing_video_url_removes_previous_cached_video_id(self):
        old_url = _query_params.get("video_url")
        old_id = _query_params.get("video_id")
        new_url = "https://www.youtube.com/watch?v=9bZkp7q19f0"
        try:
            app._save_video_state("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ")
            app._save_video_state(new_url, "")
            self.assertEqual(_query_params.get("video_url"), new_url)
            self.assertNotIn("video_id", _query_params)
        finally:
            app._save_video_state(old_url or "", old_id or "")

    def test_new_source_resets_video_owned_state_and_progress(self):
        state = {
            "result": {"transcript": "old"},
            "chat_history": [{"role": "user", "content": "old question"}],
            "pending_query": "old pending question",
            "is_processing": True,
            "_analysis_run_id": "stale-run",
            "last_source_id": "YouTube URL::old-video",
            "pipeline_steps": {"transcription": "completed", "visual_ai": "active"},
        }
        app._reset_video_session_state(state, "YouTube URL::new-video")
        self.assertIsNone(state["result"])
        self.assertEqual(state["chat_history"], [])
        self.assertIsNone(state["pending_query"])
        self.assertFalse(state["is_processing"])
        self.assertIsNone(state["_analysis_run_id"])
        self.assertEqual(state["last_source_id"], "YouTube URL::new-video")
        self.assertEqual(set(state["pipeline_steps"].values()), {"pending"})

    def test_refresh_restore_ignores_video_id_from_another_url(self):
        old_restored = _session_state.get("_restored_from_cache")
        old_last_source = _session_state.get("last_source_id")
        old_input_url = _session_state.get("youtube_url_input")
        old_url = _query_params.get("video_url")
        old_id = _query_params.get("video_id")
        new_url = "https://www.youtube.com/watch?v=9bZkp7q19f0"
        try:
            _session_state["_restored_from_cache"] = False
            _query_params["video_url"] = new_url
            _query_params["video_id"] = "dQw4w9WgXcQ"
            with patch.object(app, "_load_cached_result") as load_cache:
                app._restore_session_from_cache()
            load_cache.assert_not_called()
            self.assertNotIn("video_id", _query_params)
            self.assertEqual(
                _session_state["last_source_id"],
                "YouTube URL::9bZkp7q19f0",
            )
        finally:
            if old_restored is None:
                _session_state.pop("_restored_from_cache", None)
            else:
                _session_state["_restored_from_cache"] = old_restored
            if old_last_source is None:
                _session_state.pop("last_source_id", None)
            else:
                _session_state["last_source_id"] = old_last_source
            if old_input_url is None:
                _session_state.pop("youtube_url_input", None)
            else:
                _session_state["youtube_url_input"] = old_input_url
            if old_url is None:
                _query_params.pop("video_url", None)
            else:
                _query_params["video_url"] = old_url
            if old_id is None:
                _query_params.pop("video_id", None)
            else:
                _query_params["video_id"] = old_id

    def test_chapter_answers_keep_each_timestamp_on_its_own_line(self):
        answer = "00:00 Ohio State Penitentiary\n00:32 Gear Up for Haunting"
        rendered = app._chat_answer_markdown(answer)
        self.assertEqual(rendered.splitlines(), [
            "00:00 — Ohio State Penitentiary  ",
            "00:32 — Gear Up for Haunting  ",
        ])

    def test_non_chapter_chat_text_is_unchanged(self):
        answer = "A normal paragraph.\nAnother paragraph."
        self.assertEqual(app._chat_answer_markdown(answer), answer)


class TestNewVisualAndChapterRegressions(unittest.TestCase):
    def test_chapter_records_render_one_entry_per_line(self):
        rendered = app._chat_answer_markdown({"chapters": [
            {"seconds": 0, "title": "Introduction"},
            {"timestamp": "00:52", "title": "Main Topic"},
        ]})
        self.assertEqual(rendered.splitlines(), [
            "00:00 \u2014 Introduction  ", "00:52 \u2014 Main Topic  ",
        ])

    def test_flattened_chapter_string_is_split_and_normalized(self):
        rendered = app._chat_answer_markdown("00:00 ? Intro 00:52 ? Main Topic")
        self.assertEqual(rendered.splitlines(), [
            "00:00 \u2014 Intro  ", "00:52 \u2014 Main Topic  ",
        ])

    def test_visual_retrieval_keeps_video_and_frame_timestamp_metadata(self):
        from core.vector_store import FaissVectorStore, build_visual_vector_store
        from unittest.mock import patch

        class Index:
            ntotal = 1
            def search(self, _query, _count):
                return [[0.91]], [[0]]
        class Embedding:
            def encode(self, _texts, **_kwargs):
                return [[0.1, 0.2]]

        store = object.__new__(FaissVectorStore)
        store.index, store.embedding_model = Index(), Embedding()
        store.chunks = [{
            "chunk_id": "frame-a", "chunk_index": 0, "source": "visual",
            "video_id": "video-a", "frame_index": 3,
            "timestamp": "00:00:32", "timestamp_seconds": 32.0,
            "text": "Visible equipment: a red camera.",
        }]
        document = store.similarity_search("camera", k=1)[0]
        self.assertEqual(document.metadata["video_id"], "video-a")
        self.assertEqual(document.metadata["timestamp_seconds"], 32.0)
        self.assertEqual(document.metadata["frame_index"], 3)

        class Model:
            def get_embedding_dimension(self):
                return 2
        class CapturedStore:
            def __init__(self, **_kwargs):
                self.chunks = []
            def add_chunks(self, chunks):
                self.chunks.extend(chunks)
        with patch("core.vector_store.get_embedding_model", return_value=Model()), \
             patch("core.vector_store.FaissVectorStore", CapturedStore):
            built = build_visual_vector_store([{
                "frame_index": 7, "timestamp": "00:01:12", "timestamp_seconds": 72,
                "summary_text": "A camera is visible.",
            }], video_id="source-b")
        self.assertEqual(built.chunks[0]["video_id"], "source-b")
        self.assertEqual(built.chunks[0]["timestamp_seconds"], 72)

    def test_chain_rejects_visual_documents_from_a_different_video(self):
        from types import SimpleNamespace
        from core.rag_engine import LocalRAGChain

        wrong = SimpleNamespace(page_content="A red car is visible.",
                                metadata={"video_id": "video-a", "timestamp_seconds": 12})
        store = SimpleNamespace(
            index=SimpleNamespace(ntotal=1),
            chunks=[{"video_id": "video-a", "text": "A red car is visible.",
                     "start_time": 12, "end_time": 13}],
            similarity_search=lambda *_args, **_kwargs: [wrong],
        )
        chain = LocalRAGChain(vector_store=SimpleNamespace(), visual_store=store)
        chain.video_id = "video-b"
        self.assertEqual(chain.retrieve_visual("red car"), [])
        self.assertEqual(chain.retrieve_visual_by_timestamp(12), [])

    def test_irrelevant_visual_evidence_is_insufficient(self):
        from types import SimpleNamespace
        from core.rag_engine import ask_question

        visual = SimpleNamespace(index=SimpleNamespace(ntotal=1))
        irrelevant = SimpleNamespace(page_content="A weather report is visible.",
                                     metadata={"score": 0.01, "video_id": "video-a"})
        chain = SimpleNamespace(visual_store=visual, top_k=3,
                                retrieve_visual=lambda _q, k=4: [irrelevant])
        answer = ask_question(chain, "What equipment is visible?")
        self.assertIn("visual evidence", answer.lower())
        self.assertIn("not provide enough", answer.lower())

    def test_timeout_is_not_cached_or_fabricated_and_retry_can_succeed(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from core.rag_engine import ask_question

        evidence = SimpleNamespace(page_content="The speaker describes a blue car.", metadata={"score": .9})
        chain = SimpleNamespace(video_id="video-a", visual_store=None, top_k=4,
                                retrieve=lambda _q: [evidence])
        with patch("core.rag_engine.chat_completion", side_effect=[TimeoutError("timed out"), "A blue car."]):
            timed_out = ask_question(chain, "What color car is discussed?")
            answer = ask_question(chain, "What color car is discussed?")
        self.assertIn("timed out", timed_out.lower())
        self.assertNotIn("blue", timed_out.lower())
        self.assertEqual(answer, "A blue car.")

    def test_duplicate_answer_cache_is_scoped_to_each_video_chain(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from core.rag_engine import ask_question

        def make_chain(video_id, text):
            evidence = SimpleNamespace(page_content=text, metadata={"score": .9})
            return SimpleNamespace(video_id=video_id, visual_store=None, top_k=4,
                                   retrieve=lambda _q: [evidence])
        first = make_chain("video-a", "The transcript mentions a violin.")
        second = make_chain("video-b", "The transcript mentions a trumpet.")
        with patch("core.rag_engine.chat_completion", side_effect=["A violin.", "A violin in context.", "A trumpet."]) as model:
            a1 = ask_question(first, "What instrument is mentioned?")
            a2 = ask_question(first, "What instrument is mentioned?", conversation_history=[
                {"role": "user", "content": "What instrument is mentioned?"},
                {"role": "assistant", "content": "A violin."},
            ])
            a3 = ask_question(first, "What instrument is mentioned?", conversation_history=[
                {"role": "user", "content": "New context that changes the conversation."},
            ])
            b1 = ask_question(second, "What instrument is mentioned?")
        self.assertEqual((a1, a2, a3, b1), (
            "A violin.", "A violin.", "A violin in context.", "A trumpet.",
        ))
        self.assertEqual(model.call_count, 3)


class TestYouTubePlayerFunction(unittest.TestCase):
    """Issue 1: _render_youtube_player must exist and handle missing IDs."""

    def test_function_exists(self):
        self.assertTrue(callable(app._render_youtube_player))

    def test_render_with_empty_id_shows_warning(self):
        # _render_youtube_player calls st.warning when youtube_id is empty
        with patch.object(app.st, "warning") as mock_warning:
            app._render_youtube_player("", source_url="https://example.com")
            mock_warning.assert_called_once()

    def test_render_with_valid_id_calls_components_html(self):
        with patch.object(app.components, "html") as mock_html:
            app._render_youtube_player("dQw4w9WgXcQ", source_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ")
            mock_html.assert_called_once()
            # Verify the rendered HTML targets the correct video ID (embed URL).
            rendered = mock_html.call_args.args[0]
            self.assertIn("dQw4w9WgXcQ", rendered)
            # key= is NOT supported by the installed Streamlit components.html.
            self.assertNotIn("key", mock_html.call_args.kwargs)


class TestChapterPlayer(unittest.TestCase):
    """Issue 1: chapter player renders the correct video and is callable."""

    def test_chapter_player_renders_video_id(self):
        chapters = [{"seconds": 0, "timestamp": "00:00", "title": "Intro"}]
        with patch.object(app.components, "html") as mock_html:
            app._render_chapter_controls(chapters, youtube_id="dQw4w9WgXcQ")
            rendered = mock_html.call_args.args[0]
            # The video ID must appear in the rendered HTML/JS so the player
            # loads the correct video.
            self.assertIn("dQw4w9WgXcQ", rendered)
            # key= is not supported by the installed Streamlit version.
            self.assertNotIn("key", mock_html.call_args.kwargs)


if __name__ == "__main__":
    unittest.main(verbosity=2)
