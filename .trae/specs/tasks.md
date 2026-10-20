# Implementation Tasks

## Task 1: Fix YouTube player rendering (Issue 1)

**Status:** pending
**Priority:** high

**Files:** `app.py`

**Changes:**
1. Add a helper `_render_youtube_player(youtube_id, chapters=None)` that always renders a YouTube embed iframe (via the existing IFrame API pattern) for YouTube videos, instead of falling back to `st.video(youtube_url)`.
2. In the results dashboard, replace the `if is_youtube_video and not show_chapter_player: st.video(preview_path)` branch with a call to render the YouTube iframe player (always, regardless of chapters).
3. Add a stable `key` to `components.html` in `_render_chapter_controls` (and the new standalone player) using the YouTube video ID, so Streamlit does not recreate the iframe on reruns.
4. Add an explicit error/fallback message when `youtube_id` is empty or embedding may be restricted, with a link to open the video on YouTube.

**Test Requirements:**
- `rule`: A YouTube URL with no chapters renders a working iframe player, not a black `st.video`.
- `rule`: The player component has a stable `key` containing the video ID.
- `rule`: Local file uploads still use `st.video`/`st.audio`.
- `rule`: Chapter seeking still works (existing `_render_chapter_controls` path unchanged).

**Completion Evidence:** Run the app, submit a YouTube URL, verify the player renders and plays.

---

## Task 2: Fix Indic-script text corruption in `_clean_segment_text` (Issue 2)

**Status:** pending
**Priority:** high

**Files:** `core/transcriber.py`

**Changes:**
1. In `_clean_segment_text`, replace the unconditional `while text and unicodedata.category(text[-1]) == "Mn": text = text[:-1]` loop with a guard that only strips a trailing combining mark when it is orphaned — i.e., when the preceding character is whitespace or the string has no base character before it.
2. Keep NFC normalization and control-character filtering (these are safe for Indic scripts).

**Test Requirements:**
- `rule`: `_clean_segment_text("नहीं")` returns `"नहीं"` (not `"नह"`).
- `rule`: `_clean_segment_text("ਮੈਂ")` (Punjabi "I am") returns unchanged.
- `rule`: `_clean_segment_text("hello world ")` still strips trailing whitespace.
- `rule`: A truly orphaned trailing mark (e.g. `"abc\u093e"` after a non-letter) is still cleaned.

**Completion Evidence:** Unit tests pass.

---

## Task 3: Add Punjabi/Gurmukhi to RAG answer-language rules (Issue 2)

**Status:** pending
**Priority:** high

**Files:** `core/rag_engine.py`

**Changes:**
1. In `_prepare_rag_messages`, extend the system prompt's answer-language rules to explicitly include Punjabi (Gurmukhi script): "Punjabi question (in Gurmukhi script) → Answer strictly in Punjabi (Gurmukhi script)."
2. Keep English, Hindi, Hinglish, and "other languages" rules intact.

**Test Requirements:**
- `rule`: The system prompt string contains an explicit Punjabi/Gurmukhi answer-language clause.
- `rule`: Hindi and Hinglish clauses remain.
- `rule`: English clause remains unchanged.

**Completion Evidence:** Grep the prompt for Punjabi/Gurmukhi clause; existing tests still pass.

---

## Task 4: Persist active video URL and analysis results across refresh (Issue 3)

**Status:** pending
**Priority:** high

**Files:** `app.py`

**Changes:**
1. Add helper functions:
   - `_get_active_video_from_query()` — read `st.query_params.get("video_url")` and `st.query_params.get("video_id")`.
   - `_save_video_state(url, video_id)` — write URL to `st.query_params`.
   - `_load_cached_result(video_id)` — load analysis results from `downloads/_cache/{video_id}.json` if present.
   - `_save_cached_result(video_id, result)` — persist analysis results to the cache file (excluding non-serializable `rag_chain`).
2. On app start (before the sidebar input), if the URL input is empty and `st.query_params` has a `video_url`, restore it into session state so the input is pre-filled and `last_source_id` matches.
3. After successful analysis, call `_save_video_state` and `_save_cached_result`.
4. On a fresh session where `st.session_state.result` is None but a cached result exists for the restored video ID, load it into session state (rebuild the RAG chain from the cached transcript).
5. When a different URL is submitted, overwrite the cached state for the new video ID.

**Test Requirements:**
- `rule`: After analysis, `st.query_params["video_url"]` is set.
- `rule`: A cached JSON file is written for the video ID containing transcript, summary, etc.
- `rule`: On a simulated fresh session (clear session_state), the URL and cached result are restored.
- `rule`: Submitting a different URL clears/restores the correct cached result.
- `rule`: The RAG chain is rebuilt from the cached transcript on restore.

**Completion Evidence:** Automated tests + manual browser refresh test.

---

## Task 5: Add regression tests for all three issues

**Status:** pending
**Priority:** medium

**Files:** `tests/test_issues.py` (new)

**Changes:**
Create a pytest-compatible test file covering:
- `_youtube_video_id` extraction for standard, youtu.be, and shorts URLs.
- `_clean_segment_text` preserves Hindi/Punjabi/Hinglish and strips only orphaned marks.
- RAG system prompt contains Punjabi, Hindi, Hinglish, English clauses.
- Video state cache save/load round-trips.
- Timestamp validity after text processing.

**Test Requirements:**
- `rule`: All new tests pass.
- `rule`: No existing functionality is broken (imports succeed).

**Completion Evidence:** `pytest tests/test_issues.py -v` passes.

---

## Task 6: End-to-end verification

**Status:** pending
**Priority:** high

**Changes:** None (verification only).

**Test Requirements:**
- `rule`: App starts without errors.
- `rule`: YouTube URL analysis completes and player renders.
- `rule`: Browser refresh restores the video.
- `rule`: Hindi/Punjabi transcript text is not corrupted.
- `rule`: English transcription unchanged.

**Completion Evidence:** Manual run + test output.
