# Specification: Video Playback, Indian Multilingual Support, and Persistent Video State

## Problem

Three production issues in the AI Video Assistant Streamlit application:

1. **YouTube video player intermittently shows a black screen** after analysis completes. The transcript and analysis are available, but the player renders nothing playable.
2. **Indian-language content (Hindi, Punjabi, Hinglish) is corrupted or inconsistently handled** across transcription, RAG, and chat. The supplied Punjabi screenshot shows an inconsistent response.
3. **The current video disappears after a browser refresh.** The user must re-paste the URL and re-run analysis.

## Users & Goals

- End users pasting YouTube URLs or uploading media, expecting reliable playback, correct multilingual handling, and state continuity across refreshes.

## Non-Goals

- Rebuilding the UI or redesigning the application.
- Replacing existing models (Whisper, Qwen, Ollama) or frameworks.
- Introducing paid/cloud APIs or external services.
- Hardcoding fixes for specific videos, languages, or phrases.

---

## Issue 1 — YouTube Video Player Black Screen

### Root Cause (confirmed by code inspection)

In `app.py`, the results dashboard determines the player source:
- `preview_path = res.get("source_path", "")` — for YouTube, this is the YouTube **URL string**, not a local file.
- `is_local_video = os.path.exists(preview_path) and is_video_file(preview_path)` → **False** for YouTube (URL is not a local path).
- When `show_chapter_player` is **False** (no visual chapters), the code executes `st.video(preview_path)` with a YouTube watch URL. `st.video` cannot play YouTube watch URLs — it expects a local file or a direct media URL. This produces a black/non-functional player.
- When `show_chapter_player` is **True**, the YouTube IFrame API player renders correctly in `_render_chapter_controls`.
- Additionally, `components.html` in `_render_chapter_controls` is called **without a `key`**, so Streamlit may recreate the iframe on every rerun, causing the YouTube IFrame API to reinitialize and sometimes fail (race condition → black screen).

### Requirements

- `rule`: YouTube videos must always render a working YouTube embed/iframe player (not `st.video` with a watch URL).
- `rule`: The `components.html` player must use a stable `key` derived from the YouTube video ID to prevent recreation on rerun.
- `rule`: If a YouTube video has embedding restrictions, show a clear message and a link to open it on YouTube, rather than a black rectangle.
- `rule`: Chapter click seeking must continue to work on the correct video.
- `rule`: Local video/audio files must continue to use `st.video`/`st.audio` as before.

### Acceptance Criteria

- `rule`: Repeated analysis of a supported YouTube video does not randomly lose the player.
- `rule`: The player is associated with the correct video after analysis.
- `rule`: Chapter selections target the correct video and timestamp.
- `rule`: A player failure is distinguishable from an analysis failure (explicit error messaging).
- `rule`: Embedding-restricted videos are handled with a clear fallback explanation.

---

## Issue 2 — Indian Languages Not Processed Correctly

### Root Cause (confirmed by code inspection)

1. **`_clean_segment_text` in `core/transcriber.py` strips trailing `Mn` (Mark, Nonspacing) characters** via a `while` loop. Devanagari/Gurmukhi vowel signs (ि, ी, ु, ू, etc.) and anusvara (ं) are Unicode category `Mn`. A transcribed segment ending in a word with a trailing vowel sign or anusvara (e.g. "नहीं" → न+ह+ी+ं) gets its final marks stripped, corrupting the text to "नह".
2. **The RAG system prompt** (`core/rag_engine.py`) enumerates English, Hindi (Devanagari), and Hinglish, but only says "Other languages → that language." It does not explicitly mention Punjabi/Gurmukhi, increasing the chance the model defaults to Hindi or English for Punjabi questions.
3. **`_normalize_language`** maps only English and Hindi explicitly; other codes pass through. This is acceptable for auto-detect but means no Punjabi-specific handling exists.

### Requirements

- `rule`: `_clean_segment_text` must not strip valid trailing vowel signs or anusvara from Devanagari/Gurmukhi text. Only strip a trailing combining mark when it is truly orphaned (preceded by whitespace or start of string).
- `rule`: The RAG system prompt must explicitly include Punjabi (Gurmukhi script) in the answer-language rules, alongside Hindi and Hinglish.
- `rule`: Unicode text (Devanagari, Gurmukhi, mixed scripts) must round-trip unchanged through transcript cleaning, chunking, FAISS storage, retrieval, and prompt construction.
- `rule`: The transcript language and the user's response language must remain separate in prompts.

### Acceptance Criteria

- `rule`: Hindi Devanagari text is not corrupted by `_clean_segment_text`.
- `rule`: Punjabi Gurmukhi text is preserved through cleaning and retrieval where the model recognizes it.
- `rule`: Hinglish (mixed Latin + Devanagari) is not destroyed by post-processing.
- `rule`: A Punjabi question gets a Punjabi (Gurmukhi) answer, not Hindi or English.
- `rule`: English behavior is unchanged.

---

## Issue 3 — Current Video Disappears After Page Reload

### Root Cause (confirmed by code inspection)

- `st.session_state` does **not** survive a browser refresh — Streamlit creates a new session.
- The URL input (`st.text_input`) resets to empty on reload.
- `DEFAULT_STATE["last_source_id"]` is `""`; `result` is `None`.
- No persistence mechanism (query params, local storage, or file cache) exists for the active video URL or its analysis results.

### Requirements

- `rule`: On a normal browser refresh, the previously analyzed YouTube URL must be restored into the input field.
- `rule`: If cached analysis results for that video exist locally, they must be restored into `st.session_state.result` without re-running the pipeline.
- `rule`: If no cached results exist, the URL is restored and the user can click Analyze without re-pasting.
- `rule`: Submitting a different valid URL switches the active video and clears stale results.
- `rule`: Persistence must be local-first (no external DB/cloud). Use `st.query_params` for the URL and a local JSON cache keyed by video ID for results.
- `rule`: Temporary files that may have been deleted must be handled safely (re-create derived state from the persisted video ID).

### Acceptance Criteria

- `rule`: After analyzing a YouTube video and refreshing the browser, the same URL and player are restored.
- `rule`: Cached analysis results are reused if available.
- `rule`: Submitting a different URL switches videos and prevents stale results.
- `rule`: Normal Streamlit reruns do not clear the selected video.

---

## Constraints & Assumptions

- Streamlit ≥ 1.35 supports `st.query_params`.
- The local cache directory will be `downloads/_cache/` (gitignored via existing patterns).
- Ollama/Qwen and Whisper models remain local; no cloud calls.
- The existing `_render_chapter_controls` YouTube IFrame approach is the supported playback mechanism and is not replaced.
