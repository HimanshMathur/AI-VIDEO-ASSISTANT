import os
import sys

# Ensure stdout and stderr use UTF-8 to prevent charmap encoding errors on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

import time
import uuid
import shutil
import logging
import traceback
import html
import hashlib
import json
import re
from typing import Any
from urllib.parse import parse_qs, urlparse
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

# Load .env before importing core modules
load_dotenv()

from utils.audio_processor import (
    check_ffmpeg,
    validate_youtube_url,
    download_youtube_audio,
    download_youtube_video,
    extract_audio_from_file,
    validate_audio,
    chunk_audio,
    cleanup_temp_files,
    AudioProcessingError,
    MediaValidationError,
    ALL_SUPPORTED_EXTENSIONS,
)
from core.local_llm import check_ollama, check_model_available, OLLAMA_MODEL, OLLAMA_BASE_URL
from core.transcriber import transcribe_all, TranscriptionError, WHISPER_MODEL
from core.summarizer import summarize, generate_title, generate_video_intelligence
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question, ask_question_stream
from core.visual_analyzer import (
    is_video_file,
    analyze_video_visuals,
    check_vision_model_available,
    VISION_MODEL,
)
from core.languages import (
    build_language_selector_options,
    get_language,
    is_rtl,
    language_display_name,
    detect_language_from_text,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _format_chapter_timestamp(seconds: float) -> str:
    """Format chapter time as MM:SS, or HH:MM:SS for hour-long videos."""
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def _parse_chapter_seconds(value: Any) -> float:
    """Read an absolute chapter time, never treating percentage text as a time."""
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str) or "%" in value:
        raise ValueError(f"Invalid absolute chapter time: {value!r}")

    parts = value.strip().split(":")
    if not 1 <= len(parts) <= 3:
        raise ValueError(f"Invalid chapter timestamp: {value!r}")
    try:
        numbers = [float(part) for part in parts]
    except ValueError as exc:
        raise ValueError(f"Invalid chapter timestamp: {value!r}") from exc
    if any(part < 0 for part in numbers):
        raise ValueError(f"Invalid chapter timestamp: {value!r}")

    if len(numbers) == 1:
        return numbers[0]
    if len(numbers) == 2:
        return numbers[0] * 60 + numbers[1]
    return numbers[0] * 3600 + numbers[1] * 60 + numbers[2]


def _build_video_chapters(visual_evidence: list, duration_seconds: float) -> list:
    """Build chronological, in-range chapter entries from absolute visual event times."""
    chapters = []
    for item in visual_evidence:
        try:
            raw_seconds = item.get("timestamp_seconds")
            if raw_seconds is None:
                raw_seconds = item.get("timestamp")
            seconds = _parse_chapter_seconds(raw_seconds)
        except (AttributeError, TypeError, ValueError) as exc:
            logger.warning("Skipping visual event with invalid chapter timestamp: %s", exc)
            continue

        if not 0 <= seconds <= duration_seconds:
            logger.warning(
                "Skipping visual event at %.3fs outside video duration %.3fs",
                seconds,
                duration_seconds,
            )
            continue

        slide_title = item.get("slide_title")
        scene = item.get("scene")
        title = (
            slide_title.strip()
            if isinstance(slide_title, str) and slide_title.strip() not in ("", "None")
            else scene.strip()
            if isinstance(scene, str) and scene.strip() not in ("", "None")
            else f"Scene {len(chapters) + 1}"
        )
        chapters.append({
            "seconds": seconds,
            "timestamp": _format_chapter_timestamp(seconds),
            "title": title,
        })

    return sorted(chapters, key=lambda chapter: chapter["seconds"])


def _parse_topic_chapters(content: Any, duration_seconds: float) -> list:
    """Parse timestamped topic chapters returned by the existing chat segmentation."""
    if not isinstance(content, str) or content.startswith("Could not generate video chapters"):
        return []
    chapters = []
    pattern = re.compile(r"^\s*(\d{2}:\d{2}(?::\d{2})?)\s*(?:\u2014|\u2013|-)\s*(.+?)\s*$")
    for line in content.splitlines():
        match = pattern.match(line)
        if not match:
            continue
        try:
            seconds = _parse_chapter_seconds(match.group(1))
        except ValueError:
            continue
        title = match.group(2).strip()
        if title and 0 <= seconds <= duration_seconds:
            chapters.append({
                "seconds": seconds,
                "timestamp": _format_chapter_timestamp(seconds),
                "title": title,
            })
    # Collapse only duplicates that point to effectively the same moment.
    # Repeated titles later in the video are legitimate chapter occurrences.
    unique = []
    for chapter in sorted(chapters, key=lambda item: item["seconds"]):
        normalized_title = " ".join(chapter["title"].casefold().split())
        duplicate = any(
            " ".join(item["title"].casefold().split()) == normalized_title
            and abs(item["seconds"] - chapter["seconds"]) <= 2
            for item in unique
        )
        if not duplicate:
            unique.append(chapter)
    return unique


def _sanitize_cached_topic_chapters(chapters: Any, duration_seconds: float) -> list:
    """Validate cached chapter records and restore their numeric chronology."""
    if not isinstance(chapters, list):
        return []
    valid = []
    for item in chapters:
        if not isinstance(item, dict) or not isinstance(item.get("title"), str):
            continue
        try:
            seconds = _parse_chapter_seconds(item.get("seconds", item.get("timestamp")))
        except (TypeError, ValueError):
            continue
        title = item["title"].strip()
        if not title or not 0 <= seconds <= duration_seconds:
            continue
        valid.append({"seconds": seconds, "timestamp": _format_chapter_timestamp(seconds), "title": title})
    unique = []
    for chapter in sorted(valid, key=lambda item: item["seconds"]):
        normalized_title = " ".join(chapter["title"].casefold().split())
        if any(
            " ".join(item["title"].casefold().split()) == normalized_title
            and abs(item["seconds"] - chapter["seconds"]) <= 2
            for item in unique
        ):
            continue
        unique.append(chapter)
    return unique


def _youtube_video_id(url: str) -> str:
    """Extract a supported YouTube video ID for the seekable iframe player."""
    parsed = urlparse(url)
    host = parsed.netloc.lower().split(":")[0]
    if host.endswith("youtu.be"):
        video_id = parsed.path.strip("/").split("/")[0]
    elif host.endswith("youtube.com") or host.endswith("youtube-nocookie.com"):
        path_parts = parsed.path.strip("/").split("/")
        if path_parts and path_parts[0] in ("shorts", "embed", "live") and len(path_parts) > 1:
            video_id = path_parts[1]
        else:
            video_id = parse_qs(parsed.query).get("v", [""])[0]
    else:
        return ""
    return video_id if re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id) else ""


def _chat_answer_markdown(content: Any) -> str:
    """Render chapter records and timestamp lists as separate Markdown lines."""
    separator = "\u2014"
    if isinstance(content, dict):
        content = content.get("chapters", content.get("content", ""))
    if isinstance(content, (list, tuple)):
        chapter_lines = []
        for chapter in content:
            if not isinstance(chapter, dict):
                return str(content)
            timestamp = chapter.get("timestamp")
            if timestamp is None and chapter.get("seconds") is not None:
                timestamp = _format_chapter_timestamp(chapter["seconds"])
            elif isinstance(timestamp, (int, float)):
                timestamp = _format_chapter_timestamp(timestamp)
            title = chapter.get("title")
            if not isinstance(timestamp, str) or not isinstance(title, str) or not title.strip():
                return str(content)
            chapter_lines.append(f"{timestamp} {separator} {title.strip()}")
        return "\n".join(line + "  " for line in chapter_lines)
    if not isinstance(content, str):
        return str(content)
    # A model may flatten a chapter list onto one line. Parse only strings
    # that start with a timestamp and contain multiple timestamp/title pairs.
    timestamp = r"\d{2}:\d{2}(?::\d{2})?"
    if re.match(rf"^\s*{timestamp}\s+\S", content) and len(re.findall(timestamp, content)) > 1:
        content = re.sub(rf"\s+(?={timestamp}\s+(?:\u2014|\u2013|-|\?|:)?\s*\S)", "\n", content)
    lines = content.splitlines()
    if len(lines) > 1 and all(
        re.fullmatch(rf"{timestamp}\s*(?:\u2014|\u2013|-|\?|:)\s*.+|{timestamp}\s+.+", line.strip())
        for line in lines
    ):
        normalized_lines = [
            re.sub(
                rf"^({timestamp})\s*(?:\?|:|\u2013|-|\u2014)\s*|^({timestamp})\s+",
                lambda match: f"{match.group(1) or match.group(2)} {separator} ", line.strip(),
            )
            for line in lines
        ]
        return "\n".join(line + "  " for line in normalized_lines)
    if len(lines) == 1 and re.fullmatch(r"\d{2}:\d{2}(?::\d{2})?\s*(?:\u2014|\u2013|-|\?|:)\s*.+", lines[0].strip()):
        line = re.sub(
            r"^((?:\d{2}:){1,2}\d{2})\s*(?:\?|:|\u2013|-|\u2014)\s*",
            lambda match: f"{match.group(1)} {separator} ", lines[0].strip(),
        )
        return line + "  "
    return content


def _render_chapter_controls(chapters: list, youtube_id: str = "") -> None:
    """Render clickable chapter entries that seek, play, and request real fullscreen."""
    chapter_json = json.dumps(chapters, ensure_ascii=True).replace("<", "\\u003c")
    rows = "".join(
        f'<button class="chapter" data-index="{index}" aria-label="Play {html.escape(chapter["title"])} at {chapter["timestamp"]}">'
        f'<span class="chapter-title">{html.escape(chapter["title"])}</span>'
        f'<time>{chapter["timestamp"]}</time></button>'
        for index, chapter in enumerate(chapters)
    )

    if youtube_id:
        player_markup = """
        <div id="player-shell"><div id="yt-player"></div></div>
        <script src="https://www.youtube.com/iframe_api"></script>
        """
        container_height = "315px"
    else:
        player_markup = ""
        container_height = "auto"

    component_html = f"""
    <!doctype html>
    <html><head><meta charset="utf-8"><style>
      * {{ box-sizing: border-box; }}
      body {{ margin: 0; color: #f8fafc; background: transparent; font-family: Arial, sans-serif; }}
      #player-shell {{ width: 100%; height: {container_height}; background: #000; border-radius: 10px; overflow: hidden; }}
      #yt-player {{ width: 100%; height: 100%; }}
      #chapters {{ display: grid; gap: 8px; max-height: 360px; overflow-y: auto; margin-top: 12px; }}
      .chapter {{ display: flex; justify-content: space-between; align-items: center; gap: 16px; width: 100%; padding: 12px 14px;
        color: #f8fafc; text-align: left; background: rgba(18, 21, 35, .85); border: 1px solid rgba(255,255,255,.12);
        border-radius: 10px; cursor: pointer; }}
      .chapter:hover, .chapter:focus-visible {{ border-color: #8b5cf6; background: rgba(139,92,246,.16); }}
      .chapter-title {{ font-size: 15px; font-weight: 650; }}
      .chapter time {{ flex: 0 0 auto; color: #67e8f9; font-family: monospace; font-size: 14px; }}
      #playback-status {{ min-height: 22px; margin-top: 8px; color: #94a3b8; font-size: 13px; }}
    </style></head><body>
      {player_markup}
      <div id="chapters">{rows}</div>
      <div id="playback-status" role="status" aria-live="polite"></div>
      <script>
        const chapters = {chapter_json};
        const status = document.getElementById("playback-status");
        let ytPlayer = null;
        let ytReady = false;
        const youtubeId = {json.dumps(youtube_id)};
        if (youtubeId) {{
          const initPlayer = () => {{
            if (ytPlayer) return;
            ytPlayer = new YT.Player("yt-player", {{
              videoId: youtubeId,
              playerVars: {{ playsinline: 1, controls: 1 }},
              events: {{ onReady: () => {{ ytReady = true; }} }}
            }});
          }};
          if (window.YT && window.YT.Player) {{
            initPlayer();
          }} else {{
            window.onYouTubeIframeAPIReady = initPlayer;
          }}
        }}

        async function seekLocalVideo(target) {{
          const parentDocument = window.parent.document;
          const video = Array.from(parentDocument.querySelectorAll("video"))
            .find(candidate => candidate.readyState >= 1);
          if (!video) throw new Error("The video player is not ready. Open the video preview and try again.");

          video.controls = true;
          const seekedPromise = new Promise(resolve => {{
            let timeout;
            const finish = () => {{
              clearTimeout(timeout);
              video.removeEventListener("seeked", finish);
              resolve(video.currentTime);
            }};
            video.addEventListener("seeked", finish, {{ once: true }});
            timeout = setTimeout(finish, 2500);
            video.currentTime = target;
            if (Math.abs(video.currentTime - target) <= 0.75) finish();
          }});

          let fullscreenPromise = null;
          let fullscreenError = null;
          try {{
            if (video.requestFullscreen) fullscreenPromise = video.requestFullscreen();
            else if (video.webkitEnterFullscreen) {{
              video.webkitEnterFullscreen();
              fullscreenPromise = Promise.resolve();
            }}
          }} catch (error) {{
            fullscreenError = error;
          }}
          let playError = null;
          try {{ await video.play(); }} catch (error) {{ playError = error; }}
          const seeked = await seekedPromise;
          if (Math.abs(seeked - target) > 0.75) {{
            throw new Error(`Seek verification failed: player reached ${{seeked.toFixed(2)}}s, expected ${{target.toFixed(2)}}s.`);
          }}
          if (playError || video.paused) {{
            status.textContent = `Reached ${{seeked.toFixed(2)}}s, but playback was blocked. Use the player controls to start playback.`;
          }} else {{
            status.textContent = `Playing from ${{seeked.toFixed(2)}}s.`;
          }}
          try {{
            if (fullscreenError) throw fullscreenError;
            if (!fullscreenPromise) throw new Error("Fullscreen is unavailable");
            await fullscreenPromise;
          }} catch (error) {{
            status.textContent += " Fullscreen was blocked; use the player's fullscreen control.";
          }}
        }}

        async function seekYoutubeVideo(target) {{
          if (!ytReady || !ytPlayer) throw new Error("The YouTube player is still loading. Try the chapter again.");
          const iframe = ytPlayer.getIframe();
          let fullscreenPromise = null;
          let fullscreenError = null;
          try {{
            if (iframe.requestFullscreen) fullscreenPromise = iframe.requestFullscreen();
          }} catch (error) {{
            fullscreenError = error;
          }}
          ytPlayer.seekTo(target, true);
          ytPlayer.playVideo();
          await new Promise(resolve => setTimeout(resolve, 900));
          const reached = ytPlayer.getCurrentTime();
          if (Math.abs(reached - target) > 2) {{
            throw new Error(`Seek verification failed: player reached ${{reached.toFixed(2)}}s, expected ${{target.toFixed(2)}}s.`);
          }}
          const playing = ytPlayer.getPlayerState() === YT.PlayerState.PLAYING;
          status.textContent = playing
            ? `Playing from ${{reached.toFixed(2)}}s.`
            : `Reached ${{reached.toFixed(2)}}s, but playback was blocked. Use the player controls to start playback.`;
          try {{
            if (fullscreenError) throw fullscreenError;
            if (!fullscreenPromise) throw new Error("Fullscreen is unavailable");
            await fullscreenPromise;
          }} catch (error) {{
            status.textContent += " Fullscreen was blocked; use the player's fullscreen control.";
          }}
        }}

        document.querySelectorAll(".chapter").forEach(button => {{
          button.addEventListener("click", async () => {{
            const chapter = chapters[Number(button.dataset.index)];
            status.textContent = "Seeking…";
            try {{
              if (youtubeId) await seekYoutubeVideo(chapter.seconds);
              else await seekLocalVideo(chapter.seconds);
            }} catch (error) {{
              status.textContent = error.message;
            }}
          }});
        }});
      </script>
    </body></html>
    """
    components.html(
        component_html,
        height=600 if youtube_id else 430,
        scrolling=False,
    )


def _render_youtube_player(youtube_id: str, source_url: str = "") -> None:
    """
    Render a standalone YouTube embed player for a YouTube video that has no
    chapter list. Uses the same IFrame API approach as _render_chapter_controls
    so playback is always reliable, instead of passing a watch URL to st.video
    (which cannot play YouTube URLs).
    """
    if not youtube_id:
        st.warning(
            "Could not extract a YouTube video ID from this URL. "
            "Open the video directly on YouTube: "
            f"[Watch on YouTube]({source_url or '#'})"
        )
        return

    embed_url = f"https://www.youtube.com/embed/{youtube_id}"
    watch_url = source_url or f"https://www.youtube.com/watch?v={youtube_id}"
    player_html = f"""
    <!doctype html>
    <html><head><meta charset="utf-8"><style>
      * {{ box-sizing: border-box; }}
      body {{ margin: 0; background: transparent; }}
      #yt-standalone {{ width: 100%; aspect-ratio: 16 / 9; background: #000; border-radius: 10px; overflow: hidden; }}
      #yt-standalone iframe {{ width: 100%; height: 100%; border: 0; }}
      #yt-status {{ min-height: 22px; margin-top: 8px; color: #94a3b8; font-size: 13px; font-family: Arial, sans-serif; }}
    </style></head><body>
      <div id="yt-standalone">
        <iframe
          src="{embed_url}?playsinline=1&controls=1&rel=0"
          title="YouTube video player"
          allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
          allowfullscreen
          referrerpolicy="strict-origin-when-cross-origin">
        </iframe>
      </div>
      <div id="yt-status" role="status" aria-live="polite">
        If the player does not load, the video may restrict embedding.
        <a href="{watch_url}" target="_blank" rel="noopener" style="color:#67e8f9">Open on YouTube</a>
      </div>
    </body></html>
    """
    components.html(
        player_html,
        height=420,
        scrolling=False,
    )


# ─── Video state persistence (Issue 3: survive browser refresh) ──────────────────
CACHE_DIR = os.path.join("downloads", "_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Keys in the result dict that cannot be JSON-serialized and must be rebuilt.
_NON_SERIALIZABLE_KEYS = ("rag_chain",)


def _get_active_video_from_query() -> tuple:
    """Read the persisted video URL and ID from st.query_params. Returns (url, video_id)."""
    url = ""
    video_id = ""
    try:
        url = (st.query_params.get("video_url") or "").strip()
        video_id = (st.query_params.get("video_id") or "").strip()
    except Exception:
        pass
    return url, video_id


def _save_video_state(url: str, video_id: str) -> None:
    """Persist the active video URL and matching ID for refresh restoration."""
    try:
        if url:
            st.query_params["video_url"] = url
        else:
            st.query_params.pop("video_url", None)
        if video_id:
            st.query_params["video_id"] = video_id
        else:
            st.query_params.pop("video_id", None)
    except Exception as exc:
        logger.warning("Could not save video state to query params: %s", exc)


def _reset_video_session_state(session_state: Any, source_id: str) -> None:
    """Clear state owned by the prior video and invalidate its in-flight work."""
    session_state["result"] = None
    session_state["chat_history"] = []
    session_state["pending_query"] = None
    session_state["is_processing"] = False
    session_state["_analysis_run_id"] = None
    for key in session_state.get("pipeline_steps", {}):
        session_state["pipeline_steps"][key] = "pending"
    session_state["last_source_id"] = source_id


def _cache_path_for(video_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", video_id) or "unknown"
    return os.path.join(CACHE_DIR, f"{safe}.json")


def _save_cached_result(video_id: str, result: dict) -> None:
    """Persist analysis results (minus non-serializable objects) to a local JSON
    file keyed by video ID."""
    if not video_id or not isinstance(result, dict):
        return
    serializable = {
        k: v for k, v in result.items() if k not in _NON_SERIALIZABLE_KEYS
    }
    try:
        with open(_cache_path_for(video_id), "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.warning("Could not save cached result for %s: %s", video_id, exc)


def _load_cached_result(video_id: str) -> dict | None:
    """Load a previously persisted analysis result for the given video ID.

    The RAG chain is NOT serialized; callers must rebuild it from the cached
    transcript and visual_evidence.
    """
    if not video_id:
        return None
    path = _cache_path_for(video_id)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
        return data
    except Exception as exc:
        logger.warning("Could not load cached result for %s: %s", video_id, exc)
        return None


def _restore_session_from_cache() -> None:
    """On a fresh Streamlit session (browser refresh), restore the active video
    URL into the input field and, if a cached result exists, restore it into
    st.session_state (rebuilding the RAG chain from the cached transcript)."""
    if st.session_state.get("_restored_from_cache"):
        return

    url, video_id = _get_active_video_from_query()
    if video_id and video_id != _youtube_video_id(url):
        video_id = ""
        _save_video_state(url, "")
    if not url:
        st.session_state["_restored_from_cache"] = True
        return

    # Pre-fill the URL input on the first run of a fresh session.
    st.session_state["youtube_url_input"] = url
    restored_source_id = f"YouTube URL::{_youtube_video_id(url) or url}"
    st.session_state.last_source_id = restored_source_id

    cached = _load_cached_result(video_id) if video_id else None
    if cached and cached.get("transcript"):
        try:
            transcript = cached.get("transcript", "")
            cached_summary_refreshed = False
            old_summary_error = "Summary could not be generated from the audio recording."
            cached_summary = cached.get("summary")
            if isinstance(cached_summary, str) and cached_summary.strip() == old_summary_error:
                try:
                    refreshed_summary = summarize(
                        transcript,
                        language=cached.get("detected_language"),
                    )
                    if refreshed_summary and refreshed_summary.strip():
                        cached["summary"] = refreshed_summary.strip()
                        cached_summary_refreshed = True
                    else:
                        cached["summary"] = (
                            "Executive summary generation failed: summarize() returned an empty response."
                        )
                except Exception as summary_error:
                    logger.exception("Could not regenerate stale cached executive summary for %s.", video_id)
                    cached["summary"] = (
                        f"Executive summary generation failed: "
                        f"{type(summary_error).__name__}: {summary_error}"
                    )
            visual_evidence = cached.get("visual_evidence") or []
            duration = float(cached.get("duration_seconds") or 0.0)
            rag_chain = build_rag_chain(
                transcript,
                visual_evidence=visual_evidence,
                duration_seconds=duration,
                video_id=restored_source_id,
            )
            cached["rag_chain"] = rag_chain
            cached["_source_id"] = restored_source_id
            st.session_state.result = cached
            if cached_summary_refreshed:
                _save_cached_result(video_id, cached)
            # Mark pipeline as completed so the sidebar shows success.
            for k in st.session_state.pipeline_steps:
                st.session_state.pipeline_steps[k] = "completed"
            logger.info("Restored cached analysis for video %s", video_id)
        except Exception as exc:
            logger.warning("Could not rebuild RAG chain from cache: %s", exc)
            st.session_state.result = None

    st.session_state["_restored_from_cache"] = True


TEMP_UPLOAD_DIR = "temp_uploads"
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)

# ─── Page Configuration ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Video Assistant — Local & Private",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── High-End Design System & CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg-main: #08090f;
    --bg-surface: rgba(16, 18, 28, 0.82);
    --bg-surface-alt: rgba(22, 25, 40, 0.75);
    --bg-card: rgba(18, 21, 35, 0.7);
    --border-subtle: rgba(255, 255, 255, 0.08);
    --border-accent: #8b5cf6;
    --border-glow: rgba(139, 92, 246, 0.3);
    --accent-purple: #8b5cf6;
    --accent-purple-light: #c4b5fd;
    --accent-cyan: #06b6d4;
    --accent-cyan-light: #67e8f9;
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --status-success: #10b981;
    --status-warning: #f59e0b;
    --status-error: #ef4444;
}

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    background-color: var(--bg-main) !important;
    color: var(--text-primary) !important;
}

.stApp {
    background: var(--bg-main) !important;
}

/* Ambient glow background */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background-image:
        radial-gradient(circle at 15% 15%, rgba(139, 92, 246, 0.07) 0%, transparent 40%),
        radial-gradient(circle at 85% 85%, rgba(6, 182, 212, 0.05) 0%, transparent 40%),
        linear-gradient(rgba(139, 92, 246, 0.02) 1px, transparent 1px),
        linear-gradient(90deg, rgba(139, 92, 246, 0.02) 1px, transparent 1px);
    background-size: 100% 100%, 100% 100%, 48px 48px, 48px 48px;
    pointer-events: none;
    z-index: 0;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: rgba(13, 15, 24, 0.95) !important;
    border-right: 1px solid var(--border-subtle) !important;
    backdrop-filter: blur(20px) !important;
}

[data-testid="stSidebar"] * {
    color: var(--text-primary) !important;
}

/* Typography */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: var(--text-primary) !important;
    letter-spacing: -0.025em;
    font-weight: 700;
}

.hero-brand {
    font-size: 1.65rem;
    font-weight: 800;
    line-height: 1.2;
    background: linear-gradient(135deg, #ffffff 10%, #a78bfa 60%, #67e8f9 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.25rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.hero-sub {
    font-size: 0.73rem;
    color: var(--text-muted);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-weight: 600;
}

/* Modern Glassmorphic Cards */
.custom-card {
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: 14px;
    padding: 1.4rem;
    margin-bottom: 1.2rem;
    position: relative;
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.35);
    transition: all 0.25s ease-in-out;
}

.custom-card:hover {
    border-color: rgba(139, 92, 246, 0.35);
    box-shadow: 0 12px 36px rgba(0, 0, 0, 0.45), 0 0 20px rgba(139, 92, 246, 0.1);
}

.card-header {
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text-secondary);
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.card-body {
    font-size: 0.9rem;
    line-height: 1.75;
    color: var(--text-primary);
}

/* Badges */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.25rem 0.65rem;
    border-radius: 6px;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    backdrop-filter: blur(8px);
}

.badge-purple { background: rgba(139, 92, 246, 0.15); color: #c4b5fd; border: 1px solid rgba(139, 92, 246, 0.3); }
.badge-cyan   { background: rgba(6, 182, 212, 0.15); color: #67e8f9; border: 1px solid rgba(6, 182, 212, 0.3); }
.badge-green  { background: rgba(16, 185, 129, 0.15); color: #6ee7b7; border: 1px solid rgba(16, 185, 129, 0.3); }
.badge-amber  { background: rgba(245, 158, 11, 0.15); color: #fcd34d; border: 1px solid rgba(245, 158, 11, 0.3); }
.badge-red    { background: rgba(239, 68, 68, 0.15); color: #fca5a5; border: 1px solid rgba(239, 68, 68, 0.3); }

/* Pipeline Progress Steps */
.pipeline-step {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.65rem 0.9rem;
    background: var(--bg-surface-alt);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
    margin: 0.35rem 0;
    font-size: 0.78rem;
    transition: all 0.2s ease;
}

.pipeline-step:hover {
    border-color: rgba(139, 92, 246, 0.25);
}

.pipeline-step-name {
    display: flex;
    align-items: center;
    gap: 0.6rem;
}

.status-indicator {
    width: 9px;
    height: 9px;
    border-radius: 50%;
}

.status-pending   { background: rgba(255, 255, 255, 0.15); }
.status-active    { background: var(--accent-cyan); box-shadow: 0 0 10px var(--accent-cyan); animation: step-pulse 1.2s infinite; }
.status-completed { background: var(--status-success); box-shadow: 0 0 8px rgba(16, 185, 129, 0.4); }
.status-failed    { background: var(--status-error); box-shadow: 0 0 8px rgba(239, 68, 68, 0.4); }

@keyframes step-pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%      { opacity: 0.35; transform: scale(0.85); }
}

/* Transcript Box */
.transcript-box {
    background: var(--bg-surface-alt);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 1.4rem;
    font-size: 0.86rem;
    line-height: 1.85;
    max-height: 440px;
    overflow-y: auto;
    color: #cbd5e1;
    white-space: pre-wrap;
    word-break: break-word;
    font-family: 'Plus Jakarta Sans', sans-serif;
}

/* Chat container & bubbles */
.chat-container {
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: 14px;
    padding: 1.4rem;
    max-height: 480px;
    overflow-y: auto;
    margin-bottom: 1rem;
    display: flex;
    flex-direction: column;
    gap: 1.1rem;
    backdrop-filter: blur(16px);
}

.chat-bubble-user {
    align-self: flex-end;
    background: linear-gradient(135deg, rgba(139, 92, 246, 0.28), rgba(109, 40, 217, 0.22));
    border: 1px solid rgba(139, 92, 246, 0.45);
    border-radius: 14px 14px 2px 14px;
    padding: 0.85rem 1.2rem;
    max-width: 80%;
    font-size: 0.88rem;
    line-height: 1.65;
    box-shadow: 0 4px 16px rgba(124, 58, 237, 0.15);
}

.chat-bubble-bot {
    align-self: flex-start;
    background: var(--bg-surface-alt);
    border: 1px solid var(--border-subtle);
    border-radius: 14px 14px 14px 2px;
    padding: 0.85rem 1.2rem;
    max-width: 85%;
    font-size: 0.88rem;
    line-height: 1.65;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
}

/* Suggestion chips */
.suggestion-chip {
    display: inline-block;
    background: var(--bg-surface-alt);
    border: 1px solid rgba(139, 92, 246, 0.3);
    border-radius: 20px;
    padding: 0.4rem 0.9rem;
    font-size: 0.78rem;
    color: var(--accent-purple-light);
    cursor: pointer;
    transition: all 0.2s ease;
    margin: 0.25rem;
}

.suggestion-chip:hover {
    background: rgba(139, 92, 246, 0.2);
    border-color: var(--accent-purple);
    transform: translateY(-1px);
}

/* Primary buttons */
.stButton > button {
    background: linear-gradient(135deg, #7c3aed 0%, #4f46e5 100%) !important;
    color: #ffffff !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.03em !important;
    border-radius: 10px !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    padding: 0.65rem 1.4rem !important;
    box-shadow: 0 4px 18px rgba(124, 58, 237, 0.35) !important;
    transition: all 0.2s ease !important;
}

.stButton > button:hover {
    transform: translateY(-1.5px) !important;
    box-shadow: 0 6px 26px rgba(124, 58, 237, 0.55) !important;
    border-color: rgba(255, 255, 255, 0.3) !important;
}

/* Inputs & Form Controls */
.stTextInput > div > div > input,
.stSelectbox > div > div {
    background: var(--bg-surface-alt) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

.stTextInput > div > div > input:focus {
    border-color: var(--accent-purple) !important;
    box-shadow: 0 0 0 2px rgba(139, 92, 246, 0.25) !important;
}

/* Streamlit Tabs */
[data-baseweb="tab-list"] {
    gap: 0.5rem !important;
    background: rgba(13, 15, 24, 0.6) !important;
    padding: 0.35rem !important;
    border-radius: 12px !important;
    border: 1px solid var(--border-subtle) !important;
}

[data-baseweb="tab"] {
    border-radius: 8px !important;
    padding: 0.5rem 1rem !important;
    font-weight: 600 !important;
    font-size: 0.84rem !important;
    color: var(--text-secondary) !important;
    transition: all 0.2s ease !important;
}

[data-baseweb="tab"][aria-selected="true"] {
    background: rgba(139, 92, 246, 0.25) !important;
    color: #ffffff !important;
    border: 1px solid rgba(139, 92, 246, 0.4) !important;
}

/* Scrollbars */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-main); }
::-webkit-scrollbar-thumb { background: rgba(255, 255, 255, 0.12); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent-purple); }
</style>
""", unsafe_allow_html=True)

# ─── System Diagnostics ──────────────────────────────────────────────────────────
# Cached with TTL so ffmpeg/ollama checks (subprocess + HTTP) do NOT rerun on every Streamlit rerender.
# Otherwise every chat message causes 1x subprocess + 2x /api/tags HTTP calls = wasted latency.
@st.cache_data(ttl=30, show_spinner=False)
def _cached_system_checks():
    ffmpeg_ok = False
    ffmpeg_err = ""
    try:
        ffmpeg_ok = check_ffmpeg()
    except Exception as e:
        ffmpeg_err = str(e)
    ollama_ok = check_ollama()
    model_ok = check_model_available() if ollama_ok else False
    return ffmpeg_ok, ffmpeg_err, ollama_ok, model_ok

ffmpeg_available, ffmpeg_error_msg, ollama_available, model_available = _cached_system_checks()

# ─── Session State Initialization ───────────────────────────────────────────────
DEFAULT_STATE = {
    "result": None,
    "chat_history": [],
    "pipeline_steps": {
        "validation": "pending",
        "audio_extract": "pending",
        "audio_check": "pending",
        "transcription": "pending",
        "visual_ai": "pending",
        "intelligence": "pending",
        "rag": "pending",
    },
    "active_source_type": "YouTube URL",
    "last_source_id": "",
    "pending_query": None,
}

for k, v in DEFAULT_STATE.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Restore the active video (and cached analysis) from query params after a
# browser refresh. Must run before the sidebar input is rendered so the URL
# field is pre-filled.
_restore_session_from_cache()

PIPELINE_STAGES = [
    ("validation",    "🔍", "Input Validation"),
    ("audio_extract", "🔊", "Audio Extraction"),
    ("audio_check",   "📊", "Audio Quality Check"),
    ("transcription", "📝", "Speech Transcription"),
    ("visual_ai",     "👁️", "Visual Frame Analysis"),
    ("intelligence",  "🧠", "Video Intelligence (Ollama)"),
    ("rag",           "⚡", "Multimodal FAISS Indexing"),
]


def render_pipeline_status():
    """Render 6-stage progress tracker in sidebar."""
    for key, icon, name in PIPELINE_STAGES:
        st_state = st.session_state.pipeline_steps.get(key, "pending")
        dot_class = f"status-{st_state}"
        badge_text = st_state.capitalize()
        st.markdown(f"""
        <div class="pipeline-step">
            <div class="pipeline-step-name">
                <div class="status-indicator {dot_class}"></div>
                <span>{icon} {name}</span>
            </div>
            <span style="font-size:0.68rem;color:var(--text-muted)">{badge_text}</span>
        </div>
        """, unsafe_allow_html=True)


# ─── Sidebar: Controls & Configuration ───────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="hero-brand">🎬 AI Video Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Local AI Processing · Privacy-First</div>', unsafe_allow_html=True)
    st.markdown("---")

    # Local System Diagnostics
    if not ffmpeg_available:
        st.markdown(f"""
        <div class="badge badge-red" style="display:block;margin-bottom:0.75rem;line-height:1.4">
            ⚠️ FFmpeg is missing on PATH.
        </div>
        """, unsafe_allow_html=True)

    if not ollama_available:
        st.markdown(f"""
        <div class="badge badge-red" style="display:block;margin-bottom:0.75rem;line-height:1.4">
            ⚠️ Local Ollama server is not running.<br>Start Ollama and try again.
        </div>
        """, unsafe_allow_html=True)
    elif not model_available:
        st.markdown(f"""
        <div class="badge badge-amber" style="display:block;margin-bottom:0.75rem;line-height:1.4">
            ⚠️ {OLLAMA_MODEL} is not available locally.<br>Run: ollama pull {OLLAMA_MODEL}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="badge badge-green" style="display:block;margin-bottom:0.75rem;line-height:1.4">
            🔒 Local Ollama Online: {OLLAMA_MODEL}
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<span class="badge badge-purple">STEP 1: Choose Source</span>', unsafe_allow_html=True)
    source_type = st.radio(
        "Source Type",
        ["YouTube URL", "Upload Video/Audio"],
        label_visibility="collapsed",
    )

    source_path = ""
    source_display_name = ""
    source_identity = ""

    if source_type == "YouTube URL":
        youtube_url = st.text_input(
            "YouTube URL",
            key="youtube_url_input",
            placeholder="https://www.youtube.com/watch?v=...",
            help="Supports standard YouTube videos, shorts, and youtu.be links",
        )
        source_path = youtube_url.strip() if youtube_url else ""
        source_display_name = source_path
        source_identity = _youtube_video_id(source_path) or source_path
    else:
        uploaded_file = st.file_uploader(
            "Upload Audio or Video",
            type=["mp4", "mkv", "avi", "mov", "webm", "m4v", "mp3", "wav", "m4a", "flac", "ogg", "aac"],
            help="Common video (mp4, mkv, mov) and audio (mp3, wav, m4a, flac) formats supported",
        )
        if uploaded_file is not None:
            source_display_name = uploaded_file.name
            upload_bytes = uploaded_file.getbuffer()
            source_identity = f"{uploaded_file.name}::{hashlib.sha256(upload_bytes).hexdigest()}"
            unique_ext = os.path.splitext(uploaded_file.name)[1].lower()
            temp_save_path = os.path.join(TEMP_UPLOAD_DIR, f"upload_{uuid.uuid4().hex[:8]}{unique_ext}")
            with open(temp_save_path, "wb") as f:
                f.write(upload_bytes)
            source_path = temp_save_path

    # Reset result if the submitted source identity changes. Upload identity
    # includes content so replacing a file with the same name resets video state.
    current_source_id = f"{source_type}::{source_identity}"
    if current_source_id != st.session_state.last_source_id:
        _reset_video_session_state(st.session_state, current_source_id)
        if source_type == "YouTube URL" and source_path:
            # Replace any old video ID immediately; an in-progress new URL must
            # never restore the previous video's cached result after refresh.
            _save_video_state(source_path, "")
        else:
            _save_video_state("", "")

    st.markdown("---")
    st.markdown('<span class="badge badge-cyan">STEP 2: Speech Recognition</span>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.78rem;color:var(--text-secondary);line-height:1.5;margin-top:0.35rem">
        🎙️ <strong>Automatic Language Detection</strong><br>
        Speech language is detected automatically and preserved in its original language.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<span class="badge badge-purple">STEP 2.5: Response Language</span>', unsafe_allow_html=True)
    lang_options = build_language_selector_options()
    response_language = st.selectbox(
        "Answer language",
        options=lang_options,
        format_func=lambda code: (
            "Auto - match the question language"
            if code == "auto"
            else language_display_name(code)
        ),
        index=0,
        help="Force all answers into this language. 'Auto' answers in the user's question language by default.",
    )

    st.markdown("---")
    st.markdown('<span class="badge badge-green">STEP 3: Process</span>', unsafe_allow_html=True)
    run_button = st.button("⚡ Analyze", use_container_width=True)

    st.markdown("---")
    st.markdown('<span class="card-header" style="margin:0 0 0.5rem 0">Pipeline Status</span>', unsafe_allow_html=True)
    render_pipeline_status()

# ─── Main Content Canvas ────────────────────────────────────────────────────────
top_col1, top_col2 = st.columns([4, 1])
with top_col1:
    st.markdown('<h1>🎬 AI Video Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub" style="margin-bottom:1.5rem">Local AI Video Intelligence · Transcribe · Summarize · Chat</div>', unsafe_allow_html=True)

# ─── Pipeline Execution ─────────────────────────────────────────────────────────
status_placeholder = st.empty()

if run_button:
    if st.session_state.get("is_processing", False):
        st.warning("⚠️ An analysis is already in progress. Please wait for it to complete.")
        st.stop()

    if not source_path:
        st.error("Please enter a valid YouTube URL or upload an audio/video file.")
    elif not ffmpeg_available:
        st.error(f"Cannot start processing: {ffmpeg_error_msg}")
    elif not ollama_available:
        st.error(f"Local Ollama server is not running. Start Ollama and try again.")
    elif not model_available:
        st.error(f"Qwen3:8B is not available locally. Run in terminal:\nollama pull {OLLAMA_MODEL}")
    else:
        # Reset previous run state
        st.session_state.is_processing = True
        analysis_run_id = uuid.uuid4().hex
        st.session_state["_analysis_run_id"] = analysis_run_id
        st.session_state.result = None
        st.session_state.chat_history = []
        for k in st.session_state.pipeline_steps:
            st.session_state.pipeline_steps[k] = "pending"

        def set_step(key, state):
            if st.session_state.get("_analysis_run_id") == analysis_run_id:
                st.session_state.pipeline_steps[key] = state

        chunks_to_clean = []
        extracted_wav = None
        video_for_vision_path = None  # local video path used ONLY for vision analysis (YouTube: downloaded copy)
        t_start_total = time.time()

        try:
            # ── 1. Validation ──────────────────────────────────
            t_val_start = time.time()
            set_step("validation", "active")
            status_placeholder.info("🔍 Step 1/6: Validating source input...")
            if source_type == "YouTube URL":
                if not validate_youtube_url(source_path):
                    raise MediaValidationError(
                        f"Invalid YouTube URL format: '{source_path}'. "
                        "Please use a standard link such as https://www.youtube.com/watch?v=... or https://youtu.be/..."
                    )
            else:
                if not os.path.exists(source_path) or os.path.getsize(source_path) == 0:
                    raise MediaValidationError("The uploaded file could not be read or is empty.")
            set_step("validation", "completed")
            t_val = time.time() - t_val_start

            # ── 2. Audio Extraction ────────────────────────────
            t_extract_start = time.time()
            set_step("audio_extract", "active")
            status_placeholder.info("🔊 Step 2/6: Extracting audio track to 16kHz mono PCM...")
            if source_type == "YouTube URL":
                extracted_wav = download_youtube_audio(source_path)
            else:
                extracted_wav = extract_audio_from_file(source_path)
            chunks_to_clean.append(extracted_wav)
            set_step("audio_extract", "completed")
            t_extract = time.time() - t_extract_start

            # ── 2b. Resolve a local video path for vision analysis ──
            # For local file uploads: source_path IS the video (already on disk).
            # For YouTube URLs: download a small (≤480p) video copy for frame analysis.
            if source_type == "YouTube URL":
                try:
                    status_placeholder.info("🎞️ Preparing YouTube video copy for visual frame analysis...")
                    video_for_vision_path = download_youtube_video(source_path)
                    if video_for_vision_path:
                        chunks_to_clean.append(video_for_vision_path)
                except Exception as ve:
                    logger.warning(f"Could not download YouTube video for vision analysis: {ve}")
                    video_for_vision_path = None
                vision_input_path = video_for_vision_path
            else:
                vision_input_path = source_path

            # ── 3. Audio Quality Check ─────────────────────────
            t_check_start = time.time()
            set_step("audio_check", "active")
            status_placeholder.info("📊 Step 3/6: Validating audio duration and signal energy...")
            audio_meta = validate_audio(extracted_wav)
            chunks = chunk_audio(extracted_wav)
            chunks_to_clean.extend([c for c in chunks if c != extracted_wav])
            set_step("audio_check", "completed")
            t_check = time.time() - t_check_start

            # ── 4. Speech Transcription ────────────────────────
            t_trans_start = time.time()
            set_step("transcription", "active")
            def on_transcribe_progress(msg, current, total):
                status_placeholder.info(f"📝 Step 4/6: {msg}")

            status_placeholder.info(
                f"📝 Step 4/6: Transcribing speech locally with Whisper ({WHISPER_MODEL}) "
                f"({len(chunks)} chunk(s))..."
            )
            # Automatic language detection without manual selection
            transcript, detected_language = transcribe_all(chunks, language=None, progress_callback=on_transcribe_progress)

            if not transcript or not transcript.strip():
                raise TranscriptionError("No speech could be recognized from the recording.")
            set_step("transcription", "completed")
            t_transcribe = time.time() - t_trans_start

            # ── 5. Visual Frame Analysis (Phase D) ─────────────
            t_vis_start = time.time()
            set_step("visual_ai", "active")
            visual_evidence = []
            # Use vision_input_path (local file on disk), not the YouTube URL string.
            # vision_input_path may be None if YouTube video download failed -> skip gracefully.
            if vision_input_path and os.path.exists(vision_input_path) and is_video_file(vision_input_path) and check_vision_model_available():
                def on_vision_progress(msg, current, total):
                    status_placeholder.info(f"👁️ Visual Analysis ({current}/{total}): {msg}")

                status_placeholder.info(
                    f"👁️ Step 5/7: Analyzing video frames locally with vision model ({VISION_MODEL})..."
                )
                visual_evidence = analyze_video_visuals(vision_input_path, max_frames=20, progress_callback=on_vision_progress, language=detected_language)
            else:
                # Explain why vision was skipped for debuggability
                if source_type == "YouTube URL" and not vision_input_path:
                    logger.info("Vision skipped for YouTube: video copy was not available on disk (download failed or unsupported).")
                elif source_type != "YouTube URL":
                    logger.info(f"Vision skipped: uploaded file is audio-only ({vision_input_path}).")
                elif not check_vision_model_available():
                    logger.info(f"Vision skipped: vision model {VISION_MODEL} not available in Ollama.")
            set_step("visual_ai", "completed")
            t_vis = time.time() - t_vis_start

            # ── 6. Video Intelligence (Title, Summary, Insights in 1 consolidated call) ──
            t_intel_start = time.time()
            set_step("intelligence", "active")
            status_placeholder.info(f"🧠 Step 6/7: Generating executive dossier via local Ollama ({OLLAMA_MODEL})...")
            intel = generate_video_intelligence(transcript, language=detected_language)
            title = intel.get("title", "Video Intelligence Dossier")
            summary = intel.get("summary", "")
            action_items = intel.get("action_items", "")
            decisions = intel.get("key_decisions", "")
            questions = intel.get("open_questions", "")
            set_step("intelligence", "completed")
            t_intel = time.time() - t_intel_start

            # ── 7. Multimodal RAG Knowledge Indexing ───────────
            t_rag_start = time.time()
            set_step("rag", "active")
            status_placeholder.info("⚡ Step 7/7: Indexing transcript & visual frames into local FAISS...")
            vid_duration = audio_meta.get("duration", 0.0) if audio_meta else 0.0
            rag_chain = build_rag_chain(
                transcript,
                visual_evidence=visual_evidence,
                duration_seconds=vid_duration,
                video_id=current_source_id,
            )
            set_step("rag", "completed")
            t_rag = time.time() - t_rag_start

            t_total = time.time() - t_start_total

            logger.info("==================================================")
            logger.info(f"[PERF] 1. Source validation:      {t_val:.2f}s")
            logger.info(f"[PERF] 2. Audio extraction:       {t_extract:.2f}s")
            logger.info(f"[PERF] 3. Audio check & chunking: {t_check:.2f}s")
            logger.info(f"[PERF] 4. Whisper transcription:  {t_transcribe:.2f}s")
            logger.info(f"[PERF] 5. Visual frame analysis:  {t_vis:.2f}s")
            logger.info(f"[PERF] 6. Ollama video intel:     {t_intel:.2f}s")
            logger.info(f"[PERF] 7. FAISS vector indexing:  {t_rag:.2f}s")
            logger.info(f"[PERF] TOTAL PIPELINE TIME:       {t_total:.2f}s")
            logger.info("==================================================")

            # A source change invalidates the run token; never publish its stale result.
            if (st.session_state.get("_analysis_run_id") != analysis_run_id
                    or st.session_state.last_source_id != current_source_id):
                logger.warning("Discarding completed analysis for a superseded video source")
                st.stop()

            # Store results in session state
            st.session_state.result = {
                "_source_id": current_source_id,
                "title": title,
                "source_name": source_display_name,
                "source_type": source_type,
                "source_path": source_path,
                "duration_seconds": audio_meta.get("duration", 0),
                "transcript": transcript,
                "detected_language": detected_language,
                "visual_evidence": visual_evidence,
                "summary": summary,
                "action_items": action_items,
                "key_decisions": decisions,
                "open_questions": questions,
                "rag_chain": rag_chain,
                "timing": {
                    "validation": t_val,
                    "extract": t_extract,
                    "check": t_check,
                    "transcribe": t_transcribe,
                    "visual": t_vis,
                    "intel": t_intel,
                    "rag": t_rag,
                    "total": t_total,
                }
            }

            # Persist the active video URL and analysis results so a browser
            # refresh can restore the session without re-running the pipeline.
            if source_type == "YouTube URL":
                yt_id = _youtube_video_id(source_path)
                _save_video_state(source_path, yt_id)
                _save_cached_result(yt_id, st.session_state.result)

            status_placeholder.success(f"✅ Local Analysis completed in {t_total:.1f}s!")
            time.sleep(0.6)
            status_placeholder.empty()
            st.rerun()

        except Exception as e:
            if st.session_state.get("_analysis_run_id") != analysis_run_id:
                logger.info("Ignoring failure from a superseded video analysis run")
                st.stop()
            # Mark only the active step as failed
            found_active = False
            for step_key, _, _ in PIPELINE_STAGES:
                if st.session_state.pipeline_steps.get(step_key) == "active":
                    st.session_state.pipeline_steps[step_key] = "failed"
                    found_active = True
                elif found_active:
                    st.session_state.pipeline_steps[step_key] = "pending"

            logger.exception("Pipeline execution failed")
            status_placeholder.error(f"❌ Error: {e}")

        finally:
            if st.session_state.get("_analysis_run_id") == analysis_run_id:
                st.session_state.is_processing = False
            if chunks_to_clean:
                cleanup_temp_files(chunks_to_clean)


# ─── Results Dashboard ──────────────────────────────────────────────────────────
if (st.session_state.result
        and st.session_state.result.get("_source_id", current_source_id) == current_source_id):
    res = st.session_state.result
    duration_min = int(res["duration_seconds"] // 60)
    duration_sec = int(res["duration_seconds"] % 60)
    duration_label = f"{duration_min}m {duration_sec:02d}s" if duration_min > 0 else f"{duration_sec}s"
    word_count = len(res["transcript"].split())
    total_time_label = f"{res.get('timing', {}).get('total', 0):.1f}s"
    preview_path = res.get("source_path", "")
    is_youtube_video = res.get("source_type") == "YouTube URL"
    is_local_video = bool(preview_path and os.path.exists(preview_path) and is_video_file(preview_path))
    youtube_id = _youtube_video_id(preview_path) if is_youtube_video else ""
    duration_seconds = float(res.get("duration_seconds", 0))
    cached_chapters = _sanitize_cached_topic_chapters(res.get("topic_chapters"), duration_seconds)
    if "topic_chapters" not in res or (res.get("topic_chapters") and not cached_chapters):
        language_hint = " in Hindi" if res.get("detected_language") == "hi" else ""
        try:
            chapter_response = ask_question(
                res["rag_chain"],
                f"Generate topic-wise video chapters with timestamps{language_hint}.",
                output_language=response_language,
            ) if res.get("rag_chain") is not None else ""
            res["topic_chapters"] = _parse_topic_chapters(chapter_response, duration_seconds)
        except Exception as exc:
            logger.warning("Could not load topic chapters for the video: %s", exc)
            res["topic_chapters"] = []
    else:
        res["topic_chapters"] = cached_chapters
    chapters = res.get("topic_chapters") or []
    show_chapter_player = bool(chapters and (is_local_video or youtube_id))

    # Build detected-language badge
    detected_lang_code = res.get("detected_language")
    detected_language_badge = ""
    transcript_dir = "ltr"
    if detected_lang_code:
        _dl = get_language(detected_lang_code)
        if _dl is not None:
            _rtl_tag = " · RTL" if _dl.rtl else ""
            transcript_dir = "rtl" if _dl.rtl else "ltr"
            detected_language_badge = (
                f'<div style="font-size:0.78rem;color:var(--text-muted);'
                f'display:flex;gap:0.5rem;align-items:center;margin-top:0.35rem">'
                f'<span class="badge badge-amber">🗣️ Detected: '
                f'{_dl.native_name} ({_dl.name}){_rtl_tag}</span></div>'
            )

    # Session Title Banner with Glassmorphism
    st.markdown(f"""
    <div class="custom-card" style="border-left: 4px solid var(--accent-purple); background: linear-gradient(135deg, rgba(20, 22, 36, 0.9) 0%, rgba(15, 17, 28, 0.85) 100%);">
        <div class="card-header">
            <span>📌 Video Intelligence Dossier</span>
            <span class="badge badge-purple">{res['source_type']}</span>
            <span class="badge badge-cyan">⏱️ {duration_label}</span>
            <span class="badge badge-green">📖 {word_count:,} words</span>
            <span class="badge badge-purple">⚡ {total_time_label}</span>
            <span class="badge badge-green">🔒 100% Local AI / Zero Cloud</span>
        </div>
        <div style="font-size:1.65rem;font-weight:800;color:var(--text-primary);margin-bottom:0.4rem;line-height:1.25">
            {res['title']}
        </div>
        <div style="font-size:0.78rem;color:var(--text-muted);display:flex;gap:0.5rem;align-items:center">
            <span>Source:</span> <span style="color:var(--text-secondary);font-family:'JetBrains Mono', monospace">{res['source_name']}</span>
        </div>
        {detected_language_badge}
    </div>
    """, unsafe_allow_html=True)

    # Keep the existing preview for media without a timestamped chapter player.
    if preview_path:
        if is_youtube_video and not show_chapter_player:
            with st.expander("▶️ Watch Source Video", expanded=True):
                _render_youtube_player(youtube_id, source_url=preview_path)
        elif os.path.exists(preview_path) and not show_chapter_player and os.path.getsize(preview_path) > 0:
            with st.expander("▶️ Media Player Preview", expanded=False):
                if is_local_video:
                    st.video(preview_path)
                else:
                    st.audio(preview_path)

    # Tabs for organized analysis
    tab_overview, tab_transcript, tab_visual, tab_actions, tab_decisions, tab_questions, tab_chat = st.tabs([
        "📋 Executive Dossier",
        "📝 Full Transcript",
        "👁️ Visual Evidence",
        "✅ Action Items",
        "🔑 Strategic Decisions",
        "❓ Open Questions",
        "💬 Chat with Video",
    ])

    with tab_overview:
        if show_chapter_player:
            st.markdown("""
            <div class="custom-card">
                <div class="card-header">Video Chapters</div>
                <div style="font-size:0.84rem;color:var(--text-secondary)">
                    Select a scene to seek to its timestamp and start fullscreen playback.
                </div>
            </div>
            """, unsafe_allow_html=True)
            if youtube_id:
                _render_chapter_controls(chapters, youtube_id=youtube_id)
            else:
                st.video(preview_path)
                _render_chapter_controls(chapters)

        col_s1, col_s2 = st.columns([3, 2])
        with col_s1:
            st.markdown("""
            <div class="custom-card">
                <div class="card-header">Executive Summary</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f'<div dir="{transcript_dir}">{res["summary"]}</div>', unsafe_allow_html=True)

        with col_s2:
            timing_info = ""
            if "timing" in res:
                t_info = res["timing"]
                timing_info = (
                    f"<strong>Pipeline Time:</strong> {t_info.get('total', 0):.1f}s<br>"
                    f"&nbsp;&nbsp;• Audio Extract: {t_info.get('extract', 0):.1f}s<br>"
                    f"&nbsp;&nbsp;• Speech STT: {t_info.get('transcribe', 0):.1f}s<br>"
                    f"&nbsp;&nbsp;• Visual Vision AI: {t_info.get('visual', 0):.1f}s<br>"
                    f"&nbsp;&nbsp;• Ollama Intel: {t_info.get('intel', 0):.1f}s<br>"
                    f"&nbsp;&nbsp;• FAISS RAG: {t_info.get('rag', 0):.2f}s<br>"
                )
            vis_count = len(res.get("visual_evidence", []))
            st.markdown(f"""
            <div class="custom-card">
                <div class="card-header">⚡ Local Stack Specs & Performance</div>
                <div style="font-size:0.84rem;line-height:1.9;color:var(--text-secondary)">
                    <strong>Duration:</strong> {duration_label}<br>
                    <strong>Word Count:</strong> {word_count:,} words<br>
                    {timing_info}
                    <strong>STT Engine:</strong> Local Whisper ({WHISPER_MODEL})<br>
                    <strong>Vision Engine:</strong> Local Ollama ({VISION_MODEL} · {vis_count} frames)<br>
                    <strong>LLM Engine:</strong> Local Ollama ({OLLAMA_MODEL})<br>
                    <strong>Embeddings:</strong> paraphrase-multilingual (Local)<br>
                    <strong>Vector Search:</strong> FAISS In-Memory CPU<br>
                    <strong>Privacy:</strong> 100% Air-Gapped Local Architecture<br>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with tab_transcript:
        st.markdown(f"""
        <div class="custom-card">
            <div class="card-header">
                <span>Complete Audio Transcript</span>
                <span style="margin-left:auto;font-size:0.75rem;color:var(--text-muted)">{word_count:,} words</span>
            </div>
            <div class="transcript-box" dir="{transcript_dir}">{res['transcript']}</div>
        </div>
        """, unsafe_allow_html=True)
        st.download_button(
            "⬇️ Download Transcript (.txt)",
            data=res["transcript"],
            file_name=f"transcript_{uuid.uuid4().hex[:6]}.txt",
            mime="text/plain",
            use_container_width=True,
        )

    with tab_visual:
        v_list = res.get("visual_evidence", [])
        if not v_list:
            st.markdown("""
            <div class="custom-card" style="text-align:center;padding:2.5rem 1.5rem">
                <div style="font-size:2rem;margin-bottom:0.5rem">🎥</div>
                <div style="font-size:0.92rem;font-weight:600;color:var(--text-primary);margin-bottom:0.4rem">
                    No Visual Evidence Extracted
                </div>
                <div style="font-size:0.84rem;color:var(--text-secondary)">
                    The processed file is audio-only, or the local vision model was offline.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="custom-card">
                <div class="card-header">
                    <span>Observed Visual Evidence ({len(v_list)} event keyframes captured)</span>
                    <span style="margin-left:auto;font-size:0.75rem;color:var(--text-muted)">Vision Model: {VISION_MODEL}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            for v in v_list:
                ts_disp = v.get("timestamp_display") or v.get("timestamp", "00:00:00")
                people_str = f" · {v.get('people_count', 0)} People" if v.get("people_count", 0) > 0 else ""
                title_str = f" · '{v.get('slide_title')}'" if v.get("slide_title") and v.get("slide_title") != "None" else ""
                with st.expander(f"⏱️ [{ts_disp}]{title_str}{people_str}", expanded=False):
                    st.markdown(f"**Scene:** {v.get('scene', 'Video scene')}")
                    st.markdown(f"**Screen / UI:** {v.get('screen_ui', 'None')}")
                    if v.get('slide_title') and v.get('slide_title') != "None":
                        st.markdown(f"**Slide / Title:** {v.get('slide_title')}")
                    if v.get('charts') and v.get('charts') != "None":
                        st.markdown(f"**Charts:** {v.get('charts')}")
                    if v.get('diagrams') and v.get('diagrams') != "None":
                        st.markdown(f"**Diagrams:** {v.get('diagrams')}")
                    if v.get('visible_text') and v.get('visible_text') != "None":
                        st.markdown(f"**Visible Text:** {v.get('visible_text')}")
                    if v.get('people') and v.get('people') != "None":
                        st.markdown(f"**People:** {v.get('people')}")
                    if v.get('actions') and v.get('actions') != "None":
                        st.markdown(f"**Actions:** {v.get('actions')}")
                    if v.get('important_details') and v.get('important_details') != "None":
                        st.markdown(f"**Details:** {v.get('important_details')}")


    with tab_actions:
        st.markdown("""
        <div class="custom-card">
            <div class="card-header">Extracted Action Items & Deliverables</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(res['action_items'])

    with tab_decisions:
        st.markdown("""
        <div class="custom-card">
            <div class="card-header">Strategic & Key Decisions</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(res['key_decisions'])

    with tab_questions:
        st.markdown("""
        <div class="custom-card">
            <div class="card-header">Unresolved Questions & Follow-ups</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(res['open_questions'])

    with tab_chat:
        st.markdown('<div class="card-header">💬 Chat with Video Transcript (Cross-Lingual & Grounded)</div>', unsafe_allow_html=True)

        if res["rag_chain"] is None:
            st.warning("RAG chain could not be initialized for this session.")
        else:
            # Render chronological chat history without duplicates
            if st.session_state.chat_history:
                st.markdown('<div class="chat-container">', unsafe_allow_html=True)
                for message in st.session_state.chat_history:
                    if message["role"] == "user":
                        st.markdown(f"""
                        <div class="chat-bubble-user" dir="auto">
                            <span style="font-size:0.68rem;text-transform:uppercase;color:#c4b5fd;font-weight:700">You</span><br>
                            {message['content']}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="chat-bubble-bot" dir="auto">
                            <span style="font-size:0.68rem;text-transform:uppercase;color:#67e8f9;font-weight:700">🤖 Assistant</span><br>
                        </div>
                        """, unsafe_allow_html=True)
                        st.markdown(_chat_answer_markdown(message['content']))
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="custom-card" style="text-align:center;padding:2rem">
                    <div style="font-size:2rem;margin-bottom:0.5rem">💡</div>
                    <div style="font-size:0.92rem;font-weight:600;color:var(--text-primary);margin-bottom:0.4rem">
                        Ask Any Question About This Recording
                    </div>
                    <div style="font-size:0.84rem;color:var(--text-secondary);max-width:460px;margin:0 auto">
                        Ask in <strong>English, Hindi, Hinglish, Russian</strong> or any other language.<br>
                        <em>The assistant responds in your question's language, strictly grounded in transcript evidence.</em>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Process pending query: display complete response at once (no typewriter effect)
            if st.session_state.get("pending_query"):
                query_to_process = st.session_state.pending_query
                st.session_state.pending_query = None
                with st.spinner("Analyzing transcript evidence and generating grounded response..."):
                    prior_history = st.session_state.chat_history
                    if (prior_history and prior_history[-1].get("role") == "user"
                            and prior_history[-1].get("content") == query_to_process):
                        prior_history = prior_history[:-1]
                    full_answer = ask_question(
                        res["rag_chain"], query_to_process,
                        output_language=response_language,
                        conversation_history=prior_history,
                    )
                st.session_state.chat_history.append({"role": "assistant", "content": full_answer})
                st.rerun()

            # Chat Input Form (Enter to submit, clears automatically, no duplicate messages)
            user_query = st.chat_input("Ask questions in your preferred language...")
            if user_query and user_query.strip():
                clean_query = user_query.strip()
                st.session_state.chat_history.append({"role": "user", "content": clean_query})
                st.session_state.pending_query = clean_query
                st.rerun()

            if st.session_state.chat_history:
                if st.button("🗑️ Clear Chat History", type="secondary"):
                    st.session_state.chat_history = []
                    st.session_state.pending_query = None
                    st.rerun()

else:
    # ── Empty State ─────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;padding:5rem 2rem;text-align:center">
        <div style="font-size:3.8rem;margin-bottom:1rem">🎬</div>
        <div style="font-size:1.8rem;font-weight:800;color:var(--text-primary);margin-bottom:0.6rem;letter-spacing:-0.02em">
            Ready to Analyze Media
        </div>
        <div style="color:var(--text-secondary);font-size:0.92rem;max-width:520px;line-height:1.75;margin-bottom:2.2rem">
            Paste a YouTube URL or upload a local video/audio recording in the sidebar to transcribe, extract structured intelligence, and chat with <strong>100% local, privacy-first AI</strong>.
        </div>
        <div style="display:flex;gap:0.75rem;flex-wrap:wrap;justify-content:center">
            <span class="badge badge-purple">⚡ Local Whisper STT</span>
            <span class="badge badge-cyan">🤖 Local Ollama (Qwen3:8B)</span>
            <span class="badge badge-green">🧠 FAISS Cross-Lingual RAG</span>
            <span class="badge badge-green">🔒 100% Private / No Cloud APIs</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
