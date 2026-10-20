import os
import re
import json
import time
import base64
import shutil
import logging
import tempfile
import subprocess
from typing import List, Dict, Any, Optional
from PIL import Image
import numpy as np

from core.local_llm import get_ollama_root_url, check_ollama, _ollama_session
from core.languages import get_language, detect_language_from_text, is_rtl

logger = logging.getLogger(__name__)

VISION_MODEL = os.getenv("VISION_MODEL", "qwen3-vl:2b")
DEFAULT_MAX_FRAMES = int(os.getenv("MAX_VISION_FRAMES", "20"))


def check_vision_model_available(model_name: str = VISION_MODEL) -> bool:
    """Check if the local vision model is available in Ollama."""
    if not check_ollama():
        return False
    try:
        root_url = get_ollama_root_url()
        resp = _ollama_session.get(f"{root_url}/api/tags", timeout=3.0)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            target = model_name.strip().lower()
            for m in models:
                name = m.get("name", "").lower()
                if name == target or name.split(":")[0] == target.split(":")[0]:
                    return True
    except Exception as e:
        logger.debug(f"Error checking vision model availability: {e}")
    return False


def is_video_file(file_path: str) -> bool:
    """Detect whether a file has a video stream (vs audio-only)."""
    if not os.path.exists(file_path):
        return False
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=codec_type",
            "-of", "default=noprint_wrappers=1:nokey=1",
            file_path,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return "video" in res.stdout.lower()
    except Exception as e:
        logger.warning(f"Failed to inspect video stream in {file_path}: {e}")
        return False


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds via ffprobe."""
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        dur = float(res.stdout.strip().split("\n")[-1])
        return max(0.0, dur)
    except Exception as e:
        logger.warning(f"Error getting duration for {video_path}: {e}")
        return 0.0


def format_timestamp(seconds: float) -> str:
    """Convert float seconds to HH:MM:SS format."""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"


def detect_visual_events(
    video_path: str,
    fps: float = 1.0,
    max_keyframes: int = DEFAULT_MAX_FRAMES,
) -> tuple[List[Dict[str, Any]], int, int]:
    """
    Stage 1: Cheap Low-Cost Change Detection.
    Evaluates candidate frames across video (default 1 fps) via streaming grayscale thumbnails.
    Detects genuine visual events (slide changes, window/app switches, screen appearance/disappearance,
    UI transitions, camera cuts, person entrance/exit).
    Returns (keyframes_meta, candidate_frames_count, skipped_duplicates_count).
    """
    t0 = time.time()
    dur = get_video_duration(video_path)
    if dur <= 0.5:
        return [], 0, 0

    # Adjust sampling fps for very long videos if needed (e.g. > 30 min)
    sample_fps = fps
    if dur > 1800:
        sample_fps = 0.5

    w, h = 160, 90
    frame_bytes = w * h
    cmd = [
        "ffmpeg", "-nostdin", "-i", video_path,
        "-vf", f"fps={sample_fps},scale={w}:{h}",
        "-f", "rawvideo", "-pix_fmt", "gray",
        "-"
    ]

    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    except Exception as e:
        logger.error(f"Failed to launch ffmpeg candidate frame stream: {e}")
        return [], 0, 0

    candidate_frames = 0
    prev_frame = None
    diffs = []

    while True:
        raw = proc.stdout.read(frame_bytes)
        if len(raw) < frame_bytes:
            break
        arr = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)

        if prev_frame is not None:
            # 1. Global mean absolute difference
            g_diff = float(np.mean(np.abs(arr - prev_frame)) / 255.0)

            # 2. 4x4 grid block difference to detect localized UI/slide changes
            grid_h, grid_w = h // 4, w // 4
            curr_grid = arr.reshape(h, w)
            prev_grid = prev_frame.reshape(h, w)
            block_diffs = []
            for r in range(4):
                for c in range(4):
                    b_curr = curr_grid[r * grid_h:(r + 1) * grid_h, c * grid_w:(c + 1) * grid_w]
                    b_prev = prev_grid[r * grid_h:(r + 1) * grid_h, c * grid_w:(c + 1) * grid_w]
                    block_diffs.append(np.mean(np.abs(b_curr - b_prev)) / 255.0)
            max_b_diff = float(max(block_diffs))

            # Composite event score: combination of global scene change and regional UI shift
            score = max(g_diff, max_b_diff * 0.7)
            diffs.append((candidate_frames, candidate_frames / sample_fps, score, g_diff, max_b_diff))
        else:
            diffs.append((candidate_frames, 0.0, 0.0, 0.0, 0.0))

        prev_frame = arr
        candidate_frames += 1

    proc.wait()
    t_eval = time.time() - t0
    logger.info(f"Stage 1 candidate evaluation: {candidate_frames} candidate frames evaluated in {t_eval:.2f}s.")

    if candidate_frames == 0:
        return [], 0, 0

    # Event Selection:
    # 1. Always select the initial frame (index 0)
    selected_indices = [0]
    last_selected_ts = 0.0
    min_event_gap = 2.0  # minimum seconds between distinct events
    CHANGE_THRESHOLD = 0.032  # 3.2% perceptual change represents genuine slide/UI/scene shift

    i = 1
    while i < len(diffs):
        idx, ts, score, g_diff, b_diff = diffs[i]
        if score >= CHANGE_THRESHOLD:
            if (ts - last_selected_ts) < min_event_gap:
                i += 1
                continue

            # Lookahead to settle post-transition frame (e.g. avoid intermediate fade/motion blur)
            best_idx = idx
            best_ts = ts
            j = i + 1
            while j < min(len(diffs), i + 3):
                _, next_ts, next_score, _, _ = diffs[j]
                if next_score >= CHANGE_THRESHOLD:
                    best_idx = diffs[j][0]
                    best_ts = next_ts
                    j += 1
                else:
                    break

            selected_indices.append(best_idx)
            last_selected_ts = best_ts
            i = max(i + 1, j)
        else:
            i += 1

    # If keyframe count exceeds max_keyframes, retain highest-scoring transitions preserving temporal order
    if len(selected_indices) > max_keyframes:
        scored = [(idx, diffs[idx][2] if idx < len(diffs) else 1.0) for idx in selected_indices[1:]]
        scored.sort(key=lambda x: x[1], reverse=True)
        top_set = set([0] + [x[0] for x in scored[:max_keyframes - 1]])
        selected_indices = [idx for idx in selected_indices if idx in top_set]

    # Build keyframes metadata list with time spans
    keyframes_meta = []
    for k_order, idx in enumerate(selected_indices):
        ts_sec = round(idx / sample_fps, 1)
        if ts_sec < 0.5 and dur > 1.0:
            ts_sec = min(1.0, round(dur / 2.0, 1)) if dur < 2.0 else 1.0

        if k_order + 1 < len(selected_indices):
            next_ts = round(selected_indices[k_order + 1] / sample_fps, 1)
        else:
            next_ts = dur

        keyframes_meta.append({
            "index": k_order,
            "candidate_index": idx,
            "timestamp_seconds": ts_sec,
            "timestamp_str": format_timestamp(ts_sec),
            "end_seconds": next_ts,
            "end_timestamp_str": format_timestamp(next_ts),
            "score": diffs[idx][2] if idx < len(diffs) else 0.0,
        })

    skipped_count = candidate_frames - len(keyframes_meta)
    return keyframes_meta, candidate_frames, skipped_count


def sample_representative_frames(
    video_path: str,
    max_frames: int = DEFAULT_MAX_FRAMES,
    output_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Extract event-based representative keyframes from video based on meaningful visual changes.
    Returns list of frame metadata dicts:
    [{"index": i, "timestamp_seconds": float, "timestamp_str": "HH:MM:SS", "path": str, "is_key": bool, ...}]
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if not is_video_file(video_path):
        logger.info(f"File {video_path} does not contain a video stream (audio-only). Skipping visual extraction.")
        return []

    target_dir = output_dir or tempfile.mkdtemp(prefix="video_frames_")
    os.makedirs(target_dir, exist_ok=True)

    keyframes_meta, candidate_count, skipped_count = detect_visual_events(
        video_path, fps=1.0, max_keyframes=max_frames
    )

    if not keyframes_meta:
        logger.warning(f"No keyframe events detected in {video_path}.")
        return []

    # Print Stage 1 event detection summary
    print("\n" + "=" * 50)
    print(f"Candidate frames: {candidate_count}")
    print(f"Selected keyframes: {len(keyframes_meta)}")
    print(f"Skipped duplicates: {skipped_count}")
    print("Keyframe timestamps:")
    for kf in keyframes_meta:
        print(f"  {kf['timestamp_str']} (active until {kf['end_timestamp_str']})")
    print("=" * 50 + "\n")

    logger.info(
        f"Event Sampling: {candidate_count} candidates evaluated -> {len(keyframes_meta)} keyframes selected ({skipped_count} skipped duplicates)."
    )

    # Extract high-quality JPEG images ONLY for the selected event keyframes
    extracted_frames = []
    t0 = time.time()

    for kf in keyframes_meta:
        ts = kf["timestamp_seconds"]
        frame_name = f"keyframe_{kf['index']:03d}_{int(ts)}s.jpg"
        frame_path = os.path.join(target_dir, frame_name)

        cmd = [
            "ffmpeg", "-y", "-nostdin",
            "-ss", f"{ts:.2f}",
            "-i", video_path,
            "-frames:v", "1",
            "-q:v", "2",
            frame_path,
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if os.path.exists(frame_path) and os.path.getsize(frame_path) > 100:
            extracted_frames.append({
                "index": kf["index"],
                "candidate_index": kf["candidate_index"],
                "timestamp_seconds": ts,
                "timestamp_str": kf["timestamp_str"],
                "end_seconds": kf["end_seconds"],
                "end_timestamp_str": kf["end_timestamp_str"],
                "path": frame_path,
                "is_key": True,
                "candidate_count": candidate_count,
                "skipped_count": skipped_count,
            })

    t_ext = time.time() - t0
    logger.info(f"Extracted {len(extracted_frames)} keyframe images in {t_ext:.2f}s.")
    return extracted_frames


def parse_vision_structured_output(content: str, timestamp_str: str) -> Dict[str, Any]:
    """
    Parse structured Qwen3-VL output into normalized fields:
    SCENE, PEOPLE, ACTIONS, OBJECTS, SCREEN/UI, SLIDE/TITLE, VISIBLE TEXT, CHARTS, DIAGRAMS, IMPORTANT DETAILS.
    """
    clean_text = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()

    keys = [
        ("scene", "SCENE"),
        ("people", "PEOPLE"),
        ("actions", "ACTIONS"),
        ("objects", "OBJECTS"),
        ("screen_ui", "SCREEN/UI"),
        ("slide_title", "SLIDE/TITLE"),
        ("visible_text", "VISIBLE TEXT"),
        ("charts", "CHARTS"),
        ("diagrams", "DIAGRAMS"),
        ("important_details", "IMPORTANT VISUAL DETAILS"),
    ]

    parsed = {}
    for i, (dict_key, header) in enumerate(keys):
        next_headers = "|".join([re.escape(h) for _, h in keys[i + 1:]])
        if next_headers:
            pat = rf'(?:^|\n)\s*{re.escape(header)}\s*:\s*(.*?)(?=\n\s*(?:{next_headers})\s*:|$)'
        else:
            pat = rf'(?:^|\n)\s*{re.escape(header)}\s*:\s*(.*?)$'

        m = re.search(pat, clean_text, flags=re.DOTALL | re.IGNORECASE)
        if m:
            val = m.group(1).strip()
            val = re.sub(r'^[\[\(\-]\s*', '', val)
            val = re.sub(r'\s*[\]\)]$', '', val)
            parsed[dict_key] = val.strip()
        else:
            parsed[dict_key] = "None"

    # Fallback if unformatted text was returned
    if parsed.get("screen_ui") == "None" and parsed.get("scene") == "None":
        parsed["scene"] = clean_text[:200].strip()
        parsed["screen_ui"] = clean_text.strip()

    # Extract people count accurately
    people_str = parsed.get("people", "")
    p_count = 0
    m_p = re.search(
        r'^\s*(\d+)\s*(?:\(\s*)?(?:(?:a|an)\s+)?(?:people|persons?|participants?|men|women|man|woman|boys?|girls?)\b',
        people_str,
        re.IGNORECASE,
    )
    if m_p:
        try:
            p_count = int(m_p.group(1))
        except ValueError:
            p_count = 0
    elif any(w in people_str.lower() for w in ["one person", "1 person", "one participant", "1 participant", "single presenter"]):
        p_count = 1
    elif any(w in people_str.lower() for w in ["none", "no people", "0 people", "nobody", "no participants"]):
        p_count = 0

    parsed["people_count"] = p_count
    return parsed


def _vision_language_instruction(language: Optional[str]) -> str:
    """Build a language instruction for vision analysis outputs."""
    lang = get_language(language) if language else None
    if lang is not None:
        script = lang.scripts[0] if lang.scripts else "the original script"
        rtl_note = ""
        if lang.rtl:
            rtl_note = (
                f" {lang.name} is right-to-left (RTL); preserve the natural reading order."
            )
        return (
            f" Respond in {lang.native_name} ({lang.name}) using the {script} script."
            f"{rtl_note}"
        )
    return ""


def analyze_frame_with_vision_model(
    image_path: str,
    timestamp_str: str = "00:00:00",
    model_name: str = VISION_MODEL,
    timeout: int = 60,
    language: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run local Ollama vision model on a single key frame to extract structured visual evidence.
    """
    if not os.path.exists(image_path):
        return {}

    with open(image_path, "rb") as img_file:
        b64_image = base64.b64encode(img_file.read()).decode("utf-8")

    lang_instruction = _vision_language_instruction(language)
    prompt = (
        f"Analyze this video frame at [{timestamp_str}] and provide structured visual information in this exact format:\n"
        "SCENE: [Brief summary of the scene and setting]\n"
        "PEOPLE: [Count and description of visible people/presenters, or 'None']\n"
        "ACTIONS: [What people or UI elements are actively doing, or 'None']\n"
        "OBJECTS: [Visible physical objects or hardware, or 'None']\n"
        "SCREEN/UI: [Detailed description of screen, application window, presentation slide, or browser display]\n"
        "SLIDE/TITLE: [Slide title, headline, document header, or 'None']\n"
        "VISIBLE TEXT: [Legible headings, bullet points, code, labels, or key text]\n"
        "CHARTS: [Specific charts/graphs visible (e.g. bar chart, line chart, pie chart), or 'None']\n"
        "DIAGRAMS: [Diagrams, flowcharts, architecture diagrams, or 'None']\n"
        "IMPORTANT VISUAL DETAILS: [Any distinctive UI elements, tables, colors, indicators, or visual highlights]\n"
        "Do NOT invent or guess any unseen details."
        f"{lang_instruction}"
    )

    root_url = get_ollama_root_url()
    endpoint = f"{root_url}/api/chat"

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [b64_image],
            }
        ],
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 700,
        },
    }

    try:
        resp = _ollama_session.post(endpoint, json=payload, timeout=timeout)
        if resp.status_code != 200:
            logger.warning(f"Ollama vision inference returned HTTP {resp.status_code}: {resp.text}")
            return {}

        data = resp.json()
        content = (data.get("message", {}).get("content") or "").strip()
        thinking = (data.get("message", {}).get("thinking") or "").strip()

        if not content and thinking:
            content = thinking

        if not content:
            return {}

        parsed = parse_vision_structured_output(content, timestamp_str)
        parsed["raw"] = content
        return parsed
    except Exception as e:
        logger.error(f"Failed to analyze frame {image_path}: {e}")
        return {}


def analyze_video_visuals(
    video_path: str,
    max_frames: int = DEFAULT_MAX_FRAMES,
    progress_callback = None,
    language: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Complete local two-stage visual evidence extraction pipeline:
    1. Stage 1: Cheap candidate frame evaluation (1 fps) & event change detection
    2. Stage 2: Run local vision model (qwen3-vl:2b) on selected keyframes
    3. Return timestamped, structured visual evidence collection
    """
    if not is_video_file(video_path):
        logger.info(f"'{video_path}' is an audio-only file. Skipping visual analysis.")
        return []

    if not check_vision_model_available():
        logger.warning(f"Local vision model '{VISION_MODEL}' is not ready in Ollama. Skipping visual extraction.")
        return []

    frames = sample_representative_frames(video_path, max_frames=max_frames)
    if not frames:
        return []

    total_key = len(frames)
    logger.info(f"Starting Stage 2 vision model inference on {total_key} keyframes...")

    full_evidence: List[Dict[str, Any]] = []

    for i, kf in enumerate(frames):
        ts_str = kf["timestamp_str"]
        end_ts_str = kf.get("end_timestamp_str", ts_str)
        span_str = f"{ts_str} to {end_ts_str}" if end_ts_str != ts_str else ts_str

        msg = f"Analyzing video event {i + 1}/{total_key} at [{ts_str}]..."
        logger.info(msg)
        if progress_callback:
            progress_callback(msg, i + 1, total_key)

        try:
            t0 = time.time()
            data = analyze_frame_with_vision_model(kf["path"], timestamp_str=ts_str, language=language)
            dt = time.time() - t0
            logger.info(f"Keyframe {kf['index']} [{ts_str}] analyzed in {dt:.2f}s (people={data.get('people_count', 0)})")
        except Exception as e:
            logger.error(f"Error analyzing keyframe {kf['index']} at [{ts_str}]: {e}")
            data = {}

        # Failed inference is not visual evidence and must not enter the index.
        if not data:
            continue

        scene = data.get("scene", "Video scene")
        screen_ui = data.get("screen_ui", "None")
        slide_title = data.get("slide_title", "None")
        charts = data.get("charts", "None")
        diagrams = data.get("diagrams", "None")
        visible_text = data.get("visible_text", "None")
        people = data.get("people", "None")
        actions = data.get("actions", "None")
        objects = data.get("objects", "None")
        details = data.get("important_details", "None")
        people_count = data.get("people_count", 0)

        # Dense, structured summary text for vector indexing and LLM prompt context
        summary_text = (
            f"[VISUAL EVIDENCE at {ts_str} (Screen active {span_str})]\n"
            f"Scene: {scene}\n"
            f"Screen/UI: {screen_ui}\n"
            f"Slide/Title: {slide_title}\n"
            f"Charts: {charts}\n"
            f"Diagrams: {diagrams}\n"
            f"Visible Text: {visible_text}\n"
            f"People: {people}\n"
            f"Actions: {actions}\n"
            f"Objects: {objects}\n"
            f"Important Visual Details: {details}"
        )

        full_evidence.append({
            "frame_index": kf["index"],
            "timestamp": ts_str,
            "timestamp_display": span_str,
            "timestamp_seconds": kf["timestamp_seconds"],
            "end_seconds": kf.get("end_seconds", kf["timestamp_seconds"]),
            "scene": scene,
            "screen_ui": screen_ui,
            "slide_title": slide_title,
            "charts": charts,
            "diagrams": diagrams,
            "visible_text": visible_text,
            "people": people,
            "people_count": people_count,
            "actions": actions,
            "objects": objects,
            "important_details": details,
            "description": summary_text,
            "summary_text": summary_text,
            "is_key": True,
            # The extracted JPEG is deleted below; retain a stable source label
            # instead of returning a path that no longer exists.
            "source_file": os.path.basename(video_path),
            "raw": data,
        })

    # Clean up temporary frame image files to preserve privacy
    try:
        if frames:
            parent_dir = os.path.dirname(frames[0]["path"])
            if "video_frames_" in parent_dir and os.path.exists(parent_dir):
                shutil.rmtree(parent_dir, ignore_errors=True)
    except Exception as e:
        logger.debug(f"Could not remove temp frame directory: {e}")

    # Stage 2 visual analysis complete: free Python memory and explicitly unload vision model
    logger.info("Visual analysis complete. Releasing vision model from Ollama...")
    from core.local_llm import unload_model, log_memory_state
    log_memory_state("MEMORY BEFORE UNLOADING VISION MODEL")
    unload_model(VISION_MODEL)
    log_memory_state("MEMORY AFTER UNLOADING VISION MODEL")

    logger.info(f"Visual analysis complete: {len(full_evidence)} structured evidence points extracted.")
    return full_evidence
