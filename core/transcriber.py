import os
import re
import sys
import time
import wave
import logging
import threading
import unicodedata
import numpy as np
from dotenv import load_dotenv

from core.languages import get_whisper_code, language_display_name

load_dotenv()

logger = logging.getLogger(__name__)

# Mutex to ensure Whisper model inference is strictly thread-safe
_whisper_lock = threading.Lock()

# Model selection: honor .env; default large-v3 only when unset
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3")

# Global model cache and engine flag
_model = None
_is_faster_whisper = False
_device_in_use = "unknown"
_compute_type_in_use = "unknown"

# Overlapping decode windows (seconds). Hop = window - overlap.
# Whisper's encoder is 30s; overlapping recovers speech truncated at window edges
# without being specific to any language, speaker, or video.
_WINDOW_SECONDS = 24.0
_OVERLAP_SECONDS = 6.0
_MIN_WINDOW_SECONDS = 0.35
_MAX_SEGMENT_DURATION = 32.0

# ── Anti-hallucination / machine artifact cleanup ─────────────────────────────
# Single character repeated 7+ times (e.g. "ँँँँँँँ")
_REPEAT_CHAR_RE = re.compile(r"(.)\1{6,}", re.UNICODE)
# Short grapheme cycle (1–3 chars) repeated 6+ extra times, e.g. "गागागागागागा"
_REPEAT_CYCLE_RE = re.compile(r"(.{1,3})\1{6,}", re.UNICODE)
# Same token/short phrase 4+ times in a row. Avoids \b so Devanagari matches.
_REPEAT_PHRASE_RE = re.compile(
    r"(\S+(?:\s+\S+){0,3})(?:\s+\1){3,}",
    re.UNICODE,
)

_UNICODE_REPLACEMENT = "\ufffd"

_DECODE_KWARGS = {
    "task": "transcribe",
    "beam_size": 5,
    "vad_filter": True,
    "vad_parameters": {
        "min_silence_duration_ms": 400,
        "speech_pad_ms": 400,
    },
    "temperature": 0.0,
    "condition_on_previous_text": False,
    "no_repeat_ngram_size": 3,
    "repetition_penalty": 1.2,
    "without_timestamps": False,
    "compression_ratio_threshold": 2.0,
    "log_prob_threshold": -1.0,
    "no_speech_threshold": 0.6,
}


class TranscriptionError(Exception):
    """Raised when transcription fails or yields no speech."""
    pass


def _is_oom_error(err_msg: str) -> bool:
    """Return True when an error string indicates CUDA out-of-memory."""
    lower = err_msg.lower()
    return (
        "out of memory" in lower
        or "cudaerror" in lower
        or ("cuda" in lower and "memory" in lower)
    )


def _init_cuda_paths():
    """
    Ensure NVIDIA CUDA runtime DLLs (cublas, cudnn) installed in Python site-packages
    or system directories are discoverable on Windows.
    """
    if sys.platform != "win32":
        return

    added = []
    # Search site-packages for nvidia cublas/cudnn bin directories
    for p in sys.path:
        if "site-packages" in p:
            for root, dirs, files in os.walk(p):
                if any(f.lower().endswith(".dll") and ("cublas" in f.lower() or "cudnn" in f.lower()) for f in files):
                    if root not in added:
                        added.append(root)
                        if hasattr(os, "add_dll_directory"):
                            try:
                                os.add_dll_directory(root)
                            except Exception:
                                pass
                        if root not in os.environ.get("PATH", ""):
                            os.environ["PATH"] = root + os.pathsep + os.environ.get("PATH", "")

    # Also check standard CUDA toolkit path if set
    cuda_path = os.environ.get("CUDA_PATH", "")
    if cuda_path and os.path.exists(cuda_path):
        cuda_bin = os.path.join(cuda_path, "bin")
        if os.path.exists(cuda_bin) and cuda_bin not in added:
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(cuda_bin)
                except Exception:
                    pass
            os.environ["PATH"] = cuda_bin + os.pathsep + os.environ.get("PATH", "")


def _select_model_name() -> str:
    """Return the Whisper model name to use (default: large-v3)."""
    env_val = os.getenv("WHISPER_MODEL", "").strip()
    return env_val if env_val else "large-v3"


def load_model():
    """
    Load local faster-whisper model singleton with thread safety.
    Preferred: CUDA + large-v3 + float16.
    Fallback: CPU + large-v3 + int8 (or auto) when CUDA is unavailable.
    Zero paid/cloud fallbacks.
    """
    global _model, _is_faster_whisper, _device_in_use, _compute_type_in_use
    with _whisper_lock:
        if _model is not None:
            return _model

        _init_cuda_paths()
        model_name = _select_model_name()

        # Check CUDA capability via ctranslate2
        has_cuda = False
        try:
            import ctranslate2
            has_cuda = ctranslate2.get_cuda_device_count() > 0
        except Exception:
            try:
                import torch
                has_cuda = torch.cuda.is_available()
            except Exception:
                has_cuda = False

        from faster_whisper import WhisperModel

        # Large models exceed ~3GB in float16 and can OOM on 4GB-class GPUs.
        # int8_float16 keeps weights in int8 (~1.5GB for large-v3) while running
        # compute in float16, fitting comfortably on 4GB VRAM with near-float16
        # accuracy. Smaller models use pure float16 for maximum speed.
        is_large = model_name.startswith("large")
        cuda_compute_type = "int8_float16" if is_large else "float16"

        if has_cuda:
            logger.info(
                f"Attempting to load faster-whisper '{model_name}' on CUDA "
                f"(compute_type={cuda_compute_type})..."
            )
            try:
                _model = WhisperModel(
                    model_name,
                    device="cuda",
                    compute_type=cuda_compute_type,
                )
                _is_faster_whisper = True
                _device_in_use = "cuda"
                _compute_type_in_use = cuda_compute_type
                logger.info(
                    f"faster-whisper '{model_name}' ({cuda_compute_type}) "
                    f"successfully loaded on CUDA."
                )
                return _model
            except Exception as cuda_err:
                logger.warning(
                    f"CUDA initialization failed for '{model_name}' ({cuda_err}). "
                    f"Falling back gracefully to local CPU inference."
                )

        # CPU fallback (int8 inference with multi-threading)
        cpu_threads = min(16, max(4, os.cpu_count() or 4))
        logger.info(
            f"Loading faster-whisper '{model_name}' on CPU (compute_type=int8, cpu_threads={cpu_threads})..."
        )
        try:
            _model = WhisperModel(
                model_name,
                device="cpu",
                compute_type="int8",
                cpu_threads=cpu_threads,
            )
            _is_faster_whisper = True
            _device_in_use = "cpu"
            _compute_type_in_use = "int8"
            logger.info(f"faster-whisper '{model_name}' (int8) successfully loaded on CPU.")
            return _model
        except Exception as cpu_err:
            error_msg = (
                f"Failed to load local Whisper model '{model_name}' on CPU: {cpu_err}. "
                "Ensure sufficient system RAM is available."
            )
            logger.error(error_msg)
            raise TranscriptionError(error_msg)


def _validate_chunk_for_whisper(chunk_path: str) -> bool:
    """Verify that a chunk file exists, is valid WAV, and has audible samples."""
    if not os.path.exists(chunk_path):
        logger.warning(f"Chunk file does not exist: {chunk_path}")
        return False

    if os.path.getsize(chunk_path) < 100:
        logger.warning(f"Chunk file is too small ({os.path.getsize(chunk_path)} bytes): {chunk_path}")
        return False

    try:
        with wave.open(chunk_path, "rb") as wf:
            n_frames = wf.getnframes()
            framerate = wf.getframerate()
            duration = n_frames / float(framerate) if framerate > 0 else 0.0

            if n_frames == 0 or duration < 0.2:
                logger.warning(f"Chunk is too short ({duration:.2f}s): {chunk_path}")
                return False

            raw_bytes = wf.readframes(min(n_frames, framerate * 2))
            samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32)
            if len(samples) == 0 or np.max(np.abs(samples)) == 0:
                logger.warning(f"Chunk contains complete silence: {chunk_path}")
                return False
    except Exception as e:
        logger.warning(f"Error inspecting chunk {chunk_path}: {e}")
        return False

    return True


def _normalize_language(language) -> str | None:
    """Map caller language to a faster-whisper language code, or None for auto-detect.

    Uses the central core.languages registry so every supported language
    (all 22 Indian Eighth-Schedule languages plus international ones) maps
    to its correct whisper token. Unknown values fall through to the raw
    lowercased string; faster-whisper will reject codes it does not know.
    """
    return get_whisper_code(language)


def _audio_duration_seconds(audio_path: str) -> float:
    try:
        with wave.open(audio_path, "rb") as wf:
            rate = wf.getframerate()
            frames = wf.getnframes()
            if rate > 0:
                return frames / float(rate)
    except Exception:
        pass
    return 0.0


def _iter_source_windows(duration: float) -> list:
    """
    Build overlapping [start, end) windows that cover [0, duration].
    Short audio is a single full-file window.
    """
    if duration <= 0:
        return []
    if duration <= _WINDOW_SECONDS + 0.05:
        return [(0.0, duration)]

    hop = _WINDOW_SECONDS - _OVERLAP_SECONDS
    windows = []
    start = 0.0
    while start < duration - 1e-6:
        end = min(start + _WINDOW_SECONDS, duration)
        if end - start >= _MIN_WINDOW_SECONDS:
            windows.append((start, end))
        if end >= duration - 1e-6:
            break
        start += hop
        if start >= duration:
            break
        # Guarantee the tail is covered even if hop would skip it
        if duration - start < _MIN_WINDOW_SECONDS and windows:
            windows[-1] = (windows[-1][0], duration)
            break
    if windows and abs(windows[-1][1] - duration) > 0.05:
        tail_start = max(0.0, duration - _WINDOW_SECONDS)
        if not windows or abs(windows[-1][0] - tail_start) > 0.05:
            windows.append((tail_start, duration))
        else:
            windows[-1] = (windows[-1][0], duration)
    return windows


def _clean_segment_text(text: str) -> str:
    """
    Lightweight cleanup of a single transcribed segment.
    Removes only obvious machine corruption / encoding artifacts.
    Does NOT rewrite the speaker's words or alter vocabulary.
    """
    if not text:
        return ""

    text = unicodedata.normalize("NFC", text)
    text = text.replace(_UNICODE_REPLACEMENT, "")
    text = "".join(
        ch for ch in text
        if unicodedata.category(ch)[0] != "C" or ch in "\t\n\r \u200c\u200d"
    )
    text = _REPEAT_CHAR_RE.sub(r"\1", text)
    text = _REPEAT_CYCLE_RE.sub(r"\1", text)
    text = re.sub(r"[ \t]+", " ", text).strip()
    # Strip trailing combining marks ONLY when they are orphaned (preceded by
    # whitespace or start-of-string). Devanagari/Gurmukhi vowel signs and
    # anusvara are Unicode category Mn/Mc and must be preserved when attached
    # to a base letter (e.g. "नहीं" must not become "नह").
    while text and unicodedata.category(text[-1]) in ("Mn", "Mc"):
        if len(text) == 1 or text[-2] in " \t\n\r":
            text = text[:-1].rstrip()
        else:
            break
    return text


def _token_repeat_ratio(text: str) -> float:
    tokens = [t for t in text.split() if t]
    if len(tokens) < 8:
        return 1.0
    return len(set(tokens)) / float(len(tokens))


def _is_segment_garbage(text: str) -> bool:
    """Return True if the segment text is clearly non-speech machine corruption."""
    if not text:
        return True
    cleaned = text.strip()
    if not cleaned:
        return True
    if _REPEAT_CHAR_RE.search(cleaned) or _REPEAT_CYCLE_RE.search(cleaned):
        return True
    if not any(unicodedata.category(ch)[0] in ("L", "N") for ch in cleaned):
        return True
    tokens = cleaned.split()
    if len(tokens) >= 8 and _token_repeat_ratio(cleaned) <= 0.15:
        return True
    return False


def _collapse_pathological_token_runs(text: str) -> str:
    """
    Collapse runs of the same whitespace-separated token repeated 5+ times in a row
    down to a single occurrence. Catches single-word hallucination loops (e.g.
    "अपनी अपनी अपनी ...") that the n-gram phrase regex does not fully reduce.
    Natural doublets/triplets (≤4 repeats) are preserved.
    """
    if not text:
        return text
    tokens = text.split()
    if len(tokens) < 5:
        return text
    out = []
    i = 0
    while i < len(tokens):
        out.append(tokens[i])
        j = i + 1
        while j < len(tokens) and tokens[j] == tokens[i]:
            j += 1
        run = j - i
        if run > 4:
            pass
        else:
            for k in range(i + 1, j):
                out.append(tokens[k])
        i = j
    return " ".join(out)


def _remove_consecutive_duplicate_phrases(text: str) -> str:
    """
    Remove pathological repetition loops (phrase-level 4+ repeats, character/cycle
    repeats, and long single-token runs). Natural doublets/triplets are preserved.
    """
    if not text:
        return text
    text = _REPEAT_PHRASE_RE.sub(lambda m: m.group(1), text)
    text = _collapse_pathological_token_runs(text)
    text = _REPEAT_CHAR_RE.sub(r"\1", text)
    text = _REPEAT_CYCLE_RE.sub(r"\1", text)
    text = _collapse_pathological_token_runs(text)
    text = re.sub(r"[ \t]+", " ", text).strip()
    return text


def _stitch_chunk_texts(parts: list) -> str:
    """
    Join per-chunk transcript texts into one transcript.
    Adjacent 10-minute chunks overlap by 6 seconds, so the same phrase may appear
    at the tail of one chunk and the head of the next. We strip and join with a
    single space; pathological repeats are removed downstream by
    _remove_consecutive_duplicate_phrases. Empty parts are skipped.
    """
    cleaned = [p.strip() for p in parts if p and p.strip()]
    if not cleaned:
        return ""
    return " ".join(cleaned)


def _drop_adjacent_duplicate_segments(segments: list) -> list:
    """
    Drop immediate decoding echoes (same text emitted back-to-back).
    Legitimate repeats later in the audio are kept.
    """
    kept = []
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        if kept and kept[-1]["text"].strip() == text:
            prev_end = kept[-1].get("end", 0.0)
            cur_start = seg.get("start", 0.0)
            if cur_start <= prev_end + 1.5:
                continue
        kept.append(seg)
    return kept


def _window_owned_range(windows: list, index: int) -> tuple:
    """Absolute time range this window is allowed to contribute after overlap merge."""
    start, end = windows[index]
    half = _OVERLAP_SECONDS / 2.0
    owned_start = start if index == 0 else start + half
    owned_end = end if index == len(windows) - 1 else end - half
    if owned_end < owned_start:
        owned_end = owned_start
    return owned_start, owned_end


def _segment_in_owned_range(seg: dict, owned_start: float, owned_end: float) -> bool:
    mid = (float(seg["start"]) + float(seg["end"])) / 2.0
    return owned_start - 1e-6 <= mid <= owned_end + 1e-6


def _finalize_segments(segments: list, duration: float) -> list:
    """Clamp timestamps, drop empty/zero-duration/out-of-range segments, then de-echo."""
    cleaned = []
    dur = max(0.0, float(duration or 0.0))
    for seg in segments:
        text = _clean_segment_text(seg.get("text") or "")
        if _is_segment_garbage(text):
            continue
        start = float(seg.get("start") or 0.0)
        end = float(seg.get("end") or 0.0)
        if dur > 0:
            start = min(max(0.0, start), dur)
            end = min(max(0.0, end), dur)
        if end < start:
            start, end = end, start
        if end - start < 0.02:
            continue
        if end - start > _MAX_SEGMENT_DURATION:
            continue
        cleaned.append({
            "start": round(start, 2),
            "end": round(end, 2),
            "text": text,
            "avg_logprob": round(float(seg.get("avg_logprob") or 0.0), 4),
            "window_start": round(float(seg.get("window_start", start)), 2),
            "window_end": round(float(seg.get("window_end", end)), 2),
        })
    cleaned.sort(key=lambda s: (s["start"], s["end"]))
    return _drop_adjacent_duplicate_segments(cleaned)


def _switch_to_cpu_model():
    """
    Safely release GPU model and switch global model singleton to CPU int8 fallback.
    If already on CPU, reuses the existing model singleton without reloading.
    """
    global _model, _is_faster_whisper, _device_in_use, _compute_type_in_use
    if _model is not None and _device_in_use == "cpu":
        return _model

    if _model is not None:
        try:
            del _model
        except Exception:
            pass
        _model = None
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    from faster_whisper import WhisperModel
    cpu_threads = min(16, max(4, os.cpu_count() or 4))
    model_name = _select_model_name()
    logger.info(
        f"Loading faster-whisper '{model_name}' on CPU fallback (compute_type=int8, cpu_threads={cpu_threads})..."
    )
    _model = WhisperModel(
        model_name,
        device="cpu",
        compute_type="int8",
        cpu_threads=cpu_threads,
    )
    _is_faster_whisper = True
    _device_in_use = "cpu"
    _compute_type_in_use = "int8"
    logger.info(f"faster-whisper '{model_name}' (int8) successfully loaded on CPU fallback.")
    return _model


def _collect_window_segments(
    model,
    audio_window,
    whisper_lang,
    window_start: float,
    window_end: float,
    source_duration: float,
    progress_callback=None,
    chunk_label: str = "",
) -> tuple:
    """
    Decode one source window. Timestamps are converted from local window time
    to absolute source time. Returns (segments, detected_lang, hit_oom).
    """
    segments_gen, info = model.transcribe(
        audio_window,
        language=whisper_lang,
        **_DECODE_KWARGS,
    )
    detected_lang = info.language if info else None
    collected = []
    last_end = window_start
    hit_oom = False

    try:
        for seg in segments_gen:
            raw_text = seg.text or ""
            local_start = float(getattr(seg, "start", 0.0) or 0.0)
            local_end = float(getattr(seg, "end", 0.0) or 0.0)
            abs_start = window_start + local_start
            abs_end = window_start + local_end
            if source_duration > 0:
                abs_start = min(max(0.0, abs_start), source_duration)
                abs_end = min(max(0.0, abs_end), source_duration)
            if abs_end > window_end + 0.05:
                abs_end = min(abs_end, window_end)
            collected.append({
                "start": abs_start,
                "end": abs_end,
                "text": raw_text,
                "avg_logprob": getattr(seg, "avg_logprob", 0.0),
                "window_start": window_start,
                "window_end": window_end,
            })
            last_end = abs_end
            if progress_callback:
                preview_src = _clean_segment_text(raw_text.strip()) or raw_text.strip()
                preview = preview_src[:55] + "..." if len(preview_src) > 55 else preview_src
                label = f"{chunk_label} " if chunk_label else ""
                progress_callback(f"{label}[{last_end:.0f}s] {preview}", None, None)
    except Exception as iter_err:
        err_msg = str(iter_err)
        if _is_oom_error(err_msg) and _device_in_use == "cuda":
            logger.warning(
                f"CUDA OOM mid-iteration in window [{window_start:.1f}-{window_end:.1f}]s "
                f"at ~{last_end:.1f}s. Partial window discarded. Will retry on CPU."
            )
            hit_oom = True
            collected = []
        else:
            logger.error(
                f"Unexpected segment-iteration error in window "
                f"[{window_start:.1f}-{window_end:.1f}]s: {iter_err}"
            )
            raise

    return collected, detected_lang, hit_oom


def _detect_language(model, audio: np.ndarray, whisper_lang, sample_rate: int = 16000) -> str | None:
    """Detect language once from a short prefix, then lock it for remaining windows."""
    if whisper_lang:
        return whisper_lang
    if audio is None or len(audio) == 0:
        return None
    probe_samples = min(len(audio), int(30 * sample_rate))
    probe = audio[:probe_samples]
    try:
        _segs, info = model.transcribe(
            probe,
            language=None,
            beam_size=1,
            vad_filter=True,
            temperature=0.0,
            condition_on_previous_text=False,
            without_timestamps=True,
        )
        # Consume generator so the language result is finalized.
        for _ in _segs:
            pass
        detected = info.language if info else None
        prob = getattr(info, "language_probability", None)
        logger.info(f"Locked decode language={detected!r} (p={prob})")
        return detected
    except Exception as e:
        logger.warning(f"Language probe failed ({e}); continuing with auto-detect per window.")
        return None


def _stream_segments_safe(model, audio_path: str, whisper_lang, progress_callback=None, chunk_label: str = "") -> tuple:
    """
    Overlapping-window decode with timestamp-aware merge.
    Returns (valid_texts, seg_metadata, detected_lang, hit_oom).
    If hit_oom is True the caller must retry on CPU with a fresh model.
    """
    from faster_whisper.audio import decode_audio

    sample_rate = 16000
    try:
        audio = decode_audio(audio_path, sampling_rate=sample_rate)
    except Exception as decode_err:
        logger.warning(f"decode_audio failed for '{audio_path}' ({decode_err}); using file path.")
        audio = None

    if audio is not None:
        source_duration = float(len(audio) / float(sample_rate))
    else:
        source_duration = _audio_duration_seconds(audio_path)

    windows = _iter_source_windows(source_duration if source_duration > 0 else _WINDOW_SECONDS)
    if not windows:
        return [], [], whisper_lang, False

    locked_lang = whisper_lang
    if audio is not None and locked_lang is None:
        locked_lang = _detect_language(model, audio, whisper_lang, sample_rate=sample_rate)

    raw_by_window = []
    detected_lang = locked_lang
    hit_oom = False

    for idx, (w_start, w_end) in enumerate(windows):
        if audio is not None:
            i0 = int(round(w_start * sample_rate))
            i1 = int(round(w_end * sample_rate))
            i0 = max(0, min(i0, len(audio)))
            i1 = max(i0, min(i1, len(audio)))
            if i1 - i0 < int(_MIN_WINDOW_SECONDS * sample_rate):
                continue
            window_audio = audio[i0:i1]
        else:
            window_audio = audio_path

        collected, win_lang, win_oom = _collect_window_segments(
            model,
            window_audio,
            locked_lang,
            window_start=w_start,
            window_end=w_end,
            source_duration=source_duration,
            progress_callback=progress_callback,
            chunk_label=chunk_label,
        )
        if win_oom:
            hit_oom = True
            return [], [], detected_lang, True
        if win_lang and not detected_lang:
            detected_lang = win_lang
            locked_lang = locked_lang or win_lang
        raw_by_window.append((idx, collected))

    merged = []
    for idx, collected in raw_by_window:
        owned_start, owned_end = _window_owned_range(windows, idx)
        for seg in collected:
            if _segment_in_owned_range(seg, owned_start, owned_end):
                merged.append(seg)

    finalized = _finalize_segments(merged, source_duration)
    valid_texts = [s["text"] for s in finalized]
    return valid_texts, finalized, detected_lang, hit_oom


def transcribe_chunk_whisper(chunk_path: str, language=None, progress_callback=None) -> tuple:
    """
    Transcribe one audio chunk locally using faster-whisper:
    - overlapping 24s windows with 6s overlap and timestamp-aware merge
    - beam_size=5, vad_filter=True, temperature=0.0
    - condition_on_previous_text=False (stops 30s prompt-loop hallucinations)
    - CUDA float16 when available, CPU int8 fallback on OOM (caught even mid-iteration)
    - Calls progress_callback(msg, None, None) per segment for live UI updates

    Returns (transcript_text, detected_language).
    """
    global _model, _device_in_use, _compute_type_in_use
    if not _validate_chunk_for_whisper(chunk_path):
        return "", None

    chunk_name = os.path.basename(chunk_path)
    logger.info(f"START transcription of {chunk_name}")
    t_start = time.time()

    model = load_model()
    whisper_lang = _normalize_language(language)

    acquired = _whisper_lock.acquire(timeout=600.0)
    if not acquired:
        logger.error(f"Whisper inference lock timed out after 600s on {chunk_name}")
        raise TranscriptionError("Transcription engine is currently busy. Please try again.")

    try:
        logger.info(
            f"Transcribing {chunk_name} [engine=faster-whisper, device={_device_in_use}, "
            f"compute={_compute_type_in_use}, window={_WINDOW_SECONDS}s overlap={_OVERLAP_SECONDS}s, "
            f"beam_size=5, vad_filter=True, condition_on_previous_text=False, lang={whisper_lang!r}]"
        )

        valid_texts, _seg_meta, detected, hit_oom = _stream_segments_safe(
            model, chunk_path, whisper_lang,
            progress_callback=progress_callback,
            chunk_label=chunk_name,
        )

        # CUDA OOM fired mid-iteration — switch to CPU and retry cleanly
        if hit_oom and _device_in_use == "cuda":
            logger.warning(f"Retrying '{chunk_name}' on CPU int8 after OOM...")
            if progress_callback:
                progress_callback(f"⚠️ GPU memory full — retrying on CPU for {chunk_name}...", None, None)
            cpu_model = _switch_to_cpu_model()
            valid_texts, _seg_meta, detected, _ = _stream_segments_safe(
                cpu_model, chunk_path, whisper_lang,
                progress_callback=progress_callback,
                chunk_label=chunk_name,
            )

        text = " ".join(valid_texts).strip()
        logger.info(
            f"[PERF] Transcription ({chunk_name}): {time.time() - t_start:.2f}s | "
            f"Segments: {len(valid_texts)} | Lang: {detected}"
        )
        return text, detected

    except Exception as e:
        err_msg = str(e)
        if _is_oom_error(err_msg) and _device_in_use == "cuda":
            logger.warning(
                f"CUDA GPU memory exhausted on '{chunk_name}' ({err_msg}). "
                "Switching gracefully to local CPU fallback (int8)..."
            )
            try:
                if progress_callback:
                    progress_callback("⚠️ GPU memory full — retrying on CPU...", None, None)
                cpu_model = _switch_to_cpu_model()
                valid_texts, _seg_meta, detected, _ = _stream_segments_safe(
                    cpu_model, chunk_path, whisper_lang,
                    progress_callback=progress_callback,
                    chunk_label=chunk_name,
                )
                text = " ".join(valid_texts).strip()
                logger.info(f"CPU fallback transcription succeeded ({len(valid_texts)} segments).")
                return text, detected
            except Exception as cpu_retry_err:
                logger.error(f"CPU fallback also failed on {chunk_path}: {cpu_retry_err}")
                return "", None

        logger.error(f"Whisper transcription error on chunk {chunk_path}: {e}")
        return "", None
    finally:
        _whisper_lock.release()


def transcribe_with_segments(audio_path: str, language=None, progress_callback=None) -> dict:
    """
    Transcribe audio and return full transcript along with structured segment metadata:
    - start, end, text, avg_logprob
    Streams segments (no blocking list(segments) call).
    Includes automatic CPU int8 fallback on CUDA OOM errors, even mid-iteration.
    """
    if not _validate_chunk_for_whisper(audio_path):
        return {"text": "", "language": None, "segments": []}

    model = load_model()
    whisper_lang = _normalize_language(language)

    acquired = _whisper_lock.acquire(timeout=600.0)
    if not acquired:
        raise TranscriptionError("Transcription engine is currently busy.")

    try:
        valid_texts, seg_metadata, detected, hit_oom = _stream_segments_safe(
            model, audio_path, whisper_lang,
            progress_callback=progress_callback,
            chunk_label=os.path.basename(audio_path),
        )

        if hit_oom and _device_in_use == "cuda":
            logger.warning("CUDA OOM during segment transcription — switching to CPU and retrying...")
            if progress_callback:
                progress_callback("⚠️ GPU memory full — retrying on CPU...", None, None)
            cpu_model = _switch_to_cpu_model()
            valid_texts, seg_metadata, detected, _ = _stream_segments_safe(
                cpu_model, audio_path, whisper_lang,
                progress_callback=progress_callback,
                chunk_label=os.path.basename(audio_path),
            )
            logger.info(f"CPU fallback segment transcription succeeded ({len(seg_metadata)} segments).")

        full_text = " ".join(valid_texts).strip()
        return {"text": full_text, "language": detected, "segments": seg_metadata}

    except Exception as e:
        err_msg = str(e)
        if _is_oom_error(err_msg) and _device_in_use == "cuda":
            logger.warning(
                f"CUDA OOM at transcribe() call for '{audio_path}': {err_msg}. Retrying on CPU..."
            )
            if progress_callback:
                progress_callback("⚠️ GPU memory full — retrying on CPU...", None, None)
            cpu_model = _switch_to_cpu_model()
            valid_texts, seg_metadata, detected, _ = _stream_segments_safe(
                cpu_model, audio_path, whisper_lang,
                progress_callback=progress_callback,
                chunk_label=os.path.basename(audio_path),
            )
            full_text = " ".join(valid_texts).strip()
            logger.info(f"CPU fallback segment transcription succeeded ({len(seg_metadata)} segments).")
            return {"text": full_text, "language": detected, "segments": seg_metadata}
        else:
            logger.error(f"Whisper segment transcription error on {audio_path}: {e}")
            raise
    finally:
        _whisper_lock.release()


def transcribe_all(chunks: list, language=None, progress_callback=None) -> tuple:
    """
    Transcribe all audio chunks 100% locally with faster-whisper large-v3.
    Preserves speaker language without translation or hallucination.
    Gracefully processes each chunk; if an individual chunk encounters an error,
    logs the failure and continues with remaining chunks.

    Returns a tuple (transcript_text, detected_language).
    detected_language is the most frequent language detected across chunks,
    or the caller-supplied language if one was provided, or None.
    """
    if not chunks:
        raise TranscriptionError("No audio chunks provided for transcription.")

    full_transcript_parts = []
    detected_languages = []
    chunk_errors = []

    logger.info(
        f"Transcribing {len(chunks)} chunk(s) using faster-whisper ({WHISPER_MODEL}) locally "
        f"with {_WINDOW_SECONDS:.0f}s/{_OVERLAP_SECONDS:.0f}s overlapping windows."
    )

    for i, chunk in enumerate(chunks):
        chunk_num = i + 1
        total = len(chunks)
        start_msg = f"Chunk {chunk_num}/{total}: starting..."
        logger.info(start_msg)
        if progress_callback:
            progress_callback(f"📝 {start_msg}", chunk_num, total)

        try:
            # Pass progress_callback so per-segment updates surface in the UI
            text, det_lang = transcribe_chunk_whisper(
                chunk, language=language, progress_callback=progress_callback
            )
            if det_lang:
                detected_languages.append(det_lang)
            if text:
                full_transcript_parts.append(text)
        except Exception as chunk_err:
            err_msg = f"Error transcribing chunk {chunk_num} ({chunk}): {chunk_err}"
            logger.error(err_msg)
            chunk_errors.append(err_msg)

        comp_msg = f"Chunk {chunk_num}/{total} completed."
        logger.info(comp_msg)
        if progress_callback:
            progress_callback(f"✅ {comp_msg}", chunk_num, total)

    if progress_callback:
        progress_callback("✅ Transcription complete.", len(chunks), len(chunks))

    combined_text = _stitch_chunk_texts(full_transcript_parts)
    final_text = _remove_consecutive_duplicate_phrases(combined_text)

    if not final_text:
        if chunk_errors:
            error_details = "; ".join(chunk_errors)
            raise TranscriptionError(f"Transcription failed for all chunks: {error_details}")
        raise TranscriptionError(
            "No speech was detected or recognized in this media. "
            "Please ensure the recording contains audible, clear speech."
        )

    # Determine the most likely detected language. If the caller supplied an
    # explicit language, prefer it; otherwise pick the most frequent detected
    # language across chunks. Falls back to None if nothing was detected.
    caller_lang = _normalize_language(language)
    if caller_lang:
        detected_language = caller_lang
    elif detected_languages:
        from collections import Counter
        detected_language = Counter(detected_languages).most_common(1)[0][0]
    else:
        detected_language = None

    logger.info(
        f"Transcription complete ({len(final_text)} characters). "
        f"Detected language(s): {set(detected_languages)} | locked={detected_language}"
    )
    return final_text, detected_language
