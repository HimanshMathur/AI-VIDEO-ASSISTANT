"""Baseline Hindi/English transcription using the live production path."""
import json
import os
import sys
import time
import wave

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, ".env"))

from core.transcriber import (
    WHISPER_MODEL,
    transcribe_all,
    transcribe_with_segments,
)
from core import transcriber
from utils.audio_processor import chunk_audio, validate_audio

CASES = [
    {
        "id": "story_gappu",
        "kind": "Hindi storytelling",
        "path": os.path.join(ROOT, "downloads", "yt_10807aa488.wav"),
    },
    {
        "id": "missing_night_hi",
        "kind": "Hindi narration/drama snippet",
        "path": os.path.join(ROOT, "scratch", "missing_night_30s.wav"),
    },
    {
        "id": "hinglish_chapters",
        "kind": "Hindi mixed with English technical terms",
        "path": os.path.join(ROOT, "scratch", "test_android_rPvJ7FKqWy4.wav"),
    },
    {
        "id": "english_sample",
        "kind": "English regression",
        "path": os.path.join(ROOT, "scratch", "speech_sample_16k.wav"),
    },
]


def wav_meta(path):
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        dur = wf.getnframes() / float(sr or 1)
    return {"duration": dur, "sample_rate": sr, "channels": ch}


def main():
    out_path = os.path.join(ROOT, "scratch", "hindi_baseline_results.json")
    results = []
    print(f"Configured WHISPER_MODEL={WHISPER_MODEL}", flush=True)
    for case in CASES:
        path = case["path"]
        print("\n" + "=" * 80, flush=True)
        print(f"{case['id']}  {case['kind']}", flush=True)
        print(path, flush=True)
        if not os.path.exists(path):
            print("MISSING FILE", flush=True)
            results.append({**case, "error": "missing"})
            continue
        meta = wav_meta(path)
        audio_meta = validate_audio(path)
        t0 = time.time()
        chunks = chunk_audio(path)
        prod_text = transcribe_all(chunks, language=None)
        t_all = time.time() - t0
        t1 = time.time()
        segs = transcribe_with_segments(path, language=None)
        t_seg = time.time() - t1
        rec = {
            **case,
            "audio": meta,
            "validate": {k: audio_meta[k] for k in ("duration", "sample_rate", "channels", "rms") if k in audio_meta},
            "num_chunks": len(chunks),
            "model": WHISPER_MODEL,
            "device": transcriber._device_in_use,
            "compute": transcriber._compute_type_in_use,
            "transcribe_all_seconds": round(t_all, 2),
            "transcribe_with_segments_seconds": round(t_seg, 2),
            "detected_language": segs.get("language"),
            "num_segments": len(segs.get("segments") or []),
            "segments": segs.get("segments") or [],
            "raw_joined_from_segments": segs.get("text") or "",
            "final_transcribe_all": prod_text,
        }
        # Truncation heuristic: 30s-aligned long segments with short last token
        trunc_flags = []
        for s in rec["segments"]:
            dur = round((s.get("end") or 0) - (s.get("start") or 0), 2)
            if dur >= 29.5:
                trunc_flags.append({"start": s["start"], "end": s["end"], "tail": (s.get("text") or "")[-20:]})
        rec["possible_30s_window_cuts"] = trunc_flags
        rec["replacement_char_count"] = (prod_text or "").count("\ufffd")
        results.append(rec)
        print(f"dur={meta['duration']:.1f}s sr={meta['sample_rate']} lang={rec['detected_language']} segs={rec['num_segments']} device={rec['device']}", flush=True)
        print(f"30s-cut candidates: {len(trunc_flags)}", flush=True)
        print("FINAL:", (prod_text or "")[:400], flush=True)

    payload = {
        "model": WHISPER_MODEL,
        "device": transcriber._device_in_use,
        "compute": transcriber._compute_type_in_use,
        "results": results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
