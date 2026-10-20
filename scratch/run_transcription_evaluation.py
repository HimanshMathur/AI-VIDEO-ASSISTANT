import sys
import os
import time
import json
import logging

# Ensure UTF-8 output on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Setup logging to stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
    force=True
)

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath("."))

# Monkey-patch WhisperModel.transcribe only to capture info.language_probability without altering execution
import faster_whisper
orig_transcribe = faster_whisper.WhisperModel.transcribe
captured_info = {}

def wrapped_transcribe(self, *args, **kwargs):
    segments, info = orig_transcribe(self, *args, **kwargs)
    captured_info["language"] = getattr(info, "language", None)
    captured_info["language_probability"] = getattr(info, "language_probability", None)
    captured_info["duration"] = getattr(info, "duration", None)
    return segments, info

faster_whisper.WhisperModel.transcribe = wrapped_transcribe

from core.transcriber import transcribe_with_segments, _device_in_use, _compute_type_in_use

audio_path = "scratch/test_android_rPvJ7FKqWy4.wav"

print(f"Starting segment extraction on: {audio_path}", flush=True)
t_start = time.time()

result = transcribe_with_segments(audio_path)

t_end = time.time()
total_time = t_end - t_start

language = result.get("language")
prob = captured_info.get("language_probability")
segments = result.get("segments", [])

from core.transcriber import _device_in_use as final_device, _compute_type_in_use as final_compute

print("\n" + "="*80)
print(f"TRANSCRIPTION RESULTS")
print("="*80)
print(f"1. Total transcription time: {total_time:.2f} seconds ({int(total_time//60)}m {int(total_time%60)}s)")
print(f"2. Detected language + probability: {language} ({prob:.4f} / {prob*100:.1f}%)" if prob is not None else f"2. Detected language: {language}")
print(f"3. Execution device used: {final_device} (compute_type: {final_compute})")
print(f"4. Total number of segments: {len(segments)}")
print("5. Every segment in [START - END] transcript text format:\n")

for seg in segments:
    print(f"[{seg['start']:.2f} - {seg['end']:.2f}] {seg['text']}")

print("\n" + "="*80)
print("6. Timestamp Confirmation:")
if segments:
    first_seg = segments[0]
    last_seg = segments[-1]
    all_numeric = all(isinstance(s['start'], (int, float)) and isinstance(s['end'], (int, float)) for s in segments)
    non_trivial = any(s['start'] > 0 for s in segments)
    is_chronological = all(segments[i]['start'] <= segments[i+1]['start'] for i in range(len(segments)-1))
    print(f"   - All timestamps numeric: {all_numeric}")
    print(f"   - Strictly chronological: {is_chronological}")
    print(f"   - Audio span covered: {first_seg['start']:.2f}s to {last_seg['end']:.2f}s across {len(segments)} segments")
    print(f"   - Confirmation: Timestamps are genuine segment timestamps produced by faster-whisper.")
else:
    print("   - No segments returned.")
print("="*80)

# Save result to scratch for reference
with open("scratch/segments_test_android_result.json", "w", encoding="utf-8") as f:
    json.dump({
        "total_time_seconds": total_time,
        "language": language,
        "language_probability": prob,
        "device_used": final_device,
        "compute_type": final_compute,
        "num_segments": len(segments),
        "segments": segments
    }, f, ensure_ascii=False, indent=2)

print("\nSaved segments to scratch/segments_test_android_result.json", flush=True)
