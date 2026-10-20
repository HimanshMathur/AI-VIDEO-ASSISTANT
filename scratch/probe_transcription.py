import os
import sys
import time
import json
import logging

# Force unbuffered UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
    force=True
)

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.transcriber import (
    load_model,
    _clean_segment_text,
    _is_segment_garbage,
    _remove_consecutive_duplicate_phrases,
    _switch_to_cpu_model,
    _device_in_use,
    _compute_type_in_use,
    WHISPER_MODEL,
)

AUDIO_FILE = os.path.abspath(r"downloads\yt_ca7e92bdc1.wav")

print("="*80, flush=True)
print(f"LIGHTWEIGHT REAL-TIME SEGMENT PROBE", flush=True)
print(f"Target Audio: {AUDIO_FILE}", flush=True)
print(f"File Size: {os.path.getsize(AUDIO_FILE)} bytes", flush=True)
print("="*80, flush=True)

t_start_total = time.time()

# 1. Load model with production settings
print(f"\n[1] Loading faster-whisper model ({WHISPER_MODEL})...", flush=True)
t0 = time.time()
try:
    model = load_model()
    t_load = time.time() - t0
    from core import transcriber
    device = transcriber._device_in_use
    compute = transcriber._compute_type_in_use
    print(f"Model loaded successfully in {t_load:.2f}s [Device: {device}, Compute: {compute}]", flush=True)
except Exception as e:
    print(f"Failed to load model: {e}", flush=True)
    sys.exit(1)

# 2. Run transcription with exact production parameters: beam_size=10, vad_filter=True
print(f"\n[2] Starting transcription probe with production parameters...", flush=True)
print(f"    - beam_size=10, vad_filter=True, task='transcribe', language=None", flush=True)

cuda_oom_occurred = False
segments_collected = []
detected_language = None
language_prob = None

def transcribe_and_stream(active_model, current_device):
    global detected_language, language_prob
    print(f"\n--- Transcribing on {current_device} ---", flush=True)
    t_infer_start = time.time()
    
    segments, info = active_model.transcribe(
        AUDIO_FILE,
        task="transcribe",
        language=None,
        beam_size=10,
        vad_filter=True,
    )
    
    detected_language = getattr(info, "language", None)
    language_prob = getattr(info, "language_probability", None)
    dur = getattr(info, "duration", None)
    prob_str = f"{language_prob:.4f}" if language_prob else "N/A"
    print(f"Detected Language: {detected_language} (confidence: {prob_str})", flush=True)
    print(f"Audio Duration: {dur:.2f}s" if dur else "Audio Duration: N/A", flush=True)
    print(f"\nSTREAMING SEGMENTS IN REAL-TIME [Start - End] (Raw vs Clean):", flush=True)
    print("-" * 80, flush=True)
    
    seg_idx = 0
    raw_texts = []
    clean_texts = []
    
    for seg in segments:
        seg_idx += 1
        raw_t = seg.text or ""
        clean_t = _clean_segment_text(raw_t.strip())
        is_garbage = _is_segment_garbage(clean_t)
        
        raw_texts.append(raw_t)
        if not is_garbage:
            clean_texts.append(clean_t)
            
        seg_entry = {
            "index": seg_idx,
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "raw_text": raw_t,
            "clean_text": clean_t,
            "is_garbage": is_garbage,
            "avg_logprob": round(getattr(seg, "avg_logprob", 0.0), 3),
            "no_speech_prob": round(getattr(seg, "no_speech_prob", 0.0), 3),
            "compression_ratio": round(getattr(seg, "compression_ratio", 0.0), 2),
        }
        segments_collected.append(seg_entry)
        
        garbage_tag = " [GARBAGE REMOVED]" if is_garbage else ""
        print(
            f"[{seg.start:6.2f}s -> {seg.end:6.2f}s] (logprob: {seg.avg_logprob:5.2f}, comp: {seg.compression_ratio:4.1f}){garbage_tag}\n"
            f"    RAW:   {raw_t}\n"
            f"    CLEAN: {clean_t}",
            flush=True
        )
        
    t_infer = time.time() - t_infer_start
    print("-" * 80, flush=True)
    print(f"Inference complete: {seg_idx} segments in {t_infer:.2f}s ({t_infer/max(1, seg_idx):.2f}s/seg)", flush=True)
    return raw_texts, clean_texts, t_infer

try:
    raw_texts, clean_texts, t_infer = transcribe_and_stream(model, device)
except Exception as e:
    err_str = str(e)
    print(f"\n[EXCEPTION DURING INFERENCE]: {e}", flush=True)
    if "out of memory" in err_str.lower() or "cuda" in err_str.lower():
        cuda_oom_occurred = True
        print("\nCUDA OOM DETECTED! Testing production CPU fallback (_switch_to_cpu_model)...", flush=True)
        t_fallback_start = time.time()
        cpu_model = _switch_to_cpu_model()
        from core import transcriber
        print(f"CPU model loaded in {time.time()-t_fallback_start:.2f}s [Device: {transcriber._device_in_use}, Compute: {transcriber._compute_type_in_use}]", flush=True)
        raw_texts, clean_texts, t_infer = transcribe_and_stream(cpu_model, "cpu")
    else:
        raise

# 3. Analyze text transformations
combined_raw = " ".join(raw_texts).strip()
combined_clean = " ".join(clean_texts).strip()
final_dedup = _remove_consecutive_duplicate_phrases(combined_clean)

print("\n" + "="*80, flush=True)
print("TEXT PROCESSING TRANSFORMATION COMPARISON:", flush=True)
print("="*80, flush=True)
print(f"Raw Whisper Total Characters:   {len(combined_raw)} ({len(combined_raw.split())} words)", flush=True)
print(f"After _clean_segment_text:      {len(combined_clean)} ({len(combined_clean.split())} words)", flush=True)
print(f"After _remove_duplicate_phrases: {len(final_dedup)} ({len(final_dedup.split())} words)", flush=True)

# 4. Save results to scratch
result_summary = {
    "audio_file": AUDIO_FILE,
    "model": WHISPER_MODEL,
    "initial_device": device,
    "cuda_oom_occurred": cuda_oom_occurred,
    "final_device": transcriber._device_in_use,
    "detected_language": detected_language,
    "language_probability": language_prob,
    "total_time_seconds": time.time() - t_start_total,
    "inference_time_seconds": t_infer,
    "segment_count": len(segments_collected),
    "segments": segments_collected,
    "raw_transcript": combined_raw,
    "cleaned_transcript": combined_clean,
    "final_transcript": final_dedup,
}

output_json = os.path.abspath(r"scratch\probe_transcription_results.json")
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(result_summary, f, ensure_ascii=False, indent=2)

print(f"\nProbe completed in {time.time()-t_start_total:.2f}s. Saved to {output_json}", flush=True)
print("="*80, flush=True)
