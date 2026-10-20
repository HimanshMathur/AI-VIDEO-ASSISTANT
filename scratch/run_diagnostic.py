import sys
import os
import time
import wave
import logging
import threading

# Force unbuffered stdout/stderr
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Add current workspace to path
sys.path.insert(0, os.path.abspath("."))

# Configure logging to stdout so transcriber logs are immediately visible
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger("diagnostic")

TARGET_FILE = os.path.abspath("scratch/test_android_rPvJ7FKqWy4.wav")

def main():
    print("=== TIME-BOUNDED TRANSCRIBER DIAGNOSTIC ===", flush=True)
    t_start = time.time()
    start_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t_start))
    print(f"Start Time: {start_str}", flush=True)

    # 1. Check file existence & metadata
    exists = os.path.exists(TARGET_FILE)
    print(f"File exists: {exists} ({TARGET_FILE})", flush=True)
    if not exists:
        print("ERROR: Target audio file does not exist!", flush=True)
        return

    file_size_bytes = os.path.getsize(TARGET_FILE)
    file_size_mb = file_size_bytes / (1024 * 1024)
    print(f"File size: {file_size_bytes} bytes ({file_size_mb:.2f} MB)", flush=True)

    duration_sec = 0.0
    try:
        with wave.open(TARGET_FILE, "rb") as wf:
            n_frames = wf.getnframes()
            rate = wf.getframerate()
            duration_sec = n_frames / float(rate) if rate > 0 else 0.0
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            print(f"Audio duration: {duration_sec:.2f}s ({int(duration_sec // 60)}m {int(duration_sec % 60)}s)", flush=True)
            print(f"Audio specs: {rate} Hz, {channels} channel(s), {sampwidth * 8}-bit", flush=True)
    except Exception as we:
        print(f"Audio inspection warning: {we}", flush=True)

    # 2. Check model / device info before call
    from core.transcriber import WHISPER_MODEL, _device_in_use, _compute_type_in_use, transcribe_with_segments
    import torch
    import ctranslate2

    cuda_available = torch.cuda.is_available()
    ct2_cuda_count = ctranslate2.get_cuda_device_count()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "N/A"
    print(f"Model configured: {WHISPER_MODEL}", flush=True)
    print(f"PyTorch CUDA available: {cuda_available} ({device_name})", flush=True)
    print(f"CTranslate2 CUDA device count: {ct2_cuda_count}", flush=True)

    # 3. Time-bounded execution
    TIMEOUT_SECONDS = 180.0
    result_container = {"result": None, "error": None, "done": False}

    def target_worker():
        try:
            print("[Worker] Starting transcribe_with_segments()...", flush=True)
            res = transcribe_with_segments(TARGET_FILE)
            result_container["result"] = res
            result_container["done"] = True
            print("[Worker] transcribe_with_segments() completed successfully.", flush=True)
        except Exception as ex:
            result_container["error"] = ex
            result_container["done"] = True
            print(f"[Worker] Exception in transcribe_with_segments: {ex}", flush=True)

    worker_thread = threading.Thread(target=target_worker, daemon=True)
    worker_thread.start()

    # Monitor progress with timeout
    poll_interval = 2.0
    while time.time() - t_start < TIMEOUT_SECONDS:
        if result_container["done"]:
            break
        elapsed = time.time() - t_start
        print(f"Transcribing in progress... ({elapsed:.1f}s elapsed / timeout={TIMEOUT_SECONDS}s)", flush=True)
        time.sleep(poll_interval)

    elapsed_total = time.time() - t_start

    # 4. Handle results / timeout
    if not result_container["done"]:
        print(f"\n[FAIL] TIMEOUT: Transcription did NOT finish within {TIMEOUT_SECONDS}s!", flush=True)
        print("Model may be hanging during inference, lock contention, or slow CPU execution.", flush=True)
        sys.exit(1)

    if result_container["error"]:
        print(f"\n[FAIL] TRANSCRIPTION ERROR: {result_container['error']}", flush=True)
        import traceback
        traceback.print_exception(result_container["error"])
        sys.exit(1)

    res = result_container["result"]
    if not res:
        print("\n[FAIL] Empty result returned by transcribe_with_segments().", flush=True)
        sys.exit(1)

    # 5. Print verified outputs
    print(f"\n[SUCCESS] Transcription completed in {elapsed_total:.2f}s!", flush=True)
    lang = res.get("language")
    segments = res.get("segments", [])
    print(f"Detected Language: {lang}", flush=True)
    print(f"Total Segments Count: {len(segments)}", flush=True)

    # Save to scratch for reference
    import json
    with open("scratch/verified_segments.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("Saved verified segments to scratch/verified_segments.json", flush=True)

    print("\n--- FIRST 10 SEGMENTS ---", flush=True)
    for idx, s in enumerate(segments[:10]):
        start_sec = s.get("start", 0.0)
        end_sec = s.get("end", 0.0)
        text = s.get("text", "")
        start_m, start_s = int(start_sec // 60), int(start_sec % 60)
        end_m, end_s = int(end_sec // 60), int(end_sec % 60)
        avg_lp = s.get("avg_logprob", 0.0)
        print(f"[{idx+1:02d}] [{start_m:02d}:{start_s:02d} - {end_m:02d}:{end_s:02d}] ({start_sec:.2f}s - {end_sec:.2f}s) (logprob={avg_lp:.2f}): {text}", flush=True)

    print("\n=== DIAGNOSTIC REPORT FINISHED ===", flush=True)

if __name__ == "__main__":
    main()
