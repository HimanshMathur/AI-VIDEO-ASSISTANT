import sys
import os
import time
sys.path.insert(0, os.path.abspath("."))

from core.transcriber import load_model, _clean_segment_text

print("Loading model...", flush=True)
t0 = time.time()
model = load_model()
print(f"Model loaded in {time.time()-t0:.2f}s", flush=True)

audio_file = "scratch/test_android_rPvJ7FKqWy4.wav"
print(f"Transcribing {audio_file}...", flush=True)
t1 = time.time()
segments, info = model.transcribe(audio_file, task="transcribe", beam_size=10, vad_filter=True)
print(f"Detected language: {info.language} ({info.language_probability:.2f})", flush=True)

count = 0
for seg in segments:
    count += 1
    t_now = time.time() - t1
    print(f"Seg #{count} [{seg.start:.2f}s - {seg.end:.2f}s] (elapsed {t_now:.1f}s): {seg.text}", flush=True)
    if count >= 10:
        print("Printed first 10 segments!", flush=True)
        break

print(f"Done first 10 segments in {time.time()-t1:.2f}s total.", flush=True)
