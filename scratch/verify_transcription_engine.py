import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print("=== 1. TESTING MODULE IMPORT & SYNTAX ===")
from core.transcriber import (
    load_model,
    transcribe_chunk_whisper,
    transcribe_all,
    transcribe_with_segments,
    WHISPER_MODEL,
)
print("Successfully imported core.transcriber!")
print(f"Configured WHISPER_MODEL default: {WHISPER_MODEL}")

print("\n=== 2. LOADING MODEL SINGLETON ===")
model = load_model()
from core import transcriber
print(f"Actual device in use: {transcriber._device_in_use}")
print(f"Actual compute type in use: {transcriber._compute_type_in_use}")
assert transcriber._device_in_use == "cuda", "Expected CUDA device to be active!"
assert transcriber._compute_type_in_use == "float16", "Expected float16 on CUDA!"

print("\n=== 3. TRANSCRIBING ENGLISH SAMPLE (scratch/speech_sample_16k.wav) ===")
text_en, lang_en = transcribe_chunk_whisper("scratch/speech_sample_16k.wav")
print(f"Detected Language: {lang_en}")
print(f"English Transcript:\n{text_en}")
assert len(text_en) > 20, "English transcript too short"
assert "आप" not in text_en, "Detected unwanted Hindi hallucination in English audio!"
assert "\ufffd" not in text_en, "Detected corrupted unicode character!"

print("\n=== 4. TRANSCRIBING HINGLISH SAMPLE (scratch/hinglish_sample.wav) ===")
text_hi, lang_hi = transcribe_chunk_whisper("scratch/hinglish_sample.wav")
print(f"Detected Language: {lang_hi}")
print(f"Hinglish Transcript:\n{text_hi}")
assert len(text_hi) > 20, "Hinglish transcript too short"
assert "आप आप आप" not in text_hi, "Detected pathological repetition loop!"
assert "हूँँँँ" not in text_hi, "Detected pathological loop!"
assert "\ufffd" not in text_hi, "Detected corrupted unicode character!"

print("\n=== 5. TESTING transcribe_all WITH CHUNK LIST ===")
full_text = transcribe_all(["scratch/speech_sample_16k.wav"])
print(f"transcribe_all output length: {len(full_text)}")
assert len(full_text) > 20

print("\n=== 6. TESTING transcribe_with_segments DATA STRUCTURE ===")
seg_res = transcribe_with_segments("scratch/speech_sample_16k.wav")
print("Keys in segment result:", list(seg_res.keys()))
num_segs = len(seg_res["segments"])
print(f"Number of segments: {num_segs}")
if num_segs > 0:
    sample_seg = seg_res["segments"][0]
    print("Sample segment item:", sample_seg)
    assert "start" in sample_seg and "end" in sample_seg and "text" in sample_seg and "avg_logprob" in sample_seg

print("\n=== ALL TRANSCRIPTION VERIFICATION TESTS PASSED SUCCESSFULLY! ===")
