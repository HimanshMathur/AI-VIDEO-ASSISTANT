import os
import sys
import time
import whisper
from faster_whisper import WhisperModel

audio_path = os.path.abspath("scratch/speech_sample_16k.wav")

# 1. Warm openai-whisper small
print("Loading openai-whisper small...")
model_oa = whisper.load_model("small", device="cpu")
t0 = time.time()
res_oa = model_oa.transcribe(audio_path, task="transcribe", language=None, fp16=False, condition_on_previous_text=False, beam_size=1, best_of=1, temperature=0.0)
t_oa = time.time() - t0
text_oa = res_oa.get("text", "").strip()

# 2. Warm faster-whisper small
print("Loading faster-whisper small...")
model_fw = WhisperModel("small", device="cpu", compute_type="int8", cpu_threads=4)
t0 = time.time()
segments, info = model_fw.transcribe(audio_path, task="transcribe", language=None, beam_size=1, best_of=1, temperature=0.0, condition_on_previous_text=False)
text_fw = " ".join([s.text.strip() for s in segments]).strip()
t_fw = time.time() - t0

print("\n=== HEAD-TO-HEAD COMPARISON (MODEL: 'small') ===")
print(f"openai-whisper: {t_oa:.2f}s | Language: {res_oa.get('language')}")
print(f"faster-whisper: {t_fw:.2f}s | Language: {info.language} ({info.language_probability:.2f})")
print(f"Speedup: {t_oa / t_fw:.2f}x faster!")
print(f"\nExact Text Match: {text_oa == text_fw}")
if text_oa != text_fw:
    print("Diff:")
    print("OA:", text_oa)
    print("FW:", text_fw)
