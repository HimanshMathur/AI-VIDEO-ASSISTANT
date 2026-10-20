import os
import sys
import time

from faster_whisper import WhisperModel

audio_path = os.path.abspath("scratch/speech_sample_16k.wav")
print("Testing faster-whisper parameters on:", audio_path)

model = WhisperModel("small", device="cpu", compute_type="int8", cpu_threads=4)
t0 = time.time()
segments, info = model.transcribe(
    audio_path,
    task="transcribe",
    language=None,
    beam_size=1,
    best_of=1,
    temperature=0.0,
    condition_on_previous_text=False,
)
text = " ".join([s.text.strip() for s in segments]).strip()
dt = time.time() - t0
print(f"Transcribed in {dt:.2f}s:")
print(f"Language: {info.language} ({info.language_probability:.2f})")
print(f"Text: {text}")
