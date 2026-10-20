import os
import sys
from faster_whisper import WhisperModel

# Test multilingual auto-detection on synthetic/transcription text
print("Testing faster-whisper multilingual capability...")
model = WhisperModel("tiny", device="cpu", compute_type="int8", cpu_threads=4)
audio_path = os.path.abspath("scratch/speech_sample_16k.wav")

segments, info = model.transcribe(audio_path)
print(f"Detected: {info.language} with prob {info.language_probability:.3f}")
for seg in segments:
    print(f"[{seg.start:.1f}s -> {seg.end:.1f}s]: {seg.text}")
