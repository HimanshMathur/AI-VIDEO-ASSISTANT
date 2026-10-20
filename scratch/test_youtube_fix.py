import os
import sys
import time

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.audio_processor import download_youtube_audio, validate_audio, chunk_audio
from core.transcriber import transcribe_chunk_whisper

url = "https://youtu.be/rPvJ7FKqWy4?si="
print("=== 1. TESTING YOUTUBE DOWNLOAD & EXTRACTION ===")
t0 = time.time()
wav_path = download_youtube_audio(url)
t_dl = time.time() - t0
print(f"Download & Audio Extraction succeeded in {t_dl:.2f}s!")
print(f"WAV Path: {wav_path}")
print(f"File exists: {os.path.exists(wav_path)}")
print(f"File size: {os.path.getsize(wav_path)} bytes")

print("\n=== 2. VALIDATING AUDIO METADATA ===")
meta = validate_audio(wav_path)
print(f"Audio duration: {meta['duration']:.2f} seconds")
print(f"Sample rate: {meta['sample_rate']} Hz")
print(f"Channels: {meta['channels']}")
assert meta["duration"] > 0, "Audio duration must be > 0"

print("\n=== 3. CHUNKING AUDIO ===")
chunks = chunk_audio(wav_path)
print(f"Chunks produced: {len(chunks)}")

print("\n=== 4. PASSING TO EXISTING TRANSCRIPTION PIPELINE ===")
t_tr_start = time.time()
text, lang = transcribe_chunk_whisper(chunks[0])
t_tr = time.time() - t_tr_start
print(f"Transcription succeeded in {t_tr:.2f}s!")
print(f"Detected Language: {lang}")
print(f"Transcript preview:\n{text[:200]}...")
print("\n=== ALL VERIFICATION CRITERIA PASSED! ===")
