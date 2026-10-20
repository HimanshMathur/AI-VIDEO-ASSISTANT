import os
import sys
import time
import subprocess
import torch
import whisper
from dotenv import load_dotenv

load_dotenv()

from utils.audio_processor import (
    extract_audio_from_file,
    validate_audio,
    chunk_audio,
    cleanup_temp_files,
)
from core.transcriber import load_model, transcribe_chunk_whisper, WHISPER_MODEL
from core.summarizer import generate_video_intelligence
from core.rag_engine import build_rag_chain

def run_benchmark():
    video_path = os.path.abspath("temp_uploads/upload_056ff874.mp4")
    print(f"=== BENCHMARKING CURRENT PIPELINE ===")
    print(f"Test video: {video_path}")
    print(f"File size: {os.path.getsize(video_path) / (1024*1024):.2f} MB")
    
    # Measure audio extraction on full video
    t0 = time.time()
    extracted_wav = extract_audio_from_file(video_path, output_dir="scratch")
    t_extract = time.time() - t0
    meta = validate_audio(extracted_wav)
    duration = meta["duration"]
    print(f"1. Audio Extraction:")
    print(f"   - WAV duration: {duration:.2f}s ({duration/60:.2f} min)")
    print(f"   - Time taken: {t_extract:.2f}s")
    
    # Measure audio chunking
    t0 = time.time()
    chunks = chunk_audio(extracted_wav, chunk_minutes=10, output_dir="scratch")
    t_chunk = time.time() - t0
    print(f"2. Audio Chunking:")
    print(f"   - Number of chunks: {len(chunks)}")
    print(f"   - Time taken: {t_chunk:.2f}s")
    
    # Next, let's benchmark Whisper on a 60s sample chunk to calculate precise transcription throughput
    # Extract 60s snippet
    sample_wav = "scratch/sample_60s.wav"
    subprocess.run([
        "ffmpeg", "-y", "-i", extracted_wav, "-t", "60", "-c", "copy", sample_wav
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    print(f"\nBenchmarking Whisper '{WHISPER_MODEL}' on 60s audio slice...")
    # Warmup / load model
    t0 = time.time()
    load_model()
    t_load = time.time() - t0
    print(f"   - Whisper model load time: {t_load:.2f}s")
    
    t0 = time.time()
    text_60s, lang_60s = transcribe_chunk_whisper(sample_wav)
    t_whisper_60s = time.time() - t0
    rtf = t_whisper_60s / 60.0  # Real-time factor (seconds of processing per second of audio)
    projected_whisper_full = duration * rtf
    print(f"3. Whisper Transcription:")
    print(f"   - 60s snippet transcribed in: {t_whisper_60s:.2f}s")
    print(f"   - Real-Time Factor (RTF): {rtf:.3f}x")
    print(f"   - Detected language: {lang_60s}")
    print(f"   - Text snippet (first 100 chars): {text_60s[:100]}...")
    print(f"   - PROJECTED full 42.6min Whisper time: {projected_whisper_full:.2f}s ({projected_whisper_full/60:.2f} min)")
    
    # Also benchmark Qwen Video Intelligence
    # Let's generate a realistic transcript (e.g. 5,000 words / 30,000 chars) or use sample text
    sample_transcript = (text_60s + " ") * 20
    print(f"\n4. Qwen Video Intelligence (Ollama qwen3:8b):")
    print(f"   - Input transcript length: {len(sample_transcript)} characters")
    t0 = time.time()
    intel = generate_video_intelligence(sample_transcript)
    t_intel = time.time() - t0
    print(f"   - Time taken: {t_intel:.2f}s")
    print(f"   - Title: {intel.get('title')}")
    
    # 5. FAISS Indexing
    print(f"\n5. FAISS Indexing:")
    t0 = time.time()
    rag_chain = build_rag_chain(sample_transcript)
    t_rag = time.time() - t0
    print(f"   - Time taken: {t_rag:.2f}s")
    
    # Clean up sample files
    cleanup_temp_files([sample_wav, extracted_wav] + [c for c in chunks if c != extracted_wav])
    
    print("\n=== SUMMARY OF BENCHMARK ===")
    print(f"1. Audio extraction: {t_extract:.2f}s")
    print(f"2. Audio chunking: {t_chunk:.2f}s")
    print(f"3. Whisper transcription (60s): {t_whisper_60s:.2f}s  --> Projected full ({duration/60:.1f}m): {projected_whisper_full/60:.2f}m")
    print(f"4. Qwen video intelligence: {t_intel:.2f}s")
    print(f"5. FAISS indexing: {t_rag:.2f}s")

if __name__ == "__main__":
    run_benchmark()
