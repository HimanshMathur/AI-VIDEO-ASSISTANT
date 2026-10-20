import os
import sys
import time
import subprocess
import difflib
from dotenv import load_dotenv

load_dotenv()

from utils.audio_processor import (
    extract_audio_from_file,
    validate_audio,
    chunk_audio,
    cleanup_temp_files,
)
import whisper
from faster_whisper import WhisperModel
from core.summarizer import generate_video_intelligence
from core.rag_engine import build_rag_chain

def run_phase_b_benchmark():
    video_path = r"C:\Users\mathu\Downloads\Product Marketing Meeting (weekly) 2021-06-28.mp4"
    print("=== PHASE B: BENCHMARK & QUALITY VERIFICATION ===")
    print(f"Source video: {video_path}")
    print(f"File size: {os.path.getsize(video_path)/(1024*1024):.2f} MB")
    
    # 1. Audio Extraction
    t0 = time.time()
    wav_path = extract_audio_from_file(video_path, output_dir="scratch")
    t_extract = time.time() - t0
    meta = validate_audio(wav_path)
    duration = meta["duration"]
    print(f"\n1. Audio Extraction: {t_extract:.2f}s (WAV duration: {duration:.2f}s / {duration/60:.2f}m)")
    
    # 2. Audio Chunking
    t0 = time.time()
    chunks = chunk_audio(wav_path, chunk_minutes=10, output_dir="scratch")
    t_chunk = time.time() - t0
    print(f"2. Audio Chunking: {t_chunk:.2f}s ({len(chunks)} chunks)")
    
    # Create 60s test slice
    sample_60s = os.path.abspath("scratch/meeting_sample_60s.wav")
    subprocess.run([
        "ffmpeg", "-y", "-i", wav_path, "-t", "60", "-c", "copy", sample_60s
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # 3. Benchmark openai-whisper vs faster-whisper on identical 60s clip
    print("\n--- 3. Transcription Benchmark (60s slice of real meeting) ---")
    
    # openai-whisper
    print("Running openai-whisper (small)...")
    m_oa = whisper.load_model("small", device="cpu")
    t0 = time.time()
    res_oa = m_oa.transcribe(sample_60s, task="transcribe", language=None, fp16=False, condition_on_previous_text=False, beam_size=1, best_of=1, temperature=0.0)
    t_oa = time.time() - t0
    text_oa = res_oa.get("text", "").strip()
    lang_oa = res_oa.get("language")
    
    # faster-whisper
    print("Running faster-whisper (small / int8)...")
    m_fw = WhisperModel("small", device="cpu", compute_type="int8", cpu_threads=4)
    t0 = time.time()
    segs, info = m_fw.transcribe(sample_60s, task="transcribe", language=None, beam_size=1, best_of=1, temperature=0.0, condition_on_previous_text=False)
    t_fw = time.time() - t0
    text_fw = " ".join([s.text.strip() for s in segs]).strip()
    lang_fw = info.language
    
    rtf_oa = t_oa / 60.0
    rtf_fw = t_fw / 60.0
    proj_oa = duration * rtf_oa
    proj_fw = duration * rtf_fw
    
    print(f"\n[TIMING RESULTS]")
    print(f"openai-whisper:  {t_oa:.2f}s (RTF: {rtf_oa:.3f}x) -> Projected full {duration/60:.1f}m: {proj_oa/60:.2f}m")
    print(f"faster-whisper:  {t_fw:.2f}s (RTF: {rtf_fw:.3f}x) -> Projected full {duration/60:.1f}m: {proj_fw/60:.2f}m")
    print(f"Speedup: {t_oa / t_fw:.2f}x faster!")
    
    print(f"\n[LANGUAGE DETECTION]")
    print(f"openai-whisper: {lang_oa}")
    print(f"faster-whisper: {lang_fw} (probability: {info.language_probability:.2f})")
    
    print(f"\n[TRANSCRIPT QUALITY COMPARISON]")
    print(f"openai-whisper text ({len(text_oa)} chars):")
    print(f"  \"{text_oa}\"")
    print(f"faster-whisper text ({len(text_fw)} chars):")
    print(f"  \"{text_fw}\"")
    
    diff = list(difflib.unified_diff(text_oa.split(), text_fw.split(), fromfile="openai_whisper", tofile="faster_whisper"))
    if not diff:
        print(">> WORD DIFF: 100% IDENTICAL WORD-FOR-WORD!")
    else:
        print(">> WORD DIFF:")
        for d in diff[:10]:
            print("  ", d)
            
    # Check for repetitions/hallucinations
    words_fw = text_fw.split()
    repeats = [w for i, w in enumerate(words_fw[:-1]) if w.lower() == words_fw[i+1].lower()]
    print(f">> Consecutive repeated words in faster-whisper: {len(repeats)} ({repeats})")
    
    # 4. Qwen video intelligence on full transcript sample
    print(f"\n4. Qwen Video Intelligence (Ollama qwen3:8b):")
    t0 = time.time()
    intel = generate_video_intelligence((text_fw + " ") * 15)
    t_intel = time.time() - t0
    print(f"   - Time taken: {t_intel:.2f}s")
    print(f"   - Title: {intel.get('title')}")
    
    # 5. FAISS Indexing
    print(f"\n5. FAISS Indexing:")
    t0 = time.time()
    rag = build_rag_chain((text_fw + " ") * 15)
    t_rag = time.time() - t0
    print(f"   - Time taken: {t_rag:.2f}s")
    
    # Clean up scratch files
    cleanup_temp_files([wav_path, sample_60s] + [c for c in chunks if c != wav_path])
    
    print("\n=== TOTAL PIPELINE COMPARISON (42.68-MIN VIDEO) ===")
    old_total = t_extract + t_chunk + proj_oa + t_intel + t_rag
    new_total = t_extract + t_chunk + proj_fw + t_intel + t_rag
    print(f"Old Pipeline Total: {old_total:.1f}s ({old_total/60:.2f} minutes)")
    print(f"New Pipeline Total: {new_total:.1f}s ({new_total/60:.2f} minutes)")
    print(f"Total Pipeline Speedup: {old_total / new_total:.2f}x faster! (Saved {(old_total - new_total)/60:.2f} minutes)")

if __name__ == "__main__":
    run_phase_b_benchmark()
