import os
import sys
import time
import subprocess
import torch
from dotenv import load_dotenv

load_dotenv()

def benchmark_comparison():
    sample_wav = os.path.abspath("scratch/speech_sample_16k.wav")
    print("=== BENCHMARK: OPENAI-WHISPER vs FASTER-WHISPER ===")
    print(f"Audio file: {sample_wav}")
    
    # 1. openai-whisper
    print("\n--- 1. openai-whisper ('small' on CPU) ---")
    import whisper
    t0 = time.time()
    torch.set_num_threads(min(8, max(2, (os.cpu_count() or 4) // 2)))
    model_openai = whisper.load_model("small", device="cpu")
    t_load_openai = time.time() - t0
    print(f"Model load time: {t_load_openai:.2f}s")
    
    t0 = time.time()
    res_openai = model_openai.transcribe(
        sample_wav,
        task="transcribe",
        language=None,
        fp16=False,
        beam_size=1,
        best_of=1,
        temperature=0.0,
    )
    t_infer_openai = time.time() - t0
    lang_openai = res_openai.get("language")
    text_openai = res_openai.get("text", "").strip()
    print(f"Inference time: {t_infer_openai:.2f}s (RTF: {t_infer_openai/60:.3f}x)")
    print(f"Detected language: {lang_openai}")
    print(f"Text sample: {text_openai[:120]}...")
    
    # 2. faster-whisper
    print("\n--- 2. faster-whisper ('small' on CPU with int8 quantization) ---")
    from faster_whisper import WhisperModel
    t0 = time.time()
    # int8 on CPU uses CTranslate2 AVX2/AVX-512 optimized kernels
    model_faster = WhisperModel("small", device="cpu", compute_type="int8", cpu_threads=min(8, (os.cpu_count() or 4) // 2))
    t_load_faster = time.time() - t0
    print(f"Model load time: {t_load_faster:.2f}s")
    
    t0 = time.time()
    segments, info = model_faster.transcribe(
        sample_wav,
        beam_size=1,
        temperature=0.0,
    )
    text_faster = " ".join([seg.text.strip() for seg in segments]).strip()
    t_infer_faster = time.time() - t0
    print(f"Inference time: {t_infer_faster:.2f}s (RTF: {t_infer_faster/60:.3f}x)")
    print(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")
    print(f"Text sample: {text_faster[:120]}...")
    
    speedup = t_infer_openai / t_infer_faster if t_infer_faster > 0 else 0
    print("\n=== COMPARISON SUMMARY ===")
    print(f"openai-whisper: {t_infer_openai:.2f}s")
    print(f"faster-whisper: {t_infer_faster:.2f}s")
    print(f"Speedup factor: {speedup:.2f}x faster!")
    print(f"For a 42-minute video:")
    print(f"   - openai-whisper takes ~{(42*60*t_infer_openai/60)/60:.1f} minutes")
    print(f"   - faster-whisper takes ~{(42*60*t_infer_faster/60)/60:.1f} minutes")

if __name__ == "__main__":
    benchmark_comparison()
