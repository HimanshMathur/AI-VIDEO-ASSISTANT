import os
import sys
import time
from dotenv import load_dotenv

load_dotenv()

def run_comparison():
    audio_path = os.path.abspath("scratch/speech_sample_16k.wav")
    print(f"Testing on audio: {audio_path}")
    
    # 1. openai-whisper 'small'
    import whisper
    print("\n--- openai-whisper (small) ---")
    t0 = time.time()
    model_openai = whisper.load_model("small", device="cpu")
    t_load_openai = time.time() - t0
    
    t0 = time.time()
    res_openai = model_openai.transcribe(audio_path, task="transcribe", language=None, fp16=False)
    t_infer_openai = time.time() - t0
    print(f"Load time: {t_load_openai:.2f}s | Inference time: {t_infer_openai:.2f}s")
    print(f"Language: {res_openai.get('language')}")
    print(f"Text: {res_openai.get('text', '').strip()}")
    
    # 2. faster-whisper 'tiny'
    from faster_whisper import WhisperModel
    print("\n--- faster-whisper (tiny) ---")
    t0 = time.time()
    model_faster_tiny = WhisperModel("tiny", device="cpu", compute_type="int8", cpu_threads=4)
    t_load_ft = time.time() - t0
    
    t0 = time.time()
    segments, info = model_faster_tiny.transcribe(audio_path)
    text_ft = " ".join([s.text.strip() for s in segments])
    t_infer_ft = time.time() - t0
    print(f"Load time: {t_load_ft:.2f}s | Inference time: {t_infer_ft:.2f}s")
    print(f"Language: {info.language} ({info.language_probability:.2f})")
    print(f"Text: {text_ft}")
    
    print("\n=== SUMMARY ===")
    print(f"openai-whisper (small): {t_infer_openai:.2f}s")
    print(f"faster-whisper (tiny):  {t_infer_ft:.2f}s  ({t_infer_openai / t_infer_ft:.2f}x faster)")

if __name__ == "__main__":
    run_comparison()
