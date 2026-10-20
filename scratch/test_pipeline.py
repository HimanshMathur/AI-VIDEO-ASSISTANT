import os
import time
import uuid
import logging
from dotenv import load_dotenv

load_dotenv()

from utils.audio_processor import extract_audio_from_file, chunk_audio, validate_audio
from core.transcriber import transcribe_all
from core.visual_analyzer import analyze_video_visuals, get_video_duration
from core.rag_engine import build_rag_chain, ask_question
from core.summarizer import generate_video_intelligence

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VIDEO_PATH = r"C:\Users\mathu\OneDrive\Desktop\AI-Video-Assistant--main\AI-Video-Assistant--main\scratch\debug_test\test_scene.mp4"

def run_test():
    print("=" * 50)
    print("1. RUNNING ANALYSIS PIPELINE")
    print("=" * 50)
    t_start_total = time.time()
    
    # Validation & Duration
    duration = get_video_duration(VIDEO_PATH)
    print(f"Video duration: {duration:.2f}s")
    
    # Audio Extract & Transcribe
    t_audio_start = time.time()
    wav_path = extract_audio_from_file(VIDEO_PATH)
    chunks = chunk_audio(wav_path)
    try:
        transcript = transcribe_all(chunks)
    except Exception:
        transcript = "[Silence]"
    t_transcribe = time.time() - t_audio_start
    print(f"Whisper time: {t_transcribe:.2f}s")
    
    # Vision
    t_vis_start = time.time()
    visual_evidence = analyze_video_visuals(VIDEO_PATH, max_frames=20)
    t_vis = time.time() - t_vis_start
    
    # Count key frames and duplicate frames
    total_sampled = len(visual_evidence)
    key_frames = sum(1 for v in visual_evidence if v.get("is_key"))
    print(f"Frames sampled: {total_sampled}")
    print(f"Unique/key frames (Qwen3-VL calls): {key_frames}")
    print(f"Qwen3-VL total time: {t_vis:.2f}s")
    
    # Indexing (FAISS)
    t_rag_start = time.time()
    rag_chain = build_rag_chain(transcript, visual_evidence=visual_evidence)
    t_rag = time.time() - t_rag_start
    print(f"FAISS time: {t_rag:.2f}s")
    
    t_total = time.time() - t_start_total
    print(f"Total Analyze time: {t_total:.2f}s")
    
    # Questions
    print("\n" + "=" * 50)
    print("2. RUNNING CHAT QUESTIONS")
    print("=" * 50)
    questions = [
        "What is happening in the video?",
        "How many people are visible?",
        "What is visible on the screen?",
        "What was shown when they discussed the topic?",
        "Video mein kitne log dikh rahe hain aur screen par kya chal raha hai?",
        "Is there a blue elephant flying in the video?",
    ]
    
    for i, q in enumerate(questions):
        t_q_start = time.time()
        ans = ask_question(rag_chain, q)
        t_q = time.time() - t_q_start
        print(f"Q{i+1} response time: {t_q:.2f}s")

if __name__ == "__main__":
    run_test()
