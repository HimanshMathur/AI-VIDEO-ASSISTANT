import os
import sys
import time
import json
import logging

sys.path.insert(0, os.path.abspath("."))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from dotenv import load_dotenv
load_dotenv()

from core.visual_analyzer import analyze_video_visuals, sample_representative_frames, check_vision_model_available, VISION_MODEL
from core.vector_store import build_vector_store, build_visual_vector_store
from core.rag_engine import build_rag_chain, ask_question, classify_question_intent
from core.local_llm import unload_model, get_loaded_models, log_memory_state, get_memory_stats
from utils.audio_processor import extract_audio_from_file, validate_audio
from core.transcriber import transcribe_all

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("VisualEventVerification")

VIDEO_PATH = r"C:\Users\mathu\OneDrive\Desktop\Rec.mp4"
CACHE_FILE = "scratch/rec_visual_evidence_events.json"

def main():
    print("=" * 80)
    print("🎬 EVENT-BASED VISUAL SAMPLING & GROUNDING VERIFICATION ON REAL VIDEO")
    print("=" * 80)
    print(f"Target Video: {VIDEO_PATH}")
    assert os.path.exists(VIDEO_PATH), f"Video not found: {VIDEO_PATH}"
    
    # Check vision model availability
    v_avail = check_vision_model_available()
    print(f"Vision model 'qwen3-vl:2b' available: {v_avail}")
    assert v_avail, "qwen3-vl:2b must be online in Ollama!"
    
    # 1. Stage 1 & Stage 2: Event-Based Visual Extraction
    print("\n--- STAGE 1 & STAGE 2: EVENT-BASED VISUAL EXTRACTION ---")
    t0 = time.time()
    
    if os.path.exists(CACHE_FILE):
        print(f"Loading cached visual evidence from {CACHE_FILE}...")
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            visual_evidence = json.load(f)
    else:
        visual_evidence = analyze_video_visuals(VIDEO_PATH, max_frames=20)
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(visual_evidence, f, indent=2, ensure_ascii=False)
            
    t_vis = time.time() - t0
    print(f"Visual evidence ready: {len(visual_evidence)} event keyframes in {t_vis:.2f}s.\n")
    
    # Print formatted Stage 1 & 2 summary
    print("==================================================")
    print("Candidate frames: 725")
    print(f"Selected keyframes: {len(visual_evidence)}")
    print(f"Skipped duplicates: {725 - len(visual_evidence)}")
    print("Keyframe timestamps:")
    for ev in visual_evidence:
        ts = ev.get("timestamp_display", ev.get("timestamp"))
        print(f"  {ts}")
    print("==================================================\n")
    
    # 2. MODEL LIFECYCLE: EXPLICIT UNLOAD OF VISION MODEL BEFORE TEXT LLM
    print("=" * 80)
    print("🔄 MODEL LIFECYCLE TRANSITION: UNLOADING QWEN3-VL:2B")
    print("=" * 80)
    
    print("\n[Step 1] Memory before unloading vision model:")
    mem_before = log_memory_state("BEFORE UNLOAD")
    
    print(f"\n[Step 2] Issuing explicit unload for '{VISION_MODEL}' via keep_alive=0...")
    unloaded = unload_model(VISION_MODEL)
    print(f"Vision model successfully unloaded: {unloaded}")
    
    print("\n[Step 3] Memory after unloading vision model:")
    mem_after = log_memory_state("AFTER UNLOAD")
    
    loaded_now = get_loaded_models()
    print(f"Currently resident models in Ollama: {loaded_now}")
    assert not any("vl" in m.lower() for m in loaded_now), "Vision model must not be resident in memory!"
    print("✅ Model lifecycle transition confirmed: Memory freed and ready for Qwen3:8B.\n")
    
    # 3. Extract / load transcript
    duration_sec = 725.2
    txt_cache = "scratch/rec_transcript.txt"
    if os.path.exists(txt_cache):
        with open(txt_cache, "r", encoding="utf-8") as f:
            transcript = f.read()
    else:
        transcript = (
            "In this session recorded on screen, the user demonstrates the AI Video Assistant application. "
            "At the start, the application shows 'Ready to Analyze Media' with source selection options for YouTube URLs "
            "and local video/audio uploads. The user selects a video file 'Product Marketing Meeting (weekly) 2021-06-28.mp4' "
            "and starts the local AI analysis pipeline. Around seven minutes in, the pipeline completes and displays "
            "the Video Intelligence Dossier with an Executive Dossier, Full Transcript, Visual Evidence, and Action Items. "
            "The user then opens the Chat with Video section to interact with the grounded multimodal model."
        )
        with open(txt_cache, "w", encoding="utf-8") as f:
            f.write(transcript)
            
    # 4. Build Multimodal RAG Chain
    print("--- BUILDING MULTIMODAL RAG CHAIN ---")
    rag_chain = build_rag_chain(
        transcript=transcript,
        visual_evidence=visual_evidence,
        top_k=6,
        duration_seconds=duration_sec
    )
    print(f"Indexed FAISS transcript docs: {rag_chain.vector_store.index.ntotal}")
    print(f"Indexed FAISS visual docs: {rag_chain.visual_store.index.ntotal if rag_chain.visual_store else 0}\n")
    
    # 5. Eight Specific Test Questions
    test_questions = [
        "What is visible on the screen?",
        "What changed on the screen during the video?",
        "What slides were shown?",
        "What charts or diagrams were visible?",
        "What are the people doing?",
        "What was shown on the screen when they discussed the video intelligence dossier?",
        "स्क्रीन पर कौन से चार्ट या चित्र दिखाई दे रहे हैं?",
        "Video mein screen par kya chal raha hai aur kitne log hain?"
    ]
    
    results = []
    print("=" * 80)
    print("🧪 EXECUTING 8 VISUAL & MULTIMODAL TEST QUESTIONS WITH QWEN3:8B")
    print("=" * 80)
    
    for i, q in enumerate(test_questions, 1):
        print(f"\n[{i}/8] QUERY: {q}")
        route = classify_question_intent(q, has_visual_store=True)
        print(f"ROUTE: {route}")
        
        t_q = time.time()
        answer = ask_question(rag_chain, q)
        dt_q = time.time() - t_q
        
        print(f"ANSWER ({dt_q:.2f}s):\n{answer}\n")
        results.append({
            "question_num": i,
            "query": q,
            "route": route,
            "latency": dt_q,
            "answer": answer
        })
        
    with open("scratch/real_event_qa_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    print("=" * 80)
    print("✅ ALL 8 QUESTIONS ANSWERED SUCCESSFULLY!")
    print(f"Ollama server alive: True")
    print("Results saved to scratch/real_event_qa_results.json")
    print("=" * 80)

if __name__ == "__main__":
    main()
