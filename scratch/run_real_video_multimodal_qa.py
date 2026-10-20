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

from utils.audio_processor import extract_audio_from_file, chunk_audio, validate_audio
from core.transcriber import transcribe_all, transcribe_chunk_whisper
from core.visual_analyzer import (
    analyze_video_visuals,
    sample_representative_frames,
    analyze_frame_with_vision_model,
    check_vision_model_available,
)
from core.vector_store import build_vector_store, build_visual_vector_store
from core.rag_engine import build_rag_chain, ask_question, classify_question_intent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("RealVideoQA")

VIDEO_PATH = r"C:\Users\mathu\Downloads\Product Marketing Meeting (weekly) 2021-06-28.mp4"
CACHE_VIS_FILE = "scratch/meeting_visual_evidence.json"
CACHE_TXT_FILE = "scratch/meeting_transcript.txt"

def run_real_qa():
    print("=" * 80)
    print("🎬 PHASE D: REAL VIDEO MULTIMODAL PIPELINE & QA EVALUATION")
    print("=" * 80)
    print(f"Video Path: {VIDEO_PATH}")
    assert os.path.exists(VIDEO_PATH), f"Video not found: {VIDEO_PATH}"
    
    # 1. Vision Model Status
    print("\n--- 1. Vision Model Availability ---")
    v_avail = check_vision_model_available("qwen3-vl:2b")
    print(f"Vision model 'qwen3-vl:2b' available in Ollama: {v_avail}")
    assert v_avail, "qwen3-vl:2b must be available in Ollama!"
    
    # 2. Extract and Deduplicate Frames
    print("\n--- 2. Representative Frame Sampling (Target: 25 frames) ---")
    t0 = time.time()
    frames = sample_representative_frames(VIDEO_PATH, max_frames=25, output_dir="scratch/meeting_frames_phase_d")
    t_sampling = time.time() - t0
    key_frames = [f for f in frames if f.get("is_key")]
    print(f"Extracted {len(frames)} total frames in {t_sampling:.2f}s.")
    print(f"Deduplication: {len(key_frames)} key frames retained, {len(frames) - len(key_frames)} duplicate/static scenes identified.")
    
    # 3. Visual Analysis with qwen3-vl:2b (with disk cache)
    print(f"\n--- 3. Visual Inference (qwen3-vl:2b on {len(key_frames)} key frames) ---")
    t_visual = 0.0
    if os.path.exists(CACHE_VIS_FILE):
        with open(CACHE_VIS_FILE, "r", encoding="utf-8") as f:
            visual_evidence = json.load(f)
        print(f"Loaded {len(visual_evidence)} cached visual evidence points from {CACHE_VIS_FILE}.")
    else:
        t0 = time.time()
        # Analyze key frames directly
        evidence_by_index = {}
        for i, kf in enumerate(key_frames):
            ts_str = kf["timestamp_str"]
            print(f"Analyzing key frame {i+1}/{len(key_frames)} at [{ts_str}]...")
            t_f0 = time.time()
            data = analyze_frame_with_vision_model(kf["path"])
            dt_f = time.time() - t_f0
            print(f"  -> analyzed in {dt_f:.2f}s (people={data.get('people_count', 0)})")
            evidence_by_index[kf["index"]] = data
            
        # Build full evidence list mapping duplicates
        visual_evidence = []
        for f in frames:
            idx = f["index"]
            data = evidence_by_index.get(idx)
            if not data and "linked_key_index" in f:
                data = evidence_by_index.get(f["linked_key_index"], {})
            if not data:
                continue
            desc = data.get("description", "")
            p_cnt = data.get("people_count", 0)
            visual_evidence.append({
                "frame_index": idx,
                "timestamp": f["timestamp_str"],
                "timestamp_seconds": f["timestamp_seconds"],
                "people_count": p_cnt,
                "description": desc,
                "summary_text": f"[VISUAL EVIDENCE {f['timestamp_str']}]\n{desc}",
                "is_key": f.get("is_key", True),
                "raw": data,
            })
            
        t_visual = time.time() - t0
        with open(CACHE_VIS_FILE, "w", encoding="utf-8") as f:
            json.dump(visual_evidence, f, indent=2, ensure_ascii=False)
        print(f"Visual analysis complete & cached ({len(visual_evidence)} evidence points, {t_visual:.2f}s).")
    
    # Print sample visual evidence
    print("\n--- 4. Sample Timestamped Visual Evidence Points ---")
    for i, ev in enumerate(visual_evidence[:5]):
        print(f"\n[Evidence #{i+1} at {ev['timestamp']}] (People: {ev['people_count']}, Key: {ev.get('is_key', False)})")
        print(ev['description'].strip()[:240] + ("..." if len(ev['description'].strip()) > 240 else ""))
    
    for i, ev in enumerate(visual_evidence[5:], start=6):
        if "slide" in ev['description'].lower() or "gitlab" in ev['description'].lower() or "devops" in ev['description'].lower():
            print(f"\n[Evidence #{i} at {ev['timestamp']} (Presentation/Slide Evidence)]")
            print(ev['description'].strip()[:280] + "...")
            break

    # 4. Audio Extraction & Transcription (faster-whisper-small)
    print("\n--- 5. Audio Extraction & Production Transcription ---")
    t_extract = 0.0
    t_chunk = 0.0
    t_transcribe = 0.0
    
    if os.path.exists(CACHE_TXT_FILE):
        with open(CACHE_TXT_FILE, "r", encoding="utf-8") as f:
            transcript = f.read().strip()
        print(f"Loaded cached transcript from {CACHE_TXT_FILE} ({len(transcript)} chars, {len(transcript.split())} words).")
    else:
        # Check existing chunk files
        chunk_files = [
            f"scratch/local_a389f13521_chunk_{i}.wav" for i in range(5)
        ]
        existing_chunks = [cf for cf in chunk_files if os.path.exists(cf)]
        
        if len(existing_chunks) == 5:
            print(f"Using {len(existing_chunks)} pre-extracted audio chunks from scratch.")
            chunks = existing_chunks
        else:
            t0 = time.time()
            wav_path = extract_audio_from_file(VIDEO_PATH, output_dir="scratch")
            t_extract = time.time() - t0
            meta = validate_audio(wav_path)
            dur = meta["duration"]
            print(f"Audio extracted in {t_extract:.2f}s. Duration: {dur:.1f}s ({dur/60:.2f}m)")
            
            t0 = time.time()
            chunks = chunk_audio(wav_path, chunk_minutes=10, output_dir="scratch")
            t_chunk = time.time() - t0
            print(f"Chunked into {len(chunks)} audio chunks in {t_chunk:.2f}s.")
            
        print(f"Transcribing {len(chunks)} chunks with faster-whisper-small...")
        t0 = time.time()
        transcript = transcribe_all(chunks)
        t_transcribe = time.time() - t0
        with open(CACHE_TXT_FILE, "w", encoding="utf-8") as f:
            f.write(transcript)
        print(f"Transcription completed in {t_transcribe:.2f}s. Saved to {CACHE_TXT_FILE}.")
        
    print(f"Transcript sample:\n  \"{transcript[:200]}...\"")
    
    # 5. Build Unified Multimodal RAG Chain
    print("\n--- 6. Build Multimodal FAISS Vector Stores & RAG Chain ---")
    t0 = time.time()
    rag_chain = build_rag_chain(transcript, visual_evidence=visual_evidence)
    t_rag_build = time.time() - t0
    print(f"Multimodal RAG chain built in {t_rag_build:.2f}s.")
    print(f"Transcript FAISS docs: {rag_chain.vector_store.index.ntotal}, Visual FAISS docs: {rag_chain.visual_store.index.ntotal if rag_chain.visual_store else 0}")
    
    # 6. Real Video QA Test Suite (Required questions)
    questions = [
        ("How many people are visible in the video?", "VISUAL_ONLY"),
        ("What are the people doing?", "VISUAL_ONLY"),
        ("Who appears to be presenting?", "MULTIMODAL"),
        ("What is visible on the screen?", "VISUAL_ONLY"),
        ("What did they discuss?", "TEXT_ONLY"),
        ("What was being shown when they discussed the topic?", "MULTIMODAL"),
        ("Video mein kitne log dikh rahe hain aur screen par kya chal raha hai?", "MULTIMODAL"),
    ]
    
    print("\n" + "=" * 80)
    print("🔍 7. REAL VIDEO QA EVALUATION ON ACTUAL MEETING")
    print("=" * 80)
    
    qa_results = []
    for q_text, expected_type in questions:
        print(f"\n──────────────────────────────────────────────────")
        actual_intent = classify_question_intent(q_text)
        print(f"Q: \"{q_text}\"")
        print(f"Intent Classification: [{actual_intent}] (Expected: {expected_type})")
        
        # Check retrieved visual evidence
        v_docs = rag_chain.retrieve_visual(q_text, k=2) if rag_chain.visual_store else []
        print(f"Retrieved Visual Evidence Count: {len(v_docs)}")
        if v_docs:
            print(f"Top Visual Snippet: {v_docs[0].page_content.strip()[:180]}...")
            
        t_q0 = time.time()
        answer = ask_question(rag_chain, q_text)
        dt_ans = time.time() - t_q0
        print(f"\nAnswer ({dt_ans:.2f}s):\n{answer}\n")
        
        qa_results.append({
            "question": q_text,
            "expected_intent": expected_type,
            "actual_intent": actual_intent,
            "visual_retrieved": len(v_docs),
            "top_visual_snippet": v_docs[0].page_content.strip()[:200] if v_docs else None,
            "answer_time": dt_ans,
            "answer": answer,
        })
        
    print("\n" + "=" * 80)
    print("📊 REAL VIDEO MULTIMODAL QA SUMMARY REPORT")
    print("=" * 80)
    for res in qa_results:
        print(f"• Q: {res['question']}")
        print(f"  Intent: {res['actual_intent']} | Vis Docs: {res['visual_retrieved']} | Time: {res['answer_time']:.2f}s")
        print(f"  A: {res['answer'][:160]}...")
        print()
        
    # Save full QA results to json for auditing
    with open("scratch/real_video_qa_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "video_path": VIDEO_PATH,
            "visual_evidence_count": len(visual_evidence),
            "qa_results": qa_results,
        }, f, indent=2, ensure_ascii=False)
    print("Detailed QA results saved to scratch/real_video_qa_results.json")

if __name__ == "__main__":
    run_real_qa()
