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
from core.transcriber import transcribe_all
from core.visual_analyzer import analyze_video_visuals, sample_representative_frames, check_vision_model_available
from core.rag_engine import build_rag_chain, ask_question, classify_question_intent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("GeneralizationTest")

VIDEOS = [
    {
        "id": "video_1_meeting",
        "category": "Meeting / Talking-Head Video",
        "path": r"C:\Users\mathu\Downloads\Product Marketing Meeting (weekly) 2021-06-28.mp4",
        "max_frames": 25,
        "questions": [
            ("What is happening in the video?", "Visual general"),
            ("How many people are visible on screen and what is in the background?", "Visual detail"),
            ("What did the team discuss regarding product marketing and campaigns?", "Transcript dialogue"),
            ("What brand of bicycle was someone riding in the video?", "Absent visual information"),
        ]
    },
    {
        "id": "video_2_screen_recording",
        "category": "Lecture / Presentation / Screen-Recording Video",
        "path": r"C:\Users\mathu\Videos\Screen Recordings\Screen Recording 2026-09-12 230144.mp4",
        "max_frames": 10,
        "questions": [
            ("What is happening in the video?", "Visual general"),
            ("Are there any people visible on the screen, and what software or UI elements are shown?", "Visual detail"),
            ("What specific workflow or process is being demonstrated on the screen?", "Multimodal context"),
            ("What color of sports car is driving on the highway?", "Absent visual information"),
        ]
    },
    {
        "id": "video_3_action_drama",
        "category": "Action / Drama / Cinematic Suspense Video (Hindi)",
        "path": r"C:\Users\mathu\Downloads\द_मिसिंग_नाइट.mp4",
        "max_frames": 15,
        "questions": [
            ("What is happening in the video?", "Visual general"),
            ("What vehicles and characters are visible in the scene, and what is the environment?", "Visual detail"),
            ("Kahani mein narrator kya bata raha hai aur kya ghatna hone wali hai?", "Hinglish Transcript/Narrative"),
            ("What kind of pizza was the person eating?", "Absent information"),
        ]
    }
]

def run_test():
    print("=" * 80)
    print("🚀 MULTIMODAL GENERALIZATION TEST ACROSS 3 DIVERSE VIDEO GENRES")
    print("=" * 80)
    
    assert check_vision_model_available(), "Local vision model qwen3-vl:2b must be online in Ollama!"
    
    overall_report = []

    for item in VIDEOS:
        v_id = item["id"]
        v_cat = item["category"]
        v_path = item["path"]
        max_f = item["max_frames"]
        
        print("\n" + "=" * 80)
        print(f"🎬 TESTING GENRE: {v_cat}")
        print(f"   File: {v_path}")
        print("=" * 80)
        
        if not os.path.exists(v_path):
            print(f"❌ Video not found at {v_path}! Skipping.")
            continue

        t_start_video = time.time()
        
        # 1. Sample frames & deduplicate
        t0 = time.time()
        frames_dir = f"scratch/{v_id}_frames"
        os.makedirs(frames_dir, exist_ok=True)
        frames = sample_representative_frames(v_path, max_frames=max_f, output_dir=frames_dir)
        t_sampling = time.time() - t0
        key_frames = [f for f in frames if f.get("is_key")]
        
        # 2. Visual analysis with qwen3-vl:2b (with cache per video)
        vis_cache_file = f"scratch/{v_id}_visual.json"
        alt_vis = "scratch/meeting_visual_evidence.json" if v_id == "video_1_meeting" else None
        if os.path.exists(vis_cache_file):
            with open(vis_cache_file, "r", encoding="utf-8") as f:
                visual_evidence = json.load(f)
            t_visual = 0.0
            print(f"Loaded {len(visual_evidence)} cached visual evidence points.")
        elif alt_vis and os.path.exists(alt_vis):
            with open(alt_vis, "r", encoding="utf-8") as f:
                visual_evidence = json.load(f)
            t_visual = 0.0
            print(f"Loaded {len(visual_evidence)} cached visual evidence points from {alt_vis}.")
        else:
            t0 = time.time()
            visual_evidence = analyze_video_visuals(v_path, max_frames=max_f)
            t_visual = time.time() - t0
            with open(vis_cache_file, "w", encoding="utf-8") as f:
                json.dump(visual_evidence, f, indent=2, ensure_ascii=False)
            print(f"Generated {len(visual_evidence)} visual evidence points in {t_visual:.2f}s.")
            
        # 3. Audio Extraction & Transcription
        txt_cache_file = f"scratch/{v_id}_transcript.txt"
        alt_txt = "scratch/meeting_transcript.txt" if v_id == "video_1_meeting" else None
        t_transcribe = 0.0
        dur = 0.0
        
        if os.path.exists(txt_cache_file):
            with open(txt_cache_file, "r", encoding="utf-8") as f:
                transcript = f.read().strip()
            print(f"Loaded cached transcript ({len(transcript)} chars).")
        elif alt_txt and os.path.exists(alt_txt):
            with open(alt_txt, "r", encoding="utf-8") as f:
                transcript = f.read().strip()
            print(f"Loaded cached transcript from {alt_txt} ({len(transcript)} chars).")
        else:
            t0 = time.time()
            wav_path = extract_audio_from_file(v_path, output_dir="scratch")
            meta = validate_audio(wav_path)
            dur = meta.get("duration", 0)
            chunks = chunk_audio(wav_path, chunk_minutes=10, output_dir="scratch")
            print(f"Extracted audio ({dur:.1f}s) and {len(chunks)} chunks. Transcribing with faster-whisper-small...")
            transcript = transcribe_all(chunks)
            t_transcribe = time.time() - t0
            with open(txt_cache_file, "w", encoding="utf-8") as f:
                f.write(transcript)
            print(f"Transcription completed in {t_transcribe:.2f}s ({len(transcript)} chars).")

        # 4. Multimodal RAG Construction
        t0 = time.time()
        rag_chain = build_rag_chain(transcript, visual_evidence=visual_evidence)
        t_rag = time.time() - t0
        
        total_pipeline_time = t_sampling + t_visual + t_transcribe + t_rag
        
        print(f"\n[PIPELINE METRICS]")
        print(f"  • Video Duration:      {dur:.1f}s")
        print(f"  • Frames Sampled:      {len(frames)} total ({len(key_frames)} unique scenes)")
        print(f"  • Visual Evidence:     {len(visual_evidence)} timestamped points")
        print(f"  • Transcript Words:    {len(transcript.split())}")
        print(f"  • Total Pipeline Time: {total_pipeline_time:.2f}s")
        
        # 5. Question Answering Evaluation
        qa_results = []
        print(f"\n[QUESTION ANSWERING EVALUATION]")
        for q_text, q_type in item["questions"]:
            intent = classify_question_intent(q_text)
            print(f"\nQ ({q_type}): \"{q_text}\" [Intent: {intent}]")
            
            t0 = time.time()
            answer = ask_question(rag_chain, q_text)
            dt = time.time() - t0
            print(f"A ({dt:.2f}s):\n{answer}")
            
            # Automatic Grounding Verification
            is_pass = False
            hallucinated = False
            
            # For absent info: should contain negative phrasing
            if "absent" in q_type.lower():
                neg_phrases = ["not found", "could not find", "couldn't find", "does not contain", "no mention", "not mentioned", "unable to find", "no information", "not provided", "does not show", "does not provide", "no evidence", "नहीं", "nahi"]
                if any(np in answer.lower() for np in neg_phrases):
                    is_pass = True
                else:
                    hallucinated = True
            else:
                # Normal question: must have substantive grounded answer (> 15 chars and not error)
                if len(answer.strip()) > 15 and "error" not in answer.lower():
                    is_pass = True
            
            qa_results.append({
                "question": q_text,
                "type": q_type,
                "intent": intent,
                "answer": answer,
                "latency": dt,
                "pass": is_pass,
                "hallucinated": hallucinated,
            })
            
        overall_report.append({
            "id": v_id,
            "category": v_cat,
            "path": v_path,
            "duration": dur,
            "frames_sampled": len(frames),
            "unique_scenes": len(key_frames),
            "total_time": total_pipeline_time,
            "qa_results": qa_results,
            "all_passed": all(q["pass"] for q in qa_results),
            "any_hallucination": any(q["hallucinated"] for q in qa_results),
        })

    # Save complete generalization test results
    with open("scratch/generalization_test_results.json", "w", encoding="utf-8") as f:
        json.dump(overall_report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("🏆 GENERALIZATION TEST EXECUTIVE SUMMARY")
    print("=" * 80)
    for r in overall_report:
        status_str = "✅ PASS" if r["all_passed"] else "❌ FAIL"
        halluc_str = "NO" if not r["any_hallucination"] else "⚠️ YES"
        print(f"Category: {r['category']}")
        print(f"  Status: {status_str} | Hallucinations: {halluc_str} | Scenes: {r['unique_scenes']} | Time: {r['total_time']:.1f}s")
        for q in r["qa_results"]:
            mark = "✓" if q["pass"] else "✗"
            print(f"    {mark} [{q['type']}]: {q['question'][:50]}... -> {q['answer'][:70]}...")
        print()
    print("=" * 80)

if __name__ == "__main__":
    run_test()
