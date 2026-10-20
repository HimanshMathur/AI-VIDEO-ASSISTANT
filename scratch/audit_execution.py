import os
import sys
import time
import json
import logging

# Ensure UTF-8 output on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
    force=True
)
logger = logging.getLogger("AUDIT")

# Ensure project root is on sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.audio_processor import (
    validate_youtube_url,
    download_youtube_audio,
    download_youtube_video,
    validate_audio,
    chunk_audio,
)
from core.transcriber import (
    load_model,
    transcribe_all,
    transcribe_with_segments,
    transcribe_chunk_whisper,
    _device_in_use,
    _compute_type_in_use,
    WHISPER_MODEL,
)
from core.visual_analyzer import (
    analyze_video_visuals,
    check_vision_model_available,
    is_video_file,
    VISION_MODEL,
)
from core.summarizer import generate_video_intelligence
from core.rag_engine import build_rag_chain, ask_question

TARGET_URL = "https://youtu.be/Aw7k6LRSM14?si=baJ8pgYtmH27q8RS"
print("================================================================================")
print(f"STARTING COMPREHENSIVE AUDIT ON: {TARGET_URL}")
print("================================================================================")

audit_data = {}

# 1. Validation
t0 = time.time()
is_valid = validate_youtube_url(TARGET_URL)
print(f"Step 1 - URL Validation: {is_valid} (in {time.time()-t0:.2f}s)")
audit_data["url_valid"] = is_valid

# 2. Audio Download
t0 = time.time()
print("Step 2 - Downloading Audio via yt-dlp...")
try:
    wav_path = download_youtube_audio(TARGET_URL)
    print(f"Audio downloaded to: {wav_path} (size: {os.path.getsize(wav_path)} bytes) in {time.time()-t0:.2f}s")
    audit_data["wav_path"] = wav_path
    audit_data["wav_size"] = os.path.getsize(wav_path)
except Exception as e:
    print(f"Audio download FAILED: {e}")
    audit_data["wav_error"] = str(e)
    wav_path = None

# 2b. Video Download
t0 = time.time()
print("Step 2b - Downloading Video for Vision Analysis...")
try:
    mp4_path = download_youtube_video(TARGET_URL)
    print(f"Video downloaded to: {mp4_path} (size: {os.path.getsize(mp4_path) if mp4_path else 0} bytes) in {time.time()-t0:.2f}s")
    audit_data["mp4_path"] = mp4_path
    audit_data["mp4_size"] = os.path.getsize(mp4_path) if mp4_path else 0
except Exception as e:
    print(f"Video download FAILED: {e}")
    audit_data["mp4_error"] = str(e)
    mp4_path = None

# 3. Audio Validation & Chunking
if wav_path:
    meta = validate_audio(wav_path)
    print(f"Step 3 - Audio Metadata: duration={meta['duration']:.2f}s, rate={meta['sample_rate']}Hz, channels={meta['channels']}, rms={meta['rms']:.4f}")
    audit_data["audio_meta"] = meta
    chunks = chunk_audio(wav_path)
    print(f"Chunking: produced {len(chunks)} chunk(s)")
    audit_data["chunk_count"] = len(chunks)

# 4. Transcription - Both transcribe_all (Production path) and transcribe_with_segments
if wav_path:
    print("\nStep 4 - Running Production transcribe_all()...")
    t0 = time.time()
    try:
        prod_transcript = transcribe_all(chunks)
        t_trans = time.time() - t0
        print(f"transcribe_all() succeeded in {t_trans:.2f}s (length: {len(prod_transcript)} chars, {len(prod_transcript.split())} words)")
        print(f"Production Device in use: {_device_in_use}, compute_type: {_compute_type_in_use}")
        audit_data["prod_transcript"] = prod_transcript
        audit_data["transcribe_all_time"] = t_trans
    except Exception as e:
        print(f"transcribe_all() FAILED: {e}")
        audit_data["prod_transcript_error"] = str(e)
        prod_transcript = ""

    print("\nStep 4b - Running transcribe_with_segments() for granular segment & timestamp audit...")
    t0 = time.time()
    try:
        seg_result = transcribe_with_segments(wav_path)
        t_seg = time.time() - t0
        print(f"transcribe_with_segments() succeeded in {t_seg:.2f}s, segments: {len(seg_result['segments'])}, language: {seg_result['language']}")
        audit_data["seg_result"] = seg_result
        audit_data["transcribe_segments_time"] = t_seg
    except Exception as e:
        print(f"transcribe_with_segments() FAILED: {e}")
        audit_data["seg_result_error"] = str(e)
        seg_result = {"segments": [], "text": "", "language": None}

# 5. Visual Frame Analysis
if mp4_path and os.path.exists(mp4_path) and is_video_file(mp4_path):
    print("\nStep 5 - Running Visual Frame Analysis...")
    t0 = time.time()
    has_vision = check_vision_model_available()
    print(f"Vision model ({VISION_MODEL}) available: {has_vision}")
    if has_vision:
        try:
            visual_evidence = analyze_video_visuals(mp4_path, max_frames=20)
            t_vis = time.time() - t0
            print(f"Visual analysis succeeded in {t_vis:.2f}s, extracted {len(visual_evidence)} frames")
            audit_data["visual_evidence"] = visual_evidence
            audit_data["visual_time"] = t_vis
        except Exception as e:
            print(f"Visual analysis FAILED: {e}")
            audit_data["visual_error"] = str(e)
            visual_evidence = []
    else:
        visual_evidence = []
else:
    visual_evidence = []

# 6. Video Intelligence
if audit_data.get("prod_transcript"):
    print("\nStep 6 - Running Ollama Video Intelligence (Dossier)...")
    t0 = time.time()
    try:
        intel = generate_video_intelligence(audit_data["prod_transcript"])
        t_intel = time.time() - t0
        print(f"generate_video_intelligence() succeeded in {t_intel:.2f}s")
        audit_data["intel"] = intel
        audit_data["intel_time"] = t_intel
    except Exception as e:
        print(f"generate_video_intelligence() FAILED: {e}")
        audit_data["intel_error"] = str(e)

# 7. RAG Indexing & Multilingual Q&A Tests
if audit_data.get("prod_transcript"):
    print("\nStep 7 - Building RAG Index...")
    t0 = time.time()
    vid_dur = audit_data.get("audio_meta", {}).get("duration", 0.0)
    rag_chain = build_rag_chain(audit_data["prod_transcript"], visual_evidence=visual_evidence, duration_seconds=vid_dur)
    print(f"build_rag_chain() succeeded in {time.time()-t0:.2f}s")

    # Multilingual tests requested by user:
    # 1. Ask in Hinglish: "Mujhe video mein discuss hui sabhi baatein Russian language mein samjhao."
    # 2. Ask in English for an explanation in Russian: "Explain all topics discussed in the video in Russian."
    # 3. Ask in Hindi or Hinglish and request answer in English: "Video mein YouTube video ke chapters kaise add karein? Answer in English."
    # 4. Visual query: "What screen or UI was shown when discussing timestamps?"
    
    queries = [
        ("q1_hinglish_to_russian", "Mujhe video mein discuss hui sabhi baatein Russian language mein samjhao."),
        ("q2_english_to_russian", "Explain all topics discussed in the video in Russian language."),
        ("q3_hinglish_to_english", "Video mein YouTube video ke chapters kaise add karein? Answer in English."),
        ("q4_visual_ui", "What was visible on the screen during the video demonstration? What UI or app was shown?"),
    ]

    qa_results = {}
    for qkey, qtext in queries:
        print(f"\n--- Testing Query: {qkey} ---")
        print(f"Question: {qtext}")
        t_q = time.time()
        try:
            ans = ask_question(rag_chain, qtext)
            print(f"Answer ({time.time()-t_q:.2f}s):\n{ans}\n")
            qa_results[qkey] = {"question": qtext, "answer": ans, "time": time.time()-t_q}
        except Exception as e:
            print(f"Query {qkey} FAILED: {e}")
            qa_results[qkey] = {"question": qtext, "error": str(e)}

    audit_data["qa_results"] = qa_results

# Save raw audit data
with open("scratch/audit_full_results.json", "w", encoding="utf-8") as f:
    json.dump(audit_data, f, ensure_ascii=False, indent=2)

print("\n================================================================================")
print("AUDIT EXECUTION COMPLETE. Saved to scratch/audit_full_results.json")
print("================================================================================")
