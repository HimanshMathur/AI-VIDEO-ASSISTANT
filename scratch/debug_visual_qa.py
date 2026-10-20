import os
import sys
import time
import json
import logging

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")

from core.visual_analyzer import sample_representative_frames, analyze_video_visuals
from core.vector_store import build_vector_store, build_visual_vector_store
from core.rag_engine import build_rag_chain, classify_question_intent, _prepare_rag_messages, ask_question

VIDEO_PATH = r"C:\Users\mathu\OneDrive\Desktop\Rec.mp4"
print(f"=== TESTING VISUAL PIPELINE ON: {VIDEO_PATH} ===")

# Run visual analysis
t0 = time.time()
visual_evidence = analyze_video_visuals(VIDEO_PATH, max_frames=20)
t_vis = time.time() - t0
print(f"Visual analysis returned {len(visual_evidence)} evidence points in {t_vis:.2f}s.")

# Inspect visual evidence items
for i, ev in enumerate(visual_evidence):
    if ev.get("is_key"):
        print(f"\n[KEY FRAME {i}] ts={ev.get('timestamp')} people={ev.get('people_count')}")
        print(f"summary_text ({len(ev.get('summary_text',''))} chars): {ev.get('summary_text','')[:150]}...")

# Build RAG chain (transcript is empty or silent audio placeholder)
transcript = "[Video recording without spoken audio dialogue]"
rag_chain = build_rag_chain(transcript, visual_evidence=visual_evidence)

print("\n" + "="*80)
print(f"Transcript FAISS docs: {rag_chain.vector_store.index.ntotal}")
print(f"Visual FAISS docs: {rag_chain.visual_store.index.ntotal if rag_chain.visual_store else 0}")
print("="*80)

# 9 Visual QA Tests
test_questions = [
    "What is visible on the screen?",
    "What is happening in the video?",
    "How many people are visible?",
    "What are the people doing?",
    "What is shown on the screen?",
    "What object is visible?",
    "Did the screen/content change?",
    "स्क्रीन पर क्या दिखाई दे रहा है?",
    "Screen pe kya chal raha hai aur kitne log hain?",
    "What color of sports car is driving on the highway?",  # Definitely does NOT exist test
]

for q in test_questions:
    print("\n" + "-"*60)
    print(f"QUERY:\n{q}")
    
    # 1. Routing decision
    route = classify_question_intent(q, has_visual_store=True)
    print(f"\nROUTER DECISION:\n{route}")
    
    # 2. Visual retrieval
    v_docs = rag_chain.retrieve_visual(q, k=4)
    print(f"\nRETRIEVED VISUAL EVIDENCE ({len(v_docs)} docs):")
    for d in v_docs:
        ts = d.metadata.get("timestamp", "00:00:00")
        score = d.metadata.get("score", 0.0)
        print(f"[{ts}] (score: {score:.3f}): {d.page_content[:100]}...")
        
    # 3. Final context sent to Qwen3:8B
    msgs = _prepare_rag_messages(rag_chain, q)
    sys_prompt = msgs[0]["content"] if msgs else ""
    # find where Retrieved Video Evidence begins
    ctx_idx = sys_prompt.find("Retrieved Video Evidence:")
    ctx_sent = sys_prompt[ctx_idx:] if ctx_idx != -1 else sys_prompt
    print(f"\nFINAL CONTEXT SENT TO QWEN3:8B ({len(ctx_sent)} chars):\n{ctx_sent[:300]}...")
    
    # 4. Final Answer
    t_a0 = time.time()
    ans = ask_question(rag_chain, q)
    t_a = time.time() - t_a0
    print(f"\nFINAL ANSWER ({t_a:.2f}s):\n{ans}")
