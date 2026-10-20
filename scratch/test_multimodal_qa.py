import os
import sys
import time
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass
from dotenv import load_dotenv

load_dotenv()

from core.rag_engine import build_rag_chain, ask_question

transcript = (
    "Good morning team. Today is October 14th and we are conducting our weekly review. "
    "Sarah announced that the new payment gateway integration is scheduled for launch on November 10th. "
    "Michael agreed to take ownership of the load testing and security audits before the release. "
    "The leadership decided to allocate twenty thousand dollars for cloud infrastructure scaling. "
    "A critical question was raised regarding whether our API rate limits can handle expected traffic."
)

visual_evidence = [
    {
        "frame_index": 0,
        "timestamp": "00:00:01",
        "timestamp_seconds": 1.0,
        "people_count": 9,
        "description": (
            "A video call screen showing 9 participants arranged in a 3x3 grid. "
            "The top row includes a person with a GitLab logo on a purple background. "
            "All participants wear headphones in diverse home/office settings."
        ),
        "summary_text": (
            "[VISUAL EVIDENCE 00:00:01]\n"
            "A video call screen showing 9 participants arranged in a 3x3 grid. "
            "The top row includes a person with a GitLab logo on a purple background. "
            "All participants wear headphones in diverse home/office settings."
        ),
    },
    {
        "frame_index": 5,
        "timestamp": "00:15:00",
        "timestamp_seconds": 900.0,
        "people_count": 5,
        "description": (
            "Visible people count: 5 in the video call panel on the right. "
            "A presentation in Google Slides is displayed titled 'GitLab Platform Message - 2021-06-24 through Commit'. "
            "Main titles: 'Iterate Faster, innovate together' and 'The modern DevOps Platform'. "
            "Three pillars visible: Visibility ('A single source of truth'), Security ('End-to-end control'), and Velocity ('Move fast with confidence')."
        ),
        "summary_text": (
            "[VISUAL EVIDENCE 00:15:00]\n"
            "Visible people count: 5 in the video call panel on the right. "
            "A presentation in Google Slides is displayed titled 'GitLab Platform Message - 2021-06-24 through Commit'. "
            "Main titles: 'Iterate Faster, innovate together' and 'The modern DevOps Platform'. "
            "Three pillars visible: Visibility ('A single source of truth'), Security ('End-to-end control'), and Velocity ('Move fast with confidence')."
        ),
    }
]

print("Building multimodal RAG chain...")
rag = build_rag_chain(transcript, visual_evidence=visual_evidence)

test_questions = [
    ("Visual-Only (English)", "How many people were visible in the video call?"),
    ("Visual-Only (Hindi)", "Video mein kitne log maujood the?"),
    ("Visual-Only Slide Content", "What was displayed on the presentation slide?"),
    ("Text-Only (English)", "What did Sarah announce regarding the launch?"),
    ("Text-Only (Hinglish)", "Michael kis cheez ki responsibility le raha tha?"),
    ("Multimodal (Combined)", "Who was presenting and what slide was on screen?"),
    ("Insufficient Visual Evidence", "What brand of shoes was the speaker wearing?"),
    ("Insufficient Transcript Evidence", "What was the weather outside during the meeting?"),
]

for label, q in test_questions:
    print(f"\n==================================================")
    print(f"[{label}] Q: {q}")
    t0 = time.time()
    ans = ask_question(rag, q)
    dt = time.time() - t0
    print(f"A ({dt:.2f}s): {ans}")
