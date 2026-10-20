"""Real local-model reproduction through the same ask_question() used by app.py."""
import json
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import core.rag_engine as rag
from core.local_llm import OLLAMA_BASE_URL, OLLAMA_MODEL


DATA = ROOT / "downloads" / "_cache" / "yaS7O6FQEkc.json"
OUTPUT = ROOT / "scratch" / "live_visual_qa_retest.json"
QUESTIONS = [
    "what is happen in the video at 5:06 sec what show in the video",
    "What is visible at 5:00?",
    "What is happening visually at 5:06?",
    "What color is the kettle visible at 5:06?",
    "What is happening at 7:00?",
]


def safe_model_base(url: str) -> str:
    # This diagnostic records only the local origin/path, never any URL credentials.
    from urllib.parse import urlsplit

    parts = urlsplit(url)
    if parts.hostname not in ("localhost", "127.0.0.1", "::1"):
        raise RuntimeError("Configured Ollama endpoint is not local; refusing to send prompts.")
    return f"{parts.scheme}://{parts.hostname}:{parts.port or 80}{parts.path}"


base = safe_model_base(OLLAMA_BASE_URL).removesuffix("/v1")
tags = requests.get(base + "/api/tags", timeout=10).json()
available = [m.get("name") for m in tags.get("models", [])]
if OLLAMA_MODEL not in available:
    raise RuntimeError(f"Configured model {OLLAMA_MODEL!r} unavailable; local models={available!r}")

data = json.loads(DATA.read_text(encoding="utf-8"))
chain = rag.build_rag_chain(
    data["transcript"], visual_evidence=data["visual_evidence"],
    duration_seconds=data["duration_seconds"], video_id=data["_source_id"],
)

records = []
original_chat = rag.chat_completion


def captured_chat(messages, *args, **kwargs):
    raw = original_chat(messages, *args, **kwargs)
    records[-1]["model_calls"].append({
        "messages": messages,
        "raw_response": raw,
    })
    return raw


rag.chat_completion = captured_chat
for question in QUESTIONS:
    intent = rag.classify_question_intent(question, has_visual_store=True)
    target = rag._question_timestamp(question)
    record = {
        "question": question,
        "parsed_timestamp_seconds": target,
        "intent": intent,
        "selected_frames": [],
        "visual_evidence_in_actual_model_prompt": [],
        "model_calls": [],
        "final_ui_response": None,
    }
    records.append(record)

    # Wrap the chain methods during the actual ask_question() call so these are
    # the records retrieved by that call, not a mocked retrieval result.
    original_ts_retrieve = chain.retrieve_visual_by_timestamp
    original_visual_retrieve = chain.retrieve_visual

    def timestamp_retrieve(seconds, k=3):
        docs = original_ts_retrieve(seconds, k=k)
        record["selected_frames"].extend({
            "timestamp": d.metadata.get("timestamp"),
            "timestamp_seconds": d.metadata.get("timestamp_seconds"),
            "end_seconds": d.metadata.get("end_seconds"),
            "description": d.page_content,
        } for d in docs)
        return docs

    def visual_retrieve(query, k=4):
        docs = original_visual_retrieve(query, k=k)
        return docs

    chain.retrieve_visual_by_timestamp = timestamp_retrieve
    chain.retrieve_visual = visual_retrieve
    start = time.time()
    record["final_ui_response"] = rag.ask_question(chain, question)
    record["elapsed_seconds"] = round(time.time() - start, 2)
    chain.retrieve_visual_by_timestamp = original_ts_retrieve
    chain.retrieve_visual = original_visual_retrieve

    if record["model_calls"]:
        for call in record["model_calls"]:
            for message in call["messages"]:
                if message.get("role") == "system":
                    context = message.get("content", "")
                    section = "--- [VISUAL EVIDENCE (OBSERVED VIDEO FRAMES)] ---"
                    if section in context:
                        evidence = context.split(section, 1)[1].split("\n\n--- [", 1)[0]
                        record["visual_evidence_in_actual_model_prompt"].append(evidence)
        # The evidence as actually sent is authoritative. Drop duplicate
        # retrieval observations introduced if a focused model retry occurred.
        if record["visual_evidence_in_actual_model_prompt"]:
            record["selected_frames"] = [
                {"timestamp": d.metadata.get("timestamp"),
                 "timestamp_seconds": d.metadata.get("timestamp_seconds"),
                 "end_seconds": d.metadata.get("end_seconds"),
                 "description": d.page_content}
                for d in rag._prepare_rag_messages(chain, question)[3]
            ]

    print(json.dumps({
        "question": question,
        "timestamp_seconds": target,
        "intent": intent,
        "selected_frame_timestamps": [f["timestamp"] for f in record["selected_frames"]],
        "model_calls": len(record["model_calls"]),
        "raw_model_response": [c["raw_response"] for c in record["model_calls"]],
        "final_ui_response": record["final_ui_response"],
        "elapsed_seconds": record["elapsed_seconds"],
    }, ensure_ascii=False), flush=True)

OUTPUT.write_text(json.dumps({
    "project_root": str(ROOT),
    "ollama_model": OLLAMA_MODEL,
    "ollama_base": base,
    "available_local_models": available,
    "source_video_id": "yaS7O6FQEkc",
    "source_duration_seconds": data["duration_seconds"],
    "records": records,
}, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"RECORD={OUTPUT}", flush=True)
