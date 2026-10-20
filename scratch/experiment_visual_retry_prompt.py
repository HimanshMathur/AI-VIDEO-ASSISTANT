"""Live Ollama experiment for an interval-aware, evidence-grounded visual retry."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import core.rag_engine as rag
from core.local_llm import OLLAMA_BASE_URL, OLLAMA_MODEL

DATA = json.loads((ROOT / "downloads/_cache/yaS7O6FQEkc.json").read_text(encoding="utf-8"))
chain = rag.build_rag_chain(DATA["transcript"], visual_evidence=DATA["visual_evidence"],
                            duration_seconds=DATA["duration_seconds"], video_id=DATA["_source_id"])
questions = ["What is visible at 5:00?", "What is happening visually at 5:06?"]
results = []
for question in questions:
    target = rag._question_timestamp(question)
    messages, intent, transcript_docs, visual_docs, context = rag._prepare_rag_messages(chain, question)
    assert intent == "VISUAL_ONLY" and visual_docs and not transcript_docs
    intervals = [
        (float(doc.metadata.get("timestamp_seconds", 0)),
         float(doc.metadata.get("end_seconds", doc.metadata.get("timestamp_seconds", 0))),
         doc.metadata.get("timestamp_display") or doc.metadata.get("timestamp"))
        for doc in visual_docs
    ]
    active = [x for x in intervals if x[0] <= target <= x[1]]
    assert active, (question, intervals)
    active_text = ", ".join(f"{label} to {rag._format_chapter_time(end)}" for start, end, label in active)
    system = (
        "Answer the user's question using only the supplied visual observations. "
        "When a question broadly asks what is visible or happening, summarize the described frame: "
        "use its scene, people, actions, objects, and important-details observations. "
        "Do not treat a broad question as asking for one specific detail. "
        "If one requested detail is absent, say that detail is unknown, then state any relevant details that are present. "
        "Do not claim that no visual evidence exists when observations are supplied, and do not add outside facts.\n\n"
        f"Requested moment: {rag._format_chapter_time(target)}. "
        f"The retrieved visual record is active during {active_text}; its start time may precede the requested moment.\n\n"
        f"Retrieved visual observations:\n{context}"
    )
    candidate = [{"role": "system", "content": system}, {"role": "user", "content": question}]
    raw = rag.chat_completion(candidate, temperature=0.0, max_tokens=256, timeout=90)
    results.append({
        "question": question,
        "target_seconds": target,
        "active_intervals": intervals,
        "candidate_prompt": candidate,
        "raw_response": raw,
    })
    print(json.dumps({"question": question, "active_intervals": intervals, "response": raw}, ensure_ascii=False), flush=True)

(ROOT / "scratch/visual_retry_prompt_experiment.json").write_text(
    json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
