import os
import sys
import time
import logging
from dotenv import load_dotenv

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Load .env before any core modules
load_dotenv()

from utils.audio_processor import process_input, check_ffmpeg, cleanup_temp_files, AudioProcessingError
from core.local_llm import check_ollama, check_model_available, OLLAMA_MODEL
from core.transcriber import transcribe_all, TranscriptionError
from core.summarizer import summarize, generate_title, generate_video_intelligence
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_pipeline(source: str) -> dict:
    """Execute end-to-end 100% local pipeline from source media to FAISS RAG chain."""
    print("\n" + "=" * 60)
    print(f"🎬 AI Video Assistant — Local AI Processing ({OLLAMA_MODEL})")
    print("=" * 60)

    # 1. Verify FFmpeg
    try:
        check_ffmpeg()
    except RuntimeError as e:
        print(f"\n❌ System Requirement Error:\n{e}\n")
        sys.exit(1)

    # 2. Verify Local Ollama
    if not check_ollama():
        print(f"\n❌ Local Ollama server is not running. Start Ollama and try again.\n")
        sys.exit(1)

    if not check_model_available(OLLAMA_MODEL):
        print(f"\n❌ Model '{OLLAMA_MODEL}' is not available locally. Run: ollama pull {OLLAMA_MODEL}\n")

    chunks = []
    t_start_total = time.time()
    try:
        # 3. Process media into chunks
        print("\n[1/4] 🔊 Processing and extracting audio...")
        t0 = time.time()
        chunks = process_input(source)
        t_audio = time.time() - t0
        print(f"      ✓ Audio converted and divided into {len(chunks)} chunk(s) in {t_audio:.2f}s.")

        # 4. Local Transcribe with Whisper (automatic language detection)
        print("\n[2/4] 📝 Transcribing locally with Whisper (auto-detect language)...")
        t0 = time.time()
        transcript, detected_language = transcribe_all(chunks, language=None)
        t_trans = time.time() - t0
        print(f"      ✓ Transcription complete ({len(transcript)} characters) in {t_trans:.2f}s.")
        print(f"      ✓ Detected language: {detected_language or 'unknown'}")

        if not transcript.strip():
            print("\n❌ Error: No speech recognized. Cannot proceed with analysis.\n")
            return {}

        # 5. Local Consolidated Video Intelligence via Ollama
        print(f"\n[3/4] 🧠 Generating video intelligence dossier locally with {OLLAMA_MODEL}...")
        t0 = time.time()
        intel = generate_video_intelligence(transcript)
        t_intel = time.time() - t0
        title = intel.get("title", "Video Intelligence Dossier")
        summary = intel.get("summary", "")
        action_items = intel.get("action_items", "")
        decisions = intel.get("key_decisions", "")
        questions = intel.get("open_questions", "")
        print(f"      ✓ Title: {title}")
        print(f"      ✓ Summary and insights generated in {t_intel:.2f}s.")

        # 6. FAISS Vector Store & RAG Chain
        print("\n[4/4] ⚡ Indexing transcript into local FAISS vector store...")
        t0 = time.time()
        rag_chain = build_rag_chain(transcript)
        t_rag = time.time() - t0
        print(f"      ✓ Local FAISS RAG chain ready in {t_rag:.2f}s.")

        t_total = time.time() - t_start_total

        print("\n" + "-" * 60)
        print(f"⏱️  PERFORMANCE BREAKDOWN:")
        print(f"   • Audio Extraction & Validation: {t_audio:.2f}s")
        print(f"   • Speech Transcription (Whisper): {t_trans:.2f}s")
        print(f"   • Video Intelligence (Ollama):    {t_intel:.2f}s")
        print(f"   • FAISS Knowledge Indexing:      {t_rag:.2f}s")
        print(f"   • TOTAL PIPELINE TIME:            {t_total:.2f}s")
        print("-" * 60)

        return {
            "title": title,
            "transcript": transcript,
            "summary": summary,
            "action_items": action_items,
            "key_decisions": decisions,
            "open_questions": questions,
            "rag_chain": rag_chain,
            "timing": {
                "audio": t_audio,
                "transcription": t_trans,
                "intelligence": t_intel,
                "rag": t_rag,
                "total": t_total,
            }
        }

    except (AudioProcessingError, TranscriptionError, ValueError, RuntimeError) as e:
        print(f"\n❌ Pipeline Error: {e}\n")
        return {}
    except Exception as e:
        logger.exception("Unexpected pipeline failure")
        print(f"\n❌ Unexpected error: {e}\n")
        return {}
    finally:
        if chunks:
            cleanup_temp_files(chunks)


if __name__ == "__main__":
    source = input("Enter YouTube URL or local file path: ").strip()
    if not source:
        print("No source provided. Exiting.")
        sys.exit(0)

    result = run_pipeline(source)
    if not result or not result.get("transcript"):
        sys.exit(1)

    print("\n" + "=" * 60)
    print(f"📌 TITLE: {result['title']}")
    print("=" * 60)
    print(f"\n📋 SUMMARY:\n{result['summary']}")
    print(f"\n✅ ACTION ITEMS:\n{result['action_items']}")
    print(f"\n🔑 KEY DECISIONS:\n{result['key_decisions']}")
    print(f"\n❓ OPEN QUESTIONS:\n{result['open_questions']}")
    print("=" * 60)

    if result.get("rag_chain"):
        print("\n💬 Chat with your recording in ANY language (type 'exit' to quit)\n")
        rag_chain = result["rag_chain"]
        while True:
            try:
                question = input("You: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n👋 Goodbye!")
                break
            if question.lower() in ["exit", "quit", "q"]:
                print("👋 Goodbye!")
                break
            if not question:
                continue
            answer = ask_question(rag_chain, question)
            print(f"\n🤖 Assistant: {answer}\n")