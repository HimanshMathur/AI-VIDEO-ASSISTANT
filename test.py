import os
import sys
import wave
import uuid
import shutil
import tempfile
import subprocess
import numpy as np
from dotenv import load_dotenv

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
        sys.stderr.reconfigure(encoding='utf-8', line_buffering=True)
    except Exception:
        pass

load_dotenv()

from utils.audio_processor import (
    check_ffmpeg,
    validate_youtube_url,
    download_youtube_audio,
    extract_audio_from_file,
    validate_audio,
    chunk_audio,
    process_input,
    cleanup_temp_files,
    MediaValidationError,
    AudioProcessingError,
)
from core.transcriber import transcribe_all, transcribe_chunk_whisper, TranscriptionError
from core.local_llm import check_ollama, check_model_available, OLLAMA_MODEL
from core.summarizer import summarize, generate_title, generate_video_intelligence
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.vector_store import build_vector_store, get_retriever
from core.rag_engine import build_rag_chain, ask_question


def print_test_header(name: str):
    print("\n" + "=" * 70)
    print(f"🧪 {name}")
    print("=" * 70)


def generate_synthetic_wav(filename: str, duration_sec: float = 2.0, silent: bool = False, sample_rate: int = 16000):
    """Generate a clean synthetic WAV file for testing."""
    n_samples = int(duration_sec * sample_rate)
    if silent or n_samples == 0:
        samples = np.zeros(n_samples, dtype=np.int16)
    else:
        t = np.linspace(0, duration_sec, n_samples, False)
        tone = np.sin(2 * np.pi * 440 * t) * 16384.0
        samples = tone.astype(np.int16)

    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())


def run_all_tests():
    passed = 0
    failed = 0
    temp_dir = tempfile.mkdtemp(prefix="ai_video_local_tests_")

    try:
        # ── TEST: Check Local Ollama Availability ──────────────────
        print_test_header("TEST: Local Ollama Availability & Model Inspection")
        try:
            assert check_ollama() is True, "Ollama server is not reachable at local host"
            print(f"✓ Local Ollama is online and reachable. Configured model: '{OLLAMA_MODEL}'")
            passed += 1
        except Exception as e:
            print(f"❌ Ollama Check Failed: {e}")
            failed += 1

        # ── TEST: Check FFmpeg Availability ────────────────────────
        print_test_header("TEST: Check FFmpeg Availability")
        try:
            assert check_ffmpeg() is True
            print("✓ FFmpeg is correctly detected on system PATH.")
            passed += 1
        except Exception as e:
            print(f"❌ FFmpeg check failed: {e}")
            failed += 1

        # ── TEST: Silent Audio Rejection (Root Cause Test) ─────────
        print_test_header("TEST: Silent Audio Rejection")
        try:
            silent_wav = os.path.join(temp_dir, "silent.wav")
            generate_synthetic_wav(silent_wav, duration_sec=2.0, silent=True)
            try:
                validate_audio(silent_wav)
                assert False, "Should have rejected silent audio"
            except MediaValidationError as e:
                assert "silent" in str(e).lower()
                print(f"✓ Silent audio rejected before Whisper: {e}")
            passed += 1
        except Exception as e:
            print(f"❌ Silent audio test failed: {e}")
            failed += 1

        # ── TEST: Short Audio Handling (< 0.5s) ────────────────────
        print_test_header("TEST: Short Audio Handling (< 0.5s)")
        try:
            short_wav = os.path.join(temp_dir, "short.wav")
            generate_synthetic_wav(short_wav, duration_sec=0.1, silent=False)
            try:
                validate_audio(short_wav)
                assert False, "Should have rejected short audio"
            except MediaValidationError as e:
                assert "too short" in str(e).lower()
                print(f"✓ Short audio (<0.5s) rejected cleanly: {e}")
            passed += 1
        except Exception as e:
            print(f"❌ Short audio test failed: {e}")
            failed += 1

        # ── TEST: Empty Transcript Protection ──────────────────────
        print_test_header("TEST: Empty Transcript Protection Before LLM/RAG")
        try:
            for et in ["", "   ", "\n\t "]:
                try:
                    summarize(et)
                    assert False
                except ValueError:
                    pass

                try:
                    generate_title(et)
                    assert False
                except ValueError:
                    pass

                try:
                    extract_action_items(et)
                    assert False
                except ValueError:
                    pass

                try:
                    build_vector_store(et)
                    assert False
                except ValueError:
                    pass

            print("✓ All LLM & RAG functions strictly protect against empty transcripts.")
            passed += 1
        except Exception as e:
            print(f"❌ Empty transcript protection failed: {e}")
            failed += 1

        rag_en = None
        rag_ru = None

        # ── REGRESSION TEST 1: Full English Pipeline ───────────────
        print_test_header("REGRESSION TEST 1: Full English Flow (Ollama Local)")
        english_transcript = (
            "Good morning team. Today is October 14th and we are conducting our Q4 planning session. "
            "Sarah announced that the new payment gateway integration is scheduled for launch on November 10th. "
            "Michael agreed to take ownership of the load testing and security audits before the release. "
            "The leadership decided to allocate twenty thousand dollars for cloud infrastructure scaling. "
            "A critical question was raised by David regarding whether our API rate limits can handle expected Black Friday traffic."
        )
        try:
            title_en = generate_title(english_transcript)
            print(f"  ✓ English Title: {title_en}")
            assert len(title_en) > 0 and len(title_en.split()) <= 15

            summary_en = summarize(english_transcript)
            print(f"  ✓ English Summary (sample): {summary_en[:140]}...")
            assert len(summary_en) > 20

            actions_en = extract_action_items(english_transcript)
            print(f"  ✓ English Action Items (sample): {actions_en[:120]}...")
            assert len(actions_en) > 10

            decisions_en = extract_key_decisions(english_transcript)
            print(f"  ✓ English Decisions (sample): {decisions_en[:120]}...")
            assert len(decisions_en) > 10

            questions_en = extract_questions(english_transcript)
            print(f"  ✓ English Questions (sample): {questions_en[:120]}...")
            assert len(questions_en) > 10

            rag_en = build_rag_chain(english_transcript)
            ans_en = ask_question(rag_en, "Who is responsible for load testing and when is the payment gateway launching?")
            print(f"  ✓ English RAG Answer: {ans_en}")
            assert "Michael" in ans_en or "michael" in ans_en.lower()
            passed += 1
        except Exception as e:
            print(f"❌ REGRESSION TEST 1 FAILED: {e}")
            failed += 1

        # ── TEST: Consolidated Video Intelligence ─────────────────
        print_test_header("TEST: Consolidated Video Intelligence (Single-Pass Ollama)")
        try:
            intel = generate_video_intelligence(english_transcript)
            print(f"  ✓ Intel Title: {intel.get('title')}")
            print(f"  ✓ Intel Summary: {intel.get('summary')[:100]}...")
            print(f"  ✓ Intel Action Items: {intel.get('action_items')[:100]}...")
            assert len(intel.get("title", "")) > 0
            assert len(intel.get("summary", "")) > 20
            assert len(intel.get("action_items", "")) > 10
            passed += 1
        except Exception as e:
            print(f"❌ Consolidated Video Intelligence failed: {e}")
            failed += 1

        # ── REGRESSION TEST 2: English Video + Hindi Question ──────
        print_test_header("REGRESSION TEST 2: English Video + Hindi Question (Cross-Lingual)")
        try:
            if rag_en is None:
                rag_en = build_rag_chain(english_transcript)
            ans_hi = ask_question(rag_en, "लोड टेस्टिंग की जिम्मेदारी किसको दी गई है?")
            print(f"  ✓ Hindi Question Answer: {ans_hi}")
            assert len(ans_hi) > 5
            passed += 1
        except Exception as e:
            print(f"❌ REGRESSION TEST 2 FAILED: {e}")
            failed += 1

        # ── REGRESSION TEST 3: English Video + Hinglish Question ───
        print_test_header("REGRESSION TEST 3: English Video + Hinglish Question")
        try:
            if rag_en is None:
                rag_en = build_rag_chain(english_transcript)
            ans_hinglish = ask_question(rag_en, "Payment gateway launch kab hone wala hai?")
            print(f"  ✓ Hinglish Question Answer: {ans_hinglish}")
            assert "november" in ans_hinglish.lower() or "10" in ans_hinglish or "नवंबर" in ans_hinglish
            passed += 1
        except Exception as e:
            print(f"❌ REGRESSION TEST 3 FAILED: {e}")
            failed += 1

        # ── REGRESSION TEST 4: Non-English Video -> Same-Language Results ─
        print_test_header("REGRESSION TEST 4: Russian Video -> Russian Title/Summary (Source Language Match)")
        russian_transcript = (
            "Привет всем. Сегодня мы обсуждаем запуск новой системы безопасности. "
            "Иван предложил перевести все серверы на защищенный протокол к концу этой недели. "
            "Руководство одобрило выделение бюджета на аудит. "
            "Остается открытым вопрос о совместимости со старыми клиентами."
        )
        try:
            title_ru = generate_title(russian_transcript)
            print(f"  ✓ Russian Title: {title_ru}")
            summary_ru = summarize(russian_transcript)
            print(f"  ✓ Russian Summary (sample): {summary_ru[:120]}...")
            assert len(summary_ru) > 10
            passed += 1
        except Exception as e:
            print(f"❌ REGRESSION TEST 4 FAILED: {e}")
            failed += 1

        # ── REGRESSION TEST 5: Non-English Video + English Question ─
        print_test_header("REGRESSION TEST 5: Russian Video + English Question (Cross-Lingual)")
        try:
            if rag_ru is None:
                rag_ru = build_rag_chain(russian_transcript)
            ans_ru_en = ask_question(rag_ru, "What did Ivan propose regarding the servers?")
            print(f"  ✓ Cross-lingual Answer in English: {ans_ru_en}")
            assert any(w in ans_ru_en.lower() for w in ["ivan", "server", "protocol", "week", "secure", "иван", "сервер"])
            passed += 1
        except Exception as e:
            print(f"❌ REGRESSION TEST 5 FAILED: {e}")
            failed += 1

        # ── REGRESSION TEST 6: Non-English Video + Hindi Question ──
        print_test_header("REGRESSION TEST 6: Russian Video + Hindi Question (Cross-Lingual)")
        try:
            if rag_ru is None:
                rag_ru = build_rag_chain(russian_transcript)
            ans_ru_hi = ask_question(rag_ru, "इवान ने सर्वर के बारे में क्या प्रस्ताव दिया?")
            print(f"  ✓ Cross-lingual Answer in Hindi: {ans_ru_hi}")
            assert len(ans_ru_hi) > 5
            passed += 1
        except Exception as e:
            print(f"❌ REGRESSION TEST 6 FAILED: {e}")
            failed += 1

        # ── REGRESSION TEST 7: Absent Information (Grounding & No Hallucination) ─
        print_test_header("REGRESSION TEST 7: Absent Information (No Hallucination)")
        try:
            if rag_en is None:
                rag_en = build_rag_chain(english_transcript)
            absent_ans = ask_question(rag_en, "What is the secret recipe for chocolate cake?")
            print(f"  ✓ Grounding Test Answer: {absent_ans}")
            assert any(word in absent_ans.lower() for word in [
                "not found", "could not find", "couldn't find", "couldn't", "does not contain", "no mention",
                "not mentioned", "unable to find", "no information", "not provided",
                "नहीं", "nahi", "absent"
            ])
            passed += 1
        except Exception as e:
            print(f"❌ REGRESSION TEST 7 FAILED: {e}")
            failed += 1

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    print("\n" + "=" * 70)
    print(f"📊 ALL REGRESSION TESTS SUMMARY: {passed} PASSED, {failed} FAILED")
    print("=" * 70)
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)