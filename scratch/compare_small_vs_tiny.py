import os
import sys
import time
import difflib
import re

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from faster_whisper import WhisperModel

test_cases = [
    {
        "id": "real_english_meeting",
        "name": "Real English Meeting (90s slice from GitLab Weekly)",
        "path": os.path.abspath("scratch/real_meeting_90s.wav"),
        "key_entities": ["GitLab", "DevOps", "Commit", "marketing", "campaign", "pillar"],
    },
    {
        "id": "hinglish_meeting",
        "name": "Hinglish Mixed Meeting Audio (Hindi + English)",
        "path": os.path.abspath("scratch/hinglish_sample.wav"),
        "key_entities": ["Namaste", "Q4", "roadmap", "November", "10", "Michael", "25000", "dollars"],
    },
    {
        "id": "real_hindi",
        "name": "Real Multilingual Hindi Audio (Emergency / Distress)",
        "path": r"C:\Users\mathu\Downloads\bachao_hindi_roadrage_01 - Copy (2) - Copy_mono.wav",
        "key_entities": ["बचाओ", "मदद", "प्लीज"],
    },
    {
        "id": "synthetic_english",
        "name": "Synthetic English Meeting (Clean Sample)",
        "path": os.path.abspath("scratch/speech_sample_16k.wav"),
        "key_entities": ["Sarah", "Michael", "Tuesday", "payment", "benchmarks", "privacy"],
    }
]

def check_repetitions(text):
    words = re.findall(r'\b\w+\b', text.lower())
    repeats = []
    for i in range(len(words) - 1):
        if words[i] == words[i+1]:
            repeats.append(words[i])
    return repeats

def run_bench():
    print("=" * 75)
    print("🔬 RIGOROUS BENCHMARK: faster-whisper (small) vs faster-whisper (tiny)")
    print("=" * 75)
    
    cpu_threads = min(8, max(2, (os.cpu_count() or 4) // 2))
    print(f"Device: CPU | Compute Type: int8 | CPU Threads: {cpu_threads}\n")

    # Pre-load both models to measure load time
    print("Loading faster-whisper 'small'...")
    t0 = time.time()
    model_small = WhisperModel("small", device="cpu", compute_type="int8", cpu_threads=cpu_threads)
    t_load_small = time.time() - t0
    print(f"  ✓ 'small' loaded in {t_load_small:.2f}s")

    print("Loading faster-whisper 'tiny'...")
    t0 = time.time()
    model_tiny = WhisperModel("tiny", device="cpu", compute_type="int8", cpu_threads=cpu_threads)
    t_load_tiny = time.time() - t0
    print(f"  ✓ 'tiny' loaded in {t_load_tiny:.2f}s\n")

    overall_results = []

    for tc in test_cases:
        fpath = tc["path"]
        name = tc["name"]
        print("=" * 75)
        print(f"🎧 Test Case: {name}")
        print(f"   File: {fpath}")
        if not os.path.exists(fpath):
            print(f"   ❌ File not found! Skipping.")
            continue

        # Benchmark 'small'
        t0 = time.time()
        segs_s, info_s = model_small.transcribe(
            fpath,
            task="transcribe",
            beam_size=1,
            best_of=1,
            temperature=0.0,
            condition_on_previous_text=False
        )
        text_small = " ".join([s.text.strip() for s in segs_s if s.text.strip()]).strip()
        t_infer_small = time.time() - t0

        # Benchmark 'tiny'
        t0 = time.time()
        segs_t, info_t = model_tiny.transcribe(
            fpath,
            task="transcribe",
            beam_size=1,
            best_of=1,
            temperature=0.0,
            condition_on_previous_text=False
        )
        text_tiny = " ".join([s.text.strip() for s in segs_t if s.text.strip()]).strip()
        t_infer_tiny = time.time() - t0

        # Metrics
        speedup = t_infer_small / max(0.001, t_infer_tiny)
        dur = info_s.duration if hasattr(info_s, 'duration') else 0.0
        rtf_s = t_infer_small / dur if dur > 0 else 0
        rtf_t = t_infer_tiny / dur if dur > 0 else 0

        repeats_s = check_repetitions(text_small)
        repeats_t = check_repetitions(text_tiny)

        # Entity preservation
        entities = tc.get("key_entities", [])
        ent_s = [e for e in entities if e.lower() in text_small.lower()]
        ent_t = [e for e in entities if e.lower() in text_tiny.lower()]

        print(f"\n[TIMINGS & SPEED]")
        print(f"  Audio Duration:        {dur:.2f}s")
        print(f"  small inference:       {t_infer_small:.2f}s (RTF: {rtf_s:.3f}x)")
        print(f"  tiny inference:        {t_infer_tiny:.2f}s  (RTF: {rtf_t:.3f}x)")
        print(f"  Latency Speedup:       {speedup:.2f}x faster")

        print(f"\n[LANGUAGE DETECTION]")
        print(f"  small detected:        {info_s.language} (confidence: {info_s.language_probability:.2f})")
        print(f"  tiny detected:         {info_t.language} (confidence: {info_t.language_probability:.2f})")

        print(f"\n[TRANSCRIPT COMPARISON]")
        print(f"  small length:          {len(text_small)} chars ({len(text_small.split())} words)")
        print(f"  tiny length:           {len(text_tiny)} chars ({len(text_tiny.split())} words)")
        print(f"\n  --- 'small' text ---:\n  \"{text_small}\"")
        print(f"\n  --- 'tiny' text ---:\n  \"{text_tiny}\"")

        print(f"\n[QUALITY & INTEGRITY CHECKS]")
        print(f"  Consecutive word repetitions - small: {len(repeats_s)} {repeats_s}, tiny: {len(repeats_t)} {repeats_t}")
        print(f"  Key entities captured ({len(entities)} expected):")
        print(f"    small: {len(ent_s)}/{len(entities)} -> {ent_s}")
        print(f"    tiny:  {len(ent_t)}/{len(entities)} -> {ent_t}")

        diff_ratio = difflib.SequenceMatcher(None, text_small.lower(), text_tiny.lower()).ratio()
        print(f"  Character Sequence Similarity: {diff_ratio*100:.1f}%")

        overall_results.append({
            "id": tc["id"],
            "name": name,
            "duration": dur,
            "t_infer_small": t_infer_small,
            "t_infer_tiny": t_infer_tiny,
            "speedup": speedup,
            "lang_small": info_s.language,
            "lang_tiny": info_t.language,
            "similarity": diff_ratio,
            "entities_small": len(ent_s),
            "entities_tiny": len(ent_t),
            "total_entities": len(entities),
            "text_small": text_small,
            "text_tiny": text_tiny,
        })

    print("\n" + "=" * 75)
    print("📊 EXECUTIVE BENCHMARK SUMMARY TABLE")
    print("=" * 75)
    header = f"{'Test Case':<32} | {'small (s)':<9} | {'tiny (s)':<8} | {'Speedup':<7} | {'Lang (S/T)':<10} | {'Sim (%)':<7} | {'Entities (S/T)':<14}"
    print(header)
    print("-" * len(header))
    for r in overall_results:
        lang_str = f"{r['lang_small']}/{r['lang_tiny']}"
        ent_str = f"{r['entities_small']}/{r['entities_tiny']} of {r['total_entities']}"
        print(f"{r['name'][:32]:<32} | {r['t_infer_small']:<9.2f} | {r['t_infer_tiny']:<8.2f} | {r['speedup']:<6.2f}x | {lang_str:<10} | {r['similarity']*100:<6.1f}% | {ent_str:<14}")
    print("=" * 75)

if __name__ == "__main__":
    run_bench()
