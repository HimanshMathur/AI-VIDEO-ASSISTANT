import json, os
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
d = json.load(open("scratch/hindi_baseline_results.json", encoding="utf-8"))
lines = []
lines.append(f"MODEL={d['model']} DEVICE={d['device']} COMPUTE={d['compute']}")
for r in d["results"]:
    lines.append("")
    lines.append(f"=== {r['id']} ({r['kind']}) ===")
    lines.append(f"lang={r.get('detected_language')} segs={r.get('num_segments')} "
                 f"cuts={len(r.get('possible_30s_window_cuts', []))} "
                 f"repchars={r.get('replacement_char_count')} "
                 f"t_all={r.get('transcribe_all_seconds')}s "
                 f"t_seg={r.get('transcribe_with_segments_seconds')}s")
    final = r.get("final_transcribe_all", "")
    raw = r.get("raw_joined_from_segments", "")
    lines.append(f"FINAL ({len(final)} chars): {final}")
    lines.append(f"RAW   ({len(raw)} chars): {raw}")
open("scratch/compare_new.txt", "w", encoding="utf-8").write("\n".join(lines))
print("written", len(lines), "lines")
