import json, os
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
d = json.load(open("scratch/hindi_baseline_results.json", encoding="utf-8"))
lines = []
for r in d["results"]:
    lines.append(f"\n=== {r['id']} ===")
    segs = r.get("segments", [])
    audio_dur = r.get("audio", {}).get("duration", 0)
    lines.append(f"audio_dur={audio_dur}s num_segs={len(segs)}")
    if not segs:
        continue
    starts = sorted(s.get("start", 0) for s in segs)
    ends = sorted(s.get("end", 0) for s in segs)
    first_start = starts[0]
    last_end = ends[-1]
    zero_dur = sum(1 for s in segs if s.get("end", 0) - s.get("start", 0) <= 0)
    out_of_range = sum(1 for s in segs if s.get("start", 0) < 0 or s.get("end", 0) > audio_dur + 0.5)
    # Check coverage gaps > 1.0s between consecutive segments
    sorted_segs = sorted(segs, key=lambda x: x.get("start", 0))
    gaps = []
    for i in range(1, len(sorted_segs)):
        gap = sorted_segs[i]["start"] - sorted_segs[i-1]["end"]
        if gap > 1.0:
            gaps.append((round(sorted_segs[i-1]["end"],2), round(sorted_segs[i]["start"],2), round(gap,2)))
    overlap = sum(1 for i in range(1, len(sorted_segs)) if sorted_segs[i]["start"] < sorted_segs[i-1]["end"])
    lines.append(f"first_start={first_start:.2f}s last_end={last_end:.2f}s (coverage={last_end/audio_dur*100:.1f}%)")
    lines.append(f"zero_duration_segments={zero_dur} out_of_range_segments={out_of_range}")
    lines.append(f"gaps>1s_count={len(gaps)} first_gaps={gaps[:5]}")
    lines.append(f"overlapping_segments={overlap}")
open("scratch/verify_ts.txt", "w", encoding="utf-8").write("\n".join(lines))
print("done")
