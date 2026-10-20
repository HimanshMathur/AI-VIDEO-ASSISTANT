import sys
import os
sys.path.insert(0, os.path.abspath("."))
from core.transcriber import transcribe_with_segments
import json

res = transcribe_with_segments("scratch/test_android_rPvJ7FKqWy4.wav")
print("Language:", res["language"])
print("Num segments:", len(res["segments"]))

with open("scratch/segments_rPvJ7FKqWy4.json", "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)

for s in res["segments"][:15]:
    start_m, start_s = int(s['start'] // 60), int(s['start'] % 60)
    end_m, end_s = int(s['end'] // 60), int(s['end'] % 60)
    print(f"[{start_m:02d}:{start_s:02d} - {end_m:02d}:{end_s:02d}] ({s['start']:.1f}s - {s['end']:.1f}s): {s['text']}")
