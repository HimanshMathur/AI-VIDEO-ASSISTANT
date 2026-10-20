import os
import sys
import subprocess
import json

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

screen_dir = r"C:\Users\mathu\Videos\Screen Recordings"
files = [f for f in os.listdir(screen_dir) if f.endswith(".mp4")]

os.makedirs("scratch/screen_previews", exist_ok=True)

for f in files:
    full = os.path.join(screen_dir, f)
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration:stream=codec_type,codec_name,width,height",
        "-of", "json",
        full
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    data = json.loads(res.stdout) if res.stdout else {}
    dur = float(data.get("format", {}).get("duration", 0))
    streams = [s.get("codec_type") for s in data.get("streams", [])]
    
    mid_ts = max(1.0, dur / 2.0)
    preview = f"scratch/screen_previews/{f[:18].replace(' ', '_')}_mid.jpg"
    cmd_f = [
        "ffmpeg", "-y", "-ss", f"{mid_ts:.2f}",
        "-i", full, "-frames:v", "1", "-q:v", "2",
        preview
    ]
    subprocess.run(cmd_f, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"Screen Recording: {f}")
    print(f"  Duration: {dur:.1f}s | Streams: {streams} | Preview: {preview}")
