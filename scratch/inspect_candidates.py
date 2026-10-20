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

downloads = r"C:\Users\mathu\Downloads"
candidates = [
    "Product Marketing Meeting (weekly) 2021-06-28.mp4",
    "Untitled design.mp4",
    "marvel.mp4",
    "द_मिसिंग_नाइट.mp4",
    "12352337-hd_1920_1080_60fps.mp4",
    "14483720_1920_1080_30fps.mp4",
    "14957648_1920_1080_30fps.mp4",
]

os.makedirs("scratch/candidate_previews", exist_ok=True)

for name in candidates:
    fpath = os.path.join(downloads, name)
    if not os.path.exists(fpath):
        continue
    
    # Duration and streams
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration:stream=codec_type,codec_name,width,height",
        "-of", "json",
        fpath
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    data = json.loads(res.stdout) if res.stdout else {}
    dur = float(data.get("format", {}).get("duration", 0))
    streams = [s.get("codec_type") for s in data.get("streams", [])]
    
    # Extract middle frame
    mid_ts = max(1.0, dur / 2.0)
    clean_base = name.split(".")[0][:15].replace(" ", "_")
    frame_preview = f"scratch/candidate_previews/{clean_base}_mid.jpg"
    cmd_frame = [
        "ffmpeg", "-y", "-ss", f"{mid_ts:.2f}",
        "-i", fpath, "-frames:v", "1", "-q:v", "2",
        frame_preview
    ]
    subprocess.run(cmd_frame, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    print(f"Candidate: {name}")
    print(f"  Duration: {dur:.1f}s | Streams: {streams}")
    print(f"  Mid Frame: {frame_preview}")
