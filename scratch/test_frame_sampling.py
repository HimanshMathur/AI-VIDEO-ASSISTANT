import os
import sys
import time
import subprocess
from PIL import Image
import numpy as np

def test_frame_sampling(video_path, max_frames=20, output_dir="scratch/frames"):
    os.makedirs(output_dir, exist_ok=True)
    t0 = time.time()
    
    # Get video duration using ffprobe
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration:stream=width,height",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    lines = res.stdout.strip().split("\n")
    duration = float(lines[-1]) if lines else 0.0
    print(f"Video: {video_path}")
    print(f"Duration: {duration:.2f}s ({duration/60:.2f} minutes)")
    
    if duration <= 0:
        return []
        
    # Calculate sample timestamps
    num_frames = min(max_frames, max(3, int(duration // 10)))
    timestamps = np.linspace(1.0, max(1.0, duration - 2.0), num_frames)
    print(f"Targeting {num_frames} frames (interval: {duration/num_frames:.1f}s)")
    
    t_extract_start = time.time()
    extracted_frames = []
    
    for i, ts in enumerate(timestamps):
        frame_name = f"frame_{i:03d}_{int(ts)}s.jpg"
        frame_path = os.path.join(output_dir, frame_name)
        
        # Fast input seeking (-ss before -i)
        ff_cmd = [
            "ffmpeg", "-y", "-ss", f"{ts:.2f}",
            "-i", video_path,
            "-frames:v", "1",
            "-q:v", "3",
            frame_path
        ]
        subprocess.run(ff_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if os.path.exists(frame_path) and os.path.getsize(frame_path) > 100:
            extracted_frames.append({
                "index": i,
                "timestamp_seconds": float(ts),
                "timestamp_str": f"{int(ts//3600):02d}:{int((ts%3600)//60):02d}:{int(ts%60):02d}",
                "path": frame_path,
            })
            
    t_extract = time.time() - t_extract_start
    print(f"Extracted {len(extracted_frames)} frames in {t_extract:.2f}s (avg: {t_extract/len(extracted_frames):.3f}s/frame)")
    
    # Deduplication test
    t_dedup_start = time.time()
    retained_frames = []
    prev_thumb = None
    
    for f in extracted_frames:
        try:
            with Image.open(f["path"]) as img:
                # 32x32 grayscale thumbnail for ultra-fast perceptual diff
                thumb = np.array(img.convert("L").resize((32, 32), Image.Resampling.BILINEAR), dtype=np.float32)
                
            if prev_thumb is None:
                retained_frames.append(f)
                prev_thumb = thumb
            else:
                # Normalized mean absolute pixel difference (0.0 to 1.0)
                diff = np.mean(np.abs(thumb - prev_thumb)) / 255.0
                if diff > 0.04:  # > 4% visual scene difference
                    f["diff_from_prev"] = diff
                    retained_frames.append(f)
                    prev_thumb = thumb
                else:
                    f["is_duplicate"] = True
                    # If duplicate, we can still note it without re-running vision LLM
        except Exception as e:
            retained_frames.append(f)
            
    t_dedup = time.time() - t_dedup_start
    print(f"Deduplication finished in {t_dedup:.3f}s: {len(retained_frames)} key frames retained out of {len(extracted_frames)} (skipped {len(extracted_frames) - len(retained_frames)} near-identical slides/scenes).")
    
    for rf in retained_frames[:5]:
        print(f"  • Frame at {rf['timestamp_str']} ({rf['timestamp_seconds']:.1f}s): {os.path.basename(rf['path'])}")
        
    return retained_frames

if __name__ == "__main__":
    vid = r"C:\Users\mathu\Downloads\Product Marketing Meeting (weekly) 2021-06-28.mp4"
    test_frame_sampling(vid, max_frames=20)
