import os
import sys
import time
import subprocess
import numpy as np

VIDEO_PATH = r"C:\Users\mathu\OneDrive\Desktop\Rec.mp4"

def format_timestamp(seconds: float) -> str:
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"

def detect_visual_events(video_path: str, fps: float = 1.0, max_keyframes: int = 25):
    t0 = time.time()
    
    # Probe duration
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    dur = float(res.stdout.strip().split("\n")[-1]) if res.stdout.strip() else 0.0
    print(f"Video duration: {dur:.1f}s ({format_timestamp(dur)})")
    
    # Pipe grayscale 160x90 frames
    w, h = 160, 90
    frame_bytes = w * h
    cmd_ffmpeg = [
        "ffmpeg", "-nostdin", "-i", video_path,
        "-vf", f"fps={fps},scale={w}:{h}",
        "-f", "rawvideo", "-pix_fmt", "gray",
        "-"
    ]
    proc = subprocess.Popen(cmd_ffmpeg, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    
    candidate_frames = 0
    prev_frame = None
    diffs = []
    
    while True:
        raw = proc.stdout.read(frame_bytes)
        if len(raw) < frame_bytes:
            break
        arr = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        
        if prev_frame is not None:
            # Global mean absolute diff
            g_diff = float(np.mean(np.abs(arr - prev_frame)) / 255.0)
            
            # 4x4 block diff to catch regional UI / slide changes
            grid_h, grid_w = h // 4, w // 4
            block_diffs = []
            for r in range(4):
                for c in range(4):
                    b_curr = arr.reshape(h, w)[r*grid_h:(r+1)*grid_h, c*grid_w:(c+1)*grid_w]
                    b_prev = prev_frame.reshape(h, w)[r*grid_h:(r+1)*grid_h, c*grid_w:(c+1)*grid_w]
                    block_diffs.append(np.mean(np.abs(b_curr - b_prev)) / 255.0)
            max_b_diff = float(max(block_diffs))
            
            # Combined score
            score = max(g_diff, max_b_diff * 0.7)
            diffs.append((candidate_frames, candidate_frames / fps, score, g_diff, max_b_diff))
        else:
            diffs.append((candidate_frames, 0.0, 0.0, 0.0, 0.0))
            
        prev_frame = arr
        candidate_frames += 1
        
    proc.wait()
    t_eval = time.time() - t0
    print(f"Candidate frame evaluation done in {t_eval:.2f}s ({candidate_frames} candidates evaluated).")
    
    # Event selection:
    # 1. Always select first valid frame
    selected_indices = [0]
    last_selected_ts = 0.0
    min_event_gap = 2.0  # seconds between distinct events
    
    # Threshold for meaningful visual change
    CHANGE_THRESHOLD = 0.035
    
    # Peak / stable frame picking
    i = 1
    while i < len(diffs):
        idx, ts, score, g_diff, b_diff = diffs[i]
        
        # Check if score exceeds change threshold
        if score >= CHANGE_THRESHOLD:
            # If within min_event_gap of last selected, update if this frame has higher change or settles
            if (ts - last_selected_ts) < min_event_gap:
                i += 1
                continue
                
            # Look ahead 1-2 candidate frames to see if transition is settling
            best_idx = idx
            best_ts = ts
            j = i + 1
            while j < min(len(diffs), i + 3):
                _, next_ts, next_score, _, _ = diffs[j]
                if next_score >= CHANGE_THRESHOLD:
                    best_idx = diffs[j][0]
                    best_ts = next_ts
                    j += 1
                else:
                    break
            
            selected_indices.append(best_idx)
            last_selected_ts = best_ts
            i = max(i + 1, j)
        else:
            i += 1
            
    # If selected exceeds max_keyframes, keep the ones with largest changes while preserving temporal order
    if len(selected_indices) > max_keyframes:
        scored_events = [(idx, diffs[idx][2] if idx < len(diffs) else 1.0) for idx in selected_indices[1:]]
        scored_events.sort(key=lambda x: x[1], reverse=True)
        top_indices = set([0] + [x[0] for x in scored_events[:max_keyframes-1]])
        selected_indices = [idx for idx in selected_indices if idx in top_indices]
        
    # Build keyframe metadata
    keyframes = []
    for k_order, idx in enumerate(selected_indices):
        ts_sec = round(idx / fps, 1)
        if k_order + 1 < len(selected_indices):
            next_ts = round(selected_indices[k_order + 1] / fps, 1)
        else:
            next_ts = dur
            
        keyframes.append({
            "index": k_order,
            "candidate_index": idx,
            "timestamp_seconds": ts_sec,
            "timestamp_str": format_timestamp(ts_sec),
            "end_seconds": next_ts,
            "end_timestamp_str": format_timestamp(next_ts),
        })
        
    skipped_duplicates = candidate_frames - len(keyframes)
    
    print("\n" + "="*50)
    print(f"Candidate frames: {candidate_frames}")
    print(f"Selected keyframes: {len(keyframes)}")
    print(f"Skipped duplicates: {skipped_duplicates}")
    print("Keyframe timestamps:")
    for kf in keyframes:
        print(f"  {kf['timestamp_str']} (valid until {kf['end_timestamp_str']})")
    print("="*50)
    
    return keyframes

if __name__ == "__main__":
    detect_visual_events(VIDEO_PATH)
