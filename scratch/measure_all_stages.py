import os
import sys
import time
import subprocess
import wave
import numpy as np
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")

VIDEO_PATH = r"C:\Users\mathu\OneDrive\Desktop\Rec.mp4"
print(f"=== MEASURING PROCESSING STAGES FOR: {VIDEO_PATH} ===")
print(f"File size: {os.path.getsize(VIDEO_PATH) / (1024*1024):.2f} MB")

# 1. Video file loading / opening
t0 = time.time()
with open(VIDEO_PATH, "rb") as f:
    header = f.read(1024 * 1024)
t_video_load = time.time() - t0
print(f"1. VIDEO LOAD (open & read 1MB header): {t_video_load:.4f} sec")

# 2. Video metadata reading
t0 = time.time()
cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=width,height,codec_name", "-of", "default=noprint_wrappers=1", VIDEO_PATH]
res = subprocess.run(cmd, capture_output=True, text=True)
duration = 0.0
for line in res.stdout.strip().split("\n"):
    if line.startswith("duration="):
        duration = float(line.split("=")[1])
t_meta = time.time() - t0
print(f"2. METADATA READING (ffprobe): {t_meta:.4f} sec (Duration: {duration:.2f}s)")

# 3. Frame sampling (calculating timestamps)
t0 = time.time()
max_frames = 20
num_frames = min(max_frames, max(3, int(duration // 10)))
timestamps = np.linspace(1.0, max(1.0, duration - 2.0), num_frames)
t_frame_sampling = time.time() - t0
print(f"3. FRAME SAMPLING (calculating {num_frames} timestamps): {t_frame_sampling:.6f} sec")

# 4. Frame extraction / decoding
os.makedirs("scratch/timing_frames", exist_ok=True)
t0 = time.time()
extracted_frames = []
for i, ts in enumerate(timestamps):
    fpath = f"scratch/timing_frames/frame_{i:03d}_{int(ts)}s.jpg"
    cmd = ["ffmpeg", "-y", "-nostdin", "-ss", f"{ts:.2f}", "-i", VIDEO_PATH, "-frames:v", "1", "-q:v", "3", fpath]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if os.path.exists(fpath):
        extracted_frames.append({"index": i, "ts": ts, "path": fpath})
t_frame_extract = time.time() - t0
print(f"4. FRAME EXTRACTION ({len(extracted_frames)} frames via ffmpeg): {t_frame_extract:.4f} sec ({t_frame_extract/max(1,len(extracted_frames)):.3f}s/frame)")

# 5. Perceptual deduplication
t0 = time.time()
processed_frames = []
prev_thumb = None
for f in extracted_frames:
    with Image.open(f["path"]) as img:
        thumb = np.array(img.convert("L").resize((32, 32), Image.Resampling.BILINEAR), dtype=np.float32)
    if prev_thumb is None:
        f["is_key"] = True
        processed_frames.append(f)
        prev_thumb = thumb
    else:
        diff = float(np.mean(np.abs(thumb - prev_thumb)) / 255.0)
        if diff > 0.04:
            f["is_key"] = True
            f["diff"] = diff
            processed_frames.append(f)
            prev_thumb = thumb
        else:
            f["is_key"] = False
            processed_frames.append(f)
t_frame_dedup = time.time() - t0
key_count = sum(1 for f in processed_frames if f["is_key"])
print(f"5. FRAME DEDUP ({len(processed_frames)} frames -> {key_count} key frames, {len(processed_frames)-key_count} duplicates): {t_frame_dedup:.4f} sec")

# 6. Audio extraction
t0 = time.time()
out_wav = "scratch/timing_rec.wav"
cmd = ["ffmpeg", "-y", "-nostdin", "-i", VIDEO_PATH, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", out_wav]
subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
t_audio_extract = time.time() - t0
print(f"6. AUDIO EXTRACTION: {t_audio_extract:.4f} sec")

# 7. Audio validation & Whisper transcription
t0 = time.time()
from core.transcriber import load_model, transcribe_chunk_whisper, WHISPER_MODEL
load_model()
t_whisper_load = time.time() - t0
print(f"   Whisper model load: {t_whisper_load:.4f} sec")

t0 = time.time()
text, lang = transcribe_chunk_whisper(out_wav)
t_whisper = time.time() - t0
print(f"7. WHISPER TRANSCRIPTION: {t_whisper:.4f} sec (lang={lang}, text_len={len(text)})")

# 8. Embedding generation
t0 = time.time()
from core.vector_store import get_embedding_model
model = get_embedding_model()
sample_chunks = ["Video overview and introduction", "Discussion of architectural features", "Action items and next steps"]
embeddings = model.encode(sample_chunks, normalize_embeddings=True)
t_embeddings = time.time() - t0
print(f"8. EMBEDDING GENERATION: {t_embeddings:.4f} sec")

# 9. FAISS indexing
t0 = time.time()
import faiss
dim = embeddings.shape[1]
idx = faiss.IndexFlatIP(dim)
idx.add(np.ascontiguousarray(embeddings, dtype=np.float32))
t_faiss = time.time() - t0
print(f"9. FAISS INDEXING: {t_faiss:.6f} sec")

# 10. Visual Qwen processing on key frames
t0 = time.time()
from core.visual_analyzer import analyze_frame_with_vision_model
vision_results = []
for i, kf in enumerate([f for f in processed_frames if f["is_key"]][:3]): # test on up to 3 key frames
    t_f0 = time.time()
    v_res = analyze_frame_with_vision_model(kf["path"])
    t_f = time.time() - t_f0
    print(f"   Key frame {i+1} analyzed in {t_f:.2f}s: {v_res.get('description', '')[:80]}...")
    vision_results.append(v_res)
t_qwen_vl = time.time() - t0
print(f"10. QWEN3-VL (sampled 3 key frames): {t_qwen_vl:.4f} sec (avg {t_qwen_vl/max(1,len(vision_results)):.2f}s/frame)")

t_total = t_video_load + t_meta + t_frame_sampling + t_frame_extract + t_frame_dedup + t_audio_extract + t_whisper + t_embeddings + t_faiss + t_qwen_vl
print("\n" + "="*50)
print("PRINTED TIMINGS:")
print(f"VIDEO LOAD: {t_video_load + t_meta:.2f} sec")
print(f"FRAME SAMPLING: {t_frame_sampling + t_frame_extract:.2f} sec")
print(f"FRAME DEDUP: {t_frame_dedup:.2f} sec")
print(f"AUDIO EXTRACTION: {t_audio_extract:.2f} sec")
print(f"WHISPER: {t_whisper:.2f} sec")
print(f"EMBEDDINGS: {t_embeddings:.2f} sec")
print(f"FAISS: {t_faiss:.4f} sec")
print(f"QWEN3-VL: {t_qwen_vl:.2f} sec")
print(f"TOTAL: {t_total:.2f} sec")
print("="*50)
