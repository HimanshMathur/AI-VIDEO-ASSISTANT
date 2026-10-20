import os
import re
import sys
import uuid
import wave
import shutil
import logging
import subprocess
import numpy as np
from pydub import AudioSegment
import yt_dlp

logger = logging.getLogger(__name__)

DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v"}
SUPPORTED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"}
ALL_SUPPORTED_EXTENSIONS = SUPPORTED_VIDEO_EXTENSIONS | SUPPORTED_AUDIO_EXTENSIONS

YOUTUBE_URL_REGEX = re.compile(
    r"^(https?://)?(www\.|m\.)?(youtube\.com/(watch\?v=|embed/|v/|shorts/)|youtu\.be/)[a-zA-Z0-9_-]{6,}"
)


class AudioProcessingError(Exception):
    """Base exception for audio processing failures."""
    pass


class MediaValidationError(AudioProcessingError):
    """Raised when media validation fails (corrupt, empty, silent, or unsupported)."""
    pass


def check_ffmpeg() -> bool:
    """
    Check whether FFmpeg is installed and accessible on system PATH.
    Raises RuntimeError with clear installation advice if missing.
    """
    ffmpeg_binary = shutil.which("ffmpeg")
    if not ffmpeg_binary:
        raise RuntimeError(
            "FFmpeg is not installed or is not available on your system PATH.\n"
            "Please install FFmpeg and restart your terminal/IDE:\n"
            "  - Windows (winget): winget install Gyan.FFmpeg\n"
            "  - Windows (chocolatey): choco install ffmpeg\n"
            "  - Mac (Homebrew): brew install ffmpeg\n"
            "  - Linux: sudo apt update && sudo apt install ffmpeg"
        )
    try:
        proc = subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5,
        )
        return proc.returncode == 0
    except Exception as e:
        raise RuntimeError(f"FFmpeg check failed: {e}") from e


def extract_youtube_video_id(url: str) -> str:
    """Extract standard YouTube 11-char video ID from various YouTube URL formats."""
    if not url or not isinstance(url, str):
        return ""
    clean = url.strip()
    patterns = [
        r"(?:v=|\/v\/|\/embed\/|\/shorts\/|^youtu\.be\/|\/youtu\.be\/)([a-zA-Z0-9_-]{11})",
        r"[?&]v=([a-zA-Z0-9_-]{11})",
    ]
    for pat in patterns:
        m = re.search(pat, clean)
        if m:
            return m.group(1)
    return ""


def validate_youtube_url(url: str) -> bool:
    """Verify whether a string is a recognized YouTube video URL format."""
    if not url or not isinstance(url, str):
        return False
    clean = url.strip()
    return bool(YOUTUBE_URL_REGEX.match(clean) or extract_youtube_video_id(clean))


def download_youtube_audio(url: str, output_dir: str = DOWNLOAD_DIR) -> str:
    """
    Download audio from YouTube using yt-dlp and convert to 16kHz mono WAV.
    Returns the absolute path to the extracted WAV file.
    Uses android player client to bypass YouTube 403 Forbidden / SABR streaming restrictions.
    """
    check_ffmpeg()
    os.makedirs(output_dir, exist_ok=True)

    clean_url = url.strip()
    if not validate_youtube_url(clean_url):
        raise MediaValidationError(
            f"Invalid YouTube URL: '{clean_url}'. Please provide a valid YouTube link "
            "(e.g., https://www.youtube.com/watch?v=... or https://youtu.be/...)"
        )

    # Resolve canonical video URL if video ID is present (strips broken/tracking params like ?si=)
    vid_id = extract_youtube_video_id(clean_url)
    target_url = f"https://www.youtube.com/watch?v={vid_id}" if vid_id else clean_url

    file_id = f"yt_{uuid.uuid4().hex[:10]}"
    output_template = os.path.join(output_dir, f"{file_id}.%(ext)s")
    target_wav_path = os.path.join(output_dir, f"{file_id}.wav")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "extractor_args": {
            "youtube": {
                "player_client": ["android", "web"]
            }
        },
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "postprocessor_args": [
            "-ar", "16000",
            "-ac", "1",
        ],
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "retries": 5,
    }

    try:
        logger.info(f"Downloading YouTube audio from {target_url}...")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([target_url])
    except yt_dlp.utils.DownloadError as e:
        error_msg = str(e)
        if "Private video" in error_msg:
            raise MediaValidationError("This YouTube video is private and cannot be accessed.") from e
        elif "Sign in to confirm your age" in error_msg:
            raise MediaValidationError("This YouTube video is age-restricted and cannot be downloaded.") from e
        elif "Video unavailable" in error_msg:
            raise MediaValidationError("The requested YouTube video is unavailable or has been removed.") from e
        else:
            # Fallback attempt with generic best stream if primary extraction fails
            try:
                fallback_opts = ydl_opts.copy()
                fallback_opts["format"] = "18/best/worst"
                with yt_dlp.YoutubeDL(fallback_opts) as ydl:
                    ydl.download([target_url])
            except Exception as fb_err:
                raise AudioProcessingError(f"Failed to download YouTube audio: {error_msg}") from e
    except Exception as e:
        raise AudioProcessingError(f"Unexpected error downloading YouTube video: {e}") from e

    if not os.path.exists(target_wav_path) or os.path.getsize(target_wav_path) < 100:
        found_wav = None
        for fname in os.listdir(output_dir):
            if fname.startswith(file_id) and fname.endswith(".wav"):
                found_wav = os.path.join(output_dir, fname)
                break
        if found_wav and os.path.getsize(found_wav) >= 100:
            target_wav_path = found_wav
        else:
            raise AudioProcessingError(
                "Audio extraction failed: No valid WAV file was produced by yt-dlp/FFmpeg."
            )

    return os.path.abspath(target_wav_path)


def download_youtube_video(url: str, output_dir: str = DOWNLOAD_DIR) -> str:
    """
    Download a low-bitrate MP4 copy of a YouTube video for visual frame analysis.
    Returns absolute path to the downloaded .mp4 file on local disk.
    Prioritizes small 360p-480p video to keep download fast and disk usage low.
    """
    check_ffmpeg()
    os.makedirs(output_dir, exist_ok=True)

    clean_url = url.strip()
    if not validate_youtube_url(clean_url):
        raise MediaValidationError(
            f"Invalid YouTube URL: '{clean_url}'. Please provide a valid YouTube link "
            "(e.g., https://www.youtube.com/watch?v=... or https://youtu.be/...)"
        )

    vid_id = extract_youtube_video_id(clean_url)
    target_url = f"https://www.youtube.com/watch?v={vid_id}" if vid_id else clean_url

    file_id = f"ytv_{uuid.uuid4().hex[:10]}"
    output_template = os.path.join(output_dir, f"{file_id}.%(ext)s")
    target_mp4_path = os.path.join(output_dir, f"{file_id}.mp4")

    ydl_opts = {
        "format": "bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]/best[height<=480]/best/worst",
        "outtmpl": output_template,
        "merge_output_format": "mp4",
        "extractor_args": {
            "youtube": {
                "player_client": ["android", "web"]
            }
        },
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "retries": 5,
    }

    try:
        logger.info(f"Downloading YouTube video (480p for vision analysis) from {target_url}...")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([target_url])
    except yt_dlp.utils.DownloadError as e:
        error_msg = str(e)
        logger.warning(f"Primary YouTube video download failed: {error_msg[:200]}. Trying fallback format.")
        try:
            fallback_opts = ydl_opts.copy()
            fallback_opts["format"] = "worst[ext=mp4]/worst"
            with yt_dlp.YoutubeDL(fallback_opts) as ydl:
                ydl.download([target_url])
        except Exception as e2:
            logger.warning(f"Fallback YouTube video download also failed: {str(e2)[:200]}")
            return ""
        try:
            fallback_opts = ydl_opts.copy()
            fallback_opts["format"] = "worst[ext=mp4]/worst"
            with yt_dlp.YoutubeDL(fallback_opts) as ydl:
                ydl.download([clean_url])
        except Exception as e2:
            logger.warning(f"Fallback YouTube video download also failed: {str(e2)[:200]}")
            return ""

    if not os.path.exists(target_mp4_path) or os.path.getsize(target_mp4_path) < 1000:
        found_mp4 = None
        for fname in os.listdir(output_dir):
            if fname.startswith(file_id) and os.path.splitext(fname)[1].lower() in {".mp4", ".mkv", ".webm", ".mov"}:
                candidate = os.path.join(output_dir, fname)
                if os.path.getsize(candidate) >= 1000:
                    found_mp4 = candidate
                    break
        if found_mp4:
            target_mp4_path = found_mp4
        else:
            logger.warning(f"YouTube video file not produced on disk; vision analysis will be skipped for this URL.")
            return ""

    return os.path.abspath(target_mp4_path)


def extract_audio_from_file(input_path: str, output_dir: str = DOWNLOAD_DIR) -> str:
    """
    Extract and convert any supported local video/audio file into a 16kHz mono WAV file
    using FFmpeg streaming directly (avoids loading huge video files into Python RAM).
    """
    check_ffmpeg()
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(input_path):
        raise MediaValidationError(f"File not found: '{input_path}'")

    if os.path.getsize(input_path) == 0:
        raise MediaValidationError(f"The input file '{os.path.basename(input_path)}' is 0 bytes (empty file).")

    ext = os.path.splitext(input_path)[1].lower()
    if ext not in ALL_SUPPORTED_EXTENSIONS:
        supported_str = ", ".join(sorted(ALL_SUPPORTED_EXTENSIONS))
        raise MediaValidationError(
            f"Unsupported file format '{ext}'. Supported formats: {supported_str}"
        )

    file_id = f"local_{uuid.uuid4().hex[:10]}"
    output_wav_path = os.path.join(output_dir, f"{file_id}.wav")

    cmd = [
        "ffmpeg",
        "-y",
        "-nostdin",
        "-i", input_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        output_wav_path,
    ]

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        raise AudioProcessingError("Audio conversion timed out after 5 minutes.")

    if proc.returncode != 0:
        stderr_sample = proc.stderr[-500:] if proc.stderr else "Unknown error"
        if "does not contain any stream" in stderr_sample or "Output file is empty" in stderr_sample:
            raise MediaValidationError("The provided file does not contain any audio track.")
        raise AudioProcessingError(f"FFmpeg failed to extract audio: {stderr_sample}")

    if not os.path.exists(output_wav_path) or os.path.getsize(output_wav_path) < 100:
        raise MediaValidationError("Failed to extract audio track: Extracted audio file is missing or empty.")

    return os.path.abspath(output_wav_path)


def validate_audio(wav_path: str) -> dict:
    """
    Validate that the WAV file:
    - Exists and is non-empty
    - Has valid RIFF/WAV header
    - Has duration >= 0.5 seconds
    - Has sample count > 0
    - Contains audible sound (not completely silent/zero-amplitude)
    Returns metadata dict: duration, sample_rate, channels, rms.
    """
    if not os.path.exists(wav_path):
        raise MediaValidationError(f"WAV file does not exist: {wav_path}")

    file_size = os.path.getsize(wav_path)
    if file_size < 100:
        raise MediaValidationError(f"Audio file is too small or empty ({file_size} bytes).")

    try:
        with wave.open(wav_path, "rb") as wf:
            channels = wf.getnchannels()
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()
            duration = n_frames / float(sample_rate) if sample_rate > 0 else 0.0

            if n_frames == 0 or duration <= 0.0:
                raise MediaValidationError("Audio file contains 0 audio frames.")

            if duration < 0.5:
                raise MediaValidationError(
                    f"Audio duration is too short ({duration:.2f}s). Please provide a recording longer than 0.5 seconds."
                )

            # Read a sample block to evaluate signal amplitude/RMS
            frames_to_read = min(n_frames, sample_rate * 5)
            raw_bytes = wf.readframes(frames_to_read)
            sampwidth = wf.getsampwidth()
    except wave.Error as e:
        raise MediaValidationError(f"Invalid WAV format: {e}")

    # Inspect energy / RMS
    if sampwidth == 2:
        samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        samples = np.frombuffer(raw_bytes, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif sampwidth == 1:
        samples = (np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32)

    if len(samples) == 0:
        raise MediaValidationError("Audio track contains no samples.")

    rms = float(np.sqrt(np.mean(samples ** 2)))
    max_amplitude = float(np.max(np.abs(samples)))

    # Reject total digital silence (flat zeros)
    if max_amplitude == 0.0 or rms < 1e-6:
        logger.warning("No audible sound detected in the media file (audio is completely silent).")


    return {
        "path": wav_path,
        "duration": duration,
        "sample_rate": sample_rate,
        "channels": channels,
        "rms": rms,
        "max_amplitude": max_amplitude,
    }


def chunk_audio(wav_path: str, chunk_minutes: int = 10, output_dir: str = DOWNLOAD_DIR, overlap_seconds: float = 6.0) -> list:
    """
    Split audio into chunks strictly avoiding empty or sub-second chunks.
    Adjacent chunks overlap by overlap_seconds so speech at 10-minute cuts is not dropped.
    If the audio is shorter than chunk_minutes, returns [wav_path] directly.
    """
    os.makedirs(output_dir, exist_ok=True)
    meta = validate_audio(wav_path)
    duration_ms = int(meta["duration"] * 1000)
    chunk_ms = chunk_minutes * 60 * 1000
    overlap_ms = max(0, int(float(overlap_seconds) * 1000))
    hop_ms = max(1000, chunk_ms - overlap_ms)

    # If already shorter than chunk size, return the validated file directly
    if duration_ms <= chunk_ms:
        return [wav_path]

    audio = AudioSegment.from_wav(wav_path)
    chunks = []
    base_name = os.path.splitext(os.path.basename(wav_path))[0]

    start = 0
    i = 0
    while start < len(audio):
        end = min(start + chunk_ms, len(audio))
        chunk_slice = audio[start:end]

        # Never create empty or sub-second chunks
        if len(chunk_slice) < 500:
            logger.info(f"Skipping tail chunk {i} because duration is {len(chunk_slice)}ms (< 500ms).")
            break

        chunk_path = os.path.join(output_dir, f"{base_name}_chunk_{i}.wav")
        chunk_slice.export(chunk_path, format="wav")

        # Verify exported chunk
        if os.path.exists(chunk_path) and os.path.getsize(chunk_path) > 100:
            chunks.append(chunk_path)

        if end >= len(audio):
            break
        start += hop_ms
        i += 1

    if not chunks:
        # Fallback to single chunk if slicing filtered everything
        chunks.append(wav_path)

    return chunks


def process_input(source: str, progress_callback=None) -> list:
    """
    Unified input processor:
    1. Checks FFmpeg availability
    2. Detects YouTube URL vs local file path
    3. Extracts/converts to 16kHz mono WAV
    4. Validates audio duration and audibility
    5. Chunks audio safely
    Returns list of valid chunk WAV file paths.
    """
    check_ffmpeg()

    if not source or not source.strip():
        raise MediaValidationError("No source provided. Please enter a YouTube URL or upload a file.")

    clean_source = source.strip()

    if clean_source.startswith("http://") or clean_source.startswith("https://"):
        logger.info("Detected YouTube URL. Downloading audio...")
        if progress_callback:
            progress_callback("Downloading audio from YouTube...")
        wav_path = download_youtube_audio(clean_source)
    else:
        logger.info("Detected local file. Converting to 16kHz WAV...")
        if progress_callback:
            progress_callback("Extracting audio with FFmpeg...")
        wav_path = extract_audio_from_file(clean_source)

    if progress_callback:
        progress_callback("Validating audio stream...")
    validate_audio(wav_path)

    if progress_callback:
        progress_callback("Chunking audio for transcription...")
    chunks = chunk_audio(wav_path)
    logger.info(f"Audio ready — {len(chunks)} chunk(s) created.")
    return chunks


def cleanup_temp_files(file_paths: list):
    """Safely remove temporary audio files."""
    for p in file_paths:
        try:
            if p and os.path.exists(p):
                os.remove(p)
        except Exception as e:
            logger.warning(f"Could not clean up temp file {p}: {e}")
