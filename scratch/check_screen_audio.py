import os
import sys
sys.path.insert(0, os.path.abspath("."))
from utils.audio_processor import extract_audio_from_file, validate_audio, MediaValidationError

screen_dir = r"C:\Users\mathu\Videos\Screen Recordings"
for f in os.listdir(screen_dir):
    if f.endswith(".mp4"):
        full = os.path.join(screen_dir, f)
        try:
            wav = extract_audio_from_file(full, output_dir="scratch")
            meta = validate_audio(wav)
            print(f"{f} -> VALID AUDIO ({meta['duration']:.1f}s)")
        except MediaValidationError as e:
            print(f"{f} -> SILENT: {e}")
        except Exception as e:
            print(f"{f} -> ERROR: {e}")
