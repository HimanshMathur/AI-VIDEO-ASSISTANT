import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import unicodedata
from core.transcriber import _clean_segment_text

print("=== Unicode Cleaning Verification ===\n")

def check(label, text, expected):
    result = _clean_segment_text(text)
    ok = result == expected
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {label}")
    print(f"       input:    {text!r}")
    print(f"       expected: {expected!r}")
    print(f"       got:      {result!r}")
    assert ok, f"{label}: expected {expected!r}, got {result!r}"
    print()

# Hindi: nahi (नहीं) - must keep final anusvara + vowel sign
check("Hindi नहीं preserved", "नहीं", "नहीं")

# Punjabi: main (ਮੈਂ) - must keep final bindi/tippi
check("Punjabi ਮੈਂ preserved", "ਮੈਂ", "ਮੈਂ")

# Hindi sentence
check("Hindi sentence preserved", "मैं जा रहा हूं", "मैं जा रहा हूं")

# Punjabi sentence
check("Punjabi sentence preserved", "ਮੈਂ ਪੰਜਾਬੀ ਬੋਲਦਾ ਹਾਂ", "ਮੈਂ ਪੰਜਾਬੀ ਬੋਲਦਾ ਹਾਂ")

# Hinglish mixed script
check("Hinglish preserved", "video mein kya है", "video mein kya है")

# English trailing whitespace stripped
check("English trailing whitespace stripped", "hello world  ", "hello world")

# Orphaned mark at very start of string -> stripped
check("Orphaned mark at start stripped", "\u093e", "")

# Orphaned mark after whitespace -> stripped
check("Orphaned mark after space stripped", "hello \u093e", "hello")

# Mark after a letter is NOT stripped (safer: it may be a valid attached mark)
check("Mark after Latin letter kept (safe)", "abc\u093e", "abc\u093e")

# English plain text unchanged
check("English plain text", "the quick brown fox", "the quick brown fox")

# Empty string
check("Empty string", "", "")

print("=== ALL UNICODE CLEANING TESTS PASSED ===")
