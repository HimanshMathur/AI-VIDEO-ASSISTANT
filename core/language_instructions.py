"""Shared language-preservation wording for transcript-derived prompts."""

from __future__ import annotations

from typing import Optional

from core.languages import detect_language_from_text, get_language


def language_preservation_instruction(transcript_language: Optional[str], transcript: str) -> str:
    """Build a consistent instruction to preserve the transcript's language and script."""
    lang = get_language(transcript_language) if transcript_language else None
    if lang is None and transcript:
        lang = detect_language_from_text(transcript[:500])

    if lang is None:
        return (
            "Write EVERYTHING in the EXACT SAME language, dialect, and script as the "
            "transcript text. Do not translate to English or any other language. "
            "If the transcript is in a right-to-left script, preserve the natural RTL reading order."
        )

    script = lang.scripts[0] if lang.scripts else "the original script"
    direction_note = ""
    if lang.rtl:
        direction_note = (
            f" This language ({lang.name}) is written right-to-left (RTL); "
            "preserve the natural RTL reading order and do not reverse words."
        )
    return (
        f"Write EVERYTHING in the EXACT SAME language as the transcript: "
        f"{lang.native_name} ({lang.name}), using the {script} script."
        f"{direction_note} Do not translate to English or any other language."
    )
