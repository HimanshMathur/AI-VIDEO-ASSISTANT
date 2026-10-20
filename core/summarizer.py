import os
import json
import logging
from typing import List, Dict, Any, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from core.local_llm import (
    chat_completion,
    json_chat_completion,
    check_ollama,
    check_model_available,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
)
from core.language_instructions import language_preservation_instruction as _language_instruction

logger = logging.getLogger(__name__)


def _format_list_as_bullets(items: Any, default_empty: str = "None identified in the recording.") -> str:
    """Format list of items into clean markdown bullets or return string."""
    if not items:
        return default_empty
    if isinstance(items, str):
        return items.strip() if items.strip() else default_empty
    if isinstance(items, list):
        clean_items = [str(it).strip() for it in items if str(it).strip()]
        if not clean_items:
            return default_empty
        return "\n".join(f"{i + 1}. {it}" for i, it in enumerate(clean_items))
    return str(items)


def generate_video_intelligence(transcript: str, language: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate Title, Executive Summary, Action Items, Key Decisions, and Open Questions
    in ONE single consolidated local Qwen call using structured JSON.
    Strictly preserves the spoken language of the transcript without translating.

    `language` is the ISO 639-1 / whisper code detected by the transcriber (or None
    for auto-detect from the transcript text).
    """
    if not transcript or not transcript.strip():
        raise ValueError("Cannot generate intelligence from an empty transcript.")

    sample = transcript.strip()
    # Keep reasonable prompt length (up to 12,000 characters)
    if len(sample) > 12000:
        sample = sample[:12000]

    lang_instruction = _language_instruction(language, transcript)

    system_prompt = (
        "You are an expert video and meeting intelligence analyst.\n"
        "Analyze the transcript and respond with a structured JSON object containing exactly these keys:\n"
        "{\n"
        '  "title": "Clear punchy title under 10 words",\n'
        '  "summary": "Concise executive summary (1-2 paragraphs) followed by core highlights",\n'
        '  "action_items": ["Action item 1 with owner/timeline if mentioned", ...],\n'
        '  "key_decisions": ["Key decision 1", ...],\n'
        '  "open_questions": ["Unresolved question or follow-up 1", ...]\n'
        "}\n\n"
        "RULES:\n"
        f"1. {lang_instruction}\n"
        "2. Keep the output concise, factual, and strictly grounded in the transcript.\n"
        "3. If any category has no items in the recording, provide an empty list []. Do not invent facts."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Recording Transcript:\n{sample}"},
    ]

    intelligence_error = None
    try:
        data = json_chat_completion(messages, temperature=0.1, max_tokens=1024, timeout=120)
    except Exception as e:
        logger.exception("Consolidated JSON intelligence call failed; trying transcript summarization fallback.")
        intelligence_error = e
        data = {}

    if not isinstance(data, dict):
        data = {}

    title_value = data.get("title")
    title = title_value.strip().strip('"\'') if isinstance(title_value, str) else ""
    if not title:
        title = "Video Analysis Dossier"

    summary_value = data.get("summary")
    summary = summary_value.strip() if isinstance(summary_value, str) else ""
    if not summary:
        try:
            summary = summarize(transcript, language=language).strip()
        except Exception as fallback_error:
            logger.exception("Transcript summary fallback failed.")
            error_details = f"{type(fallback_error).__name__}: {fallback_error}"
            if intelligence_error is not None:
                error_details = (
                    f"Structured generation failed ({type(intelligence_error).__name__}: {intelligence_error}); "
                    f"summary fallback failed ({error_details})"
                )
            summary = f"Executive summary generation failed: {error_details}"

    action_items = _format_list_as_bullets(
        data.get("action_items"),
        default_empty="No specific action items were identified in the transcript.",
    )
    key_decisions = _format_list_as_bullets(
        data.get("key_decisions"),
        default_empty="No explicit decisions were recorded in the transcript.",
    )
    open_questions = _format_list_as_bullets(
        data.get("open_questions"),
        default_empty="No unresolved questions were mentioned in the transcript.",
    )

    return {
        "title": title,
        "summary": summary,
        "action_items": action_items,
        "key_decisions": key_decisions,
        "open_questions": open_questions,
        "raw": data,
    }


def split_transcript(transcript: str, chunk_size: int = 2500, chunk_overlap: int = 150) -> List[str]:
    """Split long transcript into manageable chunks for map-reduce summarization."""
    if not transcript or not transcript.strip():
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_text(transcript.strip())


def generate_title(transcript: str, language: Optional[str] = None) -> str:
    """
    Generate a concise, professional title (under 10 words) strictly in the
    SAME spoken language as the transcript.
    """
    if not transcript or not transcript.strip():
        raise ValueError("Cannot generate title from an empty transcript.")

    sample = transcript.strip()[:2000]
    lang_instruction = _language_instruction(language, transcript)
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert video content analyst.\n"
                "Generate a clear, punchy, professional title (maximum 10 words) for this recording.\n\n"
                "RULES:\n"
                f"1. {lang_instruction}\n"
                "2. Output ONLY the title itself. Do not wrap in quotes or add labels."
            ),
        },
        {"role": "user", "content": f"Transcript excerpt:\n{sample}"},
    ]

    title = chat_completion(messages, temperature=0.2, timeout=120).strip()
    title = title.strip('"\'').strip()
    if not title:
        title = "Video Analysis Dossier"
    return title


def summarize(transcript: str, language: Optional[str] = None) -> str:
    """
    Generate an executive summary and key points 100% locally via Ollama.
    Strictly preserves the spoken language of the transcript without translating.
    """
    if not transcript or not transcript.strip():
        raise ValueError("Cannot summarize an empty transcript.")

    chunks = split_transcript(transcript)
    if not chunks:
        raise ValueError("No text content available to summarize.")

    lang_instruction = _language_instruction(language, transcript)

    # For shorter recordings, summarize directly in one pass
    if len(chunks) == 1:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert video and meeting intelligence analyst.\n"
                    "Generate a concise executive summary followed by core takeaways.\n\n"
                    "RULES:\n"
                    f"1. {lang_instruction}\n"
                    "2. Structure your response with:\n"
                    "   - **Executive Summary**: A concise 1-2 paragraph synthesis.\n"
                    "   - **Core Highlights**: 3-5 bullet points covering critical insights.\n"
                    "3. Ground everything strictly in the transcript content."
                ),
            },
            {"role": "user", "content": f"Transcript:\n{chunks[0]}"},
        ]
        return chat_completion(messages, temperature=0.2, timeout=300).strip()

    # Map-Reduce for longer transcripts
    logger.info(f"Summarizing long transcript in {len(chunks)} chunk(s)...")
    partial_summaries = []
    for i, chunk in enumerate(chunks):
        logger.info(f"Processing summary chunk {i + 1}/{len(chunks)}...")
        messages = [
            {
                "role": "system",
                "content": (
                    "Summarize the key points of this transcript section. "
                    f"{lang_instruction}"
                ),
            },
            {"role": "user", "content": chunk},
        ]
        part = chat_completion(messages, temperature=0.2, timeout=180).strip()
        if part:
            partial_summaries.append(part)

    combined = "\n\n".join(partial_summaries)
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert video and meeting intelligence analyst.\n"
                "Synthesize these section summaries into one cohesive final executive summary.\n\n"
                "RULES:\n"
                f"1. {lang_instruction}\n"
                "2. Include an Executive Summary section and a Core Highlights bulleted list.\n"
                "3. Ground all facts strictly in the provided text."
            ),
        },
        {"role": "user", "content": f"Section summaries:\n{combined}"},
    ]
    return chat_completion(messages, temperature=0.2, timeout=300).strip()
