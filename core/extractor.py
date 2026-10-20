import logging
from typing import Optional
from core.local_llm import chat_completion
from core.language_instructions import language_preservation_instruction as _language_instruction

logger = logging.getLogger(__name__)


def _extract_with_prompt(transcript: str, system_prompt: str, language: Optional[str] = None) -> str:
    """Helper to run structured extraction queries against local LLM."""
    sample = transcript.strip()
    # Limit context size to avoid context-length overflow on massive transcripts
    if len(sample) > 12000:
        sample = sample[:12000]

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Recording Transcript:\n{sample}"},
    ]
    return chat_completion(messages, temperature=0.1, timeout=180).strip()


def extract_action_items(transcript: str, language: Optional[str] = None) -> str:
    """Extract action items, owners, and deadlines in the transcript's language via local Ollama."""
    if not transcript or not transcript.strip():
        raise ValueError("Cannot extract action items from an empty transcript.")

    lang_instruction = _language_instruction(language, transcript)
    prompt = (
        "You are an expert video and meeting intelligence analyst.\n"
        "From the transcript, extract all action items and tasks.\n"
        "For each item provide:\n"
        "- Task description\n"
        "- Owner (who is responsible, or 'Not specified')\n"
        "- Deadline/timeline (if mentioned, or 'Not specified')\n\n"
        "RULES:\n"
        f"1. {lang_instruction}\n"
        "2. Format as a clean numbered list.\n"
        "3. If none are found in the transcript, state clearly that no action items were found. Do not invent tasks."
    )
    return _extract_with_prompt(transcript, prompt, language=language)


def extract_key_decisions(transcript: str, language: Optional[str] = None) -> str:
    """Extract all key strategic and operational decisions made in the transcript's language via local Ollama."""
    if not transcript or not transcript.strip():
        raise ValueError("Cannot extract key decisions from an empty transcript.")

    lang_instruction = _language_instruction(language, transcript)
    prompt = (
        "You are an expert video and meeting intelligence analyst.\n"
        "From the transcript, extract all key strategic, technical, and operational decisions made.\n\n"
        "RULES:\n"
        f"1. {lang_instruction}\n"
        "2. Format as a clean numbered list.\n"
        "3. If none are found in the transcript, state clearly that no key decisions were found. Do not hallucinate."
    )
    return _extract_with_prompt(transcript, prompt, language=language)


def extract_questions(transcript: str, language: Optional[str] = None) -> str:
    """Extract all unresolved questions or follow-up topics in the transcript's language via local Ollama."""
    if not transcript or not transcript.strip():
        raise ValueError("Cannot extract questions from an empty transcript.")

    lang_instruction = _language_instruction(language, transcript)
    prompt = (
        "You are an expert video and meeting intelligence analyst.\n"
        "From the transcript, extract all unresolved questions, doubts, concerns, or open follow-ups.\n\n"
        "RULES:\n"
        f"1. {lang_instruction}\n"
        "2. Format as a clean numbered list.\n"
        "3. If none are found in the transcript, state clearly that no open questions were found."
    )
    return _extract_with_prompt(transcript, prompt, language=language)