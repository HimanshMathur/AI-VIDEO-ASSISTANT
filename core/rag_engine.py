import os
import json
import re
import time
import logging
import hashlib
import unicodedata
from typing import List, Optional, Union, Any, Iterator, Dict
from core.vector_store import build_vector_store, build_visual_vector_store, FaissVectorStore, FaissDocument
from core.local_llm import (
    chat_completion,
    chat_completion_stream,
    json_chat_completion,
    check_ollama,
    OLLAMA_MODEL,
)
from core.languages import (
    get_language,
    detect_language_from_text,
    detect_script,
    is_rtl,
    language_display_name,
)

logger = logging.getLogger(__name__)


def _remove_repeated_sentences(text: str) -> str:
    """Remove *consecutively* repeated sentences from model output.

    Conservative approach:
    - Splits on newlines only (preserving paragraph structure).
    - Within each line, detects exact sentence-level consecutive repetitions.
    - A sentence is considered a repeat only if it is identical to the
      IMMEDIATELY preceding sentence after case-folding + whitespace collapse.
    - Distinct sentences, list items, timestamps, and words that happen to
      repeat for valid reasons (e.g. "Step 1. Step 2.") are never removed.
    - Newline structure is fully preserved; lines are not joined.
    """
    if not text:
        return text

    def _dedup_line(line: str) -> str:
        """Remove immediately consecutive duplicate sentences within one line."""
        # Split on sentence-terminal punctuation + space; keep the delimiter.
        sentence_tokens = re.split(r"(?<=[।\.!?])\s+", line)
        result = []
        prev_norm = None
        for token in sentence_tokens:
            norm = re.sub(r"\s+", " ", token.casefold().strip())
            if norm and norm == prev_norm:
                continue  # exact consecutive duplicate – skip
            result.append(token)
            prev_norm = norm
        return " ".join(result)

    lines = text.split("\n")
    deduped_lines = []
    prev_line_norm = None
    for line in lines:
        deduped = _dedup_line(line)
        line_norm = re.sub(r"\s+", " ", deduped.casefold().strip())
        # Also skip a line that is an exact duplicate of the previous line.
        if line_norm and line_norm == prev_line_norm:
            continue
        deduped_lines.append(deduped)
        prev_line_norm = line_norm if line_norm else prev_line_norm
    return "\n".join(deduped_lines)


def _question_timestamp(question: str) -> Optional[float]:
    """Return the first explicit M:SS, MM:SS, or H:MM:SS timestamp in a question."""
    match = re.search(r"(?<!\d)(\d{1,2}):(\d{1,2})(?::(\d{1,2}))?(?!\d)", question or "")
    if not match:
        return None
    first, second, third = match.groups()
    if third is not None:
        hours, minutes, seconds = int(first), int(second), int(third)
        if minutes >= 60 or seconds >= 60:
            return None
        return hours * 3600 + minutes * 60 + seconds
    minutes, seconds = int(first), int(second)
    if seconds >= 60:
        return None
    return minutes * 60 + seconds


def _without_question_timestamp(question: str) -> str:
    return re.sub(r"(?<!\d)\d{1,2}:\d{1,2}(?::\d{1,2})?(?!\d)", " ", question or "").strip()


def _timestamp_intent(question: str, has_visual_store: bool) -> Optional[str]:
    """Route timestamp questions by the evidence they ask about."""
    if _question_timestamp(question) is None:
        return None
    q = question.casefold()
    visual = any(term in q for term in (
        "shown", "visible", "screen", "visual", "display", "look like",
        "what happens", "what is happening", "scene", "slide", "seen",
        "wearing", "color", "colour", "shirt", "holding", "person", "people",
    ))
    speech = any(term in q for term in (
        "speaker", "say", "said", "says", "spoken", "speech", "transcript",
        "topic", "discuss", "talk", "mentioned", "which subject",
        "orateur", "hablante", "sprecher", "\u0935\u0915\u094d\u0924\u093e",
    ))
    if visual and speech:
        return "MULTIMODAL"
    if visual:
        visual_only_details = any(term in q for term in (
            "shown", "visible", "screen", "visual", "display", "look like",
            "scene", "slide", "seen", "wearing", "color", "colour", "shirt",
            "holding", "person", "people",
        ))
        if visual_only_details:
            return "VISUAL_ONLY"
        # Neutral timestamped event questions still search both evidence sources.
        return "MULTIMODAL"
    if speech:
        return "TEXT_ONLY"
    # Neutral event questions (for example, "what happened at 2:06?") need
    # both retrieval paths so available visual evidence is not discarded.
    return "MULTIMODAL" if has_visual_store else "TEXT_ONLY"


def _visual_index_size(rag_chain: Any) -> int:
    """Return the visual index size, treating absent stores/indexes as unavailable."""
    visual_store = getattr(rag_chain, "visual_store", None)
    index = getattr(visual_store, "index", None) if visual_store is not None else None
    try:
        return max(0, int(getattr(index, "ntotal", 0))) if index is not None else 0
    except (TypeError, ValueError, AttributeError):
        return 0


def _retrieve_transcript_by_timestamp(rag_chain: Any, target_seconds: float, k: int = 4) -> List[FaissDocument]:
    """Select transcript chunks overlapping an explicit time plus nearby context."""
    store = getattr(rag_chain, "vector_store", None)
    chunks = getattr(store, "chunks", [])
    if not chunks:
        return []
    duration = float(getattr(rag_chain, "duration_seconds", 0.0) or 0.0)
    timed_chunks = []
    for position, chunk in enumerate(chunks):
        start = chunk.get("start_seconds", chunk.get("segment_start_seconds"))
        end = chunk.get("end_seconds", chunk.get("segment_end_seconds"))
        if start is None:
            start = chunk.get("timestamp_seconds", chunk.get("start_time", 0.0))
            end = chunk.get("timestamp_seconds", chunk.get("end_time", start)) if end is None else end
            # Legacy transcript stores keep percentage offsets in start_time/end_time.
            if duration > 0 and float(start or 0) <= 100 and float(end or 0) <= 100:
                start, end = float(start or 0) * duration / 100, float(end or 0) * duration / 100
        try:
            start = max(0.0, float(start or 0.0))
            end = max(start, float(end if end is not None else start))
        except (TypeError, ValueError):
            continue
        timed_chunks.append((position, start, end, chunk))

    if not timed_chunks:
        return []

    overlapping = [item for item in timed_chunks if item[1] <= target_seconds <= item[2]]
    if overlapping:
        # Preserve every directly overlapping segment, then add adjacent chunks
        # for sentence context when the source has discrete segment records.
        selected_positions = {item[0] for item in overlapping}
        for position, _, _, _ in overlapping:
            selected_positions.update((position - 1, position + 1))
        selected = [item for item in timed_chunks if item[0] in selected_positions]
        selected.sort(key=lambda item: (0 if item[1] <= target_seconds <= item[2] else 1,
                                        abs((item[1] + item[2]) / 2 - target_seconds), item[0]))
    else:
        # If the request falls in a gap, include the nearest segment on each
        # side where available, rather than selecting arbitrary distant text.
        before = [item for item in timed_chunks if item[2] < target_seconds]
        after = [item for item in timed_chunks if item[1] > target_seconds]
        nearest = []
        if before:
            nearest.append(max(before, key=lambda item: item[2]))
        if after:
            nearest.append(min(after, key=lambda item: item[1]))
        if not nearest:
            nearest = [min(timed_chunks, key=lambda item: min(abs(target_seconds - item[1]), abs(target_seconds - item[2])))]
        selected_positions = {item[0] for item in nearest}
        for position, _, _, _ in nearest:
            selected_positions.update((position - 1, position + 1))
        selected = [item for item in timed_chunks if item[0] in selected_positions]
        selected.sort(key=lambda item: (min(abs(target_seconds - item[1]), abs(target_seconds - item[2])), item[0]))

    return [
        FaissDocument(
            page_content=chunk.get("text", ""),
            metadata={key: value for key, value in chunk.items() if key != "text"},
        )
        for _, _, _, chunk in selected[:max(k, len(overlapping))]
        if chunk.get("text")
    ]


def _is_chapter_generation_request(question: str) -> bool:
    """Recognize requests to turn video content into timestamped chapters."""
    normalized = question.lower()
    requests_chapters = re.search(
        r"\b(divide|split|create|generate|make|organize|break|show|give|list|provide|prepare|"
        r"baanto|banao|banaen|banaaye|karo|batao|dikhao|likho|taiyar)\b"
        r"|(?:बांट|विभाजित|बनाइ|बनाएं|बनाये|तैयार|दिखाइ|बताइ|लिख)",
        normalized,
    )
    chapter_terms = re.search(
        r"\bchapters?\b|\btimestamps?\b|\btopic[- ]wise\b|\btopics?\b"
        r"|अध्याय|चैप्टर|टाइमस्टैम्प|विषय|टॉपिक",
        normalized,
    )
    return bool(requests_chapters and chapter_terms)


def _chapter_title_language(question: str) -> str:
    """Determine the language/script to write chapter titles in, based on the question.

    Detects the question's script and maps it to a natural language name.
    Falls back to English for Latin-script questions that are not clearly Hinglish.
    """
    # Detect the script of the question
    detected = detect_language_from_text(question)
    if detected is not None:
        return f"{detected.name} in {detected.scripts[0]} script"

    # Hinglish heuristic: common Hindi words written in Latin script
    if re.search(
        r"\b(mein|ke|ki|ko|kya|kaise|hai|hain|bana|banao|banaen|baanto|karo|dikhao|hisaab|video|screen)\b",
        question.lower(),
    ):
        return "natural Hinglish using the Latin alphabet"

    return "English"


def _format_chapter_time(seconds: float) -> str:
    """Format a stored absolute time without exposing percentage offsets."""
    total_seconds = max(0, int(round(float(seconds))))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def _chapter_words(text: str) -> list[str]:
    """Tokenize Unicode words while keeping combining marks with their base."""
    words = []
    current = []
    for char in text:
        category = unicodedata.category(char)
        if category[0] in {"L", "M", "N"}:
            current.append(char)
        elif char in {"'", "\u2019"} and current:
            current.append(char)
        else:
            if current:
                words.append("".join(current).strip("'\u2019"))
                current = []
    if current:
        words.append("".join(current).strip("'\u2019"))
    return [word for word in words if word]


def _non_latin_letter_count(text: str) -> int:
    return sum(
        1 for char in text
        if unicodedata.category(char).startswith("L")
        and "LATIN" not in unicodedata.name(char, "")
    )


def _repair_chapter_mojibake(text: str) -> str:
    """Repair byte-decoded transcript text only when script recovery is clear."""
    current = text
    for _ in range(2):
        original_script_letters = _non_latin_letter_count(current)
        repaired = None
        for encoding in ("cp1252", "latin-1"):
            try:
                candidate = current.encode(encoding).decode("utf-8")
            except (UnicodeEncodeError, UnicodeDecodeError):
                continue
            if _non_latin_letter_count(candidate) >= original_script_letters + 2:
                repaired = candidate
                break
        if repaired is None:
            break
        current = repaired
    return current


def _fallback_chapter_title(text: str) -> str:
    """Use a short grounded label, or a neutral fallback for clipped fragments."""
    words = _chapter_words(text)
    ignored = {"a", "an", "and", "at", "by", "for", "from", "in", "into", "of", "on", "or", "the", "to", "with", "um", "uh", "like", "yeah", "okay", "well", "so", "just", "really", "very", "super", "way", "more"}
    while words and words[0].casefold() in ignored:
        words.pop(0)
    while words and words[-1].casefold() in ignored | {"than", "because", "but", "then"}:
        words.pop()
    return " ".join(words[:6]) if len(words) >= 2 else "Discussion"


def _chapter_title_supported(title: str, transcript_context: str) -> bool:
    """Require meaningful transcript anchors while allowing semantic paraphrases."""
    ignored = {
        "a", "an", "and", "at", "by", "for", "from", "in", "into", "of", "on", "or", "the", "to", "with",
        "about", "after", "before", "continues", "more", "new", "some", "then",
    }

    def terms(text: str) -> set[str]:
        normalized = set()
        for raw in _chapter_words(text):
            token = raw.casefold().strip("'’")
            if token in ignored or len(token) <= 1:
                continue
            # Light English inflection normalization; Indic and other scripts
            # remain untouched and are compared exactly.
            if token.isascii():
                if len(token) > 6 and token.endswith("ation"):
                    token = token[:-5]
                    if token.endswith("or"):
                        token += "e"
                elif len(token) > 5 and token.endswith("ing"):
                    token = token[:-3]
                    if len(token) > 2 and token[-1] == token[-2]:
                        token = token[:-1]
                    if token.endswith("or"):
                        token += "e"
                elif len(token) > 4 and token.endswith("al"):
                    token = token[:-2]
                elif len(token) > 4 and token.endswith("e"):
                    token = token[:-1]
                elif len(token) > 4 and token.endswith("s"):
                    token = token[:-1]
            normalized.add(token)
        return normalized

    title_terms = terms(title)
    context_terms = terms(transcript_context)
    if not title_terms or not context_terms:
        return False
    grounded_terms = title_terms & context_terms
    minimum_anchors = 1 if len(title_terms) <= 2 else 2
    return (
        len(grounded_terms) >= minimum_anchors
        and len(grounded_terms) / len(title_terms) >= 0.35
    )


def _chapter_title_is_corrupted(title: str) -> bool:
    """Detect replacement/control characters and recoverable UTF-8 mojibake."""
    if any(unicodedata.category(char) == "Cc" and char not in "\t\n\r" for char in title):
        return True
    if "\ufffd" in title:
        return True
    return _repair_chapter_mojibake(title) != title


def _same_chapter_topic(first: str, second: str) -> bool:
    """Compare title meaning without depending on word order or inflection."""
    def normalize(title: str) -> set[str]:
        ignored = {
            "a", "an", "and", "at", "by", "for", "from", "in", "into", "of", "on", "or", "the", "to", "with",
            "about", "after", "before", "continues", "more", "new", "some", "then",
        }
        terms = set()
        for word in _chapter_words(title.casefold()):
            if word in ignored:
                continue
            if len(word) > 5 and word.endswith("ies"):
                word = word[:-3] + "y"
            elif len(word) > 6 and word.endswith("ation"):
                word = word[:-5]
            elif len(word) > 5 and word.endswith("ing"):
                word = word[:-3]
            elif len(word) > 4 and word.endswith("ed"):
                word = word[:-2]
            elif len(word) > 4 and word.endswith("s"):
                word = word[:-1]
            terms.add(word)
        return terms

    words_a = normalize(first)
    words_b = normalize(second)
    if not words_a or not words_b:
        return False
    if first.casefold().strip() == second.casefold().strip():
        return True
    shared = len(words_a & words_b)
    # Require two informative terms so a generic shared word cannot erase a
    # real boundary. Set comparison makes word order irrelevant.
    return shared >= 2 and (words_a <= words_b or words_b <= words_a or shared / len(words_a | words_b) >= 0.6)


def _is_ambiguous_transcript_fragment(text: str) -> bool:
    """Recognize short, unfinished fragments that should inherit a topic."""
    words = _chapter_words(text)
    if not words:
        return True
    if words[-1].casefold() in {
        "and", "or", "but", "then", "because", "so", "than", "with", "of", "to", "for", "from", "about",
    }:
        return True
    if len(words) > 4:
        return False
    return bool(re.search(r"(?:\.\.\.|[,;:])\s*$", text.strip()))


def _generate_chapter_list(rag_chain: Any, question: str) -> str:
    """Group ordered transcript segments into concise, topic-based chapters."""
    vector_store = getattr(rag_chain, "vector_store", None)
    chunks = getattr(vector_store, "chunks", None)
    if not chunks:
        raise ValueError("No timestamped transcript segments are available for chapter generation.")

    ordered_chunks = sorted(
        chunks,
        key=lambda chunk: (float(chunk.get("start_seconds", 0) or 0), chunk.get("chunk_index", 0)),
    )
    chapter_segments = []
    for chunk in ordered_chunks:
        seconds = chunk.get("start_seconds")
        text = _repair_chapter_mojibake(chunk.get("text", "").strip())
        if seconds is None or not text:
            raise ValueError("A transcript segment is missing its existing timestamp or text.")
        chapter_segments.append({
            "index": chunk.get("chunk_index", len(chapter_segments)),
            "timestamp": _format_chapter_time(seconds),
            "text": text,
        })

    rtl_note = ""
    transcript_context = " ".join(segment["text"] for segment in chapter_segments)
    detected = detect_language_from_text(transcript_context)
    if detected and detected.rtl:
        rtl_note = " Preserve the natural right-to-left (RTL) reading order; do not reverse words."
    transcript_language_note = (
        f"The transcript is primarily {detected.name} in {detected.scripts[0]} script. "
        if detected else "Mirror the language and script used in each transcript section. "
    )
    titles_by_index = {}
    chapter_start_indices = set()
    legacy_chapter_starts = set()
    # Small batches keep local multilingual inference inside Ollama's context
    # and request timeout while leaving neighboring chunks available as context.
    batch_size = 3
    for offset in range(0, len(chapter_segments), batch_size):
        batch = chapter_segments[offset:offset + batch_size]
        system_prompt = (
            "Create concise, meaningful, topic-based video chapters from the supplied transcript segments. "
            f"{transcript_language_note}{rtl_note} Preserve original script and mixed-language wording; do not transliterate or translate. "
            "Return JSON only in this exact shape: "
            '{"chapters":[{"start_index":0,"end_index":2,"title":"Concise topic title"}]}. '
            "Return chapter groups covering requested indices. A group may extend by one following context segment when it clearly continues the same topic across the batch boundary. "
            "Infer each chapter's central idea from all of its segments and neighboring context. Identify topic boundaries from meaningful changes in the discussion, not sentence or transcript-chunk boundaries. "
            "Group consecutive segments about the same subject in one chapter; end_index is inclusive. "
            "For each title, paraphrase the central topic in clear, human-readable language in your own words. Do not copy a short transcript phrase, quote a fragment, or use filler as the title. "
            "Use a concise descriptive phrase of about 3-10 words, preserving the discussion's language and script. Do not add claims or concepts that the segments do not support. "
            "Start a new chapter only at a meaningful subject, activity, or explanatory-section change, never at an arbitrary chunk or sentence boundary. "
            "A very short or ambiguous segment normally continues its neighbors; do not invent a subtopic from it. "
            "Every requested index must be covered by exactly one chapter group. "
            "Titles must be concise semantic paraphrases, supported by the transcript section, complete, and not generic or corrupted. "
            "Do not include timestamps."
        )
        context_start = max(0, offset - 1)
        context_end = min(len(chapter_segments), offset + len(batch) + 1)
        context_segments = chapter_segments[context_start:context_end]
        user_prompt = json.dumps(
            {
                "segments_for_context": [
                    {"index": segment["index"], "text": segment["text"]}
                    for segment in context_segments
                ],
                "return_titles_for_indices": [segment["index"] for segment in batch],
            },
            ensure_ascii=False,
        )
        try:
            result = json_chat_completion(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=192,
                timeout=90,
            )
        except Exception as exc:
            logger.warning("Chapter title model failed; using transcript-derived fallback: %s", exc)
            result = None
        generated = result.get("chapters") if isinstance(result, dict) else None
        # Older local model responses sometimes return one legacy chapter
        # object at the root instead of wrapping it in {"chapters": [...]}.
        if (not isinstance(generated, list) and isinstance(result, dict)
                and ("index" in result or "start_index" in result)):
            generated = [result]
        if not isinstance(generated, list):
            generated = []

        expected_indices = {segment["index"] for segment in batch}
        invalid_title_indices = set()
        context_positions = {segment["index"]: position for position, segment in enumerate(context_segments)}
        batch_indices = {segment["index"] for segment in batch}
        claimed_positions = set()
        grouped_ranges = set()
        explicitly_grouped_indices = set()
        for chapter in generated:
            if not isinstance(chapter, dict):
                continue
            # Handle each record by its own shape. A response may mix grouped
            # ranges with legacy per-segment entries; the explicit ranges are
            # authoritative for any segments they cover.
            if "start_index" in chapter:
                start_index = chapter.get("start_index")
                end_index = chapter.get("end_index", start_index)
                try:
                    if (start_index not in context_positions or end_index not in context_positions
                            or start_index not in batch_indices):
                        continue
                except TypeError:
                    continue
                start_position = context_positions[start_index]
                end_position = context_positions[end_index]
                if start_position > end_position or any(
                    position in claimed_positions for position in range(start_position, end_position + 1)
                ):
                    continue
                range_key = (start_index, end_index)
                if range_key in grouped_ranges:
                    continue
                grouped_ranges.add(range_key)
                positions = list(range(start_position, end_position + 1))
                segment_indices = [context_segments[position]["index"] for position in positions]
                claimed_positions.update(positions)
                chapter_index = start_index
            else:
                chapter_index = chapter.get("index")
                try:
                    if chapter_index not in expected_indices:
                        continue
                except TypeError:
                    continue
                position = next((i for i, segment in enumerate(batch) if segment["index"] == chapter_index), None)
                if position is None:
                    continue
                if chapter_index in explicitly_grouped_indices:
                    continue
                positions = [position]
                segment_indices = [chapter_index]
            title = chapter.get("title")
            if (not isinstance(title, str) or not title.strip()
                    or _chapter_title_is_corrupted(title)
                    or _is_ambiguous_transcript_fragment(title)):
                invalid_title_indices.update(segment_indices)
                continue
            title = " ".join(title.split())
            if len(title.split()) > 10:
                invalid_title_indices.update(segment_indices)
                continue
            global_start = next(i for i, item in enumerate(chapter_segments) if item["index"] == segment_indices[0])
            global_end = next(i for i, item in enumerate(chapter_segments) if item["index"] == segment_indices[-1])
            context_start = max(0, global_start - 1)
            context_end = min(len(chapter_segments), global_end + 2)
            local_context = " ".join(item["text"] for item in chapter_segments[context_start:context_end])
            if _chapter_title_supported(title, local_context):
                if "start_index" in chapter:
                    explicitly_grouped_indices.update(segment_indices)
                for segment_index in segment_indices:
                    titles_by_index[segment_index] = title
                if "start_index" in chapter:
                    # Do not let a legacy entry or a fallback create a
                    # timestamp inside a model-declared continuous chapter.
                    for segment_index in segment_indices[1:]:
                        chapter_start_indices.discard(segment_index)
                        legacy_chapter_starts.discard(segment_index)
                chapter_start_indices.add(segment_indices[0])
                if "start_index" not in chapter:
                    legacy_chapter_starts.add(segment_indices[0])
            else:
                invalid_title_indices.update(segment_indices)

        # Recover uncovered transcript runs together. Assigning a fresh
        # fallback chapter to every omitted chunk turns model timeouts into a
        # list of sentence fragments rather than topic chapters.
        valid_titles = [
            (position, titles_by_index[segment["index"]])
            for position, segment in enumerate(batch)
            if segment["index"] in titles_by_index
        ]
        if not valid_titles:
            previous_segment = chapter_segments[offset - 1] if offset else None
            run_text = " ".join(item["text"] for item in batch)
            previous_title = (
                titles_by_index.get(previous_segment["index"])
                if previous_segment is not None else None
            )
            if previous_title and _chapter_title_supported(previous_title, run_text):
                fallback_batch_title = previous_title
            else:
                fallback_batch_title = _fallback_chapter_title(run_text)
                chapter_start_indices.add(batch[0]["index"])
                legacy_chapter_starts.add(batch[0]["index"])
            for segment in batch:
                titles_by_index.setdefault(segment["index"], fallback_batch_title)
            continue

        position = 0
        while position < len(batch):
            if batch[position]["index"] in titles_by_index:
                position += 1
                continue
            run_start = position
            while position < len(batch) and batch[position]["index"] not in titles_by_index:
                position += 1
            run_end = position
            run = batch[run_start:run_end]
            run_text = " ".join(item["text"] for item in run)
            preceding = [entry for entry in valid_titles if entry[0] < run_start]
            following = [entry for entry in valid_titles if entry[0] >= run_end]
            run_has_ambiguous_entry = any(
                item["index"] in invalid_title_indices or _is_ambiguous_transcript_fragment(item["text"])
                for item in run
            )
            if preceding and (
                run_has_ambiguous_entry
                or _chapter_title_supported(preceding[-1][1], run_text)
            ):
                run_title = preceding[-1][1]
            elif following and (
                run_has_ambiguous_entry
                or _chapter_title_supported(following[0][1], run_text)
            ):
                run_title = following[0][1]
                chapter_start_indices.add(run[0]["index"])
                legacy_chapter_starts.add(run[0]["index"])
            else:
                run_title = _fallback_chapter_title(run_text)
                chapter_start_indices.add(run[0]["index"])
                legacy_chapter_starts.add(run[0]["index"])
            for item in run:
                titles_by_index[item["index"]] = run_title

    # Markdown needs a hard break; preserve each original timestamp and title.
    chapter_lines = []
    last_chapter = None
    for segment in chapter_segments:
        if segment["index"] not in chapter_start_indices:
            continue
        title = titles_by_index[segment["index"]]
        timestamp_parts = [float(part) for part in segment["timestamp"].split(":")]
        seconds = timestamp_parts[-1] + timestamp_parts[-2] * 60
        if len(timestamp_parts) == 3:
            seconds += timestamp_parts[0] * 3600
        if (last_chapter is not None
                and title.casefold().strip() != last_chapter["title"].casefold().strip()
                and segment["index"] in legacy_chapter_starts
                and last_chapter["index"] in legacy_chapter_starts
                and seconds - last_chapter["seconds"] <= 10
                and _same_chapter_topic(title, last_chapter["title"])):
            continue
        chapter_lines.append(f"{segment['timestamp']} \u2014 {title}  ")
        last_chapter = {"title": title, "seconds": seconds, "index": segment["index"]}
    return "\n".join(chapter_lines)


def _chapter_request_response(rag_chain: Any, question: str) -> str:
    try:
        return _generate_chapter_list(rag_chain, question)
    except Exception as exc:
        logger.error("Video chapter generation failed: %s", exc)
        if not check_ollama():
            return "Local Ollama server is not running. Start Ollama and try again."
        return f"Could not generate video chapters locally: {exc}"


class LocalRAGChain:
    """
    Session-scoped Multimodal RAG engine coordinating local FAISS vector search
    (across both transcript text and visual evidence) and local Ollama inference.
    """

    def __init__(
        self,
        vector_store: FaissVectorStore,
        visual_store: Optional[FaissVectorStore] = None,
        top_k: int = 5,
    ):
        self.vector_store = vector_store
        self.visual_store = visual_store
        self.top_k = top_k
        # This cache belongs to one source-scoped chain. Never share answers
        # between videos; cache only completed model responses.
        self._answer_cache: Dict[str, str] = {}

    def retrieve(self, question: str) -> List[FaissDocument]:
        """Retrieve a broad candidate set for query-aware evidence ranking."""
        return self.vector_store.similarity_search(question, k=max(self.top_k, 12))

    def retrieve_visual(self, question: str, k: int = 4) -> List[FaissDocument]:
        """Retrieve top-k relevant visual evidence points from in-memory FAISS."""
        if not self.visual_store or getattr(self.visual_store, "index", None) is None or self.visual_store.index.ntotal == 0:
            return []
        docs = self.visual_store.similarity_search(question, k=k)
        if getattr(self, "video_id", None) is not None:
            docs = [doc for doc in docs if doc.metadata.get("video_id") == self.video_id]
        return docs

    def retrieve_visual_by_timestamp(self, target_seconds: float, k: int = 3) -> List[FaissDocument]:
        """Retrieve visual evidence keyframes that were active at or nearest to target_seconds."""
        if not self.visual_store or not hasattr(self.visual_store, "chunks") or not self.visual_store.chunks:
            return []

        scored_chunks = []
        for ch in self.visual_store.chunks:
            if getattr(self, "video_id", None) is not None and ch.get("video_id") != self.video_id:
                continue
            c_start = float(ch.get("start_time", ch.get("timestamp_seconds", 0.0)))
            c_end = float(ch.get("end_time", ch.get("end_seconds", c_start)))
            if c_start <= target_seconds <= c_end:
                dist = 0.0
            elif target_seconds < c_start:
                dist = c_start - target_seconds
            else:
                dist = target_seconds - c_end
            scored_chunks.append((dist, ch))

        active_chunks = [item for item in scored_chunks if item[0] == 0.0]
        if active_chunks:
            scored_chunks = active_chunks

        scored_chunks.sort(key=lambda x: x[0])
        best_docs = []
        for dist, ch in scored_chunks[:k]:
            best_docs.append(FaissDocument(
                page_content=ch.get("text", ""),
                metadata={k: v for k, v in ch.items() if k != "text"},
            ))
        return best_docs

    def invoke(self, question: str) -> str:
        """Execute grounded cross-lingual multimodal query."""
        return ask_question(self, question)


# ── Multilingual visual/transcript keyword sets ───────────────────────────────
# Covers English, Hindi/Marathi/Nepali (Devanagari), Hinglish, Punjabi (Gurmukhi),
# Bengali, Tamil, Telugu, Kannada, Malayalam, Gujarati, Urdu/Arabic.
_VISUAL_KEYWORDS: List[str] = [
    # English
    "visible", "see", "shown", "displayed", "screen", "scene", "visual",
    "person", "people", "who is", "wearing", "holding", "object", "equipment",
    "device", "location", "where are", "doing", "action", "appears", "looks like",
    "color", "colour", "chart", "diagram", "slide", "camera", "background",
    "how many", "image", "picture", "frame", "display",
    # Hinglish (Latin-script Hindi)
    "dikh", "dikha", "dikhta", "dikhti", "dikhte", "dikh raha", "dikh rahi",
    "kaun", "pehne", "pakde", "kahan", "drishya", "kitne log", "log dikh",
    "screen par", "screen pe", "slide par", "slide pe", "chitra", "tasveer",
    "kya dikh", "kya dikhta", "kya dikhti",
    # Devanagari (Hindi/Marathi/Nepali visual)
    "दिख", "दिखता", "दिखती", "दिखाई", "दिखता है", "दिखती है",
    "दृश्य", "लोग", "व्यक्ति", "स्क्रीन", "चित्र", "छवि", "तस्वीर",
    "कितने लोग", "कौन", "पहने", "रंग", "चार्ट", "आकृति", "ग्राफ",
    # Marathi-specific
    "दिसत", "दिसते", "दाखवतो", "दाखवते", "चित्र",
    "किती लोक", "कोण", "स्क्रीनवर",
    # Punjabi (Gurmukhi)
    "ਦਿਖ", "ਦਿਖਾਈ", "ਦ੍ਰਿਸ਼", "ਲੋਕ", "ਕਿੰਨੇ ਲੋਕ", "ਸਕਰੀਨ",
    # Bengali
    "দেখা", "দৃশ্য", "মানুষ", "স্ক্রীন", "চিত্র",
    # Tamil
    "தெரிகிறது", "காட்சி", "நபர்கள்", "திரை",
    # Telugu
    "కనిపిస్తుంది", "దృశ్యం", "వ్యక్తులు", "స్క్రీన్",
    # Kannada
    "ಕಾಣಿಸುತ್ತಿದೆ", "ದೃಶ್ಯ", "ಜನ", "ಸ್ಕ್ರೀನ್",
    # Malayalam
    "കാണാം", "ദൃശ്യം", "ആളുകൾ", "സ്ക്രീൻ",
    # Gujarati
    "દેખાય", "દૃશ્ય", "લોકો", "સ્ક્રીન",
    # Urdu/Arabic
    "نظر آ", "نظارہ", "لوگ", "اسکرین",
]

_SPOKEN_KEYWORDS: List[str] = [
    # English
    "transcript", "audio", "speaker say", "said about", "said", "say", "discuss",
    "talk about", "talking about", "mentioned", "describe", "explained", "decided",
    "action items", "what did", "what was said", "what was discussed", "spoken",
    "explain", "told", "stated", "speech", "words", "topic", "content",
    "context", "summary", "summarize",
    # Hinglish
    "kya bola", "kya bol", "bata", "batao", "samjhaya", "samjhao",
    "kya hai", "kya tha", "discuss kiya", "baat ki", "mention kiya",
    # Devanagari
    "बताया", "समझाया", "बोला", "कहा", "क्या है", "क्या बताया",
    "विषय", "सामग्री", "सारांश", "चर्चा", "जानकारी",
    # Marathi
    "सांगितले", "सांगतो", "काय आहे", "काय सांगितले", "माहिती",
    "विषय", "सारांश",
    # Punjabi (Gurmukhi)
    "ਦੱਸਿਆ", "ਦੱਸਦਾ", "ਕੀ ਹੈ", "ਵਿਸ਼ਾ",
    # Bengali
    "বলেছে", "বলা হয়েছে", "বিষয়", "কী বলা",
    # Tamil
    "சொன்னார்", "விளக்கினார்", "தலைப்பு",
    # Telugu
    "చెప్పారు", "వివరించారు", "విషయం",
]

_TEMPORAL_MULTIMODAL_PATTERNS: List[str] = [
    "when they discussed", "when discussed", "when they talked", "when talking",
    "screen when", "shown when", "visible when", "slide when", "what was shown when",
    "jab discuss", "jab baat", "jab bola", "dikh raha tha jab", "dikha rahe the jab",
    "what was on screen when", "what was on the screen when",
    # Devanagari
    "जब दिख", "जब बात", "जब चर्चा",
    # Marathi
    "जेव्हा दिसत", "जेव्हा सांगत",
]

_PURE_VISUAL_PATTERNS: List[str] = [
    "what is visible", "what was visible", "visible on the screen", "shown on the screen",
    "what is on screen", "what's on screen", "what is on the screen", "what is shown",
    "what does the video show", "what does this video show", "what can be seen",
    "describe the scene", "describe what you see", "what is happening on screen",
    "what changed on", "what changed", "what slides", "which slides",
    "slide par", "slide pe", "charts", "diagrams", "graphs", "chart", "diagram", "graph",
    "people doing", "what are the people", "how many people", "kitne log",
    "screen par kya", "screen pe kya", "dikh raha hai", "dikh rahi hai",
    "kaunse charts", "kaunse diagrams", "chitra", "rekha",
    # Devanagari
    "स्क्रीन पर क्या", "क्या दिखता है", "कितने लोग", "दृश्य क्या है",
    # Marathi
    "स्क्रीनवर काय", "किती लोक", "काय दिसत",
]


def _has_non_latin_letters(text: str) -> bool:
    """Return True if the text contains any non-Latin letter characters."""
    for ch in text:
        if unicodedata.category(ch).startswith("L") and "LATIN" not in unicodedata.name(ch, ""):
            return True
    return False


def classify_question_intent(question: str, has_visual_store: bool = True) -> str:
    """Route questions with multilingual semantic similarity, without an LLM call."""
    if not question:
        return "TEXT_ONLY"

    timestamp_intent = _timestamp_intent(question, has_visual_store)
    if timestamp_intent:
        return timestamp_intent

    q_lower = question.lower()
    # A broad question about the video's story/content is transcript intent;
    # mentioning the medium alone does not ask for visual observations.
    if any(phrase in q_lower for phrase in (
        "what happens in the video", "what happened in the video",
        "what is happening in the video", "what happens in this video",
        "what happened in this video", "what is this video about",
    )):
        return "TEXT_ONLY"
    # Explicit, high-confidence cues remain a cheap fallback for common cases.
    if any(p in q_lower for p in _TEMPORAL_MULTIMODAL_PATTERNS):
        return "MULTIMODAL" if has_visual_store else "TEXT_ONLY"
    has_visual_signal = any(p in q_lower for p in _PURE_VISUAL_PATTERNS) or any(
        kw in q_lower for kw in _VISUAL_KEYWORDS
        if kw not in {"video", "film", "movie", "watch", "watching"}
    )
    has_spoken_signal = any(kw in q_lower for kw in _SPOKEN_KEYWORDS)
    if has_visual_signal and has_spoken_signal:
        return "MULTIMODAL" if has_visual_store else "TEXT_ONLY"
    if has_visual_signal and not has_spoken_signal:
        # Keep the visual intent even when frame evidence is missing so the
        # caller can return an honest visual-evidence insufficiency response.
        return "VISUAL_ONLY"
    if has_spoken_signal:
        return "TEXT_ONLY"
    if not has_visual_store:
        return "TEXT_ONLY"

    try:
        from core.vector_store import get_embedding_model
        labels = [
            "The user asks what was said, explained, mentioned, or discussed in spoken audio or the transcript.",
            "The user asks about people, objects, actions, text, slides, charts, or appearances visible in video frames.",
            "The user asks to connect or compare what was said with what was shown at the same time.",
        ]
        vectors = get_embedding_model().encode(
            [question, *labels], normalize_embeddings=True, show_progress_bar=False
        )
        scores = vectors[1:] @ vectors[0]
        best = int(scores.argmax())
        # Only choose multimodal when the relationship category itself wins;
        # near-ties between transcript and visual descriptions are ambiguous
        # and should preserve the cheaper transcript evidence path.
        if best == 2:
            return "MULTIMODAL"
        if best == 1 and float(scores[best]) < 0.40:
            return "TEXT_ONLY"
        return ("TEXT_ONLY", "VISUAL_ONLY", "MULTIMODAL")[best]
    except Exception as exc:
        logger.warning("Semantic intent routing unavailable; using text route: %s", exc)
        return "TEXT_ONLY"


def _rerank_evidence(question: str, documents: List[FaissDocument], limit: int) -> List[FaissDocument]:
    """Rerank semantic candidates by exact content overlap while retaining semantic order.

    Lexical overlap supplements semantic similarity when scripts and wording
    match. Without overlap, candidates must pass a minimum cosine score of
    0.32. Strong lexical overlap can retain lower-scoring same-script results.
    This threshold is based on the current local regression sample, not a
    universal multilingual calibration; some relevant languages can still
    fall below it.
    """
    if not documents:
        return []
    stop_words = {
        "a", "an", "and", "are", "about", "at", "did", "do", "does", "for", "from",
        "how", "in", "is", "it", "of", "on", "or", "the", "they", "this", "to", "was",
        "what", "when", "where", "which", "who", "with", "video", "show", "shown",
        "visible", "screen", "scene", "visual", "see", "looks", "appears",
    }
    query_terms = {term for term in re.findall(r"\w+", question.casefold()) if len(term) > 1 and term not in stop_words}
    # Detect whether the question is in a non-Latin script so we can apply
    # script-appropriate scoring (pure semantic for non-Latin, hybrid for Latin).
    question_is_non_latin = _has_non_latin_letters(question)
    if not query_terms:
        return [
            document for document in documents
            if float(document.metadata.get("score", 0.0) or 0.0) >= 0.32
        ][:limit]

    def score(document: FaissDocument) -> tuple:
        text_terms = set(re.findall(r"\w+", document.page_content.casefold()))
        # For non-Latin questions the lexical overlap with a (likely Latin)
        # transcript is structurally zero; use only the semantic score.
        overlap = 0.0 if question_is_non_latin else (len(query_terms & text_terms) / len(query_terms))
        semantic = float(document.metadata.get("score", 0.0) or 0.0)
        combined = 0.7 * semantic + 0.3 * overlap
        return (combined, semantic, overlap)

    scored = [(score(document), document) for document in documents]
    # A language mismatch removes lexical evidence, not the relevance floor.
    # Strong semantic evidence or substantial same-script lexical overlap is
    # required; a single generic shared token cannot rescue a weak candidate.
    minimum_semantic_score = 0.32
    minimum_lexical_overlap = 0.40
    scored = [item for item in scored if (
        item[0][1] >= minimum_semantic_score
        or (not question_is_non_latin and item[0][2] >= minimum_lexical_overlap)
    )]
    return [document for _, document in sorted(scored, key=lambda item: item[0], reverse=True)[:limit]]


def format_context(
    transcript_docs: List[FaissDocument],
    visual_docs: Optional[List[FaissDocument]] = None,
) -> str:
    """Format retrieved transcript chunks and visual evidence into clean sections for Qwen."""
    sections = []

    if transcript_docs:
        t_parts = ["--- [TRANSCRIPT EVIDENCE (SPOKEN AUDIO)] ---"]
        for i, doc in enumerate(transcript_docs[:3]):
            meta = doc.metadata
            chunk_idx = meta.get("chunk_index", i + 1)
            t_sec = meta.get("start_seconds", 0.0)
            t_parts.append(
                f"[Transcript Excerpt {i + 1} (~{int(t_sec//60):02d}:{int(t_sec%60):02d} | Offset ~{meta.get('start_time', 0)}%-{meta.get('end_time', 0)}%)]\n"
                f"{doc.page_content.strip()[:500]}"
            )
        sections.append("\n\n".join(t_parts))

    if visual_docs:
        v_parts = ["--- [VISUAL EVIDENCE (OBSERVED VIDEO FRAMES)] ---"]
        visual_limit = 20 if len(visual_docs) > 8 else 8
        content_limit = 500 if len(visual_docs) > 8 else 1800
        for i, doc in enumerate(visual_docs[:visual_limit]):
            meta = doc.metadata
            ts_display = meta.get("timestamp_display") or meta.get("timestamp", "00:00:00")
            video_id = meta.get("video_id")
            source_line = f"Video ID: {video_id}\n" if video_id else ""
            v_parts.append(
                f"[Visual Evidence {i + 1} at {ts_display}]\n"
                f"{source_line}{doc.page_content.strip()[:content_limit]}"
            )
        sections.append("\n\n".join(v_parts))

    if not sections:
        return "No relevant video or audio evidence found."

    return "\n\n" + ("\n\n" + "=" * 50 + "\n\n").join(sections)


def _associate_people_across_frames(visual_chunks: List[Dict[str, Any]]) -> str:
    """Associate per-frame person observations using only indexed visual descriptions.

    This is a local, evidence-constrained comparison. If the available frame
    descriptions cannot support reliable matching, return uncertainty instead
    of manufacturing an exact total.
    """
    observations = []
    for chunk in sorted(visual_chunks, key=lambda item: float(item.get("timestamp_seconds", 0) or 0)):
        people = str(chunk.get("people") or "").strip()
        if not people:
            match = re.search(r"(?:^|\n)People:\s*(.*?)(?=\n[A-Z][^\n]*:|$)", str(chunk.get("text") or ""), re.DOTALL)
            people = match.group(1).strip() if match else ""
        count = chunk.get("people_count")
        if not people and count in (None, 0, "0"):
            continue
        observations.append({
            "timestamp": chunk.get("timestamp_display") or chunk.get("timestamp"),
            "people_count": count,
            "people_description": people,
        })
    if not observations:
        return "Unique-person identity comparison: uncertain; indexed frames contain no usable person descriptions. Do not assert an exact video-wide total."

    try:
        result = json_chat_completion([
            {"role": "system", "content": (
                "Compare people observations across sampled video frames using ONLY the supplied descriptions. "
                "Assign temporary labels Person 1, Person 2, etc. Reuse a label only when distinguishing visual evidence "
                "strongly supports the same individual. Never merge two clearly different people in one frame. "
                "Use the best evidence-supported estimate from all sampled frames. Repeated generic descriptions may "
                "support a likely match when the frame context is consistent; do not require perfect descriptions. "
                "Mark uncertain only when the observations genuinely cannot support a useful estimate, and explain "
                "the limitation. Do not infer identity from a count alone. "
                "Return JSON only: {\"distinct_count\": integer or null, \"uncertain\": boolean, "
                "\"observations\":[{\"timestamp\":string,\"person_id\":string,\"evidence\":string}], "
                "\"reason\":string}. Each clearly visible person gets one observation per frame. These labels are not real identities."
            )},
            {"role": "user", "content": json.dumps(observations, ensure_ascii=False)},
        ], temperature=0.0, max_tokens=1500)
        if not isinstance(result, dict):
            raise ValueError("identity comparison returned no JSON object")
        records = result.get("observations")
        ids = {r.get("person_id") for r in records if isinstance(r, dict) and isinstance(r.get("person_id"), str)} if isinstance(records, list) else set()
        duplicate_frame_ids = set()
        seen = set()
        for r in records if isinstance(records, list) else []:
            if not isinstance(r, dict):
                continue
            key = (r.get("timestamp"), r.get("person_id"))
            if key in seen:
                duplicate_frame_ids.add(key)
            seen.add(key)
        count = result.get("distinct_count")
        uncertain = bool(result.get("uncertain")) or not isinstance(count, int) or count < 0
        # Missing or partial per-person records should not discard a reasonable
        # count estimate supported by per-frame counts. Contradictory duplicate
        # assignments and a count below a simultaneous frame count remain invalid.
        uncertain = uncertain or (isinstance(count, int) and count < len(ids)) or bool(duplicate_frame_ids)
        # The association cannot claim fewer distinct people than are reported
        # together in any one frame, even if the model omits observations.
        known_frame_counts = [
            int(item["people_count"]) for item in observations
            if str(item.get("people_count", "")).isdigit()
        ]
        if known_frame_counts and isinstance(count, int) and count < max(known_frame_counts):
            uncertain = True
        if uncertain:
            if known_frame_counts:
                floor = max(known_frame_counts)
                return (f"At least {floor} people are visible together in an indexed frame, but the sampled descriptions "
                        "do not support a reliable video-wide distinct-person estimate. " + str(result.get("reason") or ""))
            return ("Unique-person identity comparison: uncertain; the available sampled-frame descriptions do not "
                    "support a reliable exact distinct-person total. " + str(result.get("reason") or ""))
        summary = "; ".join(
            f"{r.get('person_id')} at {r.get('timestamp')}: {r.get('evidence')}"
            for r in records if isinstance(r, dict)
        )
        return (f"Unique-person estimate: about {count} distinct people are supported by matching the indexed frame "
                f"descriptions. This estimate is based on sampled frames, which may omit appearances between frames. {summary}")
    except Exception as exc:
        logger.warning("Local person identity comparison failed: %s", exc)
        return "Unique-person identity comparison: uncertain; identity matching could not be completed from the available visual evidence. Do not assert an exact total."


def build_rag_chain(
    transcript: str,
    visual_evidence: Optional[List[Dict[str, Any]]] = None,
    top_k: int = 6,
    duration_seconds= None,
    video_id: Optional[str] = None,
) -> LocalRAGChain:
    """
    Build a fresh session-scoped multimodal RAG chain over transcript and optional visual evidence.
    Rejects empty transcripts immediately.
    """
    if not transcript or not transcript.strip():
        raise ValueError("Cannot build RAG chain from an empty transcript.")

    vector_store = build_vector_store(transcript, duration_seconds=duration_seconds)
    visual_store = build_visual_vector_store(visual_evidence, video_id=video_id) if visual_evidence else None
    chain = LocalRAGChain(vector_store=vector_store, visual_store=visual_store, top_k=top_k)
    chain.video_id = video_id
    chain.duration_seconds = float(duration_seconds or 0.0)
    return chain


def _prepare_rag_messages(
    rag_chain: Union[LocalRAGChain, Any],
    question: str,
    output_language: Optional[str] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None,
):
    """Helper to extract retrieved multimodal context and build standard RAG messages.

    output_language: if set (and not 'auto'), overrides the default answer-in-question-language
    policy and forces the answer into the requested language/script.
    """
    t_start = time.time()
    clean_question = question.strip() if question else ""
    if not clean_question:
        return None, "TEXT_ONLY", [], []

    visual_index_size = _visual_index_size(rag_chain)
    has_visual_store = visual_index_size > 0

    # ── Stage 1: Intent classification ─────────────────────────────────────
    t0 = time.time()
    intent = classify_question_intent(clean_question, has_visual_store=has_visual_store)
    target_seconds = _question_timestamp(clean_question)
    asks_distinct_people = bool(re.search(
        r"\b(how many|number of|count|total)\b.{0,35}\b(people|persons?|individuals?)\b|"
        r"\b(people|persons?|individuals?)\b.{0,35}\b(how many|number|count|total)\b",
        clean_question, re.IGNORECASE,
    ))
    retrieval_question = _without_question_timestamp(clean_question) if target_seconds is not None else clean_question
    t_intent = time.time() - t0

    transcript_docs: List[FaissDocument] = []
    visual_docs: List[FaissDocument] = []
    timestamp_transcript_docs: List[FaissDocument] = []
    timestamp_visual_docs: List[FaissDocument] = []
    people_identity_summary = ""

    # ── Stage 2: Retrieval ─────────────────────────────────────────────────
    t_t_ret = 0.0
    t_v_ret = 0.0

    if intent == "VISUAL_ONLY":
        if has_visual_store:
            t0 = time.time()
            # Person counts require comparing the existing frame sample;
            # semantic top-k alone can omit people from other shots.
            if asks_distinct_people:
                k_vis = min(20, visual_index_size)
            else:
                k_vis = min(8, visual_index_size) if any(w in clean_question.lower() for w in ["what changed", "changed on the screen", "what slides", "all slides"]) else min(5, visual_index_size)
            if target_seconds is not None and hasattr(rag_chain, "retrieve_visual_by_timestamp"):
                visual_docs = rag_chain.retrieve_visual_by_timestamp(target_seconds, k=min(3, k_vis))
                timestamp_visual_docs = list(visual_docs)
            if not timestamp_visual_docs:
                visual_docs = rag_chain.retrieve_visual(retrieval_question or clean_question, k=k_vis)
            if asks_distinct_people:
                # Counting needs the whole indexed sample in timeline order;
                # semantic top-k can omit later reappearances or other people.
                chunks = list(getattr(rag_chain.visual_store, "chunks", []) or [])
                if chunks:
                    visual_docs = [FaissDocument(
                        page_content=item.get("text", ""),
                        metadata={key: value for key, value in item.items() if key != "text"},
                    ) for item in sorted(chunks, key=lambda item: float(item.get("timestamp_seconds", 0) or 0))]
                    people_identity_summary = _associate_people_across_frames(chunks)
            t_v_ret = time.time() - t0
    elif not has_visual_store or intent == "TEXT_ONLY":
        t0 = time.time()
        if hasattr(rag_chain, "retrieve"):
            transcript_docs = rag_chain.retrieve(retrieval_question or clean_question)
        elif hasattr(rag_chain, "vector_store"):
            transcript_docs = rag_chain.vector_store.similarity_search(retrieval_question or clean_question, k=12)
        if target_seconds is not None:
            temporal_docs = _retrieve_transcript_by_timestamp(rag_chain, target_seconds)
            timestamp_transcript_docs = list(temporal_docs)
            temporal_ids = {doc.metadata.get("chunk_id") for doc in temporal_docs}
            transcript_docs = temporal_docs + [
                doc for doc in transcript_docs if doc.metadata.get("chunk_id") not in temporal_ids
            ]
        t_t_ret = time.time() - t0
    else:  # MULTIMODAL: timestamp-aware retrieval
        t0 = time.time()
        # 1. Search transcript for the discussed topic
        if hasattr(rag_chain, "retrieve"):
            transcript_docs = rag_chain.retrieve(retrieval_question or clean_question)
        elif hasattr(rag_chain, "vector_store"):
            transcript_docs = rag_chain.vector_store.similarity_search(retrieval_question or clean_question, k=6)
        if target_seconds is not None:
            temporal_docs = _retrieve_transcript_by_timestamp(rag_chain, target_seconds)
            timestamp_transcript_docs = list(temporal_docs)
            temporal_ids = {doc.metadata.get("chunk_id") for doc in temporal_docs}
            transcript_docs = temporal_docs + [
                doc for doc in transcript_docs if doc.metadata.get("chunk_id") not in temporal_ids
            ]
        t_t_ret = time.time() - t0

        t0 = time.time()
        # 2. Semantic visual retrieval
        semantic_visual_docs = rag_chain.retrieve_visual(
            retrieval_question or clean_question, k=min(3, visual_index_size),
        ) if visual_index_size > 0 else []

        # 3. Timestamp-aware matching from transcript timestamps
        temporal_visual_docs = []
        if target_seconds is not None and hasattr(rag_chain, "retrieve_visual_by_timestamp"):
            temporal_visual_docs.extend(rag_chain.retrieve_visual_by_timestamp(target_seconds, k=2))
            timestamp_visual_docs = list(temporal_visual_docs)
        elif transcript_docs and hasattr(rag_chain, "retrieve_visual_by_timestamp"):
            for t_doc in transcript_docs[:2]:
                t_sec = t_doc.metadata.get("start_seconds", 0.0)
                if t_sec == 0.0:
                    ratio = (t_doc.metadata.get("start_time", 0.0)) / 100.0
                    max_dur = max([ch.get("end_seconds", 0.0) for ch in getattr(rag_chain.visual_store, "chunks", [])], default=0.0)
                    t_sec = ratio * max_dur if max_dur > 0 else 0.0

                matched = rag_chain.retrieve_visual_by_timestamp(t_sec, k=2)
                for m in matched:
                    if not any(v.metadata.get("chunk_id") == m.metadata.get("chunk_id") for v in temporal_visual_docs):
                        temporal_visual_docs.append(m)

        if timestamp_visual_docs:
            # Keep a timestamped question grounded in the frame active at that time.
            visual_docs = timestamp_visual_docs
        else:
            # Merge temporal and semantic visual documents without duplicates.
            combined_visual = []
            seen_ids = set()
            for doc in (temporal_visual_docs + semantic_visual_docs):
                cid = doc.metadata.get("chunk_id") or doc.metadata.get("timestamp")
                if cid not in seen_ids:
                    seen_ids.add(cid)
                    combined_visual.append(doc)
            visual_docs = combined_visual[:5]
        t_v_ret = time.time() - t0

    transcript_docs = _rerank_evidence(retrieval_question or clean_question, transcript_docs, limit=getattr(rag_chain, "top_k", 6))
    if not asks_distinct_people:
        visual_docs = _rerank_evidence(
            retrieval_question or clean_question,
            visual_docs,
            limit=5,
        )
    transcript_docs = timestamp_transcript_docs + [
        doc for doc in transcript_docs if doc.metadata.get("chunk_id") not in
        {item.metadata.get("chunk_id") for item in timestamp_transcript_docs}
    ]
    visual_docs = timestamp_visual_docs + [
        doc for doc in visual_docs if doc.metadata.get("chunk_id") not in
        {item.metadata.get("chunk_id") for item in timestamp_visual_docs}
    ]
    transcript_docs = transcript_docs[:max(getattr(rag_chain, "top_k", 6), len(timestamp_transcript_docs))]
    visual_docs = visual_docs[:max(
        12 if intent == "VISUAL_ONLY" and asks_distinct_people else 5,
        len(timestamp_visual_docs),
    )]
    context_text = format_context(transcript_docs, visual_docs)
    if people_identity_summary:
        context_text += "\n\n--- [CROSS-FRAME PERSON IDENTITY ANALYSIS] ---\n" + people_identity_summary

    # ── Detect question script for the answer-language instruction ───────────
    # Script metadata cannot identify a language for Latin or shared-script
    # text. Have the answer model mirror the complete question's language and
    # register rather than guessing from its script.
    question_script = detect_script(clean_question)
    script_note = " Preserve natural right-to-left reading order." if question_script == "Arabic" else ""
    q_lang_instruction = (
        "   - Infer the language and register from the user's complete question, "
        "including mixed-language wording, and answer in that same language/register. "
        "Do not infer a language from script alone. For clearly English questions, "
        "answer in English. For Romanized Hindi/Hinglish, mirror the same natural "
        "Latin-script style; for Devanagari, infer Hindi, Marathi, Nepali, or another "
        "language from the full wording rather than the script alone."
        + script_note
    )

    system_prompt = (
        "You are an expert, privacy-first AI Video Assistant with Unified Multimodal Intelligence.\n"
        "Answer only from the retrieved evidence belonging to the current video. Use transcript evidence only for what was spoken and visual evidence only for what is visibly shown; never treat spoken claims as visual observations.\n"
        "For factual questions, state only details directly supported by the evidence. For questions about visible events, answer from relevant visual evidence even when the transcript is silent or irrelevant; do not substitute a transcript-insufficient response for a visual answer. For questions about speech, use transcript evidence and say when it is unavailable. For questions involving both sources, use whichever relevant sources are available and distinguish spoken content from visible observations. If neither source contains relevant evidence, say so explicitly. Do not fill gaps with assumptions or outside knowledge. Conversation history may resolve references only; it is not evidence.\n\n"
        "CRITICAL MULTIMODAL & GROUNDING RULES:\n"
        "1. ANSWER LANGUAGE DETERMINED BY QUESTION (MANDATORY):\n"
        "   The language, dialect, and script of your ENTIRE answer MUST match the user's QUESTION:\n"
        + q_lang_instruction + "\n"
        "   - If the question is in Punjabi (Gurmukhi, e.g. 'ਵੀਡੀਓ ਵਿੱਚ ਕੀ ਦਿਖ ਰਿਹਾ ਹੈ?'), answer strictly in Punjabi (Gurmukhi script).\n"
        "   - If the question is in any other Indian or international language (Bengali, Tamil, Telugu, Kannada, Malayalam, Gujarati, Odia, etc.), answer strictly in that language and script.\n"
        "   - The transcript may be in a different language than the question; that is expected. Do NOT translate the question or force the answer into the transcript's language unless the question asks for a translation.\n"
        "   - CRITICAL: Do NOT repeat sentences. Write each idea once only. Keep the answer concise.\n\n"
    )

    if intent == "TEXT_ONLY":
        system_prompt += (
            "QUESTION RELEVANCE: Answer the user's requested question from transcript evidence. For a broad question about what the video is about or what happens, give a concise summary of the spoken/story content and its key ideas; do not describe frames, setting, people, or screen graphics. For a learning or lesson question, state only lessons supported by what the transcript actually says; do not infer unsupported discussion topics. If the video does not directly cover the requested application, say so and mention only a related idea explicitly present in the transcript. Do not provide general-world suggestions or advice that the video did not give, even if they seem helpful. Keep the answer focused on the requested subject.\n\n"
        )

    if target_seconds is not None and timestamp_transcript_docs:
        system_prompt += (
            f"TIMESTAMP-SPECIFIC SPEECH EVIDENCE: The user asked about { _format_chapter_time(target_seconds) }. "
            "Prioritize transcript excerpts whose stored time intervals contain that moment. Use adjacent excerpts only for sentence or topic context, "
            "and do not infer speech that is absent from those excerpts.\n\n"
        )
    if target_seconds is not None and intent in ("VISUAL_ONLY", "MULTIMODAL"):
        system_prompt += (
            f"TIMESTAMP-SPECIFIC VISUAL EVIDENCE: The user asked about { _format_chapter_time(target_seconds) }. "
            "Use the nearest retrieved visual frame(s) to describe what is visibly happening. Combine timestamp-aligned "
            "transcript evidence when present, but do not require transcript evidence to answer a visual event question. "
            "If only transcript evidence is available, label it as spoken content and do not present it as a visual observation.\n\n"
        )

    # Explicit output-language override (from the UI selector) takes precedence over the
    # "answer in question language" default policy.
    output_language_instruction = ""
    if output_language and output_language.strip() and output_language.strip().lower() not in ("auto", "match"):
        ol = get_language(output_language.strip())
        if ol is not None:
            script = ol.scripts[0] if ol.scripts else "the original script"
            rtl_note = ""
            if ol.rtl:
                rtl_note = (
                    f" {ol.name} is written right-to-left (RTL); preserve the natural RTL reading order."
                )
            system_prompt += (
                f"\nUSER-REQUESTED OUTPUT LANGUAGE OVERRIDE:\n"
                f"   The user explicitly requested the answer in {ol.native_name} ({ol.name}).\n"
                f"   Write the ENTIRE answer in {ol.native_name} using the {script} script, regardless of the question's language.{rtl_note}\n"
            )
            output_language_instruction = (
                f"\n\nAnswer entirely in {ol.name}, using the {script} script, as explicitly requested."
            )

    system_prompt += (
        "2. CHARTS, DIAGRAMS & SLIDES GROUNDING (STRICT):\n"
        "   - When asked about charts, diagrams, graphs, tables, or slides:\n"
        "     Look specifically at the 'Charts:', 'Diagrams:', 'Screen/UI:', and 'Slide/Title:' fields of the retrieved visual evidence.\n"
        "     If specific charts, diagrams, or graphs are explicitly described, state and describe them clearly.\n"
        "     If the evidence indicates 'None' or no charts/diagrams are visible in the retrieved frames, state clearly and factually that no charts or diagrams were shown on the screen/slides.\n"
        "     Never hallucinate or invent charts, diagrams, or data that are not explicitly present.\n\n"
        "3. SCREEN CHANGES & TIMELINE CHRONOLOGY:\n"
        "   - When asked what changed on screen or what slides were shown, describe the sequence chronologically using the timestamps in the visual evidence.\n\n"
        "4. TOPIC-TO-SCREEN MULTIMODAL QUESTIONS:\n"
        "   - When asked 'What was shown on screen when they discussed X?', correlate the transcript discussion of X with the visual frame active at that timestamp.\n\n"
        "5. VISUAL ACCURACY & ABSENT INFORMATION POLICY:\n"
        "   - If visual evidence is insufficient to answer the question, state: 'The video does not provide enough visual evidence to determine that.'\n"
        "   - If transcript evidence does not contain the answer, state: 'I couldn't find that information in the video transcript.'\n"
        "   - Do NOT guess, assume, or fabricate.\n\n"
        "   - For questions asking how many people appear across the video, treat every visual entry as an observation of one sampled frame and its people count as a per-frame count only. Compare descriptions across timestamps and count a person once when the evidence supports the same identity; do not add per-frame counts, count each frame as a new group, or treat a shot change alone as a new person. Do not merge generic descriptions unless there is evidence they refer to the same individual. If identity cannot be established from the available descriptions, state that the exact distinct-person total is uncertain and distinguish it from any supported per-frame count. Never invent an exact total.\n\n"
        "   - Answer the visual property the user asked about directly from relevant evidence instead of substituting a generic description of one frame.\n\n"
        "6. CONCISE ANSWERS:\n"
        "   - Keep answers direct, concise, and informative (1-4 sentences).\n"
        "   - Do not repeat the prompt or apologize.\n\n"
        "Retrieved Video Evidence:\n"
        f"{context_text}"
    )

    messages = [{"role": "system", "content": system_prompt}]
    for item in (conversation_history or [])[-4:]:
        role = item.get("role")
        content = item.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            messages.append({"role": role, "content": content[-600:]})
    messages.append({"role": "user", "content": clean_question})
    if output_language_instruction:
        messages[-1]["content"] += output_language_instruction
    elif (question_script == "Latin" and re.search(
            r"\b(mein|ke|ki|ko|kya|kaise|hai|hain|ne|se|liye|batao|bataya|samjhao|samjhaya)\b",
            clean_question.casefold(),
    )):
        messages[-1]["content"] += (
            "\n\nAnswer in natural Romanized Hindi (Hinglish) using Latin letters. "
            "Do not switch the answer to Devanagari."
        )

    t_total = time.time() - t_start
    logger.info(
        f"[CHAT TIMING] Intent={intent} | classify={t_intent*1000:.0f}ms | "
        f"transcript_retrieve(k={len(transcript_docs)})={t_t_ret*1000:.0f}ms | "
        f"visual_retrieve(k={len(visual_docs)})={t_v_ret*1000:.0f}ms | "
        f"msg_build_total={t_total*1000:.0f}ms"
    )

    return messages, intent, transcript_docs, visual_docs, context_text


def ask_question(
    rag_chain: Union[LocalRAGChain, Any],
    question: str,
    output_language: Optional[str] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """
    Query the local multimodal RAG chain with strict grounding.
    Answers strictly from retrieved evidence in the user's question language (or output_language if set).
    Performs ONLY: classify → retrieve existing FAISS → single LLM call.
    NEVER reruns Whisper/vision/FAISS-construction during chat.
    """
    t_chat_start = time.time()
    if rag_chain is None:
        return "No video or audio has been processed yet. Please analyze media first."

    clean_question = question.strip() if question else ""
    if not clean_question:
        return "Please enter a question to ask."

    target_seconds = _question_timestamp(clean_question)
    duration_seconds = float(getattr(rag_chain, "duration_seconds", 0.0) or 0.0)
    if target_seconds is not None and duration_seconds > 0 and target_seconds > duration_seconds:
        return (
            f"The requested timestamp ({_format_chapter_time(target_seconds)}) is outside "
            f"the video's duration ({_format_chapter_time(duration_seconds)}), so there is no evidence to retrieve."
        )

    if _is_chapter_generation_request(clean_question):
        return _chapter_request_response(rag_chain, clean_question)

    cache = getattr(rag_chain, "_answer_cache", None)
    if cache is None:
        cache = {}
        try:
            rag_chain._answer_cache = cache
        except Exception:
            pass
    cache_history = list(conversation_history or [])
    # A second immediate submission has the first user/assistant exchange at
    # the end of its history. Remove only that exact trailing pair; intervening
    # conversation changes remain part of the key and invalidate the cache.
    if len(cache_history) >= 2:
        prior_user, prior_assistant = cache_history[-2:]
        if (
            prior_user.get("role") == "user"
            and isinstance(prior_user.get("content"), str)
            and prior_user["content"].strip().casefold() == clean_question.casefold()
            and prior_assistant.get("role") == "assistant"
        ):
            cache_history = cache_history[:-2]
    history_key = json.dumps(cache_history[-6:], ensure_ascii=False, sort_keys=True)
    cache_material = json.dumps([
        getattr(rag_chain, "video_id", None), clean_question.casefold(),
        (output_language or "auto").casefold(), history_key,
    ], ensure_ascii=False)
    cache_key = hashlib.sha256(cache_material.encode("utf-8")).hexdigest()
    if cache_key in cache:
        logger.info("Returning cached grounded answer for repeated question on the same video")
        return cache[cache_key]

    t0 = time.time()
    messages, intent, transcript_docs, visual_docs, context_text = _prepare_rag_messages(
        rag_chain, clean_question, output_language=output_language,
        conversation_history=conversation_history,
    )
    t_prep = time.time() - t0
    if not messages:
        return "Please enter a question to ask."
    if intent == "VISUAL_ONLY" and not visual_docs:
        return "The visual evidence does not provide enough information to answer this question."
    if intent == "TEXT_ONLY" and not transcript_docs:
        return "The transcript does not provide enough information to answer this question."
    if intent == "MULTIMODAL" and not transcript_docs and not visual_docs:
        return "The transcript and visual evidence do not provide enough information to answer this question."
        
    # === TEMPORARY DEBUG OUTPUT ===
    print("\n================ TEMPORARY DEBUG OUTPUT ================")
    print(f"QUERY: {clean_question}")
    print(f"ROUTE: {intent}")
    print(f"RETRIEVED TRANSCRIPT EVIDENCE: {[d.page_content[:150] + '...' for d in transcript_docs]}")
    print(f"RETRIEVED VISUAL EVIDENCE: {[d.page_content[:150] + '...' for d in visual_docs]}")
    t_stamps = []
    for d in (visual_docs + transcript_docs):
        ts = d.metadata.get("timestamp") or d.metadata.get("start_time")
        if ts: t_stamps.append(ts)
    print(f"TIMESTAMPS: {t_stamps}")
    print(f"FINAL CONTEXT SENT TO QWEN3:8B:\n{context_text}")

    try:
        t0 = time.time()
        answer = chat_completion(messages, temperature=0.0, max_tokens=512, timeout=90)
        insufficient_visual_markers = (
            "not provide enough visual evidence",
            "visual evidence does not provide enough",
            "insufficient visual evidence",
        )
        if visual_docs and any(marker in answer.casefold() for marker in insufficient_visual_markers):
            visual_context = format_context([], visual_docs)
            active_intervals = []
            for doc in visual_docs:
                meta = doc.metadata
                start = float(meta.get("start_time", meta.get("timestamp_seconds", 0.0)) or 0.0)
                end = float(meta.get("end_time", meta.get("end_seconds", start)) or start)
                interval = (
                    f"{_format_chapter_time(start)} to {_format_chapter_time(end)}"
                    if end > start else _format_chapter_time(start)
                )
                if interval not in active_intervals:
                    active_intervals.append(interval)
            timestamp_note = ""
            if target_seconds is not None:
                active_note = "; ".join(active_intervals)
                timestamp_note = (
                    f" The requested moment is {_format_chapter_time(target_seconds)}."
                    + (f" The retrieved frame is active from {active_note}; its sample time may precede the requested moment." if active_note else "")
                )
            retry_messages = [
                {
                    "role": "system",
                    "content": (
                        "Answer using only the observed video frames below. When the user broadly asks what is visible or happening, "
                        "summarize the relevant scene using the stated scene, people, actions, objects, and important visual details; "
                        "do not treat a broad question as requesting one specific attribute. If a specific requested detail is not stated, "
                        "say that detail is unknown while still reporting relevant details that are stated. If none of the observations "
                        "supports the question, say the visual evidence is insufficient. Do not add outside facts."
                        f"{timestamp_note}\n\nRetrieved Visual Evidence:\n{visual_context}"
                    ),
                },
                {"role": "user", "content": clean_question},
            ]
            try:
                focused_answer = chat_completion(
                    retry_messages, temperature=0.0, max_tokens=256, timeout=90,
                ).strip()
                if focused_answer:
                    answer = focused_answer
            except Exception as retry_error:
                logger.warning("Focused evidence-answer retry failed: %s", retry_error)
        t_llm = time.time() - t0
        total = time.time() - t_chat_start
        logger.info(
            f"[CHAT TOTAL] Q: {clean_question[:60]}... | "
            f"retrieval_prep={t_prep:.2f}s | llm(qwen3:8b)={t_llm:.2f}s | "
            f"answer_chars={len(answer.strip())} | total_latency={total:.2f}s"
        )
        print(f"FINAL ANSWER:\n{answer.strip()}")
        print("========================================================\n")
        answer = _remove_repeated_sentences(answer.strip())
        if answer:
            cache[cache_key] = answer
            if len(cache) > 32:
                cache.pop(next(iter(cache)))
        return answer
    except Exception as e:
        logger.error(f"Local RAG invocation error: {e}", exc_info=True)
        if isinstance(e, TimeoutError) or "timed out" in str(e).casefold():
            return "The local model timed out before producing an answer. Your video evidence is unchanged; please retry."
        if not check_ollama():
            return "Local Ollama server is not running. Start Ollama and try again."
        return f"Could not generate answer locally: {e}"


def ask_question_stream(rag_chain: Union[LocalRAGChain, Any], question: str, output_language: Optional[str] = None) -> Iterator[str]:
    """
    Query the local multimodal RAG chain with streaming real-time output.
    """
    if rag_chain is None:
        yield "No video or audio has been processed yet. Please analyze media first."
        return

    clean_question = question.strip() if question else ""
    if not clean_question:
        yield "Please enter a question to ask."
        return

    if _is_chapter_generation_request(clean_question):
        yield _chapter_request_response(rag_chain, clean_question)
        return

    messages, intent, transcript_docs, visual_docs, context_text = _prepare_rag_messages(
        rag_chain, clean_question, output_language=output_language,
    )
    if not messages:
        yield "Please enter a question to ask."
        return
    if intent == "VISUAL_ONLY" and not visual_docs:
        yield "The visual evidence does not provide enough information to answer this question."
        return
    if intent == "TEXT_ONLY" and not transcript_docs:
        yield "The transcript does not provide enough information to answer this question."
        return
    if intent == "MULTIMODAL" and not transcript_docs and not visual_docs:
        yield "The transcript and visual evidence do not provide enough information to answer this question."
        return

    try:
        for token in chat_completion_stream(messages, temperature=0.0, max_tokens=512, timeout=120):
            yield token
    except Exception as e:
        logger.error(f"Local RAG streaming invocation error: {e}")
        if not check_ollama():
            yield "Local Ollama server is not running. Start Ollama and try again."
            return
        yield f"Could not generate answer locally: {e}"
