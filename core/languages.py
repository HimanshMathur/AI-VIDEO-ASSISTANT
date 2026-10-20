"""
Central language registry for the AI Video Assistant multilingual pipeline.

Defines every language the project can handle with stable metadata:
  - ISO 639 codes (1 / 2 / 3)
  - faster-whisper language token code
  - Primary script(s) and Unicode block ranges
  - RTL / LTR direction
  - Support level for each local model in the stack:
      * faster-whisper large-v3 (transcription + language detection)
      * qwen3:8b (summarization, extraction, chat, answers)
      * paraphrase-multilingual-MiniLM-L12-v2 (embeddings / retrieval)
      * qwen3-vl:2b (visual descriptions)

The registry is the single source of truth used by:
  - transcriber._normalize_language
  - summarizer / extractor prompt construction
  - rag_engine answer-language policy
  - visual_analyzer description language
  - app.py language selector and RTL rendering

Nothing here invokes a model; it is pure metadata so it can be unit-tested
offline and imported without loading any heavyweight dependency.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ── Support levels ────────────────────────────────────────────────────────────
FULL = "full"            # Model is known to handle the language well
PARTIAL = "partial"      # Model can produce output but quality is limited
WEAK = "weak"            # Model struggles; results may be unreliable
UNSUPPORTED = "unsupported"  # Model does not cover the language


@dataclass
class Language:
    """Complete metadata for one supported language."""

    name: str                          # English display name, e.g. "Hindi"
    native_name: str                   # Native-script display name, e.g. "हिन्दी"
    iso639_1: str                      # Two-letter code, e.g. "hi"
    iso639_3: str                      # Three-letter code, e.g. "hin"
    whisper_code: str                  # Code faster-whisper expects, e.g. "hi"
    scripts: List[str]                 # Primary scripts, e.g. ["Devanagari"]
    unicode_ranges: List[Tuple[int, int]] = field(default_factory=list)
    rtl: bool = False
    # Per-model support levels
    whisper_support: str = FULL
    llm_support: str = FULL
    embedding_support: str = FULL
    vision_support: str = FULL
    notes: str = ""

    @property
    def is_indian(self) -> bool:
        return self.iso639_3 in _INDIAN_ISO3

    @property
    def script_direction(self) -> str:
        return "rtl" if self.rtl else "ltr"


# ── Indian Eighth-Schedule ISO 639-3 codes ────────────────────────────────────
_INDIAN_ISO3 = {
    "asm", "ben", "brx", "doi", "guj", "hin", "kan", "kas", "kok", "mai",
    "mal", "mni", "mar", "nep", "ory", "pan", "san", "sat", "snd", "tam",
    "tel", "urd",
}


# ── The registry ──────────────────────────────────────────────────────────────
# Capability notes:
#   * faster-whisper large-v3 is trained on 99 languages; every Indian
#     Eighth-Schedule language has a whisper token and is recognised for
#     detection and transcription. Quality varies (Tamil/Telugu/Kannada/
#     Malayalam/Hindi/Bengali/Marathi/Gujarati/Urdu/Punjabi/Nepali are strong;
#     Bodo/Dogri/Kashmiri/Konkani/Maithili/Meitei/Sanskrit/Santali/Sindhi are
#     weaker because of less training data).
#   * qwen3:8b is a broadly multilingual model. It handles Indic scripts
#     natively for many languages; low-resource ones are partial/weak.
#   * paraphrase-multilingual-MiniLM-L12-v2 covers 50+ languages including
#     the major Indic ones; low-resource scripts may produce weaker vectors.
#   * qwen3-vl:2b describes frames; language of description follows the
#     text model and is limited to what qwen3-vl can generate.

_LANGUAGES: List[Language] = [
    # ── Indian Eighth Schedule (22) ──────────────────────────────────────
    Language(
        name="Assamese", native_name="অসমীয়া",
        iso639_1="as", iso639_3="asm", whisper_code="as",
        scripts=["Bengali"],
        unicode_ranges=[(0x0980, 0x09FF)],
        whisper_support=FULL, llm_support=PARTIAL, embedding_support=PARTIAL,
        vision_support=PARTIAL,
        notes="Uses Bengali script. Whisper detection/transcription available.",
    ),
    Language(
        name="Bengali", native_name="বাংলা",
        iso639_1="bn", iso639_3="ben", whisper_code="bn",
        scripts=["Bengali"],
        unicode_ranges=[(0x0980, 0x09FF)],
        whisper_support=FULL, llm_support=FULL, embedding_support=FULL,
        vision_support=FULL,
    ),
    Language(
        name="Bodo", native_name="बड़ो",
        iso639_1="brx", iso639_3="brx", whisper_code="brx",
        scripts=["Devanagari"],
        unicode_ranges=[(0x0900, 0x097F)],
        whisper_support=PARTIAL, llm_support=WEAK, embedding_support=WEAK,
        vision_support=WEAK,
        notes="Low-resource Tibeto-Burman language. Whisper token exists but accuracy is limited.",
    ),
    Language(
        name="Dogri", native_name="डोगरी",
        iso639_1="doi", iso639_3="doi", whisper_code="doi",
        scripts=["Devanagari"],
        unicode_ranges=[(0x0900, 0x097F)],
        whisper_support=PARTIAL, llm_support=WEAK, embedding_support=WEAK,
        vision_support=WEAK,
        notes="Low-resource Indo-Aryan. Whisper token present; limited training data.",
    ),
    Language(
        name="Gujarati", native_name="ગુજરાતી",
        iso639_1="gu", iso639_3="guj", whisper_code="gu",
        scripts=["Gujarati"],
        unicode_ranges=[(0x0A80, 0x0AFF)],
        whisper_support=FULL, llm_support=FULL, embedding_support=FULL,
        vision_support=FULL,
    ),
    Language(
        name="Hindi", native_name="हिन्दी",
        iso639_1="hi", iso639_3="hin", whisper_code="hi",
        scripts=["Devanagari"],
        unicode_ranges=[(0x0900, 0x097F)],
        whisper_support=FULL, llm_support=FULL, embedding_support=FULL,
        vision_support=FULL,
    ),
    Language(
        name="Kannada", native_name="ಕನ್ನಡ",
        iso639_1="kn", iso639_3="kan", whisper_code="kn",
        scripts=["Kannada"],
        unicode_ranges=[(0x0C80, 0x0CFF)],
        whisper_support=FULL, llm_support=FULL, embedding_support=FULL,
        vision_support=PARTIAL,
    ),
    Language(
        name="Kashmiri", native_name="कॉशुर / کٲشُر",
        iso639_1="ks", iso639_3="kas", whisper_code="ks",
        scripts=["Devanagari", "Arabic"],
        unicode_ranges=[(0x0900, 0x097F), (0x0600, 0x06FF)],
        rtl=True,
        whisper_support=PARTIAL, llm_support=WEAK, embedding_support=WEAK,
        vision_support=WEAK,
        notes="May be written in Devanagari (LTR) or Arabic/Perso-Arabic (RTL).",
    ),
    Language(
        name="Konkani", native_name="कोंकणी",
        iso639_1="gom", iso639_3="kok", whisper_code="kok",
        scripts=["Devanagari"],
        unicode_ranges=[(0x0900, 0x097F)],
        whisper_support=PARTIAL, llm_support=WEAK, embedding_support=WEAK,
        vision_support=WEAK,
        notes="Low-resource Indo-Aryan. Whisper token present.",
    ),
    Language(
        name="Maithili", native_name="मैथिली",
        iso639_1="mai", iso639_3="mai", whisper_code="mai",
        scripts=["Devanagari"],
        unicode_ranges=[(0x0900, 0x097F)],
        whisper_support=PARTIAL, llm_support=PARTIAL, embedding_support=WEAK,
        vision_support=WEAK,
        notes="Whisper token present; moderate training data.",
    ),
    Language(
        name="Malayalam", native_name="മലയാളം",
        iso639_1="ml", iso639_3="mal", whisper_code="ml",
        scripts=["Malayalam"],
        unicode_ranges=[(0x0D00, 0x0D7F)],
        whisper_support=FULL, llm_support=FULL, embedding_support=FULL,
        vision_support=PARTIAL,
    ),
    Language(
        name="Meitei (Manipuri)", native_name="ꯃꯩꯇꯩꯂꯣꯟ",
        iso639_1="mni", iso639_3="mni", whisper_code="mni",
        scripts=["Meetei Mayek", "Bengali"],
        unicode_ranges=[(0xABC0, 0xABFF), (0x0980, 0x09FF)],
        whisper_support=PARTIAL, llm_support=WEAK, embedding_support=WEAK,
        vision_support=WEAK,
        notes="Meetei Mayek script is rare in training data; Bengali-script Meitei more common.",
    ),
    Language(
        name="Marathi", native_name="मराठी",
        iso639_1="mr", iso639_3="mar", whisper_code="mr",
        scripts=["Devanagari"],
        unicode_ranges=[(0x0900, 0x097F)],
        whisper_support=FULL, llm_support=FULL, embedding_support=FULL,
        vision_support=FULL,
    ),
    Language(
        name="Nepali", native_name="नेपाली",
        iso639_1="ne", iso639_3="nep", whisper_code="ne",
        scripts=["Devanagari"],
        unicode_ranges=[(0x0900, 0x097F)],
        whisper_support=FULL, llm_support=FULL, embedding_support=FULL,
        vision_support=PARTIAL,
    ),
    Language(
        name="Odia", native_name="ଓଡ଼ିଆ",
        iso639_1="or", iso639_3="ory", whisper_code="or",
        scripts=["Odia"],
        unicode_ranges=[(0x0B00, 0x0B7F)],
        whisper_support=FULL, llm_support=PARTIAL, embedding_support=PARTIAL,
        vision_support=PARTIAL,
    ),
    Language(
        name="Punjabi", native_name="ਪੰਜਾਬੀ",
        iso639_1="pa", iso639_3="pan", whisper_code="pa",
        scripts=["Gurmukhi"],
        unicode_ranges=[(0x0A00, 0x0A7F)],
        whisper_support=FULL, llm_support=FULL, embedding_support=FULL,
        vision_support=PARTIAL,
    ),
    Language(
        name="Sanskrit", native_name="संस्कृतम्",
        iso639_1="sa", iso639_3="san", whisper_code="sa",
        scripts=["Devanagari"],
        unicode_ranges=[(0x0900, 0x097F)],
        whisper_support=PARTIAL, llm_support=PARTIAL, embedding_support=PARTIAL,
        vision_support=PARTIAL,
        notes="Classical language. Whisper token exists; limited conversational training data.",
    ),
    Language(
        name="Santali", native_name="ᱥᱟᱱᱛᱟᱲᱤ",
        iso639_1="sat", iso639_3="sat", whisper_code="sat",
        scripts=["Ol Chiki"],
        unicode_ranges=[(0x1C50, 0x1C7F)],
        whisper_support=PARTIAL, llm_support=WEAK, embedding_support=WEAK,
        vision_support=WEAK,
        notes="Austroasiatic language in Ol Chiki script. Very limited training data.",
    ),
    Language(
        name="Sindhi", native_name="سنڌي",
        iso639_1="sd", iso639_3="snd", whisper_code="sd",
        scripts=["Arabic", "Devanagari"],
        unicode_ranges=[(0x0600, 0x06FF), (0x0900, 0x097F)],
        rtl=True,
        whisper_support=PARTIAL, llm_support=PARTIAL, embedding_support=PARTIAL,
        vision_support=PARTIAL,
        notes="Perso-Arabic script (RTL) is standard in Pakistan; Devanagari used in India.",
    ),
    Language(
        name="Tamil", native_name="தமிழ்",
        iso639_1="ta", iso639_3="tam", whisper_code="ta",
        scripts=["Tamil"],
        unicode_ranges=[(0x0B80, 0x0BFF)],
        whisper_support=FULL, llm_support=FULL, embedding_support=FULL,
        vision_support=PARTIAL,
    ),
    Language(
        name="Telugu", native_name="తెలుగు",
        iso639_1="te", iso639_3="tel", whisper_code="te",
        scripts=["Telugu"],
        unicode_ranges=[(0x0C00, 0x0C7F)],
        whisper_support=FULL, llm_support=FULL, embedding_support=FULL,
        vision_support=PARTIAL,
    ),
    Language(
        name="Urdu", native_name="اردو",
        iso639_1="ur", iso639_3="urd", whisper_code="ur",
        scripts=["Arabic"],
        unicode_ranges=[(0x0600, 0x06FF)],
        rtl=True,
        whisper_support=FULL, llm_support=FULL, embedding_support=FULL,
        vision_support=PARTIAL,
    ),

    # ── International languages already covered by the local models ──────
    Language(
        name="English", native_name="English",
        iso639_1="en", iso639_3="eng", whisper_code="en",
        scripts=["Latin"],
        unicode_ranges=[(0x0000, 0x007F)],
        whisper_support=FULL, llm_support=FULL, embedding_support=FULL,
        vision_support=FULL,
    ),
    Language(
        name="Spanish", native_name="Español",
        iso639_1="es", iso639_3="spa", whisper_code="es",
        scripts=["Latin"],
        unicode_ranges=[(0x0000, 0x024F)],
        whisper_support=FULL, llm_support=FULL, embedding_support=FULL,
        vision_support=FULL,
    ),
    Language(
        name="French", native_name="Français",
        iso639_1="fr", iso639_3="fra", whisper_code="fr",
        scripts=["Latin"],
        unicode_ranges=[(0x0000, 0x024F)],
        whisper_support=FULL, llm_support=FULL, embedding_support=FULL,
        vision_support=FULL,
    ),
    Language(
        name="German", native_name="Deutsch",
        iso639_1="de", iso639_3="deu", whisper_code="de",
        scripts=["Latin"],
        unicode_ranges=[(0x0000, 0x024F)],
        whisper_support=FULL, llm_support=FULL, embedding_support=FULL,
        vision_support=FULL,
    ),
    Language(
        name="Russian", native_name="Русский",
        iso639_1="ru", iso639_3="rus", whisper_code="ru",
        scripts=["Cyrillic"],
        unicode_ranges=[(0x0400, 0x04FF)],
        whisper_support=FULL, llm_support=FULL, embedding_support=FULL,
        vision_support=FULL,
    ),
    Language(
        name="Arabic", native_name="العربية",
        iso639_1="ar", iso639_3="ara", whisper_code="ar",
        scripts=["Arabic"],
        unicode_ranges=[(0x0600, 0x06FF)],
        rtl=True,
        whisper_support=FULL, llm_support=FULL, embedding_support=FULL,
        vision_support=PARTIAL,
    ),
    Language(
        name="Chinese (Simplified)", native_name="简体中文",
        iso639_1="zh", iso639_3="zho", whisper_code="zh",
        scripts=["Han"],
        unicode_ranges=[(0x4E00, 0x9FFF)],
        whisper_support=FULL, llm_support=FULL, embedding_support=FULL,
        vision_support=FULL,
    ),
    Language(
        name="Japanese", native_name="日本語",
        iso639_1="ja", iso639_3="jpn", whisper_code="ja",
        scripts=["Hiragana", "Katakana", "Han"],
        unicode_ranges=[(0x3040, 0x309F), (0x30A0, 0x30FF), (0x4E00, 0x9FFF)],
        whisper_support=FULL, llm_support=FULL, embedding_support=FULL,
        vision_support=FULL,
    ),
    Language(
        name="Korean", native_name="한국어",
        iso639_1="ko", iso639_3="kor", whisper_code="ko",
        scripts=["Hangul"],
        unicode_ranges=[(0xAC00, 0xD7AF)],
        whisper_support=FULL, llm_support=FULL, embedding_support=FULL,
        vision_support=FULL,
    ),
    Language(
        name="Portuguese", native_name="Português",
        iso639_1="pt", iso639_3="por", whisper_code="pt",
        scripts=["Latin"],
        unicode_ranges=[(0x0000, 0x024F)],
        whisper_support=FULL, llm_support=FULL, embedding_support=FULL,
        vision_support=FULL,
    ),
    Language(
        name="Italian", native_name="Italiano",
        iso639_1="it", iso639_3="ita", whisper_code="it",
        scripts=["Latin"],
        unicode_ranges=[(0x0000, 0x024F)],
        whisper_support=FULL, llm_support=FULL, embedding_support=FULL,
        vision_support=FULL,
    ),
]


# ── Lookup indexes (built once at import time) ────────────────────────────────
_BY_ISO1: Dict[str, Language] = {}
_BY_ISO3: Dict[str, Language] = {}
_BY_WHISPER: Dict[str, Language] = {}
_BY_NAME: Dict[str, Language] = {}
_BY_NAME_LOWER: Dict[str, Language] = {}

for _lang in _LANGUAGES:
    _BY_ISO1[_lang.iso639_1] = _lang
    _BY_ISO3[_lang.iso639_3] = _lang
    _BY_WHISPER[_lang.whisper_code] = _lang
    _BY_NAME[_lang.name] = _lang
    _BY_NAME_LOWER[_lang.name.lower()] = _lang
    # Also index by native name (case-insensitive) for convenience.
    _BY_NAME_LOWER[_lang.native_name.lower()] = _lang


# ── Aliases / common alternative spellings ────────────────────────────────────
_ALIASES: Dict[str, str] = {
    # Indian languages
    "assamese": "as", "bengali": "bn", "bangla": "bn",
    "bodo": "brx", "dogri": "doi", "dogri": "doi",
    "gujarati": "gu", "gujrati": "gu",
    "hindi": "hi", "hin": "hi",
    "kannada": "kn", "kannad": "kn",
    "kashmiri": "ks", "koshur": "ks",
    "konkani": "gom", "konkanni": "gom",
    "maithili": "mai", "maithili": "mai",
    "malayalam": "ml", "malayali": "ml",
    "meitei": "mni", "manipuri": "mni", "meetei": "mni",
    "marathi": "mr", "marathi": "mr",
    "nepali": "ne", "nepalese": "ne",
    "odia": "or", "oriya": "or",
    "punjabi": "pa", "panjabi": "pa", "gurmukhi": "pa",
    "sanskrit": "sa", "samskrit": "sa",
    "santali": "sat", "santhali": "sat",
    "sindhi": "sd", "sindhi": "sd",
    "tamil": "ta", "thamizh": "ta",
    "telugu": "te", "thelungu": "te",
    "urdu": "ur",
    # International
    "english": "en", "spanish": "es", "espanol": "es",
    "french": "fr", "german": "de", "deutsch": "de",
    "russian": "ru", "arabic": "ar",
    "chinese": "zh", "mandarin": "zh",
    "japanese": "ja", "korean": "ko",
    "portuguese": "pt", "italian": "it",
    # Hinglish / code-switching marker (treat as Hindi for routing)
    "hinglish": "hi",
}


def all_languages() -> List[Language]:
    """Return a copy of the full registry list."""
    return list(_LANGUAGES)


def indian_languages() -> List[Language]:
    """Return only the 22 Eighth-Schedule Indian languages."""
    return [lang for lang in _LANGUAGES if lang.is_indian]


def international_languages() -> List[Language]:
    """Return non-Indian languages in the registry."""
    return [lang for lang in _LANGUAGES if not lang.is_indian]


def get_language(query: str) -> Optional[Language]:
    """
    Resolve a free-form language identifier to a Language object.

    Accepts: ISO 639-1, ISO 639-3, whisper code, English name, native name,
    or a common alias. Returns None when no match is found so callers can
    fall back to auto-detection rather than guessing.
    """
    if not query or not isinstance(query, str):
        return None
    q = query.strip().lower()
    if not q:
        return None

    if q in _BY_ISO1:
        return _BY_ISO1[q]
    if q in _BY_ISO3:
        return _BY_ISO3[q]
    if q in _BY_WHISPER:
        return _BY_WHISPER[q]
    if q in _BY_NAME_LOWER:
        return _BY_NAME_LOWER[q]
    if q in _ALIASES:
        return _BY_ISO1.get(_ALIASES[q]) or _BY_ISO3.get(_ALIASES[q])
    return None


def get_whisper_code(language: str | None) -> Optional[str]:
    """
    Map any supported language identifier to a faster-whisper language code.
    Returns None for 'auto' / empty so the model auto-detects.
    """
    if not language:
        return None
    lowered = language.strip().lower()
    if lowered in ("", "auto", "none", "null"):
        return None
    lang = get_language(language)
    if lang is not None:
        return lang.whisper_code
    # Last resort: return the raw lowercased token. faster-whisper will reject
    # unknown codes, but we avoid silently collapsing a valid code here.
    return lowered


def is_rtl(language: str | None) -> bool:
    """Return True if the language is written right-to-left."""
    lang = get_language(language)
    return bool(lang and lang.rtl)


def script_for_language(language: str | None) -> str:
    """Return the primary script name for a language, or 'Unknown'."""
    lang = get_language(language)
    if lang is None or not lang.scripts:
        return "Unknown"
    return lang.scripts[0]


def detect_script(text: str) -> Optional[str]:
    """Detect the dominant registered Unicode script, ignoring punctuation."""
    if not text:
        return None
    counts: Dict[str, int] = {}
    for ch in text:
        if not unicodedata.category(ch).startswith("L"):
            continue
        cp = ord(ch)
        scripts = set()
        for lang in _LANGUAGES:
            for i, (start, end) in enumerate(lang.unicode_ranges):
                if start <= cp <= end:
                    scripts.add(lang.scripts[i] if i < len(lang.scripts) else lang.scripts[0])
        # Kana uniquely identifies Japanese even though its registry also lists Han.
        if "Hiragana" in scripts or "Katakana" in scripts:
            scripts = {"Hiragana" if "Hiragana" in scripts else "Katakana"}
        elif "Han" in scripts:
            scripts = {"Han"}
        for script in scripts:
            counts[script] = counts.get(script, 0) + 1
    # For mixed Latin/Indic or Latin/Arabic text, report the non-Latin script
    # when present. Latin words are common in Hinglish and other mixed-script
    # transcripts and should not hide the script-specific text.
    non_latin_counts = {script: count for script, count in counts.items() if script != "Latin"}
    if non_latin_counts:
        counts = non_latin_counts
    return max(counts, key=counts.get) if counts else None


def detect_language_from_text(text: str) -> Optional[Language]:
    """Return a language only when its script uniquely identifies it."""
    script = detect_script(text)
    if script is None:
        return None
    candidates = [lang for lang in _LANGUAGES if script in lang.scripts]
    return candidates[0] if len(candidates) == 1 else None


def build_language_selector_options() -> List[str]:
    """Return stable language codes; UI labels are provided by a formatter."""
    return ["auto", *(lang.iso639_1 for lang in _LANGUAGES)]


def language_display_name(language: str | None) -> str:
    """Human-readable name for display in the UI."""
    if not language:
        return "Auto-detected"
    lang = get_language(language)
    if lang:
        return f"{lang.native_name} ({lang.name})"
    return language


def capability_summary(language: str | None) -> Dict[str, str]:
    """
    Return a per-model capability summary dict for a language.
    Keys: whisper, llm, embedding, vision.
    """
    lang = get_language(language)
    if lang is None:
        return {"whisper": UNSUPPORTED, "llm": UNSUPPORTED,
                "embedding": UNSUPPORTED, "vision": UNSUPPORTED}
    return {
        "whisper": lang.whisper_support,
        "llm": lang.llm_support,
        "embedding": lang.embedding_support,
        "vision": lang.vision_support,
    }


def overall_support_level(language: str | None) -> str:
    """
    Overall pipeline support level for a language, considering all models.
    The weakest link in transcription or LLM generation determines the level,
    because both are required for a full video-intelligence result.
    """
    caps = capability_summary(language)
    levels = [caps["whisper"], caps["llm"]]
    if WEAK in levels:
        return WEAK
    if PARTIAL in levels:
        return PARTIAL
    if UNSUPPORTED in levels:
        return UNSUPPORTED
    return FULL


def is_fully_supported(language: str | None) -> bool:
    """True only when every pipeline stage is rated FULL."""
    caps = capability_summary(language)
    return all(level == FULL for level in caps.values())


__all__ = [
    "Language",
    "all_languages",
    "indian_languages",
    "international_languages",
    "get_language",
    "get_whisper_code",
    "is_rtl",
    "script_for_language",
    "detect_script",
    "detect_language_from_text",
    "build_language_selector_options",
    "language_display_name",
    "capability_summary",
    "overall_support_level",
    "is_fully_supported",
    "FULL", "PARTIAL", "WEAK", "UNSUPPORTED",
]
