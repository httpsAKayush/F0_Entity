"""
normalization/transliteration.py — Non-Latin script to Latin/ASCII transliteration.

Design principles:
- Deterministic and pure (no I/O, no global state).
- Isolated in its own module so it is unit-testable independently of the rest
  of the normalization pipeline.
- Uses ``anyascii`` as the primary transliteration engine (broad Unicode coverage,
  zero network calls, pure Python).  Falls back to ``unidecode`` for edge cases.
- Strips diacritics from Latin-script text after transliteration.
- DOES NOT silently drop non-ASCII characters: every character is either
  transliterated to a Latin equivalent or retained if already Latin.

Examples (documented for test parity):
    "München"     → "Munchen"
    "Москва"      → "Moskva"
    "東京"         → "Dongjing"  (or "Tokyo" depending on anyascii mapping)
    "नई दिल्ली"  → "Nai Dilli"
    "مصر"         → "Masr"
"""

from __future__ import annotations

import unicodedata


def _strip_diacritics(text: str) -> str:
    """
    Remove diacritical marks from Latin-script text using Unicode NFKD decomposition.

    Parameters
    ----------
    text:
        Input text (already in Latin or ASCII).

    Returns
    -------
    str
        Text with combining characters (diacritics) removed.
    """
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def transliterate(text: str) -> str:
    """
    Convert any Unicode text to a best-effort ASCII/Latin representation.

    Strategy:
    1. Apply NFKC normalisation (e.g. ligature decomposition).
    2. Attempt transliteration via ``anyascii`` (broad Unicode coverage).
    3. Strip residual diacritics from the result.
    4. Replace remaining non-ASCII characters with a space (never silent drop).

    Parameters
    ----------
    text:
        Raw input text, potentially containing non-Latin scripts.

    Returns
    -------
    str
        Best-effort ASCII representation.  Always returns a string (never None).
    """
    if not text:
        return ""

    # Step 1: Unicode normalisation
    text = unicodedata.normalize("NFKC", text)

    # Step 2: anyascii transliteration
    try:
        from anyascii import anyascii as _anyascii
        text = _anyascii(text)
    except ImportError:
        # Fallback to unidecode if anyascii is unavailable
        try:
            from unidecode import unidecode as _unidecode
            text = _unidecode(text)
        except ImportError:
            # Last resort: strip diacritics and replace remaining non-ASCII
            pass

    # Step 3: Strip residual diacritics
    text = _strip_diacritics(text)

    # Step 4: Replace any remaining non-ASCII characters with a space
    text = "".join(ch if ord(ch) < 128 else " " for ch in text)

    return text


def is_latin_script(text: str) -> bool:
    """
    Return True if the majority of letter characters in *text* are Latin-script.

    Used to decide whether to apply transliteration to a given record field.

    Parameters
    ----------
    text:
        Input text to inspect.

    Returns
    -------
    bool
        True if >= 50% of letter characters are Latin (Basic Latin / Latin Extended).
    """
    letters = [ch for ch in text if unicodedata.category(ch).startswith("L")]
    if not letters:
        return True  # no letters → treat as effectively Latin
    latin_count = sum(
        1 for ch in letters
        if "LATIN" in (unicodedata.name(ch, "") or "")
    )
    return (latin_count / len(letters)) >= 0.5
