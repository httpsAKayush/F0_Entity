"""
normalization/normalizer.py — Public normalization API.

Public functions:
    normalize_name(name, country) -> str
    normalize_address(address, country) -> NormalizedAddress

Design:
- Pure functions (no I/O, no global state) — easily unit-testable.
- Locale rules are looked up from rules.py registry; adding a new locale's
  suffix/abbreviation rules does NOT require modifying this file.
- Transliteration is delegated to transliteration.py.
- Returns structured NormalizedAddress so callers can access the street
  number, street tokens, and landmark flag independently.
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass
from typing import Optional

from business_entity_resolution.normalization.rules import get_rules
from business_entity_resolution.normalization.transliteration import transliterate


# ---------------------------------------------------------------------------
# Structured address result
# ---------------------------------------------------------------------------


@dataclass
class NormalizedAddress:
    """
    Structured result of address normalization.

    Attributes
    ----------
    raw_normalized:
        The full normalized address string (transliterated, lowercased,
        punctuation-removed, abbreviations folded).
    street_number:
        Numeric token that appears to be a street number (first numeric-ish
        token), or None if the address has no parseable street number.
    street_name_tokens:
        Remaining word tokens after removing the street number.
    is_landmark_relative:
        True if no street number was found, indicating a landmark-relative or
        approximate address (e.g. "Near City Hall, San Jose").
    """
    raw_normalized: str
    street_number: Optional[str]
    street_name_tokens: list[str]
    is_landmark_relative: bool

    def blocking_key_prefix(self, n: int = 6) -> str:
        """Return the first *n* characters of the raw_normalized address."""
        return self.raw_normalized[:n]

    def number_key(self) -> str:
        """Return the street number as a blocking key, or empty string."""
        return self.street_number or ""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Punctuation table that replaces all punctuation with space
_PUNCT_TABLE = str.maketrans(string.punctuation, " " * len(string.punctuation))

# Regex to detect numeric tokens (street numbers, building numbers, etc.)
_NUMERIC_RE = re.compile(r"^\d+[a-z]?$")

# Multi-space collapse
_SPACE_RE = re.compile(r"\s+")


def _clean_base(text: str) -> str:
    """
    Minimal cleaning shared by both name and address normalization:
    1. Transliterate non-Latin to ASCII
    2. Lowercase
    3. Replace punctuation with spaces
    4. Collapse whitespace
    """
    text = transliterate(text)
    text = text.lower()
    text = text.translate(_PUNCT_TABLE)
    text = _SPACE_RE.sub(" ", text).strip()
    return text


def _apply_suffix_rules(tokens: list[str], suffix_map: dict[str, str]) -> list[str]:
    """
    Replace tokens that match legal suffix variants with their canonical form.

    Strategy: for efficiency, we do single-token lookups first (O(n) per token).
    Multi-word phrases are checked as a secondary pass over all rule keys that
    contain spaces (these are rarer and checked with a simple sliding window).

    Parameters
    ----------
    tokens:
        List of word tokens (lowercase, already cleaned).
    suffix_map:
        Mapping from variant (lowercase, space-joined) → canonical token.

    Returns
    -------
    list[str]
        Tokens with suffix variants replaced by canonical forms.
    """
    # Separate single-word and multi-word rules
    single_rules = {k: v for k, v in suffix_map.items() if " " not in k}
    multi_rules = sorted(
        [(k.split(), v) for k, v in suffix_map.items() if " " in k],
        key=lambda x: len(x[0]),
        reverse=True,
    )

    # First pass: single-token substitutions (fast O(n))
    result: list[str] = [single_rules.get(tok, tok) for tok in tokens]

    # Second pass: multi-word phrase substitutions (only if there are any)
    if multi_rules:
        i = 0
        while i < len(result):
            matched = False
            for phrase_tokens, canonical in multi_rules:
                n = len(phrase_tokens)
                if result[i:i + n] == phrase_tokens:
                    result[i:i + n] = [canonical]
                    matched = True
                    break
            if not matched:
                i += 1

    return result



def _apply_address_abbreviations(
    tokens: list[str], abbrev_map: dict[str, str]
) -> list[str]:
    """
    Apply address abbreviation expansions/folds to token list.

    Unlike suffix rules, we only check single-token matches (address abbreviations
    are single words in practice).
    """
    return [abbrev_map.get(tok, tok) for tok in tokens]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def normalize_name(name: str, country: str = "default") -> str:
    """
    Normalize a business name for blocking-key construction and feature extraction.

    Steps:
    1. Transliterate non-Latin scripts to ASCII.
    2. Lowercase and strip punctuation.
    3. Fold legal-entity-type suffixes to canonical forms using locale rules.
    4. Remove stopwords that add no discriminating value (articles, prepositions).
    5. Return the cleaned, space-joined string.

    Parameters
    ----------
    name:
        Raw business name string.
    country:
        ISO country code (used to select locale-specific suffix rules).

    Returns
    -------
    str
        Normalized name string.  May be empty if the input was blank.
    """
    if not name or not name.strip():
        return ""

    rules = get_rules(country)
    tokens = _clean_base(name).split()
    tokens = _apply_suffix_rules(tokens, rules.legal_suffixes)

    return " ".join(tokens).strip()


# Common stopwords to strip from addresses (not names — name stopwords would remove
# discriminating info like "The", "New", etc.)
_ADDRESS_STOPWORDS = frozenset({
    "the", "a", "an", "of", "and", "at", "in", "on", "by",
    "for", "to", "from", "with", "near",
})


def normalize_address(address: str, country: str = "default") -> NormalizedAddress:
    """
    Normalize a business address string into a structured NormalizedAddress.

    Steps:
    1. Transliterate non-Latin scripts.
    2. Lowercase, strip punctuation.
    3. Apply locale address-abbreviation folds.
    4. Detect street number (first numeric-ish token).
    5. Produce raw_normalized string and structured fields.

    Landmark-relative addresses (e.g. "Near City Hall, San Jose") that have
    no parseable street number are flagged with is_landmark_relative=True.
    Downstream code uses this flag to fall back to text-similarity-only
    comparison instead of relying on number agreement.

    Parameters
    ----------
    address:
        Raw address string.
    country:
        ISO country code for locale-specific abbreviation rules.

    Returns
    -------
    NormalizedAddress
        Structured normalization result.
    """
    if not address or not address.strip():
        return NormalizedAddress(
            raw_normalized="",
            street_number=None,
            street_name_tokens=[],
            is_landmark_relative=True,
        )

    rules = get_rules(country)
    cleaned = _clean_base(address)
    tokens = cleaned.split()
    tokens = _apply_address_abbreviations(tokens, rules.address_abbreviations)

    # Detect street number: the first token that is purely numeric (possibly
    # with a trailing letter like "123a") is taken as the street number.
    street_number: Optional[str] = None
    street_tokens: list[str] = []

    for idx, tok in enumerate(tokens):
        if street_number is None and _NUMERIC_RE.match(tok):
            street_number = tok
        else:
            street_tokens.append(tok)

    raw_normalized = " ".join(
        [street_number] + street_tokens if street_number else street_tokens
    ).strip()

    return NormalizedAddress(
        raw_normalized=raw_normalized,
        street_number=street_number,
        street_name_tokens=street_tokens,
        is_landmark_relative=(street_number is None),
    )


def normalize_country(country: str) -> str:
    """
    Normalize a country field to upper-case ISO code.

    We apply a simple mapping for full country names we know about; anything
    else is returned as-is in upper case.  This is intentionally lenient — the
    pipeline must handle data-driven country values, not a fixed allow-list.

    Parameters
    ----------
    country:
        Raw country field value.

    Returns
    -------
    str
        Normalized (upper-case) country code.
    """
    if not country or not country.strip():
        return "UNKNOWN"

    cleaned = country.strip().upper()

    # Map common full names → ISO codes
    _NAME_TO_ISO: dict[str, str] = {
        "UNITED STATES": "US",
        "UNITED STATES OF AMERICA": "US",
        "USA": "US",
        "INDIA": "IN",
        "FRANCE": "FR",
        "UNITED KINGDOM": "GB",
        "UK": "GB",
        "GREAT BRITAIN": "GB",
        "GERMANY": "DE",
        "JAPAN": "JP",
        "CHINA": "CN",
        "BRAZIL": "BR",
        "CANADA": "CA",
        "AUSTRALIA": "AU",
    }
    return _NAME_TO_ISO.get(cleaned, cleaned)
