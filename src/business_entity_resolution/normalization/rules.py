"""
normalization/rules.py — Data-driven locale rule registry for normalization.

Design principle: adding rules for a new locale or legal-entity type requires
ONLY adding entries to the dictionaries/lists in this file (or loading from a
YAML file), not modifying any logic in normalizer.py.  This satisfies the
open/closed principle for normalization rules.

Locale keys:
    "default"  — applied universally before locale-specific rules
    "US"       — United States
    "IN"       — India
    "FR"       — France
    (any ISO 3166-1 alpha-2 code may be added)

The country values in the data are normalised to upper-case before lookup.
Unknown countries fall back to the "default" rules only (no crash).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LocaleRules:
    """
    Collection of normalization rules for a single locale.

    Attributes
    ----------
    legal_suffixes:
        Mapping from variant form → canonical token.
        Applied to business names to fold legal-entity-type variants.
        E.g. {"incorporated": "inc", "corporation": "corp", ...}
    address_abbreviations:
        Mapping from abbreviated or variant form → expanded/canonical form.
        Applied to address tokens.
        E.g. {"st": "street", "ave": "avenue", ...}
    """
    legal_suffixes: dict[str, str] = field(default_factory=dict)
    address_abbreviations: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Default rules — applied universally regardless of country
# ---------------------------------------------------------------------------

_DEFAULT_LEGAL_SUFFIXES: dict[str, str] = {
    # Corporate entity types — English
    "incorporated": "inc",
    "corporation": "corp",
    "corporations": "corp",
    "company": "co",
    "companies": "co",
    "limited": "ltd",
    "limited liability company": "llc",
    "llc": "llc",
    "ltd": "ltd",
    "inc": "inc",
    "corp": "corp",
    "co": "co",
    # Partnerships / sole traders
    "partnership": "prtnr",
    "associates": "assoc",
    "associate": "assoc",
    "group": "grp",
    "holdings": "hldg",
    "holding": "hldg",
    "international": "intl",
    "industries": "ind",
    "industry": "ind",
    "services": "svc",
    "service": "svc",
    "solutions": "sol",
    "enterprises": "ent",
    "enterprise": "ent",
    "technologies": "tech",
    "technology": "tech",
    "systems": "sys",
    "system": "sys",
    "management": "mgmt",
    "brothers": "bros",
    "trading": "trdg",
}

_DEFAULT_ADDRESS_ABBREVIATIONS: dict[str, str] = {
    # Street types
    "street": "st",
    "avenue": "ave",
    "boulevard": "blvd",
    "road": "rd",
    "drive": "dr",
    "lane": "ln",
    "place": "pl",
    "court": "ct",
    "terrace": "ter",
    "circle": "cir",
    "highway": "hwy",
    "parkway": "pkwy",
    "expressway": "expy",
    "freeway": "fwy",
    "way": "wy",
    "trail": "trl",
    "alley": "aly",
    # Directionals
    "north": "n",
    "south": "s",
    "east": "e",
    "west": "w",
    "northeast": "ne",
    "northwest": "nw",
    "southeast": "se",
    "southwest": "sw",
    # Building types
    "suite": "ste",
    "apartment": "apt",
    "building": "bldg",
    "floor": "fl",
    # Common landmarks / relative addresses (fold to generic token)
    "near": "nr",
    "opposite": "opp",
    "behind": "bhnd",
    "next to": "adj",
    "adjacent": "adj",
    # Abbreviation expansions for blocking keys
    "st": "st",     # already short — keep as-is (do not expand back to street)
    "ave": "ave",
    "blvd": "blvd",
    "rd": "rd",
    "dr": "dr",
    "ln": "ln",
    "pl": "pl",
    "ct": "ct",
    "nr": "nr",
    "opp": "opp",
}

DEFAULT_RULES = LocaleRules(
    legal_suffixes=_DEFAULT_LEGAL_SUFFIXES,
    address_abbreviations=_DEFAULT_ADDRESS_ABBREVIATIONS,
)

# ---------------------------------------------------------------------------
# US-specific rules
# ---------------------------------------------------------------------------

_US_EXTRA_LEGAL: dict[str, str] = {
    "llp": "llp",          # Limited Liability Partnership
    "lp": "lp",            # Limited Partnership
    "pllc": "pllc",        # Professional LLC
    "pc": "pc",            # Professional Corporation
    "pa": "pa",            # Professional Association
    "na": "na",            # National Association (banking)
    "dba": "dba",          # Doing Business As
    "inc.": "inc",
    "corp.": "corp",
    "ltd.": "ltd",
    "llc.": "llc",
}

_US_EXTRA_ADDRESS: dict[str, str] = {
    "po box": "pobox",
    "p o box": "pobox",
    "p.o. box": "pobox",
    "post office box": "pobox",
    "route": "rte",
    "rte": "rte",
    "rural route": "rr",
    "interstate": "i",
    "us highway": "us",
    "state highway": "sr",
    "state road": "sr",
    "county road": "cr",
    "township": "twp",
}

US_RULES = LocaleRules(
    legal_suffixes={**_DEFAULT_LEGAL_SUFFIXES, **_US_EXTRA_LEGAL},
    address_abbreviations={**_DEFAULT_ADDRESS_ABBREVIATIONS, **_US_EXTRA_ADDRESS},
)

# ---------------------------------------------------------------------------
# India-specific rules
# ---------------------------------------------------------------------------

_IN_EXTRA_LEGAL: dict[str, str] = {
    "private limited": "pvt ltd",
    "pvt limited": "pvt ltd",
    "pvt. ltd.": "pvt ltd",
    "pvt. ltd": "pvt ltd",
    "pvt ltd": "pvt ltd",
    "private ltd": "pvt ltd",
    "private ltd.": "pvt ltd",
    "public limited": "pub ltd",
    "llp": "llp",
    "lp": "lp",
    "opc": "opc",           # One Person Company
    "section 8": "s8",      # Section 8 (non-profit)
    "ngo": "ngo",
    "trust": "trust",
    "society": "soc",
    "co operative": "coop",
    "cooperative": "coop",
    "co-operative": "coop",
}

_IN_EXTRA_ADDRESS: dict[str, str] = {
    "nagar": "ngr",
    "nagara": "ngr",
    "road": "rd",
    "marg": "mrg",
    "bazaar": "bazar",
    "bazar": "bazar",
    "gali": "gali",
    "mohalla": "mohalla",
    "chowk": "chowk",
    "near": "nr",
    "opp": "opp",
    "opposite": "opp",
    "beside": "adj",
    "sector": "sec",
    "phase": "ph",
    "plot": "plt",
    "shop": "shp",
    "flat": "fl",
    "wing": "wng",
    "block": "blk",
}

IN_RULES = LocaleRules(
    legal_suffixes={**_DEFAULT_LEGAL_SUFFIXES, **_IN_EXTRA_LEGAL},
    address_abbreviations={**_DEFAULT_ADDRESS_ABBREVIATIONS, **_IN_EXTRA_ADDRESS},
)

# ---------------------------------------------------------------------------
# France-specific rules
# ---------------------------------------------------------------------------

_FR_EXTRA_LEGAL: dict[str, str] = {
    "société anonyme": "sa",
    "societe anonyme": "sa",
    "société à responsabilité limitée": "sarl",
    "societe a responsabilite limitee": "sarl",
    "sarl": "sarl",
    "sa": "sa",
    "sas": "sas",           # Société par Actions Simplifiée
    "sasu": "sasu",         # SAS Unipersonnelle
    "sci": "sci",           # Société Civile Immobilière
    "snc": "snc",           # Société en Nom Collectif
    "eurl": "eurl",         # SARL Unipersonnelle
    "ei": "ei",             # Entreprise Individuelle
    "micro entreprise": "me",
    "association": "asso",
    "fondation": "fond",
    "groupement": "grpt",
    "gie": "gie",           # Groupement d'Intérêt Économique
}

_FR_EXTRA_ADDRESS: dict[str, str] = {
    "rue": "rue",
    "avenue": "ave",
    "boulevard": "blvd",
    "bd": "blvd",
    "place": "pl",
    "allée": "allee",
    "allee": "allee",
    "chemin": "chem",
    "route": "rte",
    "impasse": "imp",
    "passage": "pass",
    "villa": "vil",
    "résidence": "res",
    "residence": "res",
    "bâtiment": "bat",
    "batiment": "bat",
    "immeuble": "imm",
    "appartement": "apt",
    "boîte postale": "bp",
    "boite postale": "bp",
    "cedex": "cdx",
}

FR_RULES = LocaleRules(
    legal_suffixes={**_DEFAULT_LEGAL_SUFFIXES, **_FR_EXTRA_LEGAL},
    address_abbreviations={**_DEFAULT_ADDRESS_ABBREVIATIONS, **_FR_EXTRA_ADDRESS},
)

# ---------------------------------------------------------------------------
# Registry — keyed by upper-case ISO country code
# ---------------------------------------------------------------------------

LOCALE_REGISTRY: dict[str, LocaleRules] = {
    "default": DEFAULT_RULES,
    "US": US_RULES,
    "IN": IN_RULES,
    "FR": FR_RULES,
    # Additional locales may be registered here without touching normalizer.py
}


def get_rules(country: str) -> LocaleRules:
    """
    Return locale rules for *country* (case-insensitive ISO code).

    Falls back to default rules if the country is unknown — logs nothing
    here (the caller is responsible for logging unknown-country warnings).

    Parameters
    ----------
    country:
        ISO 3166-1 alpha-2 country code (or full country name — we normalise it).

    Returns
    -------
    LocaleRules
        The rules for the country, or default rules if not in registry.
    """
    key = country.strip().upper() if country else "default"
    return LOCALE_REGISTRY.get(key, DEFAULT_RULES)


def register_locale(country_code: str, rules: LocaleRules) -> None:
    """
    Register rules for a new locale at runtime.

    Parameters
    ----------
    country_code:
        Upper-case ISO 3166-1 alpha-2 code (e.g. "DE", "JP").
    rules:
        LocaleRules instance for this locale.
    """
    LOCALE_REGISTRY[country_code.strip().upper()] = rules
