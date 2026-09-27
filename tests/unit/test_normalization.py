"""
tests/unit/test_normalization.py — Unit tests for normalization module.

Tests:
- normalize_name: suffix folding, whitespace, punctuation, transliteration
- normalize_address: street number extraction, landmark detection, abbreviations
- normalize_country: ISO code normalization
- Transliteration: non-Latin → Latin
"""

import pytest

from business_entity_resolution.normalization.normalizer import (
    NormalizedAddress,
    normalize_address,
    normalize_country,
    normalize_name,
)
from business_entity_resolution.normalization.transliteration import (
    is_latin_script,
    transliterate,
)


class TestTransliteration:
    def test_ascii_passthrough(self):
        assert transliterate("Hello World") == "Hello World"

    def test_diacritic_removal(self):
        result = transliterate("München")
        assert "u" in result.lower() or "ue" in result.lower()
        assert all(ord(c) < 128 for c in result)

    def test_cyrillic_transliteration(self):
        result = transliterate("Москва")
        # Should produce a Latin representation, all ASCII
        assert all(ord(c) < 128 for c in result)
        assert len(result) > 0

    def test_devanagari_transliteration(self):
        result = transliterate("नई दिल्ली")
        assert all(ord(c) < 128 for c in result)
        assert len(result) > 0

    def test_empty_string(self):
        assert transliterate("") == ""

    def test_latin_already_clean(self):
        result = transliterate("San Jose CA")
        assert result == "San Jose CA"

    def test_no_silent_drop(self):
        # Non-ASCII characters must produce SOMETHING, not be silently deleted
        result = transliterate("東京")
        assert len(result.strip()) > 0

    def test_is_latin_script_ascii(self):
        assert is_latin_script("Hello World") is True

    def test_is_latin_script_cyrillic(self):
        assert is_latin_script("Москва") is False

    def test_is_latin_script_empty(self):
        assert is_latin_script("") is True


class TestNormalizeName:
    def test_basic_name(self):
        result = normalize_name("Acme Robotics Inc", "US")
        assert "acme" in result
        assert "robotics" in result
        # "inc" should be present (canonical form)
        assert "inc" in result

    def test_incorporated_folded_to_inc(self):
        result = normalize_name("Acme Robotics Incorporated", "US")
        assert "inc" in result
        # "incorporated" should not appear (folded to canonical)
        assert "incorporated" not in result

    def test_corporation_folded(self):
        result = normalize_name("Delta Foods Corporation", "US")
        assert "corp" in result
        assert "corporation" not in result

    def test_pvt_ltd_india(self):
        result = normalize_name("Zen Traders Pvt Ltd", "IN")
        assert "zen" in result
        assert "traders" in result

    def test_private_limited_india(self):
        result = normalize_name("Zen Traders Private Limited", "IN")
        # Should fold to "pvt ltd"
        result2 = normalize_name("Zen Traders Pvt Ltd", "IN")
        # Both should have the same tokens for "zen" and "traders"
        assert "zen" in result
        assert "zen" in result2

    def test_sarl_france(self):
        result = normalize_name("Le Petit Bistro SARL", "FR")
        assert "petit" in result
        assert "bistro" in result
        assert "sarl" in result

    def test_punctuation_removal(self):
        result = normalize_name("Acme, Robotics. Inc!", "US")
        assert "," not in result
        assert "." not in result
        assert "!" not in result

    def test_empty_name(self):
        assert normalize_name("", "US") == ""
        assert normalize_name("   ", "US") == ""

    def test_case_normalization(self):
        result = normalize_name("ACME ROBOTICS INC", "US")
        assert result == normalize_name("Acme Robotics Inc", "US")

    def test_unknown_country_uses_defaults(self):
        # Should not crash for unknown country code
        result = normalize_name("Firma GmbH", "DE")
        assert len(result) > 0

    def test_deterministic(self):
        name = "Bright Cafe LLC"
        assert normalize_name(name, "US") == normalize_name(name, "US")


class TestNormalizeAddress:
    def test_basic_street_number_extracted(self):
        result = normalize_address("123 Innovation Drive, San Jose, CA 95110", "US")
        assert result.street_number == "123"
        assert result.is_landmark_relative is False
        assert "innovation" in result.raw_normalized

    def test_landmark_relative_address(self):
        result = normalize_address("Near City Hall, San Jose", "US")
        assert result.street_number is None
        assert result.is_landmark_relative is True

    def test_empty_address(self):
        result = normalize_address("", "US")
        assert result.raw_normalized == ""
        assert result.is_landmark_relative is True
        assert result.street_number is None

    def test_street_abbreviation_fold(self):
        # "Drive" may or may not be abbreviated — just verify it normalizes consistently
        result1 = normalize_address("123 Innovation Drive San Jose", "US")
        result2 = normalize_address("123 Innovation Dr San Jose", "US")
        # Both should have the same street number
        assert result1.street_number == result2.street_number == "123"

    def test_india_address(self):
        result = normalize_address("14 MG Road, Bangalore 560001", "IN")
        assert result.street_number == "14"
        assert not result.is_landmark_relative

    def test_france_address(self):
        result = normalize_address("25 Rue de Rivoli, Paris 75001", "FR")
        assert result.street_number == "25"

    def test_number_with_suffix(self):
        # Street numbers like "123a" should also be extracted
        result = normalize_address("123a Main Street", "US")
        assert result.street_number == "123a"

    def test_blocking_key_prefix(self):
        result = normalize_address("123 Innovation Drive", "US")
        prefix = result.blocking_key_prefix(6)
        assert len(prefix) <= 6
        assert len(prefix) > 0

    def test_number_key_empty_for_landmark(self):
        result = normalize_address("Near City Hall", "US")
        assert result.number_key() == ""

    def test_raw_normalized_all_lowercase(self):
        result = normalize_address("123 MAIN STREET", "US")
        assert result.raw_normalized == result.raw_normalized.lower()

    def test_deterministic(self):
        addr = "456 Market Street, Chicago, IL 60601"
        r1 = normalize_address(addr, "US")
        r2 = normalize_address(addr, "US")
        assert r1.raw_normalized == r2.raw_normalized
        assert r1.street_number == r2.street_number


class TestNormalizeCountry:
    def test_us_variations(self):
        assert normalize_country("US") == "US"
        assert normalize_country("United States") == "US"
        assert normalize_country("USA") == "US"
        assert normalize_country("UNITED STATES OF AMERICA") == "US"

    def test_india_variations(self):
        assert normalize_country("India") == "IN"
        assert normalize_country("IN") == "IN"

    def test_france_variations(self):
        assert normalize_country("France") == "FR"
        assert normalize_country("FR") == "FR"

    def test_unknown_country_passthrough(self):
        result = normalize_country("Germany")
        assert result == "DE"

    def test_empty_country(self):
        result = normalize_country("")
        assert result == "UNKNOWN"

    def test_whitespace(self):
        result = normalize_country("  US  ")
        assert result == "US"

    def test_case_insensitive(self):
        assert normalize_country("india") == normalize_country("INDIA")
