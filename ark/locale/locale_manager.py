#! python3
# -*- encoding: utf-8 -*-
"""Locale management and resolution module.

This module provides the LocaleManager class which serves as both a utility
for locale resolution and a container for locale-specific metadata.

@File   : locale_manager.py
@Created: 2025/04/18 14:57
@Author : SwordJack
@Contact: https://github.com/SwordJack/
"""

from functools import cached_property
from typing import Optional, Tuple

from .phone import parse_phone_number, region_code_for_number
from .country import get_country_by_alpha2, get_language_by_alpha2

# Static mappings and aliases
DEFAULT_LOCALE_MAP = {
    "CN": "zh_cn",
    "HK": "zh_hk",
    "MO": "zh_hk",
    "TW": "zh_tw",
    "SG": "zh_sg",
    "CA": "en_ca",
    "DE": "de_de",
    "ES": "es_es",
    "FR": "fr_fr",
    "GB": "en_gb",
    "GR": "el_gr",
    "JP": "ja_jp",
    "KR": "ko_kr",
    "US": "en_us",
}

COUNTRY_CODE_ALIASES = {
    "UK": "GB",
    "EL": "GR",
}


def _normalize_locale(locale_code: str) -> str:
    """Normalizes any locale format (e.g., zh-CN, ZH_CN) to internal zh_cn."""
    return locale_code.replace("-", "_").lower()


def _format_to_bcp47(internal_code: str) -> str:
    """Formats internal lowercase_underscore locale to BCP 47 (e.g., zh-CN)."""
    if "_" not in internal_code:
        return internal_code
    lang, country = internal_code.split("_", 1)
    return f"{lang.lower()}-{country.upper()}"


class LocaleManager:
    """A combined class for locale utility and metadata.

    This class provides classmethods for resolution and validation,
    while instances of this class represent a specific locale context.
    """

    def __init__(self, locale_code: str):
        """Initializes a LocaleManager instance for a specific locale.

        Args:
            locale_code: Locale code in any common format (e.g., 'zh_cn').
        """
        self.code = _normalize_locale(locale_code)

    @cached_property
    def code_bcp47(self) -> str:
        """Standard BCP 47 tag (e.g., 'zh-CN')."""
        return _format_to_bcp47(self.code)

    @property
    def _components(self) -> Tuple[Optional[str], Optional[str]]:
        """Internal helper to split code into language and country components."""
        if "_" not in self.code:
            return None, None
        lang, country = self.code.split("_", 1)
        return lang, country

    @cached_property
    def language_name(self) -> str:
        """Full name of the language."""
        lang_code, _ = self._components
        if not lang_code:
            return ""
        lang_obj = get_language_by_alpha2(lang_code)
        return lang_obj.name if lang_obj else ""

    @cached_property
    def country_name(self) -> str:
        """Full name of the country."""
        _, country_code = self._components
        if not country_code:
            return ""
        actual_code = COUNTRY_CODE_ALIASES.get(
            country_code.upper(), country_code.upper()
        )
        country_obj = get_country_by_alpha2(actual_code)
        return country_obj.name if country_obj else ""

    @cached_property
    def flag(self) -> str:
        """Emoji flag of the country."""
        _, country_code = self._components
        if not country_code:
            return ""
        actual_code = COUNTRY_CODE_ALIASES.get(
            country_code.upper(), country_code.upper()
        )
        country_obj = get_country_by_alpha2(actual_code)
        return getattr(country_obj, "flag", "")

    @classmethod
    def is_valid(cls, locale_code: str) -> bool:
        """Checks whether a given locale code is valid.

        Args:
            locale_code: Locale code to validate.

        Returns:
            True if both language and country components are recognized.
        """
        internal_code = _normalize_locale(locale_code)
        if "_" not in internal_code:
            return False

        lang, country = internal_code.split("_", 1)
        lang_match = get_language_by_alpha2(lang)

        country_code = country.upper()
        country_code = COUNTRY_CODE_ALIASES.get(country_code, country_code)
        country_match = get_country_by_alpha2(country_code)

        return lang_match is not None and country_match is not None

    @classmethod
    def resolve_from_phone(cls, raw_phone: str) -> Optional["LocaleManager"]:
        """Resolves a LocaleManager instance from a phone number.

        Args:
            raw_phone: Raw phone number string.

        Returns:
            A LocaleManager instance or None if unresolvable.
        """
        number = parse_phone_number(raw_phone)
        if number is None:
            return None

        region_code = region_code_for_number(number)
        locale_code = DEFAULT_LOCALE_MAP.get(region_code, "en_us")

        if cls.is_valid(locale_code):
            return cls(locale_code)
        return None

    @classmethod
    def resolve_code_from_phone(
        cls, raw_phone: str, bcp47: bool = False
    ) -> Optional[str]:
        """Resolves a locale code string from a phone number.

        Args:
            raw_phone: Raw phone number string.
            bcp47: If True, returns the BCP 47 tag instead of internal format.

        Returns:
            A locale string or None if unresolvable.
        """
        instance = cls.resolve_from_phone(raw_phone)
        if not instance:
            return None
        return instance.code_bcp47 if bcp47 else instance.code

    def __repr__(self) -> str:
        return f"<LocaleManager code='{self.code}' code_bcp47='{self.code_bcp47}'>"
