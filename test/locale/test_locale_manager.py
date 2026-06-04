#! python3
# -*- coding: utf-8 -*-
"""Unit tests for the LocaleManager class.

@File   : test_locale_manager.py
@Created: 2025/04/24 16:15
@Author : SwordJack
@Contact: https://github.com/SwordJack/
"""

from ark import LocaleManager


class TestLocaleManager:
    """Tests the functionality of LocaleManager."""

    def test_is_valid(self):
        """Tests the is_valid class method."""
        assert LocaleManager.is_valid(locale_code="zh_cn")
        assert LocaleManager.is_valid(locale_code="zh-cn")
        assert LocaleManager.is_valid(locale_code="ZH-CN")
        assert LocaleManager.is_valid(locale_code="en_gb")
        assert not LocaleManager.is_valid(locale_code="ee_uu")

    def test_resolve_code_from_phone(self):
        """Tests resolution of locale code from a phone number."""
        # Default internal format
        assert LocaleManager.resolve_code_from_phone(
            raw_phone="+1 (615) 818-0310"
        ) == "en_us"
        # BCP 47 format
        assert LocaleManager.resolve_code_from_phone(
            raw_phone="+1 (615) 818-0310", bcp47=True
        ) == "en-US"
        assert LocaleManager.resolve_code_from_phone("13366889900") == "zh_cn"

    def test_resolve_from_phone(self):
        """Tests resolution of a LocaleManager instance from a phone number."""
        loc = LocaleManager.resolve_from_phone("13366889900")
        assert loc.code == "zh_cn"
        assert loc.code_bcp47 == "zh-CN"
        assert "China" in loc.country_name
        assert loc.flag == "🇨🇳"

    def test_instantiation(self):
        """Tests direct instantiation for metadata."""
        loc = LocaleManager("en-US")
        assert loc.code == "en_us"
        assert loc.code_bcp47 == "en-US"
        assert loc.country_name == "United States"
