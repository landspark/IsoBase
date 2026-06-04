#! python3
# -*- coding: utf-8 -*-
"""Country and language utility module.

This module provides helper functions to retrieve ISO country and language
information using the pycountry library.

@File   : country.py
@Created: 2025/04/23 15:48
@Author : SwordJack
@Contact: https://github.com/SwordJack/
"""

from typing import Optional, Any
import pycountry


def get_country_by_alpha2(code: str) -> Optional[Any]:
    """Retrieves country information by ISO 3166-1 alpha-2 code.

    Args:
        code: A two-letter country code (e.g., 'CN', 'US').

    Returns:
        A pycountry Country object if found, otherwise None.
    """
    return pycountry.countries.get(alpha_2=code.upper())


def get_language_by_alpha2(code: str) -> Optional[Any]:
    """Retrieves language information by ISO 639-1 alpha-2 code.

    Args:
        code: A two-letter language code (e.g., 'zh', 'en').

    Returns:
        A pycountry Language object if found, otherwise None.
    """
    # Languages often use 2-letter (alpha_2) or 3-letter codes.
    return pycountry.languages.get(alpha_2=code.lower())
