#! python3
# -*- coding: utf-8 -*-
"""Phone number utility module.

This module provides helper functions to parse, validate, and format
phone numbers using the phonenumbers library.
"""

from typing import Optional
import phonenumbers


def parse_phone_number(
    raw_phone: str, region_hint: str = "CN"
) -> Optional[phonenumbers.PhoneNumber]:
    """Parses and validates a phone number.

    Args:
        raw_phone: A phone number string.
        region_hint: Default region to assume if no country code is provided.

    Returns:
        A phonenumbers.PhoneNumber object if valid, otherwise None.
    """
    try:
        number = phonenumbers.parse(raw_phone, region_hint)
        if phonenumbers.is_valid_number(number):
            return number
        return None
    except phonenumbers.NumberParseException:
        return None


def region_code_for_number(number: phonenumbers.PhoneNumber) -> Optional[str]:
    """Retrieves the ISO 3166-1 alpha-2 region code from a phone number.

    Args:
        number: A phonenumbers.PhoneNumber object.

    Returns:
        A two-letter region code (e.g., 'CN', 'US') if valid, otherwise None.
    """
    return phonenumbers.region_code_for_number(number) if number else None


def format_e164(number_obj: phonenumbers.PhoneNumber) -> Optional[str]:
    """Formats a PhoneNumber object into E.164 format.

    Args:
        number_obj: A phonenumbers.PhoneNumber object.

    Returns:
        The E.164 formatted string, or None if the input is None.
    """
    if number_obj is None:
        return None
    return phonenumbers.format_number(
        number_obj, phonenumbers.PhoneNumberFormat.E164
    )


def format_international(number_obj: phonenumbers.PhoneNumber) -> Optional[str]:
    """Formats a PhoneNumber object into International format.

    Args:
        number_obj: A phonenumbers.PhoneNumber object.

    Returns:
        The International formatted string, or None if the input is None.
    """
    if number_obj is None:
        return None
    return phonenumbers.format_number(
        number_obj, phonenumbers.PhoneNumberFormat.INTERNATIONAL
    )


def format_national(number_obj: phonenumbers.PhoneNumber) -> Optional[str]:
    """Formats a PhoneNumber object into National format.

    Args:
        number_obj: A phonenumbers.PhoneNumber object.

    Returns:
        The National formatted string, or None if the input is None.
    """
    if number_obj is None:
        return None
    return phonenumbers.format_number(
        number_obj, phonenumbers.PhoneNumberFormat.NATIONAL
    )


def format_rfc3966(number_obj: phonenumbers.PhoneNumber) -> Optional[str]:
    """Formats a PhoneNumber object into RFC3966 format.

    Args:
        number_obj: A phonenumbers.PhoneNumber object.

    Returns:
        The RFC3966 formatted string, or None if the input is None.
    """
    if number_obj is None:
        return None
    return phonenumbers.format_number(
        number_obj, phonenumbers.PhoneNumberFormat.RFC3966
    )
