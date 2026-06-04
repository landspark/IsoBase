# Locale Utility

## Introduction

The `ark.locale` module provides tools for phone number parsing, country/language metadata retrieval, and locale resolution. It is designed to be lightweight, performant, and compliant with the Google Python Style Guide.

## Features

- **Phone Number Parsing**: Parse, validate, and format phone numbers in E.164, International, National, and RFC3966 formats using the `phonenumbers` library.
- **Locale Resolution**: Resolve locale codes (e.g., `zh_cn`) from raw phone numbers.
- **Rich Metadata**: Retrieve country names, language names, and emoji flags using the `pycountry` library.
- **BCP 47 Support**: Supports both internal format (`zh_cn`) and standard BCP 47 tags (`zh-CN`).
- **Lazy Loading**: Metadata is calculated on-demand and cached for efficiency.

## Quick Start

```python
from ark import LocaleManager

# 1. Resolve from phone number
loc = LocaleManager.resolve_from_phone("+8613366889900")
if loc:
    print(f"{loc.flag} {loc.country_name} ({loc.code_bcp47})")
    # Output: 🇨🇳 China (zh-CN)

# 2. Direct lookup by code
loc = LocaleManager("en-US")
print(loc.language_name) # Output: English

# 3. Simple code resolution
code = LocaleManager.resolve_code_from_phone("+1 615 818 0310", bcp47=True)
print(code) # Output: en-US
```

## Module Structure

- `locale_manager.py`: Core logic for resolution and metadata management.
- `phone.py`: Wrappers for `phonenumbers` library.
- `country.py`: Wrappers for `pycountry` library.
