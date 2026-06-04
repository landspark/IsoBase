# Locale 工具模块

## 简介

`ark.locale` 模块提供了电话号码解析、国家/语言元数据获取以及地区（Locale）解析的工具。该模块设计轻量、高效，并严格遵循 Google Python 代码风格指南。

## 功能特性

- **电话号码解析**：利用 `phonenumbers` 库解析、验证及转换电话号码格式（E.164, 国际, 国内, RFC3966）。
- **地区解析**：根据原始电话号码推断对应的地区码（如 `zh_cn`）。
- **丰富的元数据**：利用 `pycountry` 库获取国家名称、语言名称及国旗 Emoji。
- **BCP 47 支持**：支持内部格式（`zh_cn`）与标准 BCP 47 标签（`zh-CN`）的相互转换与兼容。
- **延迟加载**：元数据仅在访问时计算并缓存，提升大规模处理性能。

## 快速上手

```python
from ark import LocaleManager

# 1. 从手机号解析
loc = LocaleManager.resolve_from_phone("+8613366889900")
if loc:
    print(f"{loc.flag} {loc.country_name} ({loc.code_bcp47})")
    # 输出: 🇨🇳 China (zh-CN)

# 2. 通过代码直接查询
loc = LocaleManager("en-US")
print(loc.language_name) # 输出: English

# 3. 仅解析代码字符串
code = LocaleManager.resolve_code_from_phone("+1 615 818 0310", bcp47=True)
print(code) # 输出: en-US
```

## 模块结构

- `locale_manager.py`: 核心解析逻辑与元数据管理。
- `phone.py`: `phonenumbers` 库的工具封装。
- `country.py`: `pycountry` 库的工具封装。
