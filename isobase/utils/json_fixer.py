#! python3
# -*- encoding: utf-8 -*-
"""
@File   :   json_fixer_service.py
@Created:   2025/10/19 13:50
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""

# Here put the import lib.
import json
import re
import ast
from typing import List, Optional, Tuple, Any


class JsonFixer:
    """
    A robust JSON fixer that extracts, cleans, and repairs broken or noisy JSON-like text.
    Public API: `JsonFixer.clean_and_fix_json(text: str) -> str`
    """

    # --- Precompiled regular expressions (performance + readability) ---
    FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", flags=re.IGNORECASE)
    OPEN_BRACE_RE = re.compile(r'\{')
    OPEN_BRACKET_RE = re.compile(r'\[')
    CLOSE_BRACE_RE = re.compile(r'\}')
    CLOSE_BRACKET_RE = re.compile(r'\]')
    TRAILING_COMMA_RE = re.compile(r',\s*([}\]])')
    EQUALS_AS_COLON_RE = re.compile(
        r'(?P<key>(?P<q>"[^"]*"|\'[^\']*\')|(?P<u>[A-Za-z_][A-Za-z0-9_\-]*))\s*=\s*'
    )
    UNQUOTED_KEYS_RE = re.compile(r'(?P<prefix>[{\s,])(?P<key>[A-Za-z_][A-Za-z0-9_\-]*)\s*:')
    JSON_TO_PY_RE = re.compile(r'\b(true|false|null)\b')
    PY_TO_JSON_RE = re.compile(r'\b(True|False|None)\b')
    SINGLE_QUOTED_STRING_RE = re.compile(r"'([^'\\]*(?:\\.[^'\\]*)*)'")
    FIRST_BRACE_OR_BRACKET_RE = re.compile(r'[\{\[]', re.DOTALL)
    LAST_ANY_RE = re.compile(r'.*[\}\]]', re.DOTALL)
    ZERO_WIDTH_RE = re.compile(r'[\u200b-\u200f\u202a-\u202e]')
    
    # Matches single backticks that are not part of a triple-backtick sequence.
    SINGLE_BACKTICK_RE = re.compile(r'(?<!`)`(?!`)')
    # Heuristically matches unescaped double quotes inside string values.
    UNESCAPED_QUOTE_RE = re.compile(r'(?<=[^\s:,\[{])"(?=[^\s:,\]}])')

    # ========================= Public API =========================

    @classmethod
    def clean_and_fix_json(cls, text: str) -> str:
        """
        Input: Raw LLM output (expected to include a JSON object/array, possibly with noise).
        Output: Canonical, minimal JSON string (UTF-8, no spaces): separators=(',', ':'), ensure_ascii=False.

        Strategy:
        - Normalize the text (quotes, backticks, invisible chars).
        - Extract candidate JSON regions (code fences, broad slice from first '{'/'[' to last '}'/']', full text).
        - For each candidate, attempt multi-stage repairs and parse.
        - If all candidates fail, run a final brute-force repair on the full text.
        - Raises ValueError if no valid JSON can be recovered.
        """
        normalized = cls._normalize_basic(text)
        candidates = cls._extract_candidates(normalized)

        last_error: Optional[Exception] = None

        for cand in candidates:
            cand_norm = cls._normalize_basic(cand)

            # Trim to a likely JSON substring (from first { or [)
            first = cls.FIRST_BRACE_OR_BRACKET_RE.search(cand_norm)
            last = cls.LAST_ANY_RE.search(cand_norm)
            if first and last:
                cand_norm = cand_norm[first.start():]

            obj, err = cls._try_parse_variants(cand_norm)
            # Enforce that the evaluated object is a structured JSON type (dict or list),
            # preventing ast.literal_eval from mistakenly passing pure string evaluations.
            if obj is not None and isinstance(obj, (dict, list)):
                try:
                    return json.dumps(obj, ensure_ascii=False, separators=(',', ':'))
                except Exception as e:
                    last_error = e
                    continue
            else:
                last_error = err

        # Final fallback: brute-force repair over the entire normalized text
        brutal = cls._single_to_double_quotes(
            cls._quote_unquoted_keys(
                cls._fix_equals_as_colon(
                    cls._strip_trailing_commas(cls._balance_brackets(normalized))
                )
            )
        )
        obj, err = cls._try_parse_variants(brutal)
        if obj is not None and isinstance(obj, (dict, list)):
            return json.dumps(obj, ensure_ascii=False, separators=(',', ':'))

        raise ValueError(f"Failed to extract and repair a valid JSON object. Input Content: {text}; Last error: {last_error}")

    # ========================= Parsing pipeline =========================

    @classmethod
    def _try_parse_variants(cls, s: str) -> Tuple[Optional[Any], Optional[Exception]]:
        """
        Try multiple repair/parse strategies. Returns (obj, None) on success or (None, last_err) on failure.
        Order of attempts mirrors common LLM mistakes and cheapest-to-most-invasive fixes.
        """
        last_err: Optional[Exception] = None

        # Attempt 0: Direct JSON
        try:
            return json.loads(s), None
        except Exception as e:
            last_err = e

        # Attempt 1: Balance brackets + strip trailing commas -> JSON
        s1 = cls._strip_trailing_commas(cls._balance_brackets(s))
        try:
            return json.loads(s1), None
        except Exception as e:
            last_err = e

        # Attempt 2: Fix "key" = value, quote unquoted keys, strip trailing commas -> JSON
        s2 = cls._fix_equals_as_colon(s1)
        s2 = cls._quote_unquoted_keys(s2)
        s2 = cls._strip_trailing_commas(s2)
        try:
            return json.loads(s2), None
        except Exception as e:
            last_err = e

        # Attempt 2.1: Heuristic escape of internal unescaped quotes -> JSON
        # This fixes issues like: "review": "He said "hello" today."
        s_esc = cls._escape_internal_quotes(s2)
        try:
            return json.loads(s_esc), None
        except Exception as e:
            last_err = e

        # Attempt 3: Convert single-quoted strings to double-quoted -> JSON
        s3 = cls._single_to_double_quotes(s2)
        s3 = cls._strip_trailing_commas(s3)
        try:
            return json.loads(s3), None
        except Exception as e:
            last_err = e

        # Attempt 4: Python literal_eval (tolerates single quotes and True/False/None)
        s4 = cls._to_python_bools_nulls(s2)
        try:
            obj = ast.literal_eval(s4)
            return obj, None
        except Exception as e:
            last_err = e

        # Attempt 5: literal_eval on single-quote-fixed string
        s5 = cls._to_python_bools_nulls(s3)
        try:
            obj = ast.literal_eval(s5)
            return obj, None
        except Exception as e:
            last_err = e

        return None, last_err

    # ========================= Normalization & extraction =========================

    @classmethod
    def _normalize_basic(cls, s: str) -> str:
        """Remove BOM/zero-width chars, normalize smart quotes, safely replace backticks, and tidy spaces."""
        s = s.replace('\ufeff', '')
        # 将中文弯引号（" "）转为直角引号（「 」），避免与 JSON 结构引号冲突
        s = s.replace('\u201c', '\u300c').replace('\u201d', '\u300d')
        s = s.replace('\u2018', "'").replace('\u2019', "'")
        s = s.replace('\u00a0', ' ')

        # Replace single backticks to double quotes, but protect Markdown code fences (```).
        s = cls.SINGLE_BACKTICK_RE.sub('"', s)
        s = cls.ZERO_WIDTH_RE.sub('', s)
        return s

    @classmethod
    def _extract_candidates(cls, s: str) -> List[str]:
        """
        Gather candidate JSON substrings, ordered by likelihood:
        1) Markdown fenced blocks
        2) Slice from first '{'/'[' to last '}'/']'
        3) Whole text
        Deduplicated while preserving order.
        """
        cands: List[str] = []

        # Fenced code blocks (```json ... ``` or ``` ... ```)
        for blk in cls.FENCE_RE.findall(s):
            cands.append(blk.strip())

        # Slice between first opening and last closing bracket/brace
        opens = [m.start() for m in cls.OPEN_BRACE_RE.finditer(s)] + [m.start() for m in cls.OPEN_BRACKET_RE.finditer(s)]
        closes = [m.start() for m in cls.CLOSE_BRACE_RE.finditer(s)] + [m.start() for m in cls.CLOSE_BRACKET_RE.finditer(s)]
        if opens and closes:
            start = min(opens)
            end = max(closes)
            if start < end:
                cands.append(s[start:end + 1].strip())

        # Full text as fallback
        cands.append(s.strip())

        # Deduplicate while preserving order
        seen = set()
        uniq: List[str] = []
        for x in cands:
            if x and x not in seen:
                seen.add(x)
                uniq.append(x)
        return uniq

    # ========================= Structural fixes =========================

    @classmethod
    def _balance_brackets(cls, s: str) -> str:
        """
        Best-effort bracket/brace repair with ordering correction.
        - Streams through the input, maintaining a stack of openings.
        - On a mismatched closing (e.g., '}' while top is '['), it first emits the
        needed closing (']') to close the current top, then re-checks the incoming
        closing. If it now matches, it consumes it; otherwise the stray closing is dropped.
        - Finally, appends any remaining required closings.
        This is safer than simply appending all missing closings at the end.
        """
        pairs = {'{': '}', '[': ']'}
        opening = set(pairs.keys())
        closing = set(pairs.values())

        stack = []
        out_chars = []

        for ch in s:
            if ch in opening:
                stack.append(ch)
                out_chars.append(ch)
            elif ch in closing:
                if stack and pairs[stack[-1]] == ch:
                    # normal match
                    stack.pop()
                    out_chars.append(ch)
                else:
                    # mismatched closing: close what's actually open first
                    while stack and pairs[stack[-1]] != ch:
                        out_chars.append(pairs[stack.pop()])
                    if stack and pairs[stack[-1]] == ch:
                        stack.pop()
                        out_chars.append(ch)
                    else:
                        # stray unmatched closing: drop it (safer than inserting an opener)
                        # pass
                        continue
            else:
                out_chars.append(ch)

        # close any remaining openings
        while stack:
            out_chars.append(pairs[stack.pop()])

        return ''.join(out_chars)

    @classmethod
    def _strip_trailing_commas(cls, s: str) -> str:
        """Remove trailing commas before a closing '}' or ']'."""
        return cls.TRAILING_COMMA_RE.sub(r'\1', s)

    @classmethod
    def _fix_equals_as_colon(cls, s: str) -> str:
        """
        Turn key=value into "key": value.
        - If the key is quoted already ("key" or 'key'), keep it and replace '=' with ':'.
        - If the key is an unquoted identifier (foo), wrap it with double quotes.
        """
        def repl(m: re.Match) -> str:
            q = m.group('q')   # quoted key if present
            u = m.group('u')   # unquoted identifier key if present
            if q is not None:
                # Keep the quoted key as-is, just replace '=' with ':'
                return f'{q}: '
            else:
                # Add quotes for the unquoted identifier key
                return f'"{u}": '
        return cls.EQUALS_AS_COLON_RE.sub(repl, s)

    @classmethod
    def _quote_unquoted_keys(cls, s: str) -> str:
        """
        Quote unquoted object keys: `{foo: 1}` → `{"foo": 1}`.
        Only matches keys that look like identifiers and are followed by a colon.
        """
        return cls.UNQUOTED_KEYS_RE.sub(cls._quote_unquoted_keys_repl, s)

    @classmethod
    def _quote_unquoted_keys_repl(cls, m: re.Match) -> str:
        return f'{m.group("prefix")}"{m.group("key")}":'

    @classmethod
    def _escape_internal_quotes(cls, s: str) -> str:
        """
        Heuristically escape unescaped double quotes inside string values.
        Matches a double quote that is surrounded by non-boundary characters.
        """
        return cls.UNESCAPED_QUOTE_RE.sub(r'\"', s)

    # ========================= Token conversions =========================

    @classmethod
    def _to_python_bools_nulls(cls, s: str) -> str:
        """Convert JSON true/false/null to Python True/False/None (for literal_eval)."""
        return cls.JSON_TO_PY_RE.sub(cls._to_python_bools_nulls_repl, s)

    @classmethod
    def _to_python_bools_nulls_repl(cls, m: re.Match) -> str:
        word = m.group(0)
        return {'true': 'True', 'false': 'False', 'null': 'None'}[word]

    @classmethod
    def _to_json_bools_nulls(cls, s: str) -> str:
        """Convert Python True/False/None to JSON true/false/null (utility, not required in main flow)."""
        return cls.PY_TO_JSON_RE.sub(cls._to_json_bools_nulls_repl, s)

    @classmethod
    def _to_json_bools_nulls_repl(cls, m: re.Match) -> str:
        word = m.group(0)
        return {'True': 'true', 'False': 'false', 'None': 'null'}[word]

    # ========================= Quote normalization =========================

    @classmethod
    def _single_to_double_quotes(cls, s: str) -> str:
        """
        Convert single-quoted strings to double-quoted strings (conservative heuristic).
        This is NOT a full JSON parser; it targets common LLM outputs.
        """
        return cls.SINGLE_QUOTED_STRING_RE.sub(cls._single_to_double_quotes_repl, s)

    @classmethod
    def _single_to_double_quotes_repl(cls, m: re.Match) -> str:
        inner = m.group(1)
        inner = inner.replace('\\"', '"')   # Unescape existing double quotes
        inner = inner.replace('"', '\\"')   # Escape new double quotes
        return f'"{inner}"'
