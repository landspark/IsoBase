# Overload vs Dispatch in Python: A Pragmatic Engineering Guide

In dynamically typed languages like Python, developers constantly balance two competing needs:

1. **Static Developer Experience (DX)**: Providing perfect autocomplete and type warnings in IDEs (VSCode/PyCharm) before the code ever runs.
2. **Dynamic Runtime Elegance**: Writing clean, extensible logic that avoids massive `if isinstance(...)` blocks when the code actually executes.

This document serves as an architectural reference, capturing the trade-offs between `typing.overload` and `functools.singledispatch`, and how `IsoBase` makes pragmatic choices between them.

---

## 1. `@overload`: The Static "Mirage"

Introduced in Python 3.5 via the `typing` module, `@overload` solves a specific problem: **how to tell the IDE that a function's return type changes depending on its input parameters.**

### How It Works

You write multiple "stub" functions adorned with `@overload`. These stubs contain no actual logic (just `pass` or `...`). They are strictly for the type checker. You must follow them with one "fat" implementation function that actually does the work.

```python
from typing import overload, Union

@overload
def parse_data(data: str) -> dict: ...

@overload
def parse_data(data: int) -> list: ...

# The "Fat Body" (Runtime Implementation)
def parse_data(data: Union[str, int]) -> Union[dict, list]:
    if isinstance(data, str):
        return {"result": data}
    elif isinstance(data, int):
        return [data, data]
    raise TypeError("Unsupported type")
```

### The Pain Points of Overload

1. **The Spaghetti Code Problem**: While the IDE is happy, your runtime code is forced into a single, massive function body filled with `if isinstance(...)` branches. This violates the Open-Closed Principle.
2. **Type Checker Tyranny**: Strict type checkers like `mypy` demand that your implementation signature perfectly subsumes all overload signatures. Minor mismatches lead to confusing static errors.
3. **Overlapping Signatures**: If two overloads have similar parameter types but different return types (e.g., `Any` vs `int`), the IDE may fail to resolve the correct one, collapsing back to a messy `Union`.

---

## 2. `@singledispatch`: The Runtime "Router"

To solve the "Fat Body" spaghetti code issue, Python's standard library provides `@functools.singledispatch`.

### How It Works

It allows you to break a massive `if-else` block into clean, isolated functions. Python automatically routes the execution to the correct function based on the **type** of the first argument at runtime.

```python
from functools import singledispatch

@singledispatch
def parse_data(data):
    raise TypeError(f"Unsupported type: {type(data)}")

@parse_data.register(str)
def _(data: str) -> dict:
    return {"result": data}

@parse_data.register(int)
def _(data: int) -> list:
    return [data, data]
```

### The Strengths and Weaknesses

- **Strengths**: Perfect runtime decoupling. External developers can add support for their own custom classes simply by importing your function and using `@parse_data.register(CustomClass)`, completely without modifying your source code.
- **Weaknesses**: **Terrible IDE Support.** Because the entry point `parse_data(data)` lacks a specific return type, the IDE has no idea what will be returned when the user types `parse_data("hello")`.

---

## 3. The Ultimate Pattern: Overload + Dispatch Fusion

For high-end framework development (like Pydantic or FastAPI core internals) dealing with dozens of data types, the industry best practice is to **combine both**.

Use `@overload` as the "Face" (to trick the IDE into providing perfect hints) and `@singledispatch` as the "Engine" (to cleanly route execution at runtime).

```python
from functools import singledispatch
from typing import overload

# 1. The Face: Static Stubs for the IDE
@overload
def process(data: str) -> dict: ...

@overload
def process(data: int) -> list: ...

# 2. The Engine Entrypoint: Dynamic Runtime Dispatcher
@singledispatch
def process(data):
    raise TypeError("Unsupported type")

# 3. The Engine Workers: Concrete Implementations
@process.register(str)
def _(data: str) -> dict:
    return {"status": "ok", "value": data}

@process.register(int)
def _(data: int) -> list:
    return [data]
```

With this architecture:

- The IDE provides precise autocomplete based on the `@overload` stubs.
- The runtime execution routes seamlessly via `@singledispatch` without a single `isinstance` check.

---

## 4. Pragmatism in IsoBase: Why We Chose `if-else` for `ask()`

In `IsoBase` (`isobase/llm/providers/base.py`), the `ask` method's return type changes based on the `stream` parameter:

- `stream=False` ➔ returns `LLMResponse`
- `stream=True` ➔ returns `Iterator[LLMResponse]`

We used the `@overload` decorator to fix the IDE autocomplete:

```python
@overload
def ask(self, prompt: str, stream: Literal[False] = False, **kwargs) -> LLMResponse: ...

@overload
def ask(self, prompt: str, stream: Literal[True], **kwargs) -> Iterator[LLMResponse]: ...
```

However, for the actual implementation, we deliberately **avoided** dynamic dispatch and stuck to a simple `if-else`:

```python
def ask(self, prompt: str, stream: bool = False, **kwargs):
    if stream:
        return self._ask_loop_stream(...)
    else:
        return self._ask_loop(...)
```

### Why not use Dispatch here?

1. **Value vs. Type Routing**: `@singledispatch` only routes based on **types** (e.g., `str` vs `int`). It cannot route based on specific **values** (e.g., `True` vs `False`).
2. **Avoiding Over-Engineering**: While we could have implemented a Custom Strategy Pattern (e.g., mapping boolean values to functions in a dictionary: `router = {True: _ask_stream, False: _ask}`), doing so for a simple binary choice obscures the code.
3. **KISS Principle**: When faced with dozens of type branches, dispatch is king. When faced with exactly two states (`True`/`False`), `@overload` + `if-else` remains the most readable, Pythonic, and pragmatic choice.
