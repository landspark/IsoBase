# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Rule You Should Follow (Hard Constraints)

### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.
  Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.
- **Cleanup**: Remove imports/variables/functions that YOUR changes made unused. Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**
Transform tasks into verifiable goals:

- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:

```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

## Quick Context (Cheatsheet)

**What ARK is**: Agile Resource Kernel (ARK) is a Python automation framework built around a JSON-driven workflow engine with AI/LLM integration. It is a library (no `setup.py`), so code is imported relative to the repo root rather than pip-installed.

**Environment & Testing**:

- Python 3.13 via conda. Setup: `conda env create -f environment.yml && conda activate ark`
- _Note: `Pillow` is required but missing from `environment.yml`. Install it manually if needed._
- Tests use `pytest` and **must** be run from the repo root:
  - `python -m pytest` (all tests)
  - `python -m pytest test/workflow/test_workflow.py` (specific file)

**Conventions**:

- Type-annotate every parameter/return value with docstrings.
- Double quotes (`"`) are the primary string quote.
- Files begin with a docblock (`@File`, `@Created`, `@Author`, `@Contact`).
- Tests mirror source tree under `test/` (e.g. `test/core/test_logger.py`).
- Branches: `main` (release), `develop` (integration). Format: `<name>/<type>_<desc>`.

## Architectural Boundaries

### `ark/llm` — LLM Client & Tool Engine

This module is strictly separated into Data -> Logic -> Adapters to maintain vendor neutrality.

- **`entities.py` (Data DTOs)**: Defines `LLMResponse`, `TokenUsage`, and `ToolCall`. These represent pure data boundaries. `ToolCall` is the universal representation of what the model wants to do.
- **`tools/` (Engine Logic)**: `FunctionTool` and `ToolSet`. Represents local capabilities. `FunctionTool` automatically extracts JSON schemas from Python callable signatures and docstrings (manual overrides allowed). `ToolSet` blindly executes `ToolCall` requests; it has zero knowledge of OpenAI or Anthropic formats.
- **`providers/` (Translators/Adapters)**: `OpenAIChat` and `AnthropicMessages`. These handle the actual network requests, multi-turn loops, and streaming accumulation. They are responsible for translating the neutral `FunctionTool` and `ToolCall` objects into their specific vendor's JSON dialects right before transmission.

### `ark/workflow` — JSON Workflow Engine

A nested tree of execution units driven by JSON "order" files.

- **Execution Hierarchy**: `ExecutionEntity` (base, handles SQLite sync) -> `WorkUnit` (single step mapped to a Python API) -> `WorkFlow` (sequential list of units) -> `GeneralWorkUnit` (outermost entry point).
- **APIs**: Registered via `UNIT_API_MAPPER` in `config.py` at import time.
- **Data Flow**: Handled via `intermediate_data_mapper` and `outer_data_mappers`. Variables wrapped in backticks (e.g. ``"`var_name`"``) are dynamically resolved at runtime.
- **Persistence**: Statuses auto-sync to SQLite via `database.py`. `GeneralWorkUnit` can pickle snapshot state (`save_snapshot=True`) for resume/continue computing.

### Other Modules

- **`ark/core`**: OS/file/logging (`colorlog`)/image helpers. Reuse these; do not reinvent them.
- **`ark/locale`**: Resolves locales from phone numbers (`phonenumbers`) and country metadata (`pycountry`).
