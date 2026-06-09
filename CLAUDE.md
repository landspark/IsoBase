# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What ARK is

Agile Resource Kernel (ARK) is a Python automation framework for corporate resource management, built around a JSON-driven workflow engine with AI/LLM integration. It is a library (no `setup.py`/`pyproject.toml`), so code is imported relative to the repo root rather than pip-installed.

## Environment & commands

The project uses a conda environment named `ark` (Python 3.13). Set it up with:

```bash
conda env create -f environment.yml
conda activate ark
```

Tests run with pytest **from the repo root** (the `ark` package must be importable from the current directory — there is no installed package):

```bash
python -m pytest                                          # all tests
python -m pytest test/workflow/test_workflow.py           # one file
python -m pytest test/workflow/test_workflow.py::test_general_reload_pass   # one test
```

There is no build step and no configured linter. Code style follows the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) by convention (see Conventions below).

Known dependency gap: `Pillow` (`PIL`) is imported by `ark/core/image_service.py` and the LLM module but is **not** declared in `environment.yml`. A fresh env from that file will fail to `import ark.llm` until Pillow is installed.

## Module map

Four packages under `ark/`, each re-exporting its public surface through `__init__.py`:

- `ark.core` — shared infrastructure: global `LOGGER` (colorlog), `file_system` helpers, `image_service`. Per the developer guide, OS/file/logging operations belong here; reuse these rather than reinventing them.
- `ark.workflow` — the JSON workflow engine (the heart of the project).
- `ark.llm` — OpenAI-compatible chat client with tool calling and multimodal input.
- `ark.locale` — locale resolution from phone numbers and country/language metadata.

## Workflow engine architecture (`ark/workflow`)

This is the most intricate subsystem and spans `workflow.py`, `config.py`, and `database.py`. A workflow is defined by a JSON "order" file describing a nested tree of work units.

**Execution entity hierarchy** (all in `workflow.py`):

- `ExecutionEntity` — base class: identification, status, tree navigation, SQLite sync, pickle snapshot/load.
- `WorkUnit` — a single step. Its `api` key is mapped to a Python callable via the global `UNIT_API_MAPPER` (in `config.py`). Performs self-inspection of arguments against the callable's signature at init.
- `WorkFlow` — an ordered list of `WorkUnit`s; executes them sequentially, collecting errors without halting on non-fatal ones.
- `ControlWorkUnit` → `IterativeWorkUnit` → `LoopWorkUnit` / `ClusterBatchWorkUnit` — control-flow units that wrap a loop/batch body as sub-`WorkFlow`s. `ClusterBatchWorkUnit.execute` is still a stub copied from the loop logic.
- `GeneralWorkUnit` — the outermost entry point that carries the entire workflow. Use `GeneralWorkUnit.from_json_filepath(...)` to load and `.execute()` to run.

**How APIs are registered**: callables are added to `UNIT_API_MAPPER` (a module-global dict) by updating it at import time — see `WIDGET_API_MAPPER` in `config.py` and `test/workflow/pseudo_api.py` for the pattern. A `WorkUnit`'s `api` string must already be present in this dict at init or initialization fails.

**Data flow between units**: each `WorkFlow` has an `intermediate_data_mapper` (this layer's results) plus `outer_data_mappers` (inherited from parent layers). In JSON args, a value wrapped in backticks like `` "`fruit_1`" `` is a _variable reference_ resolved from these mappers at execution time (see `JsonConfigVariableFormatter` in `config.py`), not a literal. `store_as` names the key a unit's return value is written to.

**Status & control flow**: `StatusCode` (in `config.py`) defines all execution states and the status _groups_ (e.g. `error_or_pause_statuses`, `skippable_statuses`, `unexecutable_statuses`) that drive how errors and pauses propagate up the tree and which units are skipped on re-run. Setting `.status` to an executed state auto-syncs to SQLite.

**Persistence & continue-computing**:

- SQLite (via SQLAlchemy, `database.py`) records each entity's status keyed by its `identifier`, which is `"{api_key}@{locator}"` (e.g. `checkpoint_error@4:loop_1:0`). `locate_by_identifier` / `locate` walk the tree by this locator.
- `GeneralWorkUnit` with `save_snapshot=True` pickles its full state on exit. Reload an updated JSON via `reload_json_filepath(...)`; only units whose args changed are marked `SUSPECIOUS_UPDATES` and re-run — the architecture (unit tree shape) must be unchanged or reload is rejected.

**Working-directory side effects**: `GeneralWorkUnit` and iterative units `chdir` into the working directory during execution and back to the launch directory (`BASE_DIRECTORY`) afterward. When `save_snapshot=False` and `debug=False`, `from_dict` sleeps 5s with a cancel warning before proceeding.

The `ark/workflow/README.md` has a runnable quick-start order and the load/reload API. Test fixtures in `test/workflow/data/*.json` are good concrete examples.

## LLM module architecture (`ark/llm`)

- `providers/base.py` — `BaseLLMClient` abstract interface (`generate`, `generate_stream`) plus `build_user_message_content` for multimodal messages.
- `providers/openai_chat.py` — `OpenAIChat`, the concrete OpenAI-compatible client. `ask()` runs a multi-turn tool-calling loop (bounded by `max_tool_rounds`) over the lower-level `generate`/`generate_stream`; both streaming and non-streaming paths are supported and history is preserved on error.
- `entities.py` — `LLMResponse` and `TokenUsage` dataclasses; every client method returns the standardized `LLMResponse`.
- `tools/base.py` — `FunctionTool`, `FunctionToolParameter`, `ToolSet` implement OpenAI-style function calling. A tool's Python callable is supplied via `tool_function_mapper`; `ToolSet.execute_tool_calls` runs them and formats `role: "tool"` messages.
- Multimodal images go through `ark/core/image_service.convert_image_to_data_url` (PIL image → PNG data URL).

LLM tests mock the `OpenAI` client (`unittest.mock.patch` on `ark.llm.providers.openai_chat.OpenAI`); they never hit a live API.

## Locale module architecture (`ark/locale`)

`LocaleManager` (`locale_manager.py`) is both a resolver and a per-locale metadata container. It resolves a locale from a raw phone number (`phone.py`, backed by `phonenumbers`) and looks up country/language names and flags (`country.py`, backed by `pycountry`). Internal codes are lowercase-underscore (`zh_cn`); `code_bcp47` gives the `zh-CN` form.

## Conventions

From `docs/developer.md` and `.gemini/GEMINI.md`:

- Type-annotate every function/method parameter and return value, with docstrings explaining them.
- Use double quotes as the primary string quote.
- Start each Python file with a header docblock containing `@File`, `@Created`, `@Author`, `@Contact` (see any existing file, e.g. `ark/core/logger.py`).
- Favor simple, minimal changes; avoid hardcoding constant strings/numbers — many are centralized as module-level constants (e.g. the `*_KEY` / `*_LABEL` names at the top of `workflow.py`).
- Tests mirror the source tree under `test/` with a `test_` filename prefix (`ark/core/logger.py` → `test/core/test_logger.py`).

## Version control

- `develop` is the integration branch for ongoing work; `main` is the public release branch.
- Branch naming: `<your_name>/<type>_<description>`, where `<type>` is `feature`, `fix`, `hotfix`, or `docs` (e.g. `tom/feature_eat_watermelon`). `hotfix_*` branches target `main` directly; the rest target `develop`.
- Merged branches are renamed with a `zarchive/` prefix to mark them retired.
