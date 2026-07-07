# Developer Guideline

Author: [SwordJack](https://github.com/SwordJack)

[TOC]

## Multilingual

- 汉语 [developer.zh-cn.md](developer.zh-cn.md)
- English [developer.md](developer.md)

## Source Code

### About Directories

The operating system, file system, and log printing methods in this project are all stored in the [`core`](/isobase/core/) directory. For operations involving such content, please refer to it instead of "reinventing the wheel".

### Code Conventions

- The [isobase](/isobase/) directory of this project is the source code directory. The code standards refer to the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html). For each class, method, function, etc., its parameter type and return value type should be clearly marked, and effectively explained in the comments.
- For the code involving string values ​​in the project, please use double quotes as the main quotes.
- At the beginning of each Python file, please add document information. For example, in [logger.py](/isobase/core/logger.py), I added the following format information. Please refer to this example for the subsequent document information format.

```python
#! python3
# -*- encoding: utf-8 -*-
"""
Format logger output using python logging and colorlog module.

Reference: https://github.com/jyesselm/dreem/blob/main/dreem/logger.py

@File   :   logger.py
@Created:   2025/04/01 16:16
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""
```

## Unit test

- The [test](/test/) directory of this project is the unit test directory. The directory structure in the unit test directory should be consistent with the directory structure in the source code directory. For a unit test code file of a source code file, its file name should be the source code file name plus the `test_*` prefix, such as [`logger.py`](/isobase/core/logger.py), its unit test code file is [`test_logger.py`](/test/core/test_logger.py).
- Unit testing is performed through Pytest. For detailed usage, please refer to [Pytest Documentation](https://docs.pytest.org/en/stable/).
- In general, each source code file (except configuration files) should have a corresponding unit test file. Unit tests should test the general operation, exception handling, edge cases, etc. of the source code to ensure the robustness of the project code.

## Version control

- This project is version controlled through Git.
- The main branch of the project is the [`main`](https://github.com/landspark/IsoBase/tree/main) branch, which is the code version released to the public or used by external projects. It is currently controlled by the owner of the project.
- Project development is generally carried out around the [`develop`](https://github.com/landspark/IsoBase/tree/develop) branch. Developers should create new branches based on this branch for development and submit a [Pull Request](https://github.com/landspark/IsoBase/pulls) to apply for merging the created branch into the `develop` branch.
- The naming convention for the newly created branch by the developer is: `<your_name>/<type>_<description>`. For example, if **Tom** wants to add a `feature` of **eating watermelon** to the project, he can name his branch as: `tom/feature_eat_watermelon`.
- There are several types of branche prefixes:

| Prefix      | Full name     | Meaning                                                                           |
| ----------- | ------------- | --------------------------------------------------------------------------------- |
| `feature_*` | feature       | Add new features                                                                  |
| `fix_*`     | fix           | Fix a bug (low priority, will be merged into `develop`)                           |
| `hotfix_*`  | hot fix       | Urgently fix a serious bug (high priority, should be merged directly into `main`) |
| `docs_*`    | documentation | Add or improve documentation content, multilingual content, etc. to the project   |
| `test_*`    | test          | Add or modify test-related content                                                |
| `chore_*`   | chore         | Miscellaneous and configuration information maintenance                           |

- When a branch is merged, it will be renamed: for example, after the `tom/feature_eat_watermelon` branch is merged, it will be renamed to `zarchive/tom/feature_eat_watermelon` to indicate that this branch has completed its mission and should no longer be activated.

## Packaging and Distribution

This project uses [rattler-build](https://rattler-build.prefix.dev/latest/) as the primary build tool for Conda package distribution. The packaging configuration is located in the `.conda/` directory.

### Recipe Format (V1)

The project uses the V1 recipe format (`recipe.yaml`), which is the modern format defined by [CEP-13](https://github.com/conda/ceps/blob/main/cep-0013.md). Key syntax rules:

- **Context Variables**: Define variables in the `context` block for reuse throughout the recipe.
- **Jinja2 Templating**: Use `${{ variable }}` syntax for variable interpolation.
- **Environment Variables**: Access external environment variables using `env.get("VAR_NAME", default="value")`.

Example:
```yaml
context:
  name: "isobase"
  version: ${{ env.get("BUILD_VERSION", default="0.0.0") }}
  python_min: "3.12"
```

### Environment Variable Access

**Correct Syntax (V1 Recipe)**:
```yaml
version: ${{ env.get("BUILD_VERSION", default="0.0.0") }}
```

**Incorrect Syntax (Python-style, does NOT work)**:
```yaml
version: ${{ environ.get("BUILD_VERSION", default="0.0.0") }}  # Wrong!
```

The `env.get()` function is the official method for accessing environment variables in rattler-build recipes. The `environ` namespace does not exist in rattler-build's Jinja2 context.

### Build Script Syntax

The `build.script` field supports multi-line shell scripts using YAML's `|` (literal block scalar):

```yaml
build:
  noarch: python
  number: 0
  script: |
    echo "Building version: ${{ version }}"
    python -m pip install . -vv
```

### Source Specification

For local development builds, use `path`:
```yaml
source:
  path: ..
```

For conda-forge submission, use `url` with SHA256 checksum:
```yaml
source:
  url: https://github.com/landspark/IsoBase/archive/refs/tags/${{ version }}.tar.gz
  sha256: <checksum>
```

### GitHub Actions Integration

The CI/CD workflow passes the version to rattler-build via environment variables:

```yaml
- name: Build Conda Package
  env:
    BUILD_VERSION: ${{ env.VERSION }}
  run: |
    rattler-build build \
      --recipe .conda/recipe.yaml \
      --output-dir output_bld
```

This allows the recipe to dynamically read the version from the environment during build time, enabling automated version management without manual recipe editing.

### References

- [Rattler-build Documentation](https://rattler-build.prefix.dev/latest/)
- [CEP-13: Recipe Format Specification](https://github.com/conda/ceps/blob/main/cep-0013.md)
- [Conda-forge Maintainer Guide](https://conda-forge.org/docs/maintainer/)
