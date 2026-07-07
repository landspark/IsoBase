# 开发者指南

作者：[SwordJack](https://github.com/SwordJack)

[TOC]

## 多语言

- 汉语 [developer.zh-cn.md](developer.zh-cn.md)
- English [developer.md](developer.md)

## 源代码

### 关于文件夹

本项目中涉及操作系统、文件系统、日志打印的方法，均存放于 [`core`](/isobase/core/) 目录中，涉及此类内容的操作，请从中引用，而非重新“造轮子”。

### 代码规范

- 本项目的 [isobase](/isobase/) 目录为源代码目录，代码规范参照 [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) 执行，对于每一个类、方法、函数等，都应当清晰标注其参数类型、返回值类型，并在注释中加以有效说明。
- 项目中涉及字符串值的代码，请使用双引号作为主要引号。
- 在每一个 Python 文件的开头，请加注文档信息，例如在 [logger.py](/isobase/core/logger.py) 中，本人加注了如下格式信息。后续的文档信息格式请亦参考此例。

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

## 单元测试

- 本项目的 [test](/test/) 目录为单元测试目录，单元测试目录中的目录结构应与源代码目录中的目录结构相一致，对于某个源代码文件的单元测试代码文件，其文件名应为源代码文件名前加上 `test_*` 前缀，如 [`logger.py`](/isobase/core/logger.py)，其单元测试代码文件为 [`test_logger.py`](/test/core/test_logger.py)。
- 通过 Pytest 进行单元测试，其详细使用方法可参考 [Pytest Documentation](https://docs.pytest.org/en/stable/)。
- 一般情况下，每一个源代码文件（除配置文件外），都应当有一个与之相对应的单元测试文件，单元测试应当测试源代码的一般运行、异常处理、边缘案例等，以保证本项目代码的稳健性（robustness）。

## 版本控制

- 本项目通过 Git 进行版本控制。
- 项目主分支为 [`main`](https://github.com/landspark/IsoBase/tree/main) 分支，为本项目对外发布或供外部项目应用的代码版本，目前由本项目的所有者控制。
- 项目的开发一般情况下围绕 [`develop`](https://github.com/landspark/IsoBase/tree/develop) 分支进行，开发者请基于此分支创建新分支进行开发，并提交 [Pull Request](https://github.com/landspark/IsoBase/pulls) 以申请将所创建的分支合并入 `develop` 分支中。
- 开发者新创建的分支，其命名规范为：`<名字>/<类型>_<描述>`，例如：**Tom** 想要给项目添加一个 **吃西瓜（eat watermelon）** 的功能，那么，他就可以如此命名他的分支：`tom/feature_eat_watermelon`。
- 分支的类型有以下若干种：

| 前缀        | 全名          | 含义                                                        |
| ----------- | ------------- | ----------------------------------------------------------- |
| `feature_*` | feature       | 添加新功能                                                  |
| `fix_*`     | fix           | 修复某个 bug （低优先级，合并到 `develop` 中）              |
| `hotfix_*`  | hot fix       | 紧急修复某个严重 bug （高优先级，需要直接合并到 `main` 中） |
| `docs_*`    | documentation | 给项目添加或完善文档内容、多语言内容等                      |
| `test_*`    | test          | 添加或修改测试相关内容                                      |
| `chore_*`   | chore         | 杂项和配置信息维护                                          |

- 当一个分支被合并后，其会被重命名：如 `tom/feature_eat_watermelon` 分支被合并后，它会被重命名为 `zarchive/tom/feature_eat_watermelon`，以表示此分支已经完成其使命，不应再被启用。

## 打包和发布

本项目使用 [rattler-build](https://rattler-build.prefix.dev/latest/) 作为 Conda 包分发的主要构建工具。打包配置位于 `.conda/` 目录中。

### 配方格式（V1）

本项目使用 V1 配方格式（`recipe.yaml`），这是由 [CEP-13](https://github.com/conda/ceps/blob/main/cep-0013.md) 定义的现代格式。关键语法规范：

- **上下文变量（Context Variables）**：在 `context` 块中定义变量，以便在整个配方中重复使用。
- **Jinja2 模板**：使用 `${{ variable }}` 语法进行变量插值。
- **环境变量**：使用 `env.get("VAR_NAME", default="value")` 访问外部环境变量。

示例：
```yaml
context:
  name: "isobase"
  version: ${{ env.get("BUILD_VERSION", default="0.0.0") }}
  python_min: "3.12"
```

### 环境变量访问

**正确语法（V1 配方）**：
```yaml
version: ${{ env.get("BUILD_VERSION", default="0.0.0") }}
```

**错误语法（Python 风格，**不起作用**）**：
```yaml
version: ${{ environ.get("BUILD_VERSION", default="0.0.0") }}  # 错误！
```

`env.get()` 函数是在 rattler-build 配方中访问环境变量的官方方法。`environ` 命名空间在 rattler-build 的 Jinja2 上下文中不存在。

### 构建脚本语法

`build.script` 字段支持使用 YAML 的 `|`（字面块标量）的多行 shell 脚本：

```yaml
build:
  noarch: python
  number: 0
  script: |
    echo "Building version: ${{ version }}"
    python -m pip install . -vv
```

### 源代码规范

对于本地开发构建，使用 `path`：
```yaml
source:
  path: ..
```

对于 conda-forge 提交，使用带有 SHA256 校验和的 `url`：
```yaml
source:
  url: https://github.com/landspark/IsoBase/archive/refs/tags/${{ version }}.tar.gz
  sha256: <checksum>
```

### GitHub Actions 集成

CI/CD 工作流通过环境变量将版本传递给 rattler-build：

```yaml
- name: Build Conda Package
  env:
    BUILD_VERSION: ${{ env.VERSION }}
  run: |
    rattler-build build \
      --recipe .conda/recipe.yaml \
      --output-dir output_bld
```

这允许配方在构建时从环境动态读取版本，实现自动化版本管理，无需手动编辑配方。

### 参考资料

- [Rattler-build 文档](https://rattler-build.prefix.dev/latest/)
- [CEP-13：配方格式规范](https://github.com/conda/ceps/blob/main/cep-0013.md)
- [Conda-forge 维护者指南](https://conda-forge.org/docs/maintainer/)
