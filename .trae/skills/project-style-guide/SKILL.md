---
name: "project-style-guide"
description: "Enforces project-specific coding standards based on .pre-commit-config.yaml and .flake8. Invoke when generating code, formatting files, or checking style compliance."
---

# ScaleTorch Project Style Guide

此 Skill 包含了 ScaleTorch 项目的代码风格和 lint 规范，基于 `.pre-commit-config.yaml` 和 `.flake8` 配置。

## 核心规范

### 1. Python 代码风格 (Flake8 & Yapf)
*   **行长度限制**: **79 字符** (严格遵守，与 PEP 8 一致)。
*   **忽略规则**:
    *   `W503`: Line break before binary operator
    *   `W504`: Line break after binary operator
    *   `E251`: Unexpected spaces around keyword / parameter equals
    *   `E501`: Line too long (虽然设置了 max-line-length，但某些情况可能忽略)
    *   `E126`: Continuation line over-indented for hanging indent
*   **Import 排序**: 必须使用 `isort` 风格进行排序。
    *   Application import names: `scaletorch`
*   **格式化工具**: 使用 `yapf` 风格。

### 2. 文件格式与通用规则 (Pre-commit hooks)
*   **换行符**: 强制使用 **LF** (`mixed-line-ending --fix=lf`)。
*   **文件结尾**: 必须以且仅以一个换行符结尾 (`end-of-file-fixer`)。
*   **尾部空格**: 必须移除行尾多余空格 (`trailing-whitespace`)。
*   **编码声明**: **移除** 文件头的编码声明 (如 `# -*- coding: utf-8 -*-`) (`fix-encoding-pragma --remove`)。
*   **字符串引号**: 统一字符串引号风格 (`double-quote-string-fixer`)。

### 3. 特定文件检查
*   **YAML**: 必须符合 YAML 语法 (`check-yaml`)。
*   **Requirements.txt**: 必须排序 (`requirements-txt-fixer`)。
*   **合并冲突**: 检查并禁止提交包含合并冲突标记的代码 (`check-merge-conflict`).

## 使用建议
当生成或修改代码时，请务必：
1.  **检查行长**：确保生成的代码不超过 79 列。
2.  **排序 Imports**：将标准库、第三方库和本地库 (`scaletorch`) 分组并排序。
3.  **清理格式**：确保没有尾部空格，且文件以 LF 结尾。
4.  **遵循 Yapf 风格**：在复杂的表达式或参数列表中遵循 Yapf 的格式化习惯。
