# ScaleTorch Project Rules

## Project Overview
ScaleTorch is a PyTorch Training Toolkit focused on implementing and demonstrating distributed training strategies (FSDP, DDP, Tensor Parallelism, Pipeline Parallelism, etc.).

## Technology Stack
- **Language**: Python 3.x
- **Framework**: PyTorch
- **Dependencies**: numpy, tqdm, pyyaml, packaging

## Coding Standards

### Style & Formatting
- **Linter**: `flake8`
  - Max line length: 79 characters
  - Ignored errors: W503, W504, E251, E501, E126
- **Formatter**: `yapf` (via pre-commit) and `isort` for imports.
- **Quotes**: Use double quotes (`"`) for strings.
- **Imports**:
  - Absolute imports preferred.
  - Sorted by `isort`.
  - Application import names: `scaletorch`

### Pre-commit Hooks
The project uses `pre-commit` to enforce standards. Ensure hooks pass before committing:
- `trailing-whitespace`
- `check-yaml`
- `end-of-file-fixer`
- `requirements-txt-fixer`
- `double-quote-string-fixer`
- `check-merge-conflict`
- `mixed-line-ending` (LF)

## Project Structure
- `scaletorch/`: Main package containing core logic.
  - `dist/`: Distributed training utilities.
  - `parallel/`: Parallelism implementations (CP, DP, PP, TP).
  - `models/`: Model architectures (Llama, MoE, etc.).
  - `trainer/`: Training loop and configuration.
  - `data/`: Data loading and processing.
- `examples/`: Example scripts for different strategies (FSDP, ImageNet, etc.).
- `tests/`: Unit and integration tests (`test_*.py`).
- `scripts/`: Utility and launch scripts.
- `doc/`: Documentation.

## Workflow & Commands

### Setup
1. Install dependencies: `pip install -r requirements.txt` (if available) or `pip install -e .`
2. Install pre-commit hooks: `pre-commit install`

### Testing
Run tests using the provided script:
```bash
python run_tests.py
```
Or manually with unittest:
```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

### Development
- **New Features**: Add corresponding tests in `tests/` and examples in `examples/` if applicable.
- **Documentation**: Update `README.md` or `doc/` when adding significant features.
- **Parallelism**: When working on parallelism modules (`scaletorch/parallel`), ensure compatibility with `pg_manager` and standard PyTorch distributed primitives.

## Key Conventions
- **Distributed Context**: Use `scaletorch.dist` utilities for handling distributed environments.
- **Models**: Models should inherit from `torch.nn.Module`.
- **Config**: Configuration is often handled via YAML or argument parsing in examples.
