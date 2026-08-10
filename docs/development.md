# Development Guide

## Environment

The project uses `uv` for Python dependency and environment management.

Synchronize the environment:

```bash
uv sync
```

Run Python commands:

```bash
uv run python <script>
```

## Testing

Run all tests:

```bash
uv run pytest
```

Run a specific component:

```bash
uv run pytest src/ml_platform/data -v
```

```bash
uv run pytest src/ml_platform/training -v
```

## Code Quality

### Ruff

```bash
uv run ruff check .
```

### Black

Check formatting:

```bash
uv run black --check .
```

Apply formatting:

```bash
uv run black .
```

### MyPy

```bash
uv run mypy src/
```

## DVC

DVC pipelines are located under:

```text
src/pipelines/
```

Run F1:

```bash
cd src/pipelines/f1
uv run dvc repro
```

Run California Housing:

```bash
cd src/pipelines/ca_house_prediction
uv run dvc repro
```

Check pipeline status:

```bash
uv run dvc status
```

## Development Workflow

A typical development cycle is:

```text
Create / Update Story
        ↓
Implement Code
        ↓
Add Tests
        ↓
Run Ruff
        ↓
Run Black
        ↓
Run MyPy
        ↓
Run Pytest
        ↓
Run DVC Pipeline
        ↓
Commit
        ↓
Pull Request
        ↓
CI Validation
```

## CI

GitHub Actions validates pull requests.

The CI pipeline runs:

```text
Install Dependencies
        ↓
Ruff
        ↓
Black
        ↓
MyPy
        ↓
Pytest
```

A pull request should pass all checks before being merged.

## Adding a Platform Component

When adding a reusable platform capability:

1. Create the component under `src/ml_platform/`.
2. Define a clear public interface.
3. Integrate centralized configuration when appropriate.
4. Integrate centralized logging when appropriate.
5. Use centralized exceptions.
6. Add unit tests.
7. Add component documentation.
8. Update the root README if the architecture changes.

## Adding a New Pipeline

Create a new directory:

```text
src/pipelines/<project>/
```

A typical pipeline contains:

```text
<project>/
├── README.md
├── dvc.yaml
├── get_data.py
├── prepare_data.py
├── train_model.py
└── evaluate_model.py
```

The pipeline should consume reusable components from `ml_platform` instead of duplicating platform functionality.

## Documentation

Documentation follows the same separation of concerns as the code.

```text
README.md
    ↓
Project overview

docs/
    ↓
Cross-cutting documentation

ml_platform/<component>/README.md
    ↓
Component documentation

pipelines/<project>/README.md
    ↓
Model-specific documentation
```
