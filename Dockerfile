# syntax=docker/dockerfile:1

# ---- Build stage: resolve and install locked runtime dependencies ----
FROM python:3.12-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv

RUN pip install --no-cache-dir uv==0.11.32

WORKDIR /app

# Install dependencies before copying the source so this layer is reused
# when only application code changes. Only the main dependencies are
# installed: dev tools and pipeline tools (DVC, Kaggle) are left out.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-default-groups --no-install-project

# ---- Runtime stage: only Python, the virtualenv and the source ----
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app/src

RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --home-dir /home/app --create-home app

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY src/ml_platform ./src/ml_platform

USER app

# Run from the user's home so relative paths (the default sqlite:///mlflow.db,
# an optional .env) are writable without making /app writable.
WORKDIR /home/app

EXPOSE 8000

# Liveness only: /health never loads the model or runs inference.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-m", "ml_platform.api.healthcheck"]

CMD ["python", "-m", "ml_platform.api"]
