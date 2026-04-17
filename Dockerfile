# syntax=docker/dockerfile:1.7

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy \
    UV_CACHE_DIR=/root/.cache/uv \
    UV_SYSTEM_PYTHON=true \
    UV_BREAK_SYSTEM_PACKAGES=true

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

COPY pyproject.toml README.md ./
COPY src ./src

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system ".[real]"

RUN useradd --create-home --shell /bin/bash appuser && \
    mkdir -p /app/outputs/runs && \
    chown -R appuser:appuser /app
# For the public demo we prefer reliable writes to bind-mounted outputs/
# over running as an unprivileged user, because host-side ownership varies.

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import sys, urllib.request; urllib.request.urlopen('http://127.0.0.1:7860', timeout=3); sys.exit(0)"

CMD ["aigraph", "serve", "--host", "0.0.0.0", "--port", "7860", "--runs-dir", "outputs/runs"]
