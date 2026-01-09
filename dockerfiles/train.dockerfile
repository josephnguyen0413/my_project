FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim as base

# Only needed if you truly must compile packages from source
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential gcc \
 && rm -rf /var/lib/apt/lists/*

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY README.md README.md
COPY LICENSE LICENSE
COPY src src/
COPY data/ data/
COPY models/ models/
COPY reports/ reports

# WORKDIR /
# RUN uv sync --locked --no-cache --no-install-project

WORKDIR /
ENV UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-install-project

ENTRYPOINT ["uv", "run", "src/messi/train.py"]