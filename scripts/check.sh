#!/usr/bin/env sh
# The done-check: lint, format, types, tests. All four must pass.
set -e
cd "$(dirname "$0")/.."
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
