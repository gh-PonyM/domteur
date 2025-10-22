#!/usr/bin/env bash

set -eu
uv run ruff format && uv run ruff check --fix