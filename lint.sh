#!/usr/bin/env bash

set -eu
ruff format && ruff check --fix