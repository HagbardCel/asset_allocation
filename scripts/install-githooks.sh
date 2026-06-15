#!/usr/bin/env bash
# Point this repository at the versioned hooks in .githooks/.
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

chmod +x .githooks/*
git config core.hooksPath .githooks

echo "Installed git hooks from .githooks/"
