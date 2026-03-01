#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

# Heuristic scan for common API key/token/password patterns in source files.
PATTERN='(sk-[A-Za-z0-9]{16,}|AKIA[0-9A-Z]{16}|AIza[A-Za-z0-9_-]{20,}|xox[baprs]-[A-Za-z0-9-]{10,}|(?i)(api[_-]?key|token|secret|password)\s*[:=]\s*["'"'"'][^"'"'"'\n]{8,}["'"'"'])'

if rg -n -S \
  --glob '!.git/**' \
  --glob '!.venv/**' \
  --glob '!env/**' \
  --glob '!venv/**' \
  --glob '!.env' \
  --glob '!.env.*' \
  --glob '!.env.example' \
  "$PATTERN" .; then
  echo
  echo "Potential secrets found. Move real values to .env and keep placeholders in tracked files."
  exit 1
fi

echo "No obvious secrets found."
