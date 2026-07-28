#!/usr/bin/env bash
# Start the AI Dubbing Studio locally.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"

if [ ! -d .venv ] && [ "${DUB_NO_VENV:-0}" != "1" ]; then
  echo "→ creating virtualenv in .venv"
  "$PYTHON" -m venv .venv
fi
if [ -d .venv ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
  PYTHON=python
fi

echo "→ installing requirements"
"$PYTHON" -m pip install --quiet --upgrade pip
"$PYTHON" -m pip install --quiet -r requirements.txt

if [ ! -f data/sample.wav ]; then
  echo "→ generating demo clip at data/sample.wav"
  "$PYTHON" scripts/make_sample.py data/sample.wav || true
fi

HOST="${DUB_HOST:-127.0.0.1}"
PORT="${DUB_PORT:-8000}"
echo "→ open http://${HOST}:${PORT}"
exec "$PYTHON" -m uvicorn backend.app.main:app --host "$HOST" --port "$PORT"
