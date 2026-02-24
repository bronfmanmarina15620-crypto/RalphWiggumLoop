#!/usr/bin/env bash
set -euo pipefail

banner() { printf '\n=== %s ===\n' "$1"; }

banner "1/6  pip install -e .[dev]"
pip install -e ".[dev]" --quiet

banner "2/6  python -m pytest -q"
python -m pytest -q

banner "3/6  python -c \"import smaps; print('ok')\""
python -c "import smaps; print('ok')"

banner "4/6  make test"
make test

banner "5/6  make smoke"
make smoke

banner "6/6  git status --porcelain (expect empty)"
output=$(git status --porcelain)
if [ -n "$output" ]; then
  echo "FAIL: working tree is dirty:"
  echo "$output"
  exit 1
fi
echo "(clean)"

printf '\n*** ALL CHECKS PASSED ***\n'
