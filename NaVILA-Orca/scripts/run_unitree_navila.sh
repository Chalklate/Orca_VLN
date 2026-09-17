#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNITREE_PYTHON="${UNITREE_PYTHON:-python3}"

if [[ -n "${UNITREE_SDK2_ROOT:-}" ]]; then
  export PYTHONPATH="${UNITREE_SDK2_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
fi

exec "${UNITREE_PYTHON}" "${PROJECT_ROOT}/scripts/run_unitree_navila.py" "$@"
