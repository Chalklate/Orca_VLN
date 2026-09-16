#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARCHIVE_DIR="${PROJECT_ROOT}/outputs/deployment"
ARCHIVE_PATH="${ARCHIVE_DIR}/unitree-navila-client.tar.gz"

mkdir -p "${ARCHIVE_DIR}"

tar -czf "${ARCHIVE_PATH}" \
  -C "${PROJECT_ROOT}" \
  pyproject.toml \
  docs/UNITREE_DEPLOYMENT.md \
  scripts/run_unitree_navila.py \
  scripts/run_unitree_navila.sh \
  scripts/check_navvlm_endpoint.py \
  scripts/start_navila_aws_tunnel.sh \
  src/navila_orca/__init__.py \
  src/navila_orca/actions.py \
  src/navila_orca/contracts.py \
  src/navila_orca/frames.py \
  src/navila_orca/vlm_client.py \
  src/navila_orca/hardware/__init__.py \
  src/navila_orca/hardware/unitree_gateway.py

printf 'bundle: %s\n' "${ARCHIVE_PATH}"
