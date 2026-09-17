#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARCHIVE_DIR="${PROJECT_ROOT}/outputs/deployment"
ARCHIVE_PATH="${ARCHIVE_DIR}/unitree-navila-client.tar.gz"

mkdir -p "${ARCHIVE_DIR}"

BUNDLE_FILES=(
  pyproject.toml
  docs/UNITREE_DEPLOYMENT.md
  assets/memory_guide_catalog.json
  scripts/run_unitree_navila.py
  scripts/run_unitree_navila.sh
  scripts/run_unitree_memory_guide.sh
  scripts/check_navvlm_endpoint.py
  scripts/start_navila_aws_tunnel.sh
  src/navila_orca/__init__.py
  src/navila_orca/actions.py
  src/navila_orca/contracts.py
  src/navila_orca/frames.py
  src/navila_orca/memory_guide.py
  src/navila_orca/openai_router.py
  src/navila_orca/query_router.py
  src/navila_orca/semantic_map.py
  src/navila_orca/vlm_client.py
  src/navila_orca/hardware/__init__.py
  src/navila_orca/hardware/unitree_gateway.py
  scripts/run_unitree_landmark_scan.py
)

# The Memory Guide launcher may be created with mode 0644 in a checkout.  Stage
# the bundle so its executable bit is guaranteed after extraction.
STAGE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/unitree-navila-bundle.XXXXXX")"
cleanup() {
  rm -rf -- "${STAGE_DIR}"
}
trap cleanup EXIT

for path in "${BUNDLE_FILES[@]}"; do
  mkdir -p "${STAGE_DIR}/$(dirname "${path}")"
  cp -a "${PROJECT_ROOT}/${path}" "${STAGE_DIR}/${path}"
done
chmod 0755 "${STAGE_DIR}/scripts/run_unitree_memory_guide.sh"
chmod 0755 "${STAGE_DIR}/scripts/run_unitree_landmark_scan.py"

tar -czf "${ARCHIVE_PATH}" -C "${STAGE_DIR}" "${BUNDLE_FILES[@]}"

printf 'bundle: %s\n' "${ARCHIVE_PATH}"
