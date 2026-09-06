#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_DIR="${PROJECT_ROOT}/scripts"
source "${SCRIPT_DIR}/orcalab_env.sh"
navila_orca_resolve_runtime
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

RUN_ID="$(date -u +%Y%m%dT%H%M%S)-$$"
OUTPUT_DIR="${NAVILA_ORCA_TELEOP_OUTPUT:-${PROJECT_ROOT}/outputs/teleop_runs/${RUN_ID}}"

# This command reuses the current authored scene. It does not start OrcaLab,
# publish a replacement layout, or contact the NaVILA VLM service.
exec "${NAVILA_ORCA_PYTHON}" -m navila_orca.cli teleop \
  --render-backend orcalab \
  --orcagym-address 127.0.0.1:50051 \
  --orcalab-edit-address 127.0.0.1:50151 \
  --camera-actor-name mujococamera1080 \
  --camera-asset-path prefabs/mujococamera1080 \
  --orcalab-camera-mode mujoco-png \
  --camera-transport grpc-png \
  --camera-mount-position 0.1 0 1.2 \
  --stabilize-camera-horizon \
  --no-publish \
  --robot-actor-name auto \
  --anchor-existing-scene \
  --scene-profile orca-train \
  --strict-scene-alignment \
  --manual-xml-override \
  --warmup-steps 100 \
  --max-control-steps 0 \
  --output "${OUTPUT_DIR}" \
  "$@"
