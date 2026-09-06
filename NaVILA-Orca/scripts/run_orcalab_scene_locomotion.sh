#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_DIR="${PROJECT_ROOT}/scripts"
source "${SCRIPT_DIR}/orcalab_env.sh"
navila_orca_resolve_runtime
ORCA_PYTHON="${NAVILA_ORCA_PYTHON}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

EXPECTED_ORCA_VERSION="26.7.1"
for ORCA_DISTRIBUTION in orca-lab orca-gym; do
  if ! INSTALLED_ORCA_VERSION="$("${ORCA_PYTHON}" -c "from importlib.metadata import version; print(version('${ORCA_DISTRIBUTION}'))" 2>/dev/null)"; then
    echo "未在 ${ORCA_PYTHON} 中找到 ${ORCA_DISTRIBUTION}；需要 ${EXPECTED_ORCA_VERSION}。" >&2
    exit 2
  fi
  if [[ "${INSTALLED_ORCA_VERSION}" != "${EXPECTED_ORCA_VERSION}" ]]; then
    echo "不支持的 ${ORCA_DISTRIBUTION} ${INSTALLED_ORCA_VERSION}；常驻 RGB 相机要求 ${EXPECTED_ORCA_VERSION}。" >&2
    echo "执行: ${ORCA_PYTHON} -m pip install --upgrade --force-reinstall 'orca-lab==26.7.1' 'orca-gym==26.7.1'" >&2
    exit 2
  fi
done

DEFAULT_INSTRUCTION_FILE="${NAVILA_ORCA_INSTRUCTION_FILE:-${PROJECT_ROOT}/prompts/orcalab_scene_locomotion.txt}"
INSTRUCTION_ARGS=()
HAS_INSTRUCTION_OVERRIDE=false
HAS_OUTPUT_OVERRIDE=false
for ARG in "$@"; do
  case "${ARG}" in
    --instruction|--instruction=*|--instruction-file|--instruction-file=*|--waypoint-instruction-file|--waypoint-instruction-file=*)
      HAS_INSTRUCTION_OVERRIDE=true
      ;;
    --output|--output=*)
      HAS_OUTPUT_OVERRIDE=true
      ;;
  esac
done
if [[ "${HAS_INSTRUCTION_OVERRIDE}" == false ]]; then
  INSTRUCTION_ARGS=(--instruction-file "${DEFAULT_INSTRUCTION_FILE}")
fi
OUTPUT_ARGS=()
if [[ "${HAS_OUTPUT_OVERRIDE}" == false ]]; then
  RUN_ID="$(date -u +%Y%m%dT%H%M%S)-$$"
  OUTPUT_ARGS=(
    --output
    "${NAVILA_ORCA_OUTPUT:-${PROJECT_ROOT}/outputs/scene_locomotion_runs/${RUN_ID}}"
  )
fi

# This command never launches or republishes OrcaLab. It reuses the current
# layout, requires exactly one complete Go2 in the downloaded combined MJCF,
# and fails before motion if actor/XML/scene-option alignment is incomplete.
exec "${ORCA_PYTHON}" -m navila_orca.cli run \
  --render-backend orcalab \
  --orcagym-address 127.0.0.1:50051 \
  --orcalab-edit-address 127.0.0.1:50151 \
  --camera-actor-name mujococamera1080 \
  --camera-asset-path prefabs/mujococamera1080 \
  --orcalab-camera-mode mujoco-png \
  --camera-transport grpc-png \
  --no-publish \
  --robot-actor-name auto \
  --anchor-existing-scene \
  --scene-profile orca-train \
  --strict-scene-alignment \
  --manual-xml-override \
  --vlm-backend tcp \
  --vlm-host 127.0.0.1 \
  --vlm-port 54321 \
  --image-interval 0.5 \
  --state-stream-interval 0.04 \
  --live-monitor \
  --monitor-interval 0.1 \
  --decouple-vlm \
  --warmup-steps 100 \
  --max-decisions 0 \
  --max-control-steps 0 \
  "${OUTPUT_ARGS[@]}" \
  "${INSTRUCTION_ARGS[@]}" \
  "$@"
