#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_DIR="${PROJECT_ROOT}/scripts"
source "${SCRIPT_DIR}/orcalab_env.sh"
navila_orca_resolve_runtime
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

QUERY=""
INVENTORY="${NAVILA_MEMORY_INVENTORY:-${PROJECT_ROOT}/outputs/memory_guide/inventory.json}"
PLAN_OUTPUT="${PROJECT_ROOT}/outputs/memory_guide/latest_plan.json"
WAYPOINT_OUTPUT="${PROJECT_ROOT}/outputs/memory_guide/latest_waypoints.txt"
PLAN_ONLY=false
RUN_ARGS=()
CAMERA_ARGS=(
  --camera-mount-position 0.1 0 1.2
  --stabilize-camera-horizon
)

usage() {
  cat <<EOF
Usage: $0 --query "Where are my glasses?" [memory options] [runner options]

Memory options:
  --query TEXT              Resident request to route (required)
  --memory-inventory PATH   Persistent inventory JSON
  --plan-output PATH        Structured mission-plan output
  --waypoint-output PATH    Generated NaVILA waypoint file
  --plan-only               Generate the plan without starting locomotion

The Memory Guide camera defaults to a stabilized 1.2 m vertical mount offset.
All other arguments are forwarded to run_orcalab_scene_locomotion.sh, and
later camera arguments override these defaults.
EOF
}

while (($#)); do
  case "$1" in
    --query)
      [[ $# -ge 2 ]] || { echo "--query requires a value" >&2; exit 2; }
      QUERY="$2"
      shift 2
      ;;
    --query=*)
      QUERY="${1#*=}"
      shift
      ;;
    --memory-inventory)
      [[ $# -ge 2 ]] || { echo "--memory-inventory requires a value" >&2; exit 2; }
      INVENTORY="$2"
      shift 2
      ;;
    --plan-output)
      [[ $# -ge 2 ]] || { echo "--plan-output requires a value" >&2; exit 2; }
      PLAN_OUTPUT="$2"
      shift 2
      ;;
    --waypoint-output)
      [[ $# -ge 2 ]] || { echo "--waypoint-output requires a value" >&2; exit 2; }
      WAYPOINT_OUTPUT="$2"
      shift 2
      ;;
    --plan-only)
      PLAN_ONLY=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      RUN_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "${QUERY}" ]]; then
  usage >&2
  exit 2
fi

"${NAVILA_ORCA_PYTHON}" -m navila_orca.memory_guide \
  --inventory "${INVENTORY}" \
  plan \
  --query "${QUERY}" \
  --plan-output "${PLAN_OUTPUT}" \
  --waypoint-output "${WAYPOINT_OUTPUT}"

echo "Memory Guide plan: ${PLAN_OUTPUT}"
echo "NaVILA waypoints: ${WAYPOINT_OUTPUT}"
if [[ "${PLAN_ONLY}" == true ]]; then
  exit 0
fi

exec "${SCRIPT_DIR}/run_orcalab_scene_locomotion.sh" \
  --waypoint-instruction-file "${WAYPOINT_OUTPUT}" \
  "${CAMERA_ARGS[@]}" \
  "${RUN_ARGS[@]}"
