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
SEMANTIC_MAP="${NAVILA_SEMANTIC_MAP:-${PROJECT_ROOT}/outputs/memory_guide/semantic_map.json}"
TELEOP_JSON="${NAVILA_TELEOP_JSON:-${PROJECT_ROOT}/outputs/memory_guide/latest-teleop/teleop.json}"
USE_SEMANTIC_MAP=true
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
  --semantic-map PATH       Use a pose-tagged teleop map (auto-built by default)
  --teleop-json PATH        Teleop collection used to rebuild the semantic map
  --no-semantic-map         Use the legacy text-only patrol plan

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
    --semantic-map)
      [[ $# -ge 2 ]] || { echo "--semantic-map requires a value" >&2; exit 2; }
      SEMANTIC_MAP="$2"
      shift 2
      ;;
    --semantic-map=*)
      SEMANTIC_MAP="${1#*=}"
      shift
      ;;
    --teleop-json)
      [[ $# -ge 2 ]] || { echo "--teleop-json requires a value" >&2; exit 2; }
      TELEOP_JSON="$2"
      shift 2
      ;;
    --teleop-json=*)
      TELEOP_JSON="${1#*=}"
      shift
      ;;
    --no-semantic-map)
      USE_SEMANTIC_MAP=false
      shift
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

MAP_ARGS=()
MAP_RUN_ARGS=()
if [[ "${USE_SEMANTIC_MAP}" == true ]]; then
  if [[ -f "${TELEOP_JSON}" ]]; then
    "${NAVILA_ORCA_PYTHON}" -m navila_orca.memory_guide \
      map-build \
      --teleop-json "${TELEOP_JSON}" \
      --output "${SEMANTIC_MAP}"
  fi
  if [[ -f "${SEMANTIC_MAP}" ]]; then
    MAP_ARGS=(--semantic-map "${SEMANTIC_MAP}")
    # Recorded views guide coverage inside one location waypoint.
    # Options supplied after these defaults can still override them.
    MAP_RUN_ARGS=(--max-decisions 64 --max-control-steps 2000)
  else
    echo "Semantic map not found; using legacy text-only patrol plan: ${SEMANTIC_MAP}" >&2
  fi
fi

"${NAVILA_ORCA_PYTHON}" -m navila_orca.memory_guide \
  --inventory "${INVENTORY}" \
  plan \
  --query "${QUERY}" \
  "${MAP_ARGS[@]}" \
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
  "${MAP_RUN_ARGS[@]}" \
  "${RUN_ARGS[@]}"
