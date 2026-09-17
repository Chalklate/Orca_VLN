#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNITREE_PYTHON="${UNITREE_PYTHON:-python3}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
if [[ -n "${UNITREE_SDK2_ROOT:-}" ]]; then
  export PYTHONPATH="${UNITREE_SDK2_ROOT}:${PYTHONPATH}"
fi

QUERY=""
CATALOG="${NAVILA_MEMORY_CATALOG:-${PROJECT_ROOT}/assets/memory_guide_catalog.json}"
INVENTORY="${NAVILA_MEMORY_INVENTORY:-${PROJECT_ROOT}/outputs/memory_guide/inventory.json}"
PLAN_OUTPUT="${NAVILA_MEMORY_PLAN:-${PROJECT_ROOT}/outputs/memory_guide/latest_plan.json}"
WAYPOINT_OUTPUT="${NAVILA_MEMORY_WAYPOINTS:-${PROJECT_ROOT}/outputs/memory_guide/latest_waypoints.txt}"
LANDMARK_MAP="${NAVILA_LANDMARK_MAP:-}"
LLM_MODE="${NAVILA_MEMORY_LLM_MODE:-deterministic}"
OPENAI_MODEL_ID="${NAVILA_OPENAI_MODEL:-gpt-5.6-luna}"
OPENAI_BASE_URL="${NAVILA_OPENAI_BASE_URL:-}"
MAX_LANDMARK_WAYPOINTS="${NAVILA_MAX_LANDMARK_WAYPOINTS:-6}"
PLAN_ONLY=false
RUN_ARGS=()

resolve_project_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "${PROJECT_ROOT}" "$1" ;;
  esac
}

usage() {
  cat <<EOF
Usage: $0 --query "Where is my bread?" [memory options] [Unitree options]

Memory options:
  --query TEXT              Typed resident request
  --catalog PATH            Item/place catalog JSON
  --inventory PATH          Persistent item-memory JSON
  --plan-output PATH        Structured plan JSON
  --waypoint-output PATH    Generated one-waypoint-per-line file
  --llm-mode MODE           deterministic or openai (default: ${LLM_MODE})
  --openai-model-id ID      OpenAI model ID (default: ${OPENAI_MODEL_ID})
  --openai-base-url URL     Optional OpenAI-compatible API base URL
  --landmark-map PATH       Visual scan map for site-specific search waypoints
  --max-landmark-waypoints N  Maximum scan landmarks to inspect (default: ${MAX_LANDMARK_WAYPOINTS})
  --plan-only               Generate and print the plan without moving

Unitree options are forwarded to run_unitree_navila.sh.  With a waypoint file,
--max-decisions is the maximum number of VLM decisions per waypoint.  The
default is ${NAVILA_WAYPOINT_MAX_DECISIONS:-8}.

Example:
  $0 --query "Where is my bread?" \\
    --robot-model go2 --network-interface eth10 \\
    --vlm-host 100.x.y.z --vlm-port 54321 \\
    --execute-actions --max-forward-mps 0.20 --max-action-seconds 0.75
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
    --catalog)
      [[ $# -ge 2 ]] || { echo "--catalog requires a path" >&2; exit 2; }
      CATALOG="$2"
      shift 2
      ;;
    --catalog=*)
      CATALOG="${1#*=}"
      shift
      ;;
    --inventory)
      [[ $# -ge 2 ]] || { echo "--inventory requires a path" >&2; exit 2; }
      INVENTORY="$2"
      shift 2
      ;;
    --inventory=*)
      INVENTORY="${1#*=}"
      shift
      ;;
    --plan-output)
      [[ $# -ge 2 ]] || { echo "--plan-output requires a path" >&2; exit 2; }
      PLAN_OUTPUT="$2"
      shift 2
      ;;
    --plan-output=*)
      PLAN_OUTPUT="${1#*=}"
      shift
      ;;
    --waypoint-output)
      [[ $# -ge 2 ]] || { echo "--waypoint-output requires a path" >&2; exit 2; }
      WAYPOINT_OUTPUT="$2"
      shift 2
      ;;
    --waypoint-output=*)
      WAYPOINT_OUTPUT="${1#*=}"
      shift
      ;;
    --llm-mode)
      [[ $# -ge 2 ]] || { echo "--llm-mode requires a value" >&2; exit 2; }
      LLM_MODE="$2"
      shift 2
      ;;
    --llm-mode=*)
      LLM_MODE="${1#*=}"
      shift
      ;;
    --openai-model-id)
      [[ $# -ge 2 ]] || { echo "--openai-model-id requires a value" >&2; exit 2; }
      OPENAI_MODEL_ID="$2"
      shift 2
      ;;
    --openai-model-id=*)
      OPENAI_MODEL_ID="${1#*=}"
      shift
      ;;
    --openai-base-url)
      [[ $# -ge 2 ]] || { echo "--openai-base-url requires a value" >&2; exit 2; }
      OPENAI_BASE_URL="$2"
      shift 2
      ;;
    --openai-base-url=*)
      OPENAI_BASE_URL="${1#*=}"
      shift
      ;;
    --landmark-map)
      [[ $# -ge 2 ]] || { echo "--landmark-map requires a path" >&2; exit 2; }
      LANDMARK_MAP="$2"
      shift 2
      ;;
    --landmark-map=*)
      LANDMARK_MAP="${1#*=}"
      shift
      ;;
    --max-landmark-waypoints)
      [[ $# -ge 2 ]] || { echo "--max-landmark-waypoints requires a value" >&2; exit 2; }
      MAX_LANDMARK_WAYPOINTS="$2"
      shift 2
      ;;
    --max-landmark-waypoints=*)
      MAX_LANDMARK_WAYPOINTS="${1#*=}"
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
  echo "--query is required" >&2
  usage >&2
  exit 2
fi

case "${LLM_MODE}" in
  deterministic|openai) ;;
  *) echo "--llm-mode must be deterministic or openai" >&2; exit 2 ;;
esac
if [[ "${LLM_MODE}" == openai && -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OpenAI mode requires OPENAI_API_KEY in the environment" >&2
  exit 2
fi

# The launcher is often invoked from one directory above the extracted
# client. Keep relative memory paths stable by resolving them against the
# client root rather than the caller's current working directory.
CATALOG="$(resolve_project_path "${CATALOG}")"
INVENTORY="$(resolve_project_path "${INVENTORY}")"
PLAN_OUTPUT="$(resolve_project_path "${PLAN_OUTPUT}")"
WAYPOINT_OUTPUT="$(resolve_project_path "${WAYPOINT_OUTPUT}")"
if [[ -n "${LANDMARK_MAP}" ]]; then
  LANDMARK_MAP="$(resolve_project_path "${LANDMARK_MAP}")"
fi

ROUTER_ARGS=(--llm-mode "${LLM_MODE}")
if [[ "${LLM_MODE}" == openai ]]; then
  ROUTER_ARGS+=(--openai-model-id "${OPENAI_MODEL_ID}")
  if [[ -n "${OPENAI_BASE_URL}" ]]; then
    ROUTER_ARGS+=(--openai-base-url "${OPENAI_BASE_URL}")
  fi
fi
LANDMARK_ARGS=(--max-landmark-waypoints "${MAX_LANDMARK_WAYPOINTS}")
if [[ -n "${LANDMARK_MAP}" ]]; then
  LANDMARK_ARGS+=(--landmark-map "${LANDMARK_MAP}")
fi

"${UNITREE_PYTHON}" -m navila_orca.memory_guide \
  --catalog "${CATALOG}" \
  --inventory "${INVENTORY}" \
  plan \
  --query "${QUERY}" \
  "${ROUTER_ARGS[@]}" \
  "${LANDMARK_ARGS[@]}" \
  --plan-output "${PLAN_OUTPUT}" \
  --waypoint-output "${WAYPOINT_OUTPUT}"

echo "Memory Guide plan: ${PLAN_OUTPUT}"
echo "Unitree waypoints: ${WAYPOINT_OUTPUT}"
if [[ "${PLAN_ONLY}" == true ]]; then
  exit 0
fi

exec "${PROJECT_ROOT}/scripts/run_unitree_navila.sh" \
  --waypoint-instruction-file "${WAYPOINT_OUTPUT}" \
  --max-decisions "${NAVILA_WAYPOINT_MAX_DECISIONS:-8}" \
  "${RUN_ARGS[@]}"
