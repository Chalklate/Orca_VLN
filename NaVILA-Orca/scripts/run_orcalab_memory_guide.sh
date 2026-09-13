#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_DIR="${PROJECT_ROOT}/scripts"
source "${SCRIPT_DIR}/orcalab_env.sh"
navila_orca_resolve_runtime
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

QUERY=""
VOICE_REQUESTED=false
VOICE_FILE=""
VOICE_DURATION="${NAVILA_VOICE_DURATION:-8}"
VOICE_DEVICE="${NAVILA_VOICE_DEVICE:-default}"
VOICE_BACKEND="${NAVILA_VOICE_BACKEND:-firered}"
VOICE_ENDPOINT="${NAVILA_VOICE_ENDPOINT:-}"
VOICE_TRANSLATE="${NAVILA_VOICE_TRANSLATE_TO_ENGLISH:-false}"
VOICE_GOOGLE_PROJECT="${NAVILA_GOOGLE_CLOUD_PROJECT:-${GOOGLE_CLOUD_PROJECT:-}}"
VOICE_CPU=false
LLM_MODE="${NAVILA_MEMORY_LLM_MODE:-deterministic}"
BEDROCK_REGION="${NAVILA_BEDROCK_REGION:-${AWS_REGION:-${AWS_DEFAULT_REGION:-ap-southeast-1}}}"
BEDROCK_MODEL_ID="${NAVILA_BEDROCK_MODEL_ID:-amazon.nova-micro-v1:0}"
BEDROCK_PROFILE="${NAVILA_BEDROCK_PROFILE:-683803166476_hack2026_IsbUsersPS}"
OPENAI_MODEL_ID="${NAVILA_OPENAI_MODEL:-gpt-5.6-luna}"
OPENAI_BASE_URL="${NAVILA_OPENAI_BASE_URL:-}"
INVENTORY="${NAVILA_MEMORY_INVENTORY:-${PROJECT_ROOT}/outputs/memory_guide/inventory.json}"
PLAN_OUTPUT="${PROJECT_ROOT}/outputs/memory_guide/latest_plan.json"
WAYPOINT_OUTPUT="${PROJECT_ROOT}/outputs/memory_guide/latest_waypoints.txt"
SEMANTIC_MAP="${NAVILA_SEMANTIC_MAP:-${PROJECT_ROOT}/outputs/memory_guide/semantic_map.json}"
TELEOP_JSON="${NAVILA_TELEOP_JSON:-${PROJECT_ROOT}/outputs/memory_guide/latest-teleop/teleop.json}"
CURRENT_LOCATION="${NAVILA_CURRENT_LOCATION:-}"
USE_SEMANTIC_MAP=true
PLAN_ONLY=false
RUN_ARGS=()
CAMERA_ARGS=(
  --camera-mount-position 0.1 0 1.0
  --stabilize-camera-horizon
)

usage() {
  cat <<EOF
Usage: $0 (--query "Where are my glasses?" | --voice | --voice-file PATH) [memory options] [runner options]

Memory options:
  --query TEXT              Typed resident request to route
  --voice                   Record a query from the default ALSA microphone
  --voice-file PATH         Transcribe an existing WAV file as the query
  --voice-duration SECONDS  Recording length for --voice (default: ${VOICE_DURATION})
  --voice-device DEVICE     ALSA capture device for --voice (default: ${VOICE_DEVICE})
  --voice-backend NAME      firered (local) or http (default: ${VOICE_BACKEND})
  --voice-endpoint URL      HTTP transcription endpoint for --voice-backend http
  --voice-translate         Translate Han-character transcripts to English via Google Cloud
  --voice-cpu               Run local FireRed inference on CPU
  --llm-mode MODE           Query router: deterministic, bedrock, or openai
  --bedrock-region REGION   AWS region for Nova Micro
  --bedrock-model-id ID     Bedrock model ID (default: ${BEDROCK_MODEL_ID})
  --bedrock-profile NAME    Optional boto3 profile for Bedrock
  --openai-model-id ID      OpenAI model ID (default: ${OPENAI_MODEL_ID})
  --openai-base-url URL     Optional OpenAI-compatible API base URL
  --memory-inventory PATH   Persistent inventory JSON
  --plan-output PATH        Structured mission-plan output
  --waypoint-output PATH    Generated NaVILA waypoint file
  --plan-only               Generate the plan without starting locomotion
  --semantic-map PATH       Use a pose-tagged teleop map (auto-built by default)
  --teleop-json PATH        Teleop collection used to rebuild the semantic map
  --current-location NAME   Known current semantic location; enables route graph planning
  --no-semantic-map         Use the legacy text-only patrol plan

The Memory Guide camera defaults to a stabilized 1.0 m base-frame mount offset
(approximately 1.4 m above the floor in the standing scene) with a 20-degree
downward pitch.
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
    --voice)
      VOICE_REQUESTED=true
      shift
      ;;
    --voice-file)
      [[ $# -ge 2 ]] || { echo "--voice-file requires a value" >&2; exit 2; }
      VOICE_REQUESTED=true
      VOICE_FILE="$2"
      shift 2
      ;;
    --voice-file=*)
      VOICE_REQUESTED=true
      VOICE_FILE="${1#*=}"
      shift
      ;;
    --voice-duration)
      [[ $# -ge 2 ]] || { echo "--voice-duration requires a value" >&2; exit 2; }
      VOICE_DURATION="$2"
      shift 2
      ;;
    --voice-duration=*)
      VOICE_DURATION="${1#*=}"
      shift
      ;;
    --voice-device)
      [[ $# -ge 2 ]] || { echo "--voice-device requires a value" >&2; exit 2; }
      VOICE_DEVICE="$2"
      shift 2
      ;;
    --voice-device=*)
      VOICE_DEVICE="${1#*=}"
      shift
      ;;
    --voice-backend)
      [[ $# -ge 2 ]] || { echo "--voice-backend requires a value" >&2; exit 2; }
      VOICE_BACKEND="$2"
      shift 2
      ;;
    --voice-backend=*)
      VOICE_BACKEND="${1#*=}"
      shift
      ;;
    --voice-endpoint)
      [[ $# -ge 2 ]] || { echo "--voice-endpoint requires a value" >&2; exit 2; }
      VOICE_ENDPOINT="$2"
      shift 2
      ;;
    --voice-endpoint=*)
      VOICE_ENDPOINT="${1#*=}"
      shift
      ;;
    --voice-translate)
      VOICE_TRANSLATE=true
      shift
      ;;
    --voice-cpu)
      VOICE_CPU=true
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
    --bedrock-region)
      [[ $# -ge 2 ]] || { echo "--bedrock-region requires a value" >&2; exit 2; }
      BEDROCK_REGION="$2"
      shift 2
      ;;
    --bedrock-region=*)
      BEDROCK_REGION="${1#*=}"
      shift
      ;;
    --bedrock-model-id)
      [[ $# -ge 2 ]] || { echo "--bedrock-model-id requires a value" >&2; exit 2; }
      BEDROCK_MODEL_ID="$2"
      shift 2
      ;;
    --bedrock-model-id=*)
      BEDROCK_MODEL_ID="${1#*=}"
      shift
      ;;
    --bedrock-profile)
      [[ $# -ge 2 ]] || { echo "--bedrock-profile requires a value" >&2; exit 2; }
      BEDROCK_PROFILE="$2"
      shift 2
      ;;
    --bedrock-profile=*)
      BEDROCK_PROFILE="${1#*=}"
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
    --current-location)
      [[ $# -ge 2 ]] || { echo "--current-location requires a value" >&2; exit 2; }
      CURRENT_LOCATION="$2"
      shift 2
      ;;
    --current-location=*)
      CURRENT_LOCATION="${1#*=}"
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

if [[ "${VOICE_REQUESTED}" == true && -n "${QUERY}" ]]; then
  echo "Choose exactly one of --query, --voice, or --voice-file." >&2
  exit 2
fi

if [[ "${VOICE_REQUESTED}" == true ]]; then
  if [[ -n "${VOICE_FILE}" ]]; then
    VOICE_AUDIO="${VOICE_FILE}"
    [[ -f "${VOICE_AUDIO}" ]] || {
      echo "Voice audio file does not exist: ${VOICE_AUDIO}" >&2
      exit 2
    }
  else
    [[ "${VOICE_DURATION}" =~ ^[1-9][0-9]*$ ]] || {
      echo "--voice-duration must be a positive integer number of seconds" >&2
      exit 2
    }
    command -v arecord >/dev/null 2>&1 || {
      echo "--voice requires arecord; install ALSA utilities or use --voice-file" >&2
      exit 2
    }
    VOICE_TEMP="$(mktemp --suffix=.wav "${TMPDIR:-/tmp}/navila-voice-query.XXXXXX")"
    trap 'rm -f "${VOICE_TEMP}"' EXIT
    echo "Recording ${VOICE_DURATION}s from ALSA device ${VOICE_DEVICE}..." >&2
    arecord -q \
      --device "${VOICE_DEVICE}" \
      --format S16_LE \
      --rate 16000 \
      --channels 1 \
      --duration "${VOICE_DURATION}" \
      "${VOICE_TEMP}"
    VOICE_AUDIO="${VOICE_TEMP}"
  fi

  case "${VOICE_BACKEND}" in
    firered|http) ;;
    *)
      echo "--voice-backend must be firered or http" >&2
      exit 2
      ;;
  esac
  VOICE_PYTHON="${NAVILA_VOICE_PYTHON:-${NAVILA_ORCA_PYTHON}}"
  [[ -x "${VOICE_PYTHON}" ]] || {
    echo "Voice Python does not exist or is not executable: ${VOICE_PYTHON}" >&2
    echo "Set NAVILA_VOICE_PYTHON to the FireRedASR2S environment's Python." >&2
    exit 2
  }
  VOICE_ARGS=(
    -m navila_orca.voice_query
    --audio-file "${VOICE_AUDIO}"
    --backend "${VOICE_BACKEND}"
  )
  if [[ -n "${VOICE_ENDPOINT}" ]]; then
    VOICE_ARGS+=(--endpoint "${VOICE_ENDPOINT}")
  fi
  if [[ "${VOICE_CPU}" == true ]]; then
    VOICE_ARGS+=(--cpu)
  fi
  if [[ "${VOICE_TRANSLATE}" == true ]]; then
    VOICE_ARGS+=(--translate-to-english)
    if [[ -n "${VOICE_GOOGLE_PROJECT}" ]]; then
      VOICE_ARGS+=(--google-project "${VOICE_GOOGLE_PROJECT}")
    fi
  fi
  QUERY="$("${VOICE_PYTHON}" "${VOICE_ARGS[@]}")"
  QUERY="${QUERY//$'\n'/ }"
  [[ -n "${QUERY//[[:space:]]/}" ]] || {
    echo "Voice recognition returned an empty query." >&2
    exit 2
  }
  echo "Voice query: ${QUERY}" >&2
fi

if [[ -z "${QUERY}" ]]; then
  usage >&2
  exit 2
fi

case "${LLM_MODE}" in
  deterministic|bedrock|openai) ;;
  *)
    echo "--llm-mode must be deterministic, bedrock, or openai" >&2
    exit 2
    ;;
esac

LLM_ARGS=(--llm-mode "${LLM_MODE}")
if [[ "${LLM_MODE}" == bedrock ]]; then
  if [[ -n "${BEDROCK_REGION}" ]]; then
    LLM_ARGS+=(--bedrock-region "${BEDROCK_REGION}")
  fi
  LLM_ARGS+=(--bedrock-model-id "${BEDROCK_MODEL_ID}")
  if [[ -n "${BEDROCK_PROFILE}" ]]; then
    LLM_ARGS+=(--bedrock-profile "${BEDROCK_PROFILE}")
  fi
elif [[ "${LLM_MODE}" == openai ]]; then
  LLM_ARGS+=(--openai-model-id "${OPENAI_MODEL_ID}")
  if [[ -n "${OPENAI_BASE_URL}" ]]; then
    LLM_ARGS+=(--openai-base-url "${OPENAI_BASE_URL}")
  fi
fi

MAP_ARGS=()
MAP_RUN_ARGS=()
if [[ "${USE_SEMANTIC_MAP}" == true ]]; then
  if [[ -f "${TELEOP_JSON}" ]]; then
    "${NAVILA_ORCA_PYTHON}" -m navila_orca.memory_guide \
      map-build \
      --include-existing-sources \
      --teleop-json "${TELEOP_JSON}" \
      --output "${SEMANTIC_MAP}"
  fi
  if [[ -f "${SEMANTIC_MAP}" ]]; then
    MAP_ARGS=(--semantic-map "${SEMANTIC_MAP}")
    if [[ -n "${CURRENT_LOCATION}" ]]; then
      MAP_ARGS+=(--current-location "${CURRENT_LOCATION}")
    fi
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
  "${LLM_ARGS[@]}" \
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
