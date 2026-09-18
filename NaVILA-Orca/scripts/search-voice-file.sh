#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Keep the OpenAI key out of this script.  The Modal file supplies the existing
# ASR endpoint; it contains no API key.
source "${HOME}/.openaikey"
if [[ -z "${NAVILA_VOICE_ENDPOINT:-}" ]]; then
  source "${PROJECT_ROOT}/modal/modal.env"
fi

VOICE_FILE="${PROJECT_ROOT}/../dethread/sounds/我的面包在哪里.wav"
if [[ $# -eq 1 && "$1" == "--help" ]]; then
  echo "Usage: $0 [WAV_FILE]"
  echo "       $0 --voice-file WAV_FILE"
  echo "With no argument, uses: ${VOICE_FILE}"
  exit 0
elif [[ $# -eq 1 ]]; then
  VOICE_FILE="$1"
elif [[ $# -eq 2 && "$1" == "--voice-file" ]]; then
  VOICE_FILE="$2"
elif [[ $# -ne 0 ]]; then
  echo "Usage: $0 [WAV_FILE]" >&2
  echo "       $0 --voice-file WAV_FILE" >&2
  exit 2
fi

[[ -f "${VOICE_FILE}" ]] || {
  echo "Voice WAV file does not exist: ${VOICE_FILE}" >&2
  exit 2
}

"${SCRIPT_DIR}/run_unitree_memory_guide.sh" \
  --voice-file "${VOICE_FILE}" \
  --voice-backend http \
  --voice-endpoint "${NAVILA_VOICE_ENDPOINT}" \
  --voice-translate \
  --llm-mode openai \
  --robot-model go2 \
  --execute-actions \
  --balance-stand \
  --landmark-seer \
  --max-decisions 128 \
  --max-forward-mps 0.50 \
  --max-yaw-rps 0.524 \
  --max-action-seconds 2.00 \
  --landmark-seer-check-interval 0.5 \
  --landmark-seer-max-forward-action-seconds 2.00 \
  --goal-seer-max-inspection-turns 24 \
  --network-interface enx00e04c680456 \
  --vlm-host 127.0.0.1 \
  --vlm-port 54321
