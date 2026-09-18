#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Keep credentials and the ASR endpoint out of this script.  The endpoint is
# normally supplied by modal/modal.env when it is not already exported.
source "${HOME}/.openaikey"
if [[ -z "${NAVILA_VOICE_ENDPOINT:-}" && -f "${PROJECT_ROOT}/modal/modal.env" ]]; then
  source "${PROJECT_ROOT}/modal/modal.env"
fi

"${SCRIPT_DIR}/run_unitree_memory_guide.sh" \
  --voice-interactive \
  --voice-duration "${NAVILA_VOICE_DURATION:-12}" \
  --voice-backend http \
  --voice-endpoint "${NAVILA_VOICE_ENDPOINT:?Set NAVILA_VOICE_ENDPOINT or configure modal/modal.env}" \
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
  --landmark-seer-turn-degrees 30 \
  --landmark-seer-check-interval 0.5 \
  --landmark-seer-max-forward-action-seconds 2.00 \
  --goal-seer-max-inspection-turns 24 \
  --network-interface enx00e04c680456 \
  --vlm-host 127.0.0.1 \
  --vlm-port 54321
