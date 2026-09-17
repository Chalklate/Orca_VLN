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

"${SCRIPT_DIR}/run_unitree_memory_guide.sh" \
  --voice-file "${PROJECT_ROOT}/../dethread/sounds/我的面包在哪里.wav" \
  --voice-backend http \
  --voice-endpoint "${NAVILA_VOICE_ENDPOINT}" \
  --voice-translate \
  --llm-mode openai \
  --robot-model go2 \
  --plan-only \
  --balance-stand \
  --landmark-seer \
  --max-decisions 8 \
  --max-forward-mps 0.20 \
  --max-yaw-rps 0.35 \
  --max-action-seconds 0.75 \
  --landmark-seer-check-interval 0.5 \
  --landmark-seer-max-forward-action-seconds 0.25 \
  --goal-seer-max-inspection-turns 24 \
  --network-interface enx00e04c680456 \
  --vlm-host 127.0.0.1 \
  --vlm-port 54321
