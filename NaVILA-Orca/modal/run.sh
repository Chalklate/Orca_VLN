#!/bin/bash
source ${HOME}/Orca_VLN/NaVILA-Orca/modal/modal.env;
# bash ${HOME}/Orca_VLN/NaVILA-Orca/scripts/run_orcalab_memory_guide.sh \
#   --voice-file "/home/kohming/Orca_VLN/dethread/sounds/where_is_my_bread.wav" \
#   --voice-translate \
#   --voice-backend http \
#   --voice-endpoint "$NAVILA_VOICE_ENDPOINT" \
#   --llm-mode bedrock \
#   --bedrock-profile "$AWS_PROFILE" \
#   --bedrock-region "$AWS_DEFAULT_REGION" \
#   --bedrock-model-id global.anthropic.claude-haiku-4-5-20251001-v1:0

bash ${HOME}/Orca_VLN/NaVILA-Orca/scripts/run_orcalab_memory_guide.sh \
  --voice-file "/home/kohming/Orca_VLN/dethread/sounds/where_is_my_bread.wav" \
  --voice-translate \
  --voice-backend http \
  --voice-endpoint "$NAVILA_VOICE_ENDPOINT" \
  --llm-mode openai
