#!/bin/bash
source ~/.openaikey && \

./run_unitree_memory_guide.sh \
  --query "Where is my white water bottle?" \
  --llm-mode openai \
  --robot-model go2 \
  --execute-actions \
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
  --vlm-port 54321;
