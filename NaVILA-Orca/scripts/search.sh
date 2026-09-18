#!/bin/bash
source ~/.openaikey && \

./run_unitree_memory_guide.sh \
  --query "Where is my white water bottle?" \
  --llm-mode openai \
  --robot-model go2 \
  --print-timings \
  --execute-actions \
  --balance-stand \
  --landmark-seer \
  --max-decisions 128 \
  --max-forward-mps 1.00 \
  --max-yaw-rps 1.00 \
  --max-action-seconds 2.00 \
  --landmark-seer-turn-degrees 30 \
  --landmark-seer-check-interval 1.0 \
  --landmark-seer-max-forward-action-seconds 2.00 \
  --goal-seer-max-inspection-turns 24 \
  --network-interface enx00e04c680456 \
  --vlm-host 127.0.0.1 \
  --vlm-port 54321 \
  --scene-id school;
