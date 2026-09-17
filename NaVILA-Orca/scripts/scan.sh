#!/bin/bash
source ~/.openaikey;

unset LD_PRELOAD;
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}";
export UNITREE_PYTHON="$CONDA_PREFIX/bin/python";

python3 ./run_unitree_landmark_scan.py \
--robot-model go2 \
--network-interface enx00e04c680456 \
--scan-direction right \
--scan-views 9 \
--scan-yaw-rps 1.2 \
--image-brightness 1.60 \
--execute-actions;
