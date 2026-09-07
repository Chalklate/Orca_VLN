#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
DEFAULT_NAVVLM_PYTHON="${WORKSPACE_ROOT}/.conda/envs/navila/bin/python"
if [[ -x "${DEFAULT_NAVVLM_PYTHON}" ]]; then
  NAVVLM_PYTHON="${NAVVLM_PYTHON:-${DEFAULT_NAVVLM_PYTHON}}"
else
  NAVVLM_PYTHON="${NAVVLM_PYTHON:-python}"
fi
NAVILA_SERVER_SCRIPT="${NAVILA_SERVER_SCRIPT:-${PROJECT_ROOT}/scripts/navila_vlm_server.py}"
MODEL_PATH="${NAVVLM_MODEL_PATH:-${WORKSPACE_ROOT}/models/navila-llama3-8b-8f}"
MODEL_BASE="${NAVVLM_MODEL_BASE:-}"

if [[ ! -f "${NAVILA_SERVER_SCRIPT}" ]]; then
  echo "NaVILA server file does not exist: ${NAVILA_SERVER_SCRIPT}" >&2
  exit 2
fi

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "NavVLM model or adapter is missing or incomplete (config.json missing): ${MODEL_PATH}" >&2
  exit 2
fi
if [[ -n "${MODEL_BASE}" ]]; then
  if [[ ! -f "${MODEL_BASE}/config.json" ]]; then
    echo "NavVLM base model is missing config.json: ${MODEL_BASE}" >&2
    exit 2
  fi
  if ! find "${MODEL_BASE}" -maxdepth 2 -type f \
    \( -name '*.safetensors' -o -name '*.safetensors.index.json' \) \
    -print -quit | grep -q .; then
    echo "NavVLM base model has no safetensors weights: ${MODEL_BASE}" >&2
    exit 2
  fi
  if [[ ! -f "${MODEL_PATH}/adapter_config.json" ]]; then
    echo "LoRA model_path is missing adapter_config.json: ${MODEL_PATH}" >&2
    exit 2
  fi
  if ! find "${MODEL_PATH}" -maxdepth 2 -type f \
    \( -name 'adapter_model.safetensors' -o -name 'adapter_model.bin' \) \
    -print -quit | grep -q .; then
    echo "LoRA model_path has no adapter weights: ${MODEL_PATH}" >&2
    exit 2
  fi
else
  if ! find "${MODEL_PATH}" -maxdepth 2 -type f \
    \( -name '*.safetensors' -o -name '*.safetensors.index.json' \) \
    -print -quit | grep -q .; then
    echo "NavVLM model has no safetensors weights: ${MODEL_PATH}" >&2
    echo "Rerun ${PROJECT_ROOT}/scripts/download_navila_model.sh to resume and verify the download." >&2
    exit 2
  fi
fi

if ! "${NAVVLM_PYTHON}" -c \
  'import torch, transformers; from llava.model.builder import load_pretrained_model' \
  >/dev/null 2>&1; then
  echo "NaVILA runtime is incomplete in: ${NAVVLM_PYTHON}" >&2
  echo "Expected torch, transformers, deepspeed, and the editable NaVILA llava package." >&2
  echo "Run ${PROJECT_ROOT}/scripts/setup_navila_env.sh to repair it." >&2
  exit 2
fi

if ! "${NAVVLM_PYTHON}" "${PROJECT_ROOT}/scripts/check_navila_cuda.py"; then
  exit 2
fi

# NaVILA remains an explicit, separately installed dependency.  This OrcaLab
# teaching project owns the simulator/navigation adapter, not NaVILA's model.
MODEL_BASE_ARGS=()
if [[ -n "${MODEL_BASE}" ]]; then
  MODEL_BASE_ARGS=(--model_base "${MODEL_BASE}")
fi

exec "${NAVVLM_PYTHON}" "${NAVILA_SERVER_SCRIPT}" \
  --host "${NAVVLM_HOST:-127.0.0.1}" \
  --port "${NAVVLM_PORT:-54321}" \
  --model_path "${MODEL_PATH}" \
  "${MODEL_BASE_ARGS[@]}" \
  "$@"
