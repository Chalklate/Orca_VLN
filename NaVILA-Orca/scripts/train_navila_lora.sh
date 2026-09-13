#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
NAVILA_ROOT="${NAVILA_ROOT:-${WORKSPACE_ROOT}/NaVILA}"
if [[ -z "${NAVILA_PYTHON:-}" ]]; then
  if [[ -n "${CONDA_PREFIX:-}" ]]; then
    NAVILA_PYTHON="${CONDA_PREFIX}/bin/python"
  else
    NAVILA_PYTHON="${WORKSPACE_ROOT}/.conda/envs/navila/bin/python"
  fi
fi
DATASET_DIR="${ORCA_VLN_DATASET_DIR:-${PROJECT_ROOT}/outputs/training_records/navila_orca_built}"
BASE_MODEL="${NAVILA_BASE_MODEL:-${WORKSPACE_ROOT}/models/navila-llama3-8b-8f}"
OUTPUT_DIR="${NAVILA_LORA_OUTPUT:-${WORKSPACE_ROOT}/models/orca_navila_lora_candidate}"
GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-29517}"
NUM_EPOCHS="${NAVILA_NUM_EPOCHS:-1}"
LEARNING_RATE="${NAVILA_LEARNING_RATE:-1e-5}"
GRAD_ACCUM="${NAVILA_GRADIENT_ACCUMULATION_STEPS:-1}"
ALLOW_SMALL="${NAVILA_ALLOW_SMALL_DATASET:-0}"

usage() {
  cat <<EOF
Usage: $0

Build first:
  python scripts/build_navila_training.py --output-dir ${DATASET_DIR}

Environment overrides:
  NAVILA_ROOT, NAVILA_PYTHON, ORCA_VLN_DATASET_DIR, NAVILA_BASE_MODEL,
  NAVILA_LORA_OUTPUT, GPUS_PER_NODE, MASTER_PORT, NAVILA_NUM_EPOCHS,
  NAVILA_LEARNING_RATE, NAVILA_GRADIENT_ACCUMULATION_STEPS

This launcher trains an LLM-only LoRA adapter. It refuses the current tiny,
single-episode dataset unless NAVILA_ALLOW_SMALL_DATASET=1 is set explicitly.
It does not perform a full-model retrain.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

for required in "${NAVILA_ROOT}/llava/train/train_mem.py" "${BASE_MODEL}/config.json" "${DATASET_DIR}/train.jsonl"; do
  if [[ ! -e "${required}" ]]; then
    echo "missing required path: ${required}" >&2
    exit 2
  fi
done

if [[ "${ALLOW_SMALL}" != "1" ]]; then
  DATASET_DIR_ENV="${DATASET_DIR}" "${NAVILA_PYTHON}" - <<'PY'
import json, os, sys
from pathlib import Path
root = Path(os.environ["DATASET_DIR_ENV"])
info = json.loads((root / "dataset_info.json").read_text(encoding="utf-8"))
report_path = root / "collection_report.json"
report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
episodes = report.get("episodes", [])
records = int(info.get("num_records", 0))
if len(episodes) < 2 or records < 32:
    print(
        f"refusing small dataset: {records} unique records across {len(episodes)} episodes; "
        "set NAVILA_ALLOW_SMALL_DATASET=1 only for a diagnostic run",
        file=sys.stderr,
    )
    raise SystemExit(2)
PY
fi

if [[ ! -x "${NAVILA_PYTHON}" ]]; then
  echo "NAVILA_PYTHON is not executable: ${NAVILA_PYTHON}" >&2
  echo "Activate the NaVILA training environment or set NAVILA_PYTHON explicitly." >&2
  exit 2
fi

export PYTHONPATH="${NAVILA_ROOT}:${PYTHONPATH:-}"
export ORCA_VLN_DATA_PATH="${DATASET_DIR}/train.jsonl"
export ORCA_VLN_IMAGE_PATH="${DATASET_DIR}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT

cd "${NAVILA_ROOT}"
exec "${NAVILA_PYTHON}" -m torch.distributed.run \
  --nproc_per_node="${GPUS_PER_NODE}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  llava/train/train_mem.py \
  --model_name_or_path "${BASE_MODEL}" \
  --version llama_3 \
  --data_mixture orca_vln \
  --vision_tower google/siglip-so400m-patch14-384 \
  --mm_projector mlp_downsample \
  --mm_vision_select_feature cls_patch \
  --mm_vision_select_layer -2 \
  --num_video_frames 8 \
  --image_aspect_ratio resize \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --lora_enable True \
  --lora_llm True \
  --lora_vt False \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --lora_bias none \
  --tune_language_model False \
  --tune_vision_tower False \
  --tune_mm_projector False \
  --bf16 True \
  --tf32 True \
  --output_dir "${OUTPUT_DIR}" \
  --num_train_epochs "${NUM_EPOCHS}" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps "${GRAD_ACCUM}" \
  --do_eval False \
  --save_strategy steps \
  --save_steps 50 \
  --save_total_limit 2 \
  --learning_rate "${LEARNING_RATE}" \
  --weight_decay 0.0 \
  --warmup_ratio 0.1 \
  --lr_scheduler_type cosine \
  --logging_steps 1 \
  --report_to none \
  --model_max_length 4096 \
  --gradient_checkpointing True \
  --dataloader_num_workers 2 \
  --lazy_preprocess True
