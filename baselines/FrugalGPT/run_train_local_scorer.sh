#!/usr/bin/env bash
set -euo pipefail

# Fine-tune a local encoder/embedding model (default: gte_Qwen2-7B-Instruct) as a
# FrugalGPT-compatible scorer using LLMRouterBench AvengersPro JSONL splits
# (prompt/cost/score + per-model usage cost). Supports single GPU or DeepSpeed ZeRO-2.
#
# Usage:
#   ./run_train_local_scorer.sh /path/to/local/base_model_dir [output_dir] [train_jsonl] [test_jsonl]
#
# Notes:
# - If tokenizer is in the same directory as the base model, you don't need to
#   specify --local-tokenizer (the script defaults it).
# - You can run this from the FrugalGPT directory (recommended) or anywhere;
#   the script resolves paths relative to its own location.

# Optional: activate your environment
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate frugalgpt

# Resolve script directory and repo root
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

BATCH_SIZE=4
EVAL_BATCH_SIZE=16
GRAD_ACCUM_STEPS=1
LOG_INTERVAL="${LOG_INTERVAL:-50}"
NUM_GPUS="${NUM_GPUS:-4}"
BUDGET="${BUDGET:-0.02}"
CASCADE_MAX_DEPTH="${CASCADE_MAX_DEPTH:-2}"
MAX_PERMUTATIONS="${MAX_PERMUTATIONS:-5000}"

LOCAL_BASE="${1:-"/path/to/gte_Qwen2-7B-instruct"}"
OUTPUT_DIR="${2:-${script_dir}/strategy/custom_scorer}"
CASCADE_CONFIG="${OUTPUT_DIR}/cascade_config.json"
TRAIN_JSONL="${3:-original_data/seed42_split0.7/train.jsonl}"
TEST_JSONL="${4:-original_data/seed42_split0.7/test.jsonl}"

DEEPSPEED_CONFIG=""
extra_args=()

if [[ -z "${LOCAL_BASE}" ]]; then
  echo "Usage: $0 /path/to/local/base_model_dir [output_dir] [train_jsonl] [test_jsonl]" >&2
  exit 1
fi

# Ensure Python entry exists
if [[ ! -f "${script_dir}/train_router_from_results.py" ]]; then
  echo "Cannot find ${script_dir}/train_router_from_results.py" >&2
  exit 1
fi

if [[ ! -d "${LOCAL_BASE}" ]]; then
  echo "Local base model directory not found: ${LOCAL_BASE}" >&2
  exit 1
fi

if [[ ! -f "${TRAIN_JSONL}" ]]; then
  echo "Train JSONL not found: ${TRAIN_JSONL}" >&2
  exit 1
fi

if [[ ! -f "${TEST_JSONL}" ]]; then
  echo "Test JSONL not found: ${TEST_JSONL}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

if [[ ! "${NUM_GPUS}" =~ ^[0-9]+$ ]]; then
  echo "[warning] NUM_GPUS='${NUM_GPUS}' is not numeric; defaulting to 1." >&2
  NUM_GPUS=1
fi

launcher="python"
launcher_args=()
if (( NUM_GPUS > 1 )); then
  if command -v deepspeed >/dev/null 2>&1; then
    DEEPSPEED_CONFIG="${OUTPUT_DIR}/deepspeed_zero2_config.json"
    cat > "${DEEPSPEED_CONFIG}" <<JSON
{
  "train_micro_batch_size_per_gpu": ${BATCH_SIZE},
  "gradient_accumulation_steps": ${GRAD_ACCUM_STEPS},
  "zero_optimization": {
    "stage": 2,
    "reduce_scatter": true,
    "contiguous_gradients": true,
    "overlap_comm": true,
    "allgather_partitions": true,
    "allgather_bucket_size": 200000000.0,
    "reduce_bucket_size": 200000000.0
  },
  "bf16": {
    "enabled": true
  },
  "fp16": {
    "enabled": false,
    "loss_scale": 0,
    "loss_scale_window": 1000,
    "hysteresis": 2,
    "initial_scale_power": 16,
    "min_loss_scale": 1
  }
}
JSON
    launcher="deepspeed"
    launcher_args=(--num_gpus "${NUM_GPUS}")
    extra_args+=(--deepspeed "${DEEPSPEED_CONFIG}")
  else
    echo "[warning] NUM_GPUS=${NUM_GPUS} requested but 'deepspeed' executable not found; falling back to python." >&2
    NUM_GPUS=1
  fi
fi

"${launcher}" "${launcher_args[@]}" "${script_dir}/train_router_from_results.py" \
  --train-jsonl "${TRAIN_JSONL}" \
  --test-jsonl "${TEST_JSONL}" \
  --local-base "${LOCAL_BASE}" \
  --output-dir "${OUTPUT_DIR}" \
  --backbone-type embedding \
  --pooling last-token \
  --score-threshold 0.5 \
  --prob-threshold 0.5 \
  --epochs 2 \
  --batch-size "${BATCH_SIZE}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --lr 3e-5 \
  --weight-decay 0.01 \
  --warmup-ratio 0.03 \
  --grad-accum "${GRAD_ACCUM_STEPS}" \
  --max-length 2048 \
  --random-state 42 \
  --seed 42 \
  --cascade \
  --cascade-max-depth "${CASCADE_MAX_DEPTH}" \
  --budget "${BUDGET}" \
  --max-permutations "${MAX_PERMUTATIONS}" \
  --cascade-config "${CASCADE_CONFIG}" \
  "${extra_args[@]}" \
  --log-interval "${LOG_INTERVAL}" \
  --bf16 \
  --truncation-side left

printf '\nDone. Fine-tuned scorer saved (if requested) to: %s\n' "${OUTPUT_DIR}"
printf 'Cascade config (order/thresholds/cost) written to: %s\n' "${CASCADE_CONFIG}"
if [[ -n "${DEEPSPEED_CONFIG}" ]]; then
  printf 'DeepSpeed ZeRO-2 config written to: %s\n' "${DEEPSPEED_CONFIG}"
fi
printf 'Launcher: %s (NUM_GPUS=%s)\n' "${launcher}" "${NUM_GPUS}"
