#!/usr/bin/env bash
# --------------------------------------------------------------------
#  run_vllm.sh  ─  Launch the vLLM OpenAI-compatible API server
#
#  USAGE
#    ./run_vllm.sh [PORT] [MODEL_PATH]
#
#  DEFAULTS
#    PORT        8000
#    MODEL_PATH  /fs-computility/mabasic/shared/models/Qwen2.5-72B-Instruct
#
#  LOGGING
#    All console output is appended to vllm_logs/<date>/server.log
# --------------------------------------------------------------------

set -euo pipefail

# ---------- user-tunable variables ----------------------------------
PORT=${1:-8000}
MODEL=${2:-/fs-computility/mabasic/shared/models/Llama-3.1-8B-Instruct}

VLLM_DIR="./logs"
# CONDA_ENV="base"

LOG_ROOT="vllm_logs/$(date +%Y-%m-%d)"
LOG_FILE="${LOG_ROOT}/server.log"
mkdir -p "$LOG_ROOT"

# ---------- environment initialisation ------------------------------
echo "[INFO] Activating conda environment..."
export PATH="/root/miniconda3/bin:$PATH"
source /root/miniconda3/etc/profile.d/conda.sh
eval "$(/root/miniconda3/bin/conda shell.bash hook)"
conda activate base
python -m pip list 

cd "$VLLM_DIR"

# ---------- detect GPU count ----------------------------------------
GPU_COUNTS="$(nvidia-smi --query-gpu=count --format=csv,noheader | wc -l)"
echo "[INFO] Detected ${GPU_COUNTS} GPU(s)"

# ---------- common vLLM arguments (array avoids quoting issues) -----
COMMON_ARGS=(
  --model "$MODEL"
  --trust-remote-code
  --seed 42
  --enforce-eager
  --tensor-parallel-size "$GPU_COUNTS"
  --enable-prefix-caching
  --enable-chunked-prefill
  --max-model-len 8192 
  --enable-auto-tool-choice
  --tool-call-parser hermes
  --gpu-memory-utilization 0.9
  --served-model-name "$(basename "$MODEL")"
)

# ---------- launch the server ---------------------------------------
echo "[INFO] Starting vLLM server on port ${PORT}"
# Append both stdout and stderr to the log file and keep printing to console
python -m vllm.entrypoints.openai.api_server \
       "${COMMON_ARGS[@]}" \
       --port "$PORT" \
       2>&1 | tee -a "$LOG_FILE"