#!/usr/bin/env bash

set -euo pipefail

# ─── Defaults ────────────────────────────────────────────────────────────────
DATA_DIR=${1:-"./original_data/subset_logs_42"}
TRAIN_OUT=${2:-"./data/seed42/train.json"}
TEST_OUT=${3:-"./data/seed42/test.json"}
THRESHOLD=${4:-"0"}
# Optional OOD output path (for test-only tasks)
OOD_OUT=${5:-}

# ─── Detect split type (normal vs OOD) ───────────────────────────────────────
echo "[construct_dataset] DATA_DIR=$DATA_DIR"

# ─── Invoke Python merger (handles per-task OOD internally) ────────────────
python "./construct_dataset.py" \
  -d "$DATA_DIR" \
  --train-out "$TRAIN_OUT" \
  --test-out  "$TEST_OUT"  \
  --ood-out   "$OOD_OUT" \
  --threshold "$THRESHOLD"
