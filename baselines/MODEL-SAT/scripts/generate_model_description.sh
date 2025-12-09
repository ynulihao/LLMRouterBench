#!/usr/bin/env bash

INPUT_FILE=${1:-"./original_data/subset_logs_42/mmlupro_train.json"}
OUTPUT_FILE=${2:-"./data/seed42/model_description.json"}

# Execute the Python script
python ./generate_model_descriptions.py \
  --input  "$INPUT_FILE" \
  --output "$OUTPUT_FILE"

echo "Results written to: $OUTPUT_FILE"