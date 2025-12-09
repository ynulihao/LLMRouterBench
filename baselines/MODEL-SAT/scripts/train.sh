#!/usr/bin/env bash

# ------------- Paths & filenames -------------
BASE_MODEL_PATH=${BASE_MODEL_PATH:-"/path/to/Qwen2.5-7B-Instruct"} # replace with the path of your Qwen2.5-7B-Instruct model
EMBED_MODEL_PATH=${EMBED_MODEL_PATH:-"/path/to/gte_Qwen2-7B-instruct"} # replace with the path of your gte_Qwen2-7B-instruct model
EXPERTS_FILE=${EXPERTS_FILE:-"./data/seed42/model_description.json"}

TRAIN_FILE=${TRAIN_FILE:-"./data/seed42/train.json"}
VAL_FILE=${VAL_FILE:-"./data/seed42/test.json"}
OOD_FILE=${OOD_FILE:-}

OUTPUT_DIR=${OUTPUT_DIR:-"./checkpoints"}

# ------------- Training hyper-parameters -------------
BATCH_SIZE=${BATCH_SIZE:-4}
MAX_LENGTH=${MAX_LENGTH:-1024}

PROJ_LR=${PROJ_LR:-1e-5}
LLM_UNFREEZE_STEP=${LLM_UNFREEZE_STEP:-1000}
LLM_LR=${LLM_LR:-1e-6}
EMBED_LR=${EMBED_LR:-1e-6}

PROJECTOR_PATH=${PROJECTOR_PATH:-""}      
NUM_EPOCHS=${NUM_EPOCHS:-5}
PROJECTOR_TYPE=${PROJECTOR_TYPE:-"nonlinear"}

# ------------- DeepSpeed launch vars -------------
NUM_GPUS=${NUM_GPUS:-8}                     # number of GPUs DeepSpeed will use
MODEL_SCRIPT=${MODEL_SCRIPT:-"./model_sat_train.py"}  # training entrypoint

# ------------- Assemble dataset_name JSON -------------
# Only include OOD split if the file exists
if [[ -n "${OOD_FILE}" && -f "${OOD_FILE}" ]]; then
  DATASET_NAME="{\"train\":\"${TRAIN_FILE}\",\"validation\":\"${VAL_FILE}\",\"ood\":\"${OOD_FILE}\"}"
else
  echo "[info] OOD file not found or unset; proceeding without OOD split."
  DATASET_NAME="{\"train\":\"${TRAIN_FILE}\",\"validation\":\"${VAL_FILE}\"}"
fi

# ------------- Kick off training -------------
deepspeed --num_gpus=${NUM_GPUS} "${MODEL_SCRIPT}" \
  --base_model_name_or_path       "${BASE_MODEL_PATH}" \
  --embed_model_name_or_path      "${EMBED_MODEL_PATH}" \
  --experts_information_file      "${EXPERTS_FILE}" \
  --dataset_name                  "${DATASET_NAME}" \
  --output_dir                    "${OUTPUT_DIR}" \
  --batch_size                    "${BATCH_SIZE}" \
  --max_length                    "${MAX_LENGTH}" \
  --proj_lr                       "${PROJ_LR}" \
  --llm_unfreeze_step             "${LLM_UNFREEZE_STEP}" \
  --llm_lr                        "${LLM_LR}" \
  --embed_lr                      "${EMBED_LR}" \
  $( [[ -n "${PROJECTOR_PATH}" ]] && echo "--projector_path ${PROJECTOR_PATH}" ) \
  --num_train_epochs              "${NUM_EPOCHS}" \
  --projector_type                "${PROJECTOR_TYPE}" \
  "${@}"                            # forward any extra CLI args

echo "DeepSpeed training launched with ${NUM_GPUS} GPU(s). Check logs for progress; checkpoints will be saved to ${OUTPUT_DIR}."
