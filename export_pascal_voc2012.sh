#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status.
set -e

export CUDA_VISIBLE_DEVICES=''
# Update PYTHONPATH.
export PYTHONPATH=$PYTHONPATH:`pwd`

MODEL_TYPE='deeplab-v3-plus'
# Set up the working environment.
DATASET_DIR="datasets"
CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}"

# Set up the working directories.
PASCAL_FOLDER="pascal_voc2012"
EXP_FOLDER="exp"
TRAIN_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${PASCAL_FOLDER}/${EXP_FOLDER}/${MODEL_TYPE}/train"
mkdir -p "${TRAIN_LOGDIR}"

PASCAL_DATASET="${WORK_DIR}/${DATASET_DIR}/${PASCAL_FOLDER}/tfrecord"

# Export model
EXPORT_DIR="${WORK_DIR}/${DATASET_DIR}/${PASCAL_FOLDER}/${EXP_FOLDER}/${MODEL_TYPE}/export"
rm -rf "${EXPORT_DIR}"
mkdir -p "${EXPORT_DIR}"

python run.py --dataset_dir="${PASCAL_DATASET}"\
  --dataset_name="pascal_voc2012" \
  --logdir="${TRAIN_LOGDIR}" \
  --model_type="${MODEL_TYPE}" \
  --mode=export \
  --export_dir="${EXPORT_DIR}"

# freeze
python freeze.py --model_dir="${EXPORT_DIR}" \
  --output_node_names=Output \
  --output_dir="${EXPORT_DIR}"
