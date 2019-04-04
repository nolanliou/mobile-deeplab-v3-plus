#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status.
set -e

export CUDA_VISIBLE_DEVICES=''
# Update PYTHONPATH.
export PYTHONPATH=$PYTHONPATH:`pwd`

MODEL_TYPE='deeplab-v3-plus'
# Set up the working environment.
CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}"

# Run model_test first to make sure the PYTHONPATH is correctly set.
# python "${WORK_DIR}"/deeplab_v3_plus_test.py -v

# Go to datasets folder and download PASCAL VOC 2012 segmentation dataset.
DATASET_DIR="datasets"

# Set up the working directories.
PASCAL_FOLDER="pascal_voc2012"
EXP_FOLDER="exp"
TRAIN_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${PASCAL_FOLDER}/${EXP_FOLDER}/${MODEL_TYPE}/train"
EXPORT_DIR="${WORK_DIR}/${DATASET_DIR}/${PASCAL_FOLDER}/${EXP_FOLDER}/${MODEL_TYPE}/export"
rm -rf "${EXPORT_DIR}"
mkdir -p "${EXPORT_DIR}"

python run.py --dataset_dir="${PASCAL_DATASET}"\
  --logdir="${TRAIN_LOGDIR}" \
  --model_type="${MODEL_TYPE}" \
  --mode=export \
  --export_dir="${EXPORT_DIR}"

# freeze
python freeze.py --model_dir="${EXPORT_DIR}" \
  --output_node_names=Output \
  --output_dir="${EXPORT_DIR}"



