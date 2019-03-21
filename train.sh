#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Update PYTHONPATH.
export PYTHONPATH=$PYTHONPATH:`pwd`

# Set up the working environment.
CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}"

# Run model_test first to make sure the PYTHONPATH is correctly set.
# python "${WORK_DIR}"/deeplab_v3_plus_test.py -v

# Go to datasets folder and download PASCAL VOC 2012 segmentation dataset.
DATASET_DIR="datasets"
cd "${WORK_DIR}/${DATASET_DIR}"
#sh download_and_convert_voc2012.sh

# Go back to original directory.
cd "${CURRENT_DIR}"

# Set up the working directories.
PASCAL_FOLDER="pascal_voc2012"
EXP_FOLDER="exp/train_on_trainval_set_mobilenetv2"
#INIT_FOLDER="${WORK_DIR}/${DATASET_DIR}/${PASCAL_FOLDER}/init_models"
TRAIN_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${PASCAL_FOLDER}/${EXP_FOLDER}/train"
#EVAL_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${PASCAL_FOLDER}/${EXP_FOLDER}/eval"
#VIS_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${PASCAL_FOLDER}/${EXP_FOLDER}/vis"
#EXPORT_DIR="${WORK_DIR}/${DATASET_DIR}/${PASCAL_FOLDER}/${EXP_FOLDER}/export"
#mkdir -p "${INIT_FOLDER}"
mkdir -p "${TRAIN_LOGDIR}"
#mkdir -p "${EVAL_LOGDIR}"
#mkdir -p "${VIS_LOGDIR}"
#mkdir -p "${EXPORT_DIR}"

PASCAL_DATASET="${WORK_DIR}/${DATASET_DIR}/${PASCAL_FOLDER}/tfrecord"

python train.py --dataset_dir="${PASCAL_DATASET}"\
  --train_logdir="${TRAIN_LOGDIR}" \
  --train_epochs=1
