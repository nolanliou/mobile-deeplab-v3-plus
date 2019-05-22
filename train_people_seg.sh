#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status.
set -e

export CUDA_VISIBLE_DEVICES=3
# Update PYTHONPATH.
export PYTHONPATH=$PYTHONPATH:`pwd`

MODEL_TYPE='deeplab-v3-plus'
PRETRAINED_MODEL_DIR='pretrained_model'
PRETRAINED_BACKBONE_MODEL_DIR='pretrained_backbone_model'
# Set up the working environment.
CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}"

# Go to datasets folder and download PASCAL VOC 2012 segmentation dataset.
DATASET_DIR="datasets"
cd "${WORK_DIR}/${DATASET_DIR}"
sh download_and_convert_people_segmentation.sh

# Go back to original directory.
cd "${CURRENT_DIR}"

# Set up the working directories.
PS_FOLDER="people_segmentation"
EXP_FOLDER="exp"
TRAIN_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${PS_FOLDER}/${EXP_FOLDER}/${MODEL_TYPE}/train"
mkdir -p "${TRAIN_LOGDIR}"

PS_DATASET="${WORK_DIR}/${DATASET_DIR}/${PS_FOLDER}/tfrecord"

python run.py --dataset_dir="${PS_DATASET}"\
  --logdir="${TRAIN_LOGDIR}" \
  --model_type="${MODEL_TYPE}" \
  --backbone="MobilenetV2" \
  --dataset_name="people_segmentation" \
  --train_subset="train" \
  --base_learning_rate=0.05 \
  --num_clones=1 \
  --training_number_of_steps=150000 \
  --decoder_output_stride=4 \
  --model_input_size=513 \
  --model_input_size=513 \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --quant_friendly=True

# --pretrained_backbone_model_dir="${PRETRAINED_BACKBONE_MODEL_DIR}"
#  --pretrained_model_dir="${PRETRAINED_MODEL_DIR}" \
