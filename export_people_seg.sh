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
PS_FOLDER="people_segmentation"
EXP_FOLDER="exp"
TRAIN_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${PS_FOLDER}/${EXP_FOLDER}/${MODEL_TYPE}/train"

PS_DATASET="${WORK_DIR}/${DATASET_DIR}/${PS_FOLDER}/tfrecord"

#echo 'Evaluation'
#python run.py --dataset_dir="${PS_DATASET}"\
#  --logdir="${TRAIN_LOGDIR}" \
#  --dataset_name="people_segmentation" \
#  --model_type="${MODEL_TYPE}" \
#  --mode=eval

echo 'Export model'
EXPORT_DIR="${WORK_DIR}/${DATASET_DIR}/${PS_FOLDER}/${EXP_FOLDER}/${MODEL_TYPE}/export"
rm -rf "${EXPORT_DIR}"
mkdir -p "${EXPORT_DIR}"

python run.py --dataset_dir="${PS_DATASET}"\
  --logdir="${TRAIN_LOGDIR}" \
  --dataset_name="people_segmentation" \
  --model_type="${MODEL_TYPE}" \
  --backbone="MobilenetV2" \
  --decoder_output_stride=4 \
  --mode=export \
  --export_dir="${EXPORT_DIR}" \
  --model_input_size=513 \
  --model_input_size=513 \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --quant_friendly=True

# freeze
python freeze.py --model_dir="${EXPORT_DIR}" \
  --output_node_names=Output \
  --output_dir="${EXPORT_DIR}"
