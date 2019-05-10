#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status.
set -e

export CUDA_VISIBLE_DEVICES=3
# Update PYTHONPATH.
export PYTHONPATH=$PYTHONPATH:`pwd`

# Set up the working environment.
CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}"
PRETRAINED_BACKBONE_MODEL_DIR='pretrained_backbone_model'
TMP_MODEL_DIR='tmp'
mkdir -p ${TMP_MODEL_DIR}

MODEL_TAR_NAME='pretrained_mobilenet_v2_1.0_224.tar.gz'
PRETRAINED_MODEL_PATH="http://cnbj1-fds.api.xiaomi.net/code/models/${MODEL_TAR_NAME}"
wget -q ${PRETRAINED_MODEL_PATH} -O ${TMP_MODEL_DIR}/${MODEL_TAR_NAME}

cd ${TMP_MODEL_DIR}
tar xvf ${MODEL_TAR_NAME}
cd ${WORK_DIR}

# Adapt pretrained mobilenet-v2 model
python utils/adapt_mobilenet_v2.py --pretrained_model_dir="${TMP_MODEL_DIR}" --output_dir="${PRETRAINED_BACKBONE_MODEL_DIR}"

#PRETRAINED_MODEL_DIR='pretrained_model'
#MODEL_TAR_NAME='deeplabv3_mnv2_pascal_train_aug.tar.gz'
#PRETRAINED_MODEL_PATH="http://cnbj1-fds.api.xiaomi.net/code/models/${MODEL_TAR_NAME}"
#wget -q ${PRETRAINED_MODEL_PATH} -O ${TMP_MODEL_DIR}/${MODEL_TAR_NAME}
#
#cd ${TMP_MODEL_DIR}
#tar xvf ${MODEL_TAR_NAME}
#cd ${WORK_DIR}
#
## Adapt pretrained deeplab-v3-plus model for validation
#python utils/adapt_deeplab_mnv2.py --pretrained_model_dir="${TMP_MODEL_DIR}" --output_dir="${PRETRAINED_MODEL_DIR}"

# clear
rm -rf "${TMP_MODEL_DIR}"

echo "============Successful============="
