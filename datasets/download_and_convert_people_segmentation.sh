#!/bin/bash
# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# Script to download and preprocess the people segmentation dataset.
#
# Usage:
#   bash ./download_and_convert_people_segmentation.sh
#
# The folder structure is assumed to be:
#  + datasets
#     - build_data.py
#     - download_and_convert_people_segmentation.sh
#     - remove_gt_colormap.py
#     + people_segmentation
#       + images
#       + masks
#

# Exit immediately if a command exits with a non-zero status.
set -e

export CUDA_VISIBLE_DEVICES=''

CURRENT_DIR=$(pwd)
WORK_DIR="./people_segmentation"
mkdir -p "${WORK_DIR}"
cd "${WORK_DIR}"

# Helper function to unpack people segmentation dataset.
uncompress() {
  local FILENAME=${1}

  echo "Uncompressing ${FILENAME}"
  tar -xf "${FILENAME}"
}

# Download the images.
FILENAME="people_segmentation.tar.gz"

uncompress "${FILENAME}"

cd "${CURRENT_DIR}"

# Root path for people segmentation dataset.
DS_ROOT="${WORK_DIR}"

# Remove the colormap in the ground truth annotations.
SEG_FOLDER="${DS_ROOT}/masks"

# Build TFRecords of the dataset.
# First, create output directory for storing TFRecords.
OUTPUT_DIR="${WORK_DIR}/tfrecord"
mkdir -p "${OUTPUT_DIR}"

IMAGE_FOLDER="${DS_ROOT}/images"
LIST_FOLDER="${DS_ROOT}/segmentation"

echo "Converting people segmentation dataset..."
python ./build_people_segmentation.py \
  --image_folder="${IMAGE_FOLDER}" \
  --semantic_segmentation_folder="${SEG_FOLDER}" \
  --list_folder="${LIST_FOLDER}" \
  --image_format="jpg" \
  --label_format="png" \
  --output_dir="${OUTPUT_DIR}"
