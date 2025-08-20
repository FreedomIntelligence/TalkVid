#!/bin/bash

# Get current path
CURRENT_PATH=$(pwd)
export PYTHONPATH="${PYTHONPATH}:${CURRENT_PATH}"

# Defining config file paths
CONFIG_PATH="${CURRENT_PATH}/../config/config_0_0.yaml"

# cd ./2_video_rough_segmentation
python -m video_clip --config ${CONFIG_PATH}
python -m duration_and_vocal_check --config ${CONFIG_PATH}
cd ../
