#! /bin/bash
# config_file=$1

# Get current path
CURRENT_PATH=$(pwd)
export PYTHONPATH="${PYTHONPATH}:${CURRENT_PATH}"

# Defining config file paths
CONFIG_PATH="${CURRENT_PATH}/../config/config_0_0.yaml"

cd ./co-tracker

# # cotracker
python multi_gpu_runner_new.py \
    --config ${CONFIG_PATH} \

