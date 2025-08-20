#!/bin/bash
export NO_ALBUMENTATIONS_UPDATE="1"

### ----- Configuration -----
export STAGE=3  # Training stage
export TRAIN="TalkVid-Core"  # Training dataset
export GPU="0,1"    # GPU ID to use

config=./configs/$TRAIN/stage_${STAGE}.yaml
zero_config=./configs/zero2_config.json

export CHIEF_IP=127.0.0.1
export CHIEF_PORT=21001
export HOST_NUM=1
export INDEX=0

export HOST_GPU_NUM=2
export NCCL_IB_DISABLE=1
PROCESS_NUM=$((HOST_GPU_NUM * HOST_NUM))
echo "Total GPUS: ${PROCESS_NUM}"
echo "STAGE: ${STAGE}, TRAIN: ${TRAIN}, GPU id: ${GPU}"

LOG="./logs/train/stage_$STAGE-$TRAIN.log"
mkdir -p $(dirname $LOG)

accelerate launch --gpu_ids $GPU --use_deepspeed --num_processes ${PROCESS_NUM} \
    --deepspeed_config_file $zero_config \
    --num_machines "${HOST_NUM}" --machine_rank "${INDEX}" --main_process_ip "${CHIEF_IP}" --main_process_port "${CHIEF_PORT}" \
    --deepspeed_multinode_launcher standard \
    train.py  --config $config \
    > $LOG 2>&1
