#!/bin/bash

# ==== 可配置参数 ====
ROOT="/TalkVid"    # 项目根目录
cd "$ROOT"

DATASET="TalkVid-test-10k"      # 训练的数据集名称, e.g., TalkVid, HDTF
DATA_PREFIX="/mnt/nvme2n1/smith/xidong/codebase/data"             # 存放数据集的路径前缀
INPUT_DIR="$DATA_PREFIX/$DATASET/videos-crop"   # 先进行face-crop，作为输入视频目录
OUTPUT_DIR="$DATA_PREFIX/$DATASET/new_face_info"   # 输出目录，用于存储提取的面部信息
LOG_DIR="$ROOT/logs/extract_face_info"  # 日志目录
MODEL_ROOT="$ROOT/model_ckpts/insightface_models"
NUM_GPUS=6  # 使用的GPU数量
NUM_WORKERS=16  # 每个GPU的工作进程数
SHARD="true"  # 是否对输入的视频文件夹进行多卡并行处理
echo -e "数据集: $DATASET, \n输入目录: $INPUT_DIR, \n输出目录: $OUTPUT_DIR"

# ==== 设置可见的GPU设备 ====
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
cd "$(dirname "$0")"  # 切换到脚本所在目录
mkdir -p "$OUTPUT_DIR"  # 创建输出目录
mkdir -p "$LOG_DIR"  # 创建日志目录
mkdir -p "$MODEL_ROOT"  # 确保模型目录存在

# ==== 启动每张卡的任务 ====
for (( i=0; i<$NUM_GPUS; i++ ))
do
    echo "GPU $i → 视频数量: $VIDEOS_PER_SHARD → 日志: $LOG_DIR/extract_face_info_$i.log"
    CUDA_VISIBLE_DEVICES=$i python extract_face_info.py \
        --input_dir "$INPUT_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --model_root "$MODEL_ROOT" \
        --gpu_id $i \
        --num_workers $NUM_WORKERS \
        --num_gpus $NUM_GPUS \
        --shard $SHARD \
        > $LOG_DIR/extract_face_info_$i.log 2>&1 &  # 每张卡输出日志
done

# ==== 等待所有任务完成 ====
wait
echo "✅ extract-face-info 所有任务执行完成"
