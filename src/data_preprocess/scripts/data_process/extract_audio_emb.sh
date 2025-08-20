#!/bin/bash
# ==== 检查ffmpeg是否已安装 ====
if ! command -v ffmpeg >/dev/null 2>&1; then
    apt update && apt install -y ffmpeg
fi

# ==== 可配置参数 ====
ROOT="/TalkVid"    # 项目根目录
cd "$ROOT"

DATASET="TalkVid-test-10k"      # 训练的数据集名称, e.g., TalkVid, HDTF
DATA_PREFIX="/mnt/nvme2n1/smith/xidong/codebase/data"             # 存放数据集的路径前缀
INPUT_DIR="$DATA_PREFIX/$DATASET/videos"   # 绝对路径，未进行face-crop的视频目录，作为输入视频目录
AUDIO_DIR="$DATA_PREFIX/$DATASET/audios"  # 绝对路径，原始音频文件目录
OUTPUT_DIR="$DATA_PREFIX/$DATASET/short_clip_aud_embeds"   # 绝对路径，输出目录，用于存储提取的音频特征
LOG_DIR="$ROOT/logs/extract_audio_emb"  # 日志目录
MODEL_ROOT="$ROOT/model_ckpts/wav2vec2-base-960h"   # 音频特征提取模型路径，hf: facebook/wav2vec2-base-960h
NUM_GPUS=6  # 使用的GPU数量
NUM_WORKERS=4  # 每个GPU的工作进程数，每个进程显存占用~2GB，workers=4/8效率较高
SHARD="true"  # 是否对输入的视频文件夹进行多卡并行处理
echo -e "数据集: $DATASET, \n输入目录: $INPUT_DIR, \n音频目录: $AUDIO_DIR, \n输出目录: $OUTPUT_DIR"

# ==== 设置可见的GPU设备 ====
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
# export HF_ENDPOINT=https://hf-mirror.com  # 可选
mkdir -p "$OUTPUT_DIR"  # 创建输出目录
mkdir -p "$LOG_DIR"  # 创建日志目录
if [[ "$DATASET" == *TalkVid* ]]; then
    MODE_AUDIO="true"
elif [[ "$DATASET" == *HDTF* || "$DATASET" == *hdtf* || "$DATASET" == *hallo3* ]]; then
    MODE_AUDIO="false"
else
    echo "Unsupported dataset!"
    exit 1
fi

# ==== 启动每张卡的任务 ====
cd "$(dirname "$0")"  # 切换到脚本所在目录
for (( i=0; i<$NUM_GPUS; i++ ))
do
    echo "GPU $i → 日志: $LOG_DIR/extract_audio_emb_$i.log"
    CUDA_VISIBLE_DEVICES=$i python extract_audio_emb.py \
        --input_dir "$INPUT_DIR" \
        --audio_dir "$AUDIO_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --model_root "$MODEL_ROOT" \
        --gpu_id $i \
        --num_workers $NUM_WORKERS \
        --num_gpus $NUM_GPUS \
        --shard $SHARD \
        --mode_audio $MODE_AUDIO \
        > $LOG_DIR/extract_audio_emb_$i.log 2>&1 &  # 每张卡输出日志
done

# ==== 等待所有任务完成 ====
wait
echo "✅ extract-audio-emb 所有任务执行完成"
