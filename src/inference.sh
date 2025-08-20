#!/bin/bash

# ==== Configuration ====
ROOT="/TalkVid"
cd "$ROOT"

TEST_DATASET="TalkVid-bench"
TRAINING_DATASET="TalkVid-Core"
DATA_PREFIX="$ROOT/evaluation/testset"
STEPS=50000
INPUT_IMG_DIR="$DATA_PREFIX/$TEST_DATASET/data/imgs"
INPUT_AUDIO_DIR="$DATA_PREFIX/$TEST_DATASET/data/audios"
INPUT_VKPS_DIR="$DATA_PREFIX/$TEST_DATASET/data/vkps"
OUTPUT_DIR="$DATA_PREFIX/$TEST_DATASET/trainset/$TRAINING_DATASET/videos"
MODEL_DIR="$ROOT/exp/$TRAINING_DATASET"
MODEL_ROOT="$ROOT/model_ckpts/insightface_models"

mkdir -p "$OUTPUT_DIR"

# ==== Model checkpoint paths ====
denoising_unet_path="$MODEL_DIR/stage_3/denoising_unet-$STEPS.pth"
reference_net_path="$MODEL_DIR/stage_3/reference_net-$STEPS.pth"
v_kps_guider_path="$MODEL_DIR/stage_3/v_kps_guider-$STEPS.pth"
audio_projection_path="$MODEL_DIR/stage_3/audio_projection-$STEPS.pth"
motion_module_path="$MODEL_DIR/stage_3/motion_module-$STEPS.pth"

# ==== Single sample path ====
reference_image_path="./test_samples/short_case/tys/ref.jpg"
audio_path="./test_samples/short_case/tys/aud.mp3"
vkps_path="./test_samples/short_case/tys/kps.pth"

base_name=$(basename "$reference_image_path" .jpg)
output_path="$OUTPUT_DIR/${base_name}.mp4"

# ==== Run inference ====
python inference.py \
    --reference_image_path "$reference_image_path" \
    --audio_path "$audio_path" \
    --kps_path "$vkps_path" \
    --output_path "$output_path" \
    --denoising_unet_path "$denoising_unet_path" \
    --reference_net_path "$reference_net_path" \
    --v_kps_guider_path "$v_kps_guider_path" \
    --audio_projection_path "$audio_projection_path" \
    --motion_module_path "$motion_module_path" \
    --retarget_strategy "naive_retarget" \
    --num_inference_steps 25 \
    --guidance_scale 3.5 \
    --audio_attention_weight 2.0 \
    --context_frames 24 \
    --test_stage "stage_3"
