#!/bin/bash
# ===== Configuration =====
ROOT="/TalkVid"
HF_REPO="tk93/V-Express"
TARGET_DIR="$ROOT/model_ckpts"
NESTED_DIR="$TARGET_DIR/model_ckpts"

# ===== Download model to the target directory =====
mkdir -p "$TARGET_DIR"
# huggingface-cli download "$HF_REPO" --local-dir "$TARGET_DIR"
if ! huggingface-cli download "$HF_REPO" --local-dir "$TARGET_DIR"; then
  echo "huggingface-cli download failed, aborting"
  exit 1
fi

# ===== Move top-level *.bin files to v-express (if exists) =====
VEXPRESS_DIR=$(find "$NESTED_DIR" -type d -name "v-express" | head -n 1)
if [ -n "$VEXPRESS_DIR" ]; then
  echo "Found v-express dir: $VEXPRESS_DIR"
  find "$TARGET_DIR" -maxdepth 1 -type f -name '*.bin' -exec mv {} "$VEXPRESS_DIR"/ \;
else
  echo "v-express directory not found, skipping *.bin move"
fi

# ===== Promote model_ckpts/model_ckpts/* to model_ckpts/ =====
if [ -d "$NESTED_DIR" ]; then
  echo "Promoting nested directory contents..."
  find "$NESTED_DIR" -mindepth 1 -maxdepth 1 -exec mv -t "$TARGET_DIR" {} +
  rm -rf "$NESTED_DIR"
else
  echo "Nested directory $NESTED_DIR not found"
fi

# ===== Completion status =====
echo "✅ Model directory organization complete. Current contents:"
ls -lh "$TARGET_DIR"
