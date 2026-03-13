#!/usr/bin/env bash
set -e

DATASET_DIR=./dataset/text_image_orientation
CONFIG=model_config.yaml
OUTPUT=./output

echo "===== [1/2] 数据集检查 ====="
python PaddleX/main.py -c "$CONFIG" \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir="$DATASET_DIR" \
    -o Global.output="$OUTPUT"

echo ""
echo "===== 数据集检查通过，开始训练 ====="
echo "===== [2/2] 模型训练 ====="
python PaddleX/main.py -c "$CONFIG" \
    -o Global.mode=train \
    -o Global.dataset_dir="$DATASET_DIR" \
    -o Global.output="$OUTPUT"