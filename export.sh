#!/usr/bin/env bash
set -e

CONFIG=model_config.yaml
OUTPUT=./output

echo "===== 导出模型为 Inference 格式 ====="
python PaddleX/main.py -c "$CONFIG" \
    -o Global.mode=export \
    -o Global.output="$OUTPUT" \
    -o Export.weight_path="$OUTPUT/best_model/best_model.pdparams"

echo ""
echo "导出完成，inference 模型保存在: $OUTPUT/best_model/inference"
