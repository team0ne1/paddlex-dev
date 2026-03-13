#!/usr/bin/env bash
set -e

python augment_rotation.py \
    -i ./new_images \
    -o ./dataset/text_image_orientation/images \
    --train-txt ./dataset/text_image_orientation/train.txt \
    --val-txt ./dataset/text_image_orientation/val.txt \
    --val-split 20 \
    --oversample 5 \
    --seed 42
