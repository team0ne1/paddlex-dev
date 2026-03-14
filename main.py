#!/usr/bin/env python3
"""
全流程演示脚本
步骤：
  1. 检查 new_images/ 目录是否存在新增数据
  2. 运行 augment_rotation.py 对新数据进行旋转增广并写入数据集标注
  3. 训练模型（check_dataset + train）
  4. 导出模型为 Inference 格式
  5. 与官方预训练模型进行预测效果对比
"""

import os
import subprocess
import sys

# ─── 路径配置 ────────────────────────────────────────────────────────────────
NEW_IMAGES_DIR      = "./new_images"
DATASET_IMAGES_DIR  = "./dataset/text_image_orientation/images"
TRAIN_TXT           = "./dataset/text_image_orientation/train.txt"
VAL_TXT             = "./dataset/text_image_orientation/val.txt"
DATASET_DIR         = "./dataset/text_image_orientation"
CONFIG              = "model_config.yaml"
OUTPUT_DIR          = "./output"
RESUME_PATH         = f"{OUTPUT_DIR}/latest/latest.pdparams"
BEST_WEIGHT         = f"{OUTPUT_DIR}/best_model/best_model.pdparams"
INFERENCE_DIR       = f"{OUTPUT_DIR}/best_model/inference"
PADDLEX_MAIN        = "PaddleX/main.py"


def run(cmd: list[str], step: str):
    """运行子命令，失败时打印错误并退出"""
    print(f"\n{'=' * 60}")
    print(f"[{step}] 执行: {' '.join(cmd)}")
    print("=" * 60)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n[错误] 步骤「{step}」失败，退出码 {result.returncode}")
        sys.exit(result.returncode)


# ─── Step 1: 检查新增数据 ────────────────────────────────────────────────────
def step1_check_new_images() -> list[str]:
    print("\n" + "=" * 60)
    print("[Step 1] 检查 new_images/ 目录")
    print("=" * 60)

    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    if not os.path.isdir(NEW_IMAGES_DIR):
        print(f"目录不存在: {NEW_IMAGES_DIR}，跳过增广步骤。")
        return []

    new_files = [
        f for f in os.listdir(NEW_IMAGES_DIR)
        if os.path.splitext(f)[1].lower() in image_exts
    ]

    if new_files:
        print(f"发现 {len(new_files)} 张新图片：")
        for f in new_files:
            print(f"  - {f}")
    else:
        print("未发现新图片，跳过增广步骤。")

    return new_files


# ─── Step 2: 增广旋转 ────────────────────────────────────────────────────────
def step2_augment():
    run(
        [
            sys.executable, "augment_rotation.py",
            "-i", NEW_IMAGES_DIR,
            "-o", DATASET_IMAGES_DIR,
            "-t", TRAIN_TXT,
            "--val-txt", VAL_TXT,
            "--val-split", "20",
            "--oversample", "5",
            "--seed", "42",
        ],
        step="Step 2 | 旋转增广 & 更新标注",
    )


# ─── Step 3: 训练 ────────────────────────────────────────────────────────────
def step3_train():
    # 3a. 数据集检查
    run(
        [
            sys.executable, PADDLEX_MAIN,
            "-c", CONFIG,
            "-o", "Global.mode=check_dataset",
            "-o", f"Global.dataset_dir={DATASET_DIR}",
            "-o", f"Global.output={OUTPUT_DIR}",
        ],
        step="Step 3a | 数据集检查",
    )
    # 3b. 模型训练
    run(
        [
            sys.executable, PADDLEX_MAIN,
            "-c", CONFIG,
            "-o", "Global.mode=train",
            "-o", f"Global.dataset_dir={DATASET_DIR}",
            "-o", f"Train.resume_path={RESUME_PATH}",
            "-o", f"Global.output={OUTPUT_DIR}",
        ],
        step="Step 3b | 模型训练",
    )


# ─── Step 4: 导出 ────────────────────────────────────────────────────────────
def step4_export():
    if not os.path.isfile(BEST_WEIGHT):
        print(f"\n[Step 4] 未找到最优权重 {BEST_WEIGHT}，跳过导出。")
        return
    run(
        [
            sys.executable, PADDLEX_MAIN,
            "-c", CONFIG,
            "-o", "Global.mode=export",
            "-o", f"Global.output={OUTPUT_DIR}",
            "-o", f"Export.weight_path={BEST_WEIGHT}",
        ],
        step="Step 4 | 导出 Inference 模型",
    )


# ─── Step 5: 对比 ────────────────────────────────────────────────────────────
def step5_compare():
    print("\n" + "=" * 60)
    print("[Step 5] 官方模型 vs 微调模型 效果对比")
    print("=" * 60)
    run(
        [sys.executable, "compare.py", "--max", "50"],
        step="Step 5 | 预测对比",
    )


# ─── 主流程 ──────────────────────────────────────────────────────────────────
def main():
    # Step 1
    new_files = step1_check_new_images()

    # Step 2（仅当有新图片时执行）
    if new_files:
        step2_augment()
    else:
        print("\n[Step 2] 无新数据，跳过增广。")

    # Step 3
    step3_train()

    # Step 4
    step4_export()

    # Step 5
    step5_compare()

    print("\n" + "=" * 60)
    print("全流程完成！")
    print(f"  最优权重:      {BEST_WEIGHT}")
    print(f"  Inference 模型: {INFERENCE_DIR}")
    print(f"  对比结果:      {OUTPUT_DIR}/res_official.json")
    if os.path.isdir(INFERENCE_DIR):
        print(f"                 {OUTPUT_DIR}/res_finetuned.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
