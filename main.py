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

import argparse
import os
import shutil
import subprocess
import sys
import hashlib
from datetime import datetime

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
MAIN_LOG_FILE       = f"{OUTPUT_DIR}/main_pipeline.log"


def log_print(msg: str):
    """同时输出到控制台和日志文件"""
    print(msg)
    os.makedirs(os.path.dirname(MAIN_LOG_FILE), exist_ok=True)
    with open(MAIN_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def get_file_md5(filepath: str) -> str:
    """计算文件的 MD5 值"""
    if not os.path.isfile(filepath):
        return ""
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def run(cmd: list[str], step: str):
    """运行子命令，同时将输出流写入控制台和日志文件，失败时打印错误并退出"""
    log_print(f"\n{'=' * 60}")
    log_print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [{step}] 执行: {' '.join(cmd)}")
    log_print("=" * 60)
    
    os.makedirs(os.path.dirname(MAIN_LOG_FILE), exist_ok=True)
    with open(MAIN_LOG_FILE, "a", encoding="utf-8") as log_f:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            sys.stdout.write(line)
            log_f.write(line)
        process.wait()
        
    if process.returncode != 0:
        log_print(f"\n[错误] 步骤「{step}」失败，退出码 {process.returncode}")
        sys.exit(process.returncode)


# ─── Step 1: 检查新增数据 ────────────────────────────────────────────────────
def step1_check_new_images() -> list[str]:
    log_print("\n" + "=" * 60)
    log_print("[Step 1] 检查 new_images/ 目录")
    log_print("=" * 60)

    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    if not os.path.isdir(NEW_IMAGES_DIR):
        log_print(f"目录不存在: {NEW_IMAGES_DIR}，跳过增广步骤。")
        return []

    new_files = [
        f for f in os.listdir(NEW_IMAGES_DIR)
        if os.path.splitext(f)[1].lower() in image_exts
    ]

    if new_files:
        log_print(f"发现 {len(new_files)} 张新图片：")
        for f in new_files:
            log_print(f"  - {f}")
    else:
        log_print("未发现新图片，跳过增广步骤。")

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
    txt_hash = get_file_md5(TRAIN_TXT)
    config_hash = get_file_md5(CONFIG)
    combined_hash = f"{txt_hash}:{config_hash}"
    hash_file = os.path.join(OUTPUT_DIR, ".train_hash")
    old_hash = ""
    if os.path.isfile(hash_file):
        with open(hash_file, "r", encoding="utf-8") as f:
            old_hash = f.read().strip()

    train_cmd = [
        sys.executable, PADDLEX_MAIN,
        "-c", CONFIG,
        "-o", "Global.mode=train",
        "-o", f"Global.dataset_dir={DATASET_DIR}",
        "-o", f"Global.output={OUTPUT_DIR}",
    ]

    force_flag = not os.path.isdir(OUTPUT_DIR)  # output 已被清空则视为 force
    if not force_flag and combined_hash == old_hash and combined_hash != ":" and os.path.isfile(RESUME_PATH):
        log_print(f"\n[提示] 数据集与配置文件均未发生变化，将从 {RESUME_PATH} 恢复训练...")
        train_cmd.append("-o")
        train_cmd.append(f"Train.resume_path={RESUME_PATH}")
    else:
        log_print("\n[提示] 数据集或配置文件发生变化，或无历史权重，将从头开始训练...")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(hash_file, "w", encoding="utf-8") as f:
            f.write(combined_hash)

    run(train_cmd, step="Step 3b | 模型训练")


# ─── Step 4: 导出 ────────────────────────────────────────────────────────────
def step4_export():
    if not os.path.isfile(BEST_WEIGHT):
        log_print(f"\n[Step 4] 未找到最优权重 {BEST_WEIGHT}，跳过导出。")
        return
    # Export Inference
    run(
        [
            sys.executable, PADDLEX_MAIN,
            "-c", CONFIG,
            "-o", "Global.mode=export",
            "-o", f"Global.output={OUTPUT_DIR}",
            "-o", f"Export.weight_path={BEST_WEIGHT}",
        ],
        step="Step 4a | 导出 Inference 模型",
    )
    # Export ONNX
    onnx_dir = os.path.join(os.path.dirname(INFERENCE_DIR), "onnx")
    run(
        [
            "paddlex",
            "--paddle2onnx",
            "--paddle_model_dir", INFERENCE_DIR,
            "--onnx_model_dir", onnx_dir,
            "--opset_version", "11"
        ],
        step="Step 4b | 导出 ONNX 模型",
    )


# ─── Step 5: 对比 ────────────────────────────────────────────────────────────
def step5_compare():
    log_print("\n" + "=" * 60)
    log_print("[Step 5] 官方模型 vs 微调模型 效果对比")
    log_print("=" * 60)
    run(
        [sys.executable, "compare.py", "--max", "100"],
        step="Step 5 | 预测对比",
    )


# ─── 主流程 ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="全流程演示脚本")
    parser.add_argument("-f", "--force", action="store_true",
                        help="强制清空 output 目录，从头开始训练（不断点续训）")
    args = parser.parse_args()

    log_print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 启动 main.py 工作流...")

    # -f: 清空 output 目录
    if args.force:
        if os.path.isdir(OUTPUT_DIR):
            log_print(f"\n[--force] 清空输出目录: {OUTPUT_DIR}")
            shutil.rmtree(OUTPUT_DIR)
        log_print("[--force] 将从头开始训练，不使用断点续训。")

    # Step 1
    new_files = step1_check_new_images()

    # Step 2（仅当有新图片时执行）
    if new_files:
        step2_augment()
    else:
        log_print("\n[Step 2] 无新数据，跳过增广。")

    # Step 3
    step3_train()

    # Step 4
    step4_export()

    # Step 5
    step5_compare()

    log_print("\n" + "=" * 60)
    log_print("全流程完成！")
    log_print(f"  最优权重:      {BEST_WEIGHT}")
    log_print(f"  Inference 模型: {INFERENCE_DIR}")
    log_print(f"  对比结果:      {OUTPUT_DIR}/res_official.json")
    if os.path.isdir(INFERENCE_DIR):
        log_print(f"                 {OUTPUT_DIR}/res_finetuned.json")
    log_print("=" * 60)


if __name__ == "__main__":
    main()
