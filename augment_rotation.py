#!/usr/bin/env python3
"""
图像旋转数据增强工具
功能：将正放的图片旋转为0度、90度、180度、270度，并生成对应的标签文件
支持将新数据按比例分割写入 train.txt 和 val.txt，并对训练集过采样
"""

import os
import random
import time
from PIL import Image
from pathlib import Path
import argparse


def load_existing_records(*txt_paths):
    """从多个 txt 文件中加载已有记录，用于去重"""
    existing_records = set()
    start_time = time.time()
    total_lines = 0
    for txt_path in txt_paths:
        if not os.path.exists(txt_path):
            continue
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    existing_records.add(line)
                    total_lines += 1
                    if total_lines % 100000 == 0:
                        print(f"  已加载 {total_lines:,} 条记录...")
    if total_lines > 0:
        load_time = time.time() - start_time
        print(f"已加载现有记录: {len(existing_records):,} 条 (耗时: {load_time:.2f}秒)")
    return existing_records


def rotate_and_save_images(input_dir, output_dir, train_txt_path, val_txt_path,
                            val_split, oversample, seed):
    """
    将输入目录中的图片旋转并保存，同时按比例更新 train.txt 和 val.txt

    Args:
        input_dir:      输入图片目录
        output_dir:     输出图片目录（应位于数据集 images/ 下）
        train_txt_path: train.txt 文件路径
        val_txt_path:   val.txt 文件路径
        val_split:      验证集百分比，按源图像划分，取值 1~99（整数）
        oversample:     训练集过采样倍数（val 集不重复）
        seed:           随机种子，保证可复现
    """
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    rotations = {
        0: 0,    # 0度   -> 标签0
        90: 1,   # 90度  -> 标签1
        180: 2,  # 180度 -> 标签2
        270: 3   # 270度 -> 标签3
    }

    input_path = Path(input_dir)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = sorted([f for f in input_path.iterdir()
                          if f.suffix.lower() in image_extensions])

    if not image_files:
        print(f"在 {input_dir} 中没有找到图片文件")
        return

    print(f"找到 {len(image_files)} 张源图片")
    print(f"参数: val_split={val_split}, oversample={oversample}x, seed={seed}")

    # 读取两个文件中的现有记录，统一去重
    print("正在加载现有记录...")
    existing_records = load_existing_records(train_txt_path, val_txt_path)

    new_train_lines = []
    new_val_lines = []
    skipped_train = 0
    skipped_val = 0

    for idx, img_file in enumerate(image_files, 1):
        try:
            img = Image.open(img_file)
            original_name = img_file.stem

            # 按源图像决定整组旋转图归属，防止数据泄露
            is_val = random.random() * 100 < val_split

            group_records = []
            for angle, label in rotations.items():
                rotated_img = img if angle == 0 else img.rotate(-angle, expand=True)

                new_filename = f"img_rot{angle}_{original_name}.jpg"
                output_path = os.path.join(output_dir, new_filename)
                rotated_img.save(output_path, 'JPEG', quality=95)

                record = f"images/{new_filename} {label}"
                group_records.append(record)

            if is_val:
                for record in group_records:
                    if record in existing_records:
                        skipped_val += 1
                    else:
                        new_val_lines.append(record + "\n")
            else:
                for record in group_records:
                    if record in existing_records:
                        skipped_train += 1
                    else:
                        # 训练集过采样：重复写入 oversample 次
                        new_train_lines.extend([record + "\n"] * oversample)

            split_tag = "val" if is_val else f"train(×{oversample})"
            print(f"[{idx}/{len(image_files)}] {img_file.name} → {split_tag}")

        except Exception as e:
            print(f"处理 {img_file.name} 时出错: {e}")

    # 写入 train.txt
    if new_train_lines:
        mode = 'a' if os.path.exists(train_txt_path) else 'w'
        with open(train_txt_path, mode, encoding='utf-8') as f:
            f.writelines(new_train_lines)

    # 写入 val.txt
    if new_val_lines:
        mode = 'a' if os.path.exists(val_txt_path) else 'w'
        with open(val_txt_path, mode, encoding='utf-8') as f:
            f.writelines(new_val_lines)

    print(f"\n处理完成！")
    actual_new_train = len(new_train_lines) // oversample if oversample > 0 else len(new_train_lines)
    print(f"- train.txt 新增: {actual_new_train} 条记录 × {oversample} 倍 = {len(new_train_lines)} 行，跳过重复: {skipped_train}")
    print(f"- val.txt   新增: {len(new_val_lines)} 条记录，跳过重复: {skipped_val}")
    print(f"- 图片保存位置: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='图像旋转数据增强工具（支持 train/val 分割与过采样）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例用法:
  # 默认：val_split=20（即20%进入验证集），训练集过采样5倍
  python augment_rotation.py -i ./new_images -o ./dataset/text_image_orientation/images

  # 自定义验证集比例（15%）和过采样倍数
  python augment_rotation.py -i ./new_images -o ./dataset/text_image_orientation/images \\
      --val-split 15 --oversample 8

  # 固定随机种子保证 train/val 划分可复现
  python augment_rotation.py -i ./new_images -o ./dataset/text_image_orientation/images \\
      --seed 42
        '''
    )

    parser.add_argument('-i', '--input', required=True,
                        help='输入图片目录路径')
    parser.add_argument('-o', '--output', required=True,
                        help='输出图片目录路径（应为数据集的 images/ 目录）')
    parser.add_argument('-t', '--train-txt',
                        default='./dataset/text_image_orientation/train.txt',
                        help='train.txt 文件路径 (默认: ./dataset/text_image_orientation/train.txt)')
    parser.add_argument('--val-txt',
                        default='./dataset/text_image_orientation/val.txt',
                        help='val.txt 文件路径 (默认: ./dataset/text_image_orientation/val.txt)')
    parser.add_argument('--val-split', type=int, default=20,
                        help='验证集百分比，按源图像划分，取值 1~99 (默认: 20，即 20%%)')
    parser.add_argument('--oversample', type=int, default=5,
                        help='训练集过采样倍数，val 集不重复 (默认: 5)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子，保证 train/val 划分可复现 (默认: 42)')

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"错误: 输入目录不存在: {args.input}")
        return

    if not (0 < args.val_split < 100):
        print(f"错误: --val-split 必须在 1 到 99 之间，当前值: {args.val_split}")
        return

    if args.oversample < 1:
        print(f"错误: --oversample 必须 >= 1，当前值: {args.oversample}")
        return

    rotate_and_save_images(
        input_dir=args.input,
        output_dir=args.output,
        train_txt_path=args.train_txt,
        val_txt_path=args.val_txt,
        val_split=args.val_split,
        oversample=args.oversample,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()