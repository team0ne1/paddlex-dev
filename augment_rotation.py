#!/usr/bin/env python3
"""
图像旋转数据增强工具
功能：将正放的图片旋转为0度、90度、180度、270度，并生成对应的标签文件
"""

import os
import time
from PIL import Image
from pathlib import Path
import argparse


def rotate_and_save_images(input_dir, output_dir, train_txt_path):
    """
    将输入目录中的图片旋转并保存，同时更新train.txt
    
    Args:
        input_dir: 输入图片目录
        output_dir: 输出图片目录
        train_txt_path: train.txt文件路径
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 旋转角度和对应的标签
    rotations = {
        0: 0,      # 0度 -> 标签0
        90: 1,     # 90度 -> 标签1
        180: 2,    # 180度 -> 标签2
        270: 3     # 270度 -> 标签3
    }
    
    # 获取所有图片文件
    input_path = Path(input_dir)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = [f for f in input_path.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"在 {input_dir} 中没有找到图片文件")
        return
    
    print(f"找到 {len(image_files)} 张图片，开始处理...")
    
    # 读取现有的train.txt中的记录，用于去重
    existing_records = set()
    if os.path.exists(train_txt_path):
        print(f"正在加载现有记录...")
        start_time = time.time()
        try:
            with open(train_txt_path, 'r', encoding='utf-8') as f:
                line_count = 0
                for line in f:
                    line = line.strip()
                    if line:  # 忽略空行
                        existing_records.add(line)
                        line_count += 1
                        # 每10万条显示一次进度
                        if line_count % 100000 == 0:
                            print(f"  已加载 {line_count:,} 条记录...")
            
            load_time = time.time() - start_time
            print(f"已加载现有记录: {len(existing_records):,} 条 (耗时: {load_time:.2f}秒)")
            
            # 内存使用估算
            if len(existing_records) > 0:
                avg_size = sum(len(r.encode('utf-8')) for r in list(existing_records)[:min(1000, len(existing_records))]) / min(1000, len(existing_records))
                estimated_memory_mb = (len(existing_records) * avg_size) / (1024 * 1024)
                print(f"估算内存占用: {estimated_memory_mb:.1f} MB")
                
        except Exception as e:
            print(f"读取现有train.txt时出错: {e}")
    
    # 用于存储新的训练数据行
    new_train_lines = []
    skipped_count = 0
    
    # 处理每张图片
    for idx, img_file in enumerate(image_files, 1):
        try:
            # 打开原始图片
            img = Image.open(img_file)
            
            # 获取原始文件名（不含扩展名）
            original_name = img_file.stem
            
            # 对每个角度进行旋转
            for angle, label in rotations.items():
                # 旋转图片
                if angle == 0:
                    rotated_img = img
                else:
                    # PIL的rotate是逆时针旋转，我们需要顺时针，所以用负值
                    rotated_img = img.rotate(-angle, expand=True)
                
                # 生成新的文件名
                new_filename = f"img_rot{angle}_{original_name}.jpg"
                output_path = os.path.join(output_dir, new_filename)
                
                # 保存旋转后的图片
                rotated_img.save(output_path, 'JPEG', quality=95)
                
                # 添加到训练数据列表（相对路径）
                relative_path = f"images/{new_filename}"
                record = f"{relative_path} {label}"
                
                # 检查是否已存在，避免重复
                if record in existing_records:
                    skipped_count += 1
                else:
                    new_train_lines.append(record + "\n")
            
            print(f"[{idx}/{len(image_files)}] 处理完成: {img_file.name}")
            
        except Exception as e:
            print(f"处理 {img_file.name} 时出错: {e}")
    
    # 写入train.txt（追加模式）
    if new_train_lines:
        mode = 'a' if os.path.exists(train_txt_path) else 'w'
        with open(train_txt_path, mode, encoding='utf-8') as f:
            f.writelines(new_train_lines)
        print(f"\n处理完成！")
        print(f"- 新增记录数量: {len(new_train_lines)}")
        print(f"- 跳过重复记录: {skipped_count}")
        print(f"- 保存位置: {output_dir}")
        print(f"- 标签文件: {train_txt_path} ({'追加' if mode == 'a' else '创建'})")
    else:
        print(f"\n处理完成！")
        print(f"- 所有记录均已存在，未新增任何数据")
        print(f"- 跳过重复记录: {skipped_count}")


def main():
    parser = argparse.ArgumentParser(
        description='图像旋转数据增强工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例用法:
  python augment_rotation.py -i ./my_images -o ./dataset/text_image_orientation/images
  python augment_rotation.py -i ./my_images -o ./output/images -t [train.txt](http://_vscodecontentref_/1)
        '''
    )
    
    parser.add_argument('-i', '--input', required=True,
                        help='输入图片目录路径')
    parser.add_argument('-o', '--output', required=True,
                        help='输出图片目录路径')
    parser.add_argument('-t', '--train-txt', 
                        default='./dataset/text_image_orientation/train.txt',
                        help='train.txt文件路径 (默认: ./dataset/text_image_orientation/train.txt)')
    
    args = parser.parse_args()
    
    # 检查输入目录
    if not os.path.exists(args.input):
        print(f"错误: 输入目录不存在: {args.input}")
        return
    
    # 执行处理
    rotate_and_save_images(args.input, args.output, args.train_txt)


if __name__ == '__main__':
    main()