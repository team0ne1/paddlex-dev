"""
计算数据集的 mean 和 std 示例代码
用于自定义数据集的归一化参数计算
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


def calculate_mean_std(image_dir, sample_size=None):
    """
    计算数据集的均值和标准差
    
    Args:
        image_dir: 图片目录路径
        sample_size: 采样数量（None表示使用全部图片）
    
    Returns:
        mean: [R, G, B] 三通道均值
        std: [R, G, B] 三通道标准差
    """
    image_paths = list(Path(image_dir).rglob("*.jpg")) + \
                  list(Path(image_dir).rglob("*.png"))
    
    if sample_size and len(image_paths) > sample_size:
        image_paths = np.random.choice(image_paths, sample_size, replace=False)
    
    print(f"计算 {len(image_paths)} 张图片的统计值...")
    
    # 方法1: 逐像素计算（精确但慢）
    pixel_values = []
    for img_path in tqdm(image_paths):
        img = cv2.imread(str(img_path))
        if img is not None:
            # 转为RGB并归一化到[0,1]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            pixel_values.append(img.reshape(-1, 3))
    
    all_pixels = np.vstack(pixel_values)
    
    mean = np.mean(all_pixels, axis=0)
    std = np.std(all_pixels, axis=0)
    
    print(f"\n计算结果:")
    print(f"mean: [{mean[0]:.3f}, {mean[1]:.3f}, {mean[2]:.3f}]")
    print(f"std:  [{std[0]:.3f}, {std[1]:.3f}, {std[2]:.3f}]")
    
    return mean, std


def calculate_mean_std_fast(image_dir, sample_size=1000):
    """
    快速计算（使用采样和在线统计）
    
    适用于大规模数据集
    """
    image_paths = list(Path(image_dir).rglob("*.jpg")) + \
                  list(Path(image_dir).rglob("*.png"))
    
    if len(image_paths) > sample_size:
        image_paths = np.random.choice(image_paths, sample_size, replace=False)
    
    print(f"快速计算模式：采样 {len(image_paths)} 张图片")
    
    # 在线计算均值和方差（Welford算法）
    n = 0
    mean = np.zeros(3)
    M2 = np.zeros(3)
    
    for img_path in tqdm(image_paths):
        img = cv2.imread(str(img_path))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            
            # 计算图片级别的均值
            img_mean = np.mean(img.reshape(-1, 3), axis=0)
            
            n += 1
            delta = img_mean - mean
            mean += delta / n
            delta2 = img_mean - mean
            M2 += delta * delta2
    
    variance = M2 / (n - 1) if n > 1 else M2
    std = np.sqrt(variance)
    
    print(f"\n快速计算结果:")
    print(f"mean: [{mean[0]:.3f}, {mean[1]:.3f}, {mean[2]:.3f}]")
    print(f"std:  [{std[0]:.3f}, {std[1]:.3f}, {std[2]:.3f}]")
    
    return mean, std


if __name__ == "__main__":
    # 示例用法
    # mean, std = calculate_mean_std("./dataset/text_image_orientation/images")
    
    # 或者使用快速模式
    # mean, std = calculate_mean_std_fast("./dataset/text_image_orientation/images")
    
    # ImageNet标准值（参考）
    print("ImageNet标准值:")
    print("mean: [0.485, 0.456, 0.406]")
    print("std:  [0.229, 0.224, 0.225]")
    print("\n提示：如果使用预训练模型，建议直接使用ImageNet的值")
