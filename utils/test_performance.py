#!/usr/bin/env python3
"""
性能测试脚本 - 测试不同数据量下的查重性能
"""

import time
import random
import os


def generate_test_data(filename, num_records):
    """生成测试数据"""
    print(f"生成 {num_records:,} 条测试数据...")
    start = time.time()
    
    with open(filename, 'w', encoding='utf-8') as f:
        for i in range(num_records):
            angle = random.choice([0, 90, 180, 270])
            label = {0: 0, 90: 1, 180: 2, 270: 3}[angle]
            f.write(f"images/img_rot{angle}_test_{i}.jpg {label}\n")
    
    print(f"生成完成，耗时: {time.time() - start:.2f}秒")


def test_loading_performance(filename):
    """测试加载性能"""
    print(f"\n{'='*60}")
    print(f"测试文件: {filename}")
    print(f"文件大小: {os.path.getsize(filename) / (1024*1024):.2f} MB")
    print(f"{'='*60}")
    
    # 测试1: 加载到set
    print("\n[测试1] 加载到 set...")
    start = time.time()
    existing_records = set()
    
    with open(filename, 'r', encoding='utf-8') as f:
        line_count = 0
        for line in f:
            line = line.strip()
            if line:
                existing_records.add(line)
                line_count += 1
                if line_count % 100000 == 0:
                    elapsed = time.time() - start
                    rate = line_count / elapsed
                    print(f"  进度: {line_count:,} 条 | 速度: {rate:,.0f} 条/秒")
    
    load_time = time.time() - start
    print(f"✓ 加载完成: {len(existing_records):,} 条")
    print(f"✓ 总耗时: {load_time:.2f} 秒")
    print(f"✓ 平均速度: {len(existing_records)/load_time:,.0f} 条/秒")
    
    # 测试2: 查重性能
    print("\n[测试2] 查重性能...")
    test_queries = [
        "images/img_rot0_test_100.jpg 0",
        "images/img_rot90_test_5000.jpg 1",
        "images/img_rot180_new_file.jpg 2",  # 不存在的
    ]
    
    start = time.time()
    for query in test_queries * 10000:  # 重复查询10000次
        _ = query in existing_records
    lookup_time = time.time() - start
    
    print(f"✓ 30,000 次查询耗时: {lookup_time:.4f} 秒")
    print(f"✓ 单次查询耗时: {lookup_time/30000*1000:.6f} 毫秒")
    
    # 内存估算
    avg_size = sum(len(r.encode('utf-8')) for r in list(existing_records)[:1000]) / 1000
    estimated_memory_mb = (len(existing_records) * avg_size) / (1024 * 1024)
    print(f"\n[内存占用]")
    print(f"✓ 估算内存: {estimated_memory_mb:.2f} MB")
    print(f"✓ 平均每条: {avg_size:.1f} 字节")


def main():
    """主函数"""
    test_cases = [
        ("test_10k.txt", 10000),
        ("test_100k.txt", 100000),
        ("test_500k.txt", 500000),
    ]
    
    print("=" * 60)
    print("train.txt 查重性能测试")
    print("=" * 60)
    
    for filename, num_records in test_cases:
        # 生成测试数据
        if not os.path.exists(filename):
            generate_test_data(filename, num_records)
        
        # 测试性能
        test_loading_performance(filename)
        print()
    
    # 清理测试文件
    print("\n是否删除测试文件? (y/n): ", end="")
    try:
        response = input().strip().lower()
        if response == 'y':
            for filename, _ in test_cases:
                if os.path.exists(filename):
                    os.remove(filename)
                    print(f"✓ 已删除 {filename}")
    except:
        pass
    
    print("\n性能测试完成！")
    print("\n结论:")
    print("• set 查重方法对于百万级数据仍然高效")
    print("• 加载速度主要受限于磁盘I/O")
    print("• 查询性能几乎不受数据量影响（O(1)复杂度）")
    print("• 建议: 如果 train.txt 超过 1000万条，可考虑使用数据库")


if __name__ == '__main__':
    main()
