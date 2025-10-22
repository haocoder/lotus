#!/usr/bin/env python3
"""
图片压缩测试脚本

测试图片压缩效果和base64编码大小优化
"""

import os
import sys
import time
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

def test_image_compression():
    """测试图片压缩功能"""
    print("=== 图片压缩测试 ===")
    
    try:
        from lotus.utils.image_optimizer import ImageOptimizer, optimize_image_for_processing
        
        # 创建测试图片（如果没有PIL，跳过实际测试）
        try:
            from PIL import Image
            import numpy as np
            
            # 创建测试图片
            test_image = Image.new('RGB', (2048, 2048), color='red')
            print(f"✓ 创建测试图片: {test_image.size}")
            
            # 测试不同压缩级别
            compression_configs = [
                {"max_size": (1024, 1024), "quality": 85, "format": "JPEG", "name": "标准压缩"},
                {"max_size": (512, 512), "quality": 70, "format": "JPEG", "name": "中等压缩"},
                {"max_size": (256, 256), "quality": 50, "format": "JPEG", "name": "高压缩"},
                {"max_size": (128, 128), "quality": 30, "format": "JPEG", "name": "极高压缩"},
            ]
            
            print("\n--- 压缩效果测试 ---")
            for config in compression_configs:
                start_time = time.time()
                
                # 创建优化器
                optimizer = ImageOptimizer(
                    max_size=config["max_size"],
                    quality=config["quality"],
                    format=config["format"]
                )
                
                # 压缩图片
                compressed = optimizer._compress_image(test_image)
                
                end_time = time.time()
                compression_time = end_time - start_time
                
                # 计算压缩效果
                original_size = len(test_image.tobytes())
                compressed_size = len(compressed)
                compression_ratio = (1 - compressed_size / original_size) * 100
                
                print(f"[{config['name']}]:")
                print(f"   原始大小: {original_size:,} bytes")
                print(f"   压缩后: {compressed_size:,} bytes")
                print(f"   压缩率: {compression_ratio:.1f}%")
                print(f"   处理时间: {compression_time:.3f}s")
                print()
            
            # 测试渐进式压缩
            print("--- 渐进式压缩测试 ---")
            optimizer = ImageOptimizer(max_size=(1024, 1024), quality=85)
            start_time = time.time()
            result = optimizer.optimize_image(test_image)
            end_time = time.time()
            
            print(f"✓ 渐进式压缩完成")
            print(f"  处理时间: {end_time - start_time:.3f}s")
            print(f"  最终大小: {len(result):,} bytes")
            
        except ImportError as e:
            print(f"PIL不可用，跳过实际压缩测试: {e}")
            print("   但可以测试URL检测和缓存功能")
            
            # 测试URL检测
            optimizer = ImageOptimizer()
            test_urls = [
                "https://example.com/image.jpg",
                "http://test.com/pic.png",
                "s3://bucket/image.webp",
                "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD",
                "/path/to/local/image.jpg"
            ]
            
            print("\n--- URL检测测试 ---")
            for url in test_urls:
                is_url = optimizer.is_url_image(url)
                is_encoded = optimizer.is_encoded_image(url)
                is_file = optimizer.is_file_path(url)
                
                print(f"URL: {url[:50]}...")
                print(f"  是URL: {is_url}")
                print(f"  已编码: {is_encoded}")
                print(f"  文件路径: {is_file}")
                print()
        
        # 测试缓存功能
        print("--- 缓存功能测试 ---")
        optimizer = ImageOptimizer(enable_cache=True, cache_size=10)
        
        # 模拟缓存操作
        test_data = "test_image_data"
        for i in range(15):  # 超过缓存大小
            key = f"image_{i}"
            optimizer._cache[key] = f"compressed_{i}"
            optimizer._cache_access_times[key] = time.time()
            optimizer._cache_sizes[key] = 1000
        
        print(f"缓存大小: {len(optimizer._cache)}")
        print(f"缓存统计: {optimizer.get_cache_stats()}")
        
        # 清理缓存
        optimizer.clear_cache()
        print(f"清理后缓存大小: {len(optimizer._cache)}")
        
        print("\n图片压缩测试完成")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()

def test_base64_optimization():
    """测试base64编码优化"""
    print("\n=== Base64编码优化测试 ===")
    
    try:
        from lotus.utils.image_optimizer import ImageOptimizer
        
        # 测试不同格式的base64编码
        test_cases = [
            {
                "name": "小图片",
                "size": (100, 100),
                "quality": 85,
                "format": "JPEG"
            },
            {
                "name": "中等图片", 
                "size": (500, 500),
                "quality": 85,
                "format": "JPEG"
            },
            {
                "name": "大图片",
                "size": (1000, 1000), 
                "quality": 85,
                "format": "JPEG"
            },
            {
                "name": "PNG格式",
                "size": (500, 500),
                "quality": 85,
                "format": "PNG"
            }
        ]
        
        print("--- Base64编码大小对比 ---")
        for case in test_cases:
            try:
                from PIL import Image
                
                # 创建测试图片
                test_image = Image.new('RGB', case["size"], color='blue')
                
                # 创建优化器
                optimizer = ImageOptimizer(
                    max_size=(800, 800),
                    quality=case["quality"],
                    format=case["format"]
                )
                
                # 压缩并编码
                compressed = optimizer._compress_image(test_image)
                
                # 计算大小
                base64_size = len(compressed)
                original_size = len(test_image.tobytes())
                reduction = (1 - base64_size / original_size) * 100
                
                print(f"[{case['name']}] ({case['size'][0]}x{case['size'][1]}):")
                print(f"   原始大小: {original_size:,} bytes")
                print(f"   Base64大小: {base64_size:,} bytes")
                print(f"   减少: {reduction:.1f}%")
                print()
                
            except ImportError:
                print(f"PIL不可用，跳过 {case['name']} 测试")
                continue
            except Exception as e:
                print(f"{case['name']} 测试失败: {e}")
                continue
        
        print("Base64编码优化测试完成")
        
    except Exception as e:
        print(f"Base64测试失败: {e}")

def main():
    """主测试函数"""
    print("开始图片压缩和Base64优化测试")
    print("=" * 50)
    
    # 测试图片压缩
    test_image_compression()
    
    # 测试base64优化
    test_base64_optimization()
    
    print("\n" + "=" * 50)
    print("所有测试完成")

if __name__ == "__main__":
    main()
