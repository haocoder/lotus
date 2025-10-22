#!/usr/bin/env python3
"""
图片优化功能测试脚本

测试新的图片优化模块是否正常工作，包括：
1. URL图片直接传递
2. 本地图片智能压缩
3. 已编码图片优化
4. 性能对比测试
"""

import sys
import time
import os
from pathlib import Path
from typing import List, Dict, Any
import tempfile
from PIL import Image
import numpy as np

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

import lotus
from lotus.utils.image_optimizer import ImageOptimizer, optimize_image_for_processing, is_url_image
from lotus.utils import fetch_image


def create_test_images() -> Dict[str, Any]:
    """
    创建测试图片数据
    """
    print("Creating test images...")
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    
    # 1. 创建大尺寸测试图片
    large_image = Image.new('RGB', (2048, 2048), color='red')
    large_image_path = os.path.join(temp_dir, 'large_image.jpg')
    large_image.save(large_image_path, 'JPEG', quality=95)
    
    # 2. 创建中等尺寸图片
    medium_image = Image.new('RGB', (512, 512), color='blue')
    medium_image_path = os.path.join(temp_dir, 'medium_image.png')
    medium_image.save(medium_image_path, 'PNG')
    
    # 3. 创建小尺寸图片
    small_image = Image.new('RGB', (256, 256), color='green')
    small_image_path = os.path.join(temp_dir, 'small_image.jpg')
    small_image.save(small_image_path, 'JPEG', quality=85)
    
    # 4. 创建numpy数组图片
    numpy_image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    
    # 5. 创建base64编码图片
    import base64
    import io
    buffered = io.BytesIO()
    medium_image.save(buffered, format='PNG')
    base64_image = "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    return {
        'url_images': [
            'https://example.com/image1.jpg',
            'http://example.com/image2.png',
            's3://bucket/image3.jpg'
        ],
        'file_images': [
            large_image_path,
            medium_image_path,
            small_image_path
        ],
        'numpy_image': numpy_image,
        'base64_image': base64_image,
        'temp_dir': temp_dir
    }


def test_url_image_handling(test_data: Dict[str, Any]) -> None:
    """
    测试URL图片处理
    """
    print("\n" + "="*50)
    print("TEST 1: URL Image Handling")
    print("="*50)
    
    for url in test_data['url_images']:
        print(f"Testing URL: {url}")
        
        # 测试URL识别
        is_url = is_url_image(url)
        print(f"  Is URL: {is_url}")
        
        # 测试优化器处理
        optimizer = ImageOptimizer()
        result = optimizer.optimize_image(url)
        print(f"  Result: {result[:50]}..." if len(result) > 50 else f"  Result: {result}")
        print(f"  Same as input: {result == url}")
        print()


def test_file_image_optimization(test_data: Dict[str, Any]) -> None:
    """
    测试文件图片优化
    """
    print("\n" + "="*50)
    print("TEST 2: File Image Optimization")
    print("="*50)
    
    optimizer = ImageOptimizer(max_size=(512, 512), quality=80)
    
    for file_path in test_data['file_images']:
        print(f"Testing file: {os.path.basename(file_path)}")
        
        # 获取原始文件大小
        original_size = os.path.getsize(file_path)
        print(f"  Original size: {original_size:,} bytes")
        
        # 优化图片
        start_time = time.time()
        optimized = optimizer.optimize_image(file_path)
        end_time = time.time()
        
        # 计算优化后大小
        if optimized.startswith('data:image'):
            # 提取base64数据计算大小
            import base64
            _, data = optimized.split('base64,', 1)
            optimized_size = len(base64.b64decode(data))
        else:
            optimized_size = len(optimized)
        
        print(f"  Optimized size: {optimized_size:,} bytes")
        print(f"  Compression ratio: {optimized_size/original_size:.2%}")
        print(f"  Processing time: {end_time - start_time:.3f}s")
        print()


def test_numpy_image_optimization(test_data: Dict[str, Any]) -> None:
    """
    测试numpy数组图片优化
    """
    print("\n" + "="*50)
    print("TEST 3: Numpy Image Optimization")
    print("="*50)
    
    optimizer = ImageOptimizer(max_size=(400, 400), quality=85)
    
    print(f"Testing numpy array: {test_data['numpy_image'].shape}")
    
    start_time = time.time()
    optimized = optimizer.optimize_image(test_data['numpy_image'])
    end_time = time.time()
    
    print(f"  Optimized result: {optimized[:50]}..." if len(optimized) > 50 else f"  Optimized result: {optimized}")
    print(f"  Processing time: {end_time - start_time:.3f}s")
    print()


def test_base64_image_optimization(test_data: Dict[str, Any]) -> None:
    """
    测试base64图片优化
    """
    print("\n" + "="*50)
    print("TEST 4: Base64 Image Optimization")
    print("="*50)
    
    optimizer = ImageOptimizer(max_size=(300, 300), quality=75)
    
    print(f"Testing base64 image: {len(test_data['base64_image'])} characters")
    
    start_time = time.time()
    optimized = optimizer.optimize_image(test_data['base64_image'])
    end_time = time.time()
    
    print(f"  Original length: {len(test_data['base64_image'])}")
    print(f"  Optimized length: {len(optimized)}")
    print(f"  Compression ratio: {len(optimized)/len(test_data['base64_image']):.2%}")
    print(f"  Processing time: {end_time - start_time:.3f}s")
    print()


def test_fetch_image_integration(test_data: Dict[str, Any]) -> None:
    """
    测试fetch_image集成
    """
    print("\n" + "="*50)
    print("TEST 5: fetch_image Integration")
    print("="*50)
    
    # 测试URL图片
    url_image = test_data['url_images'][0]
    result = fetch_image(url_image, "base64")
    print(f"URL image result: {result == url_image}")
    
    # 测试文件图片
    file_image = test_data['file_images'][0]
    result = fetch_image(file_image, "base64")
    print(f"File image result: {result.startswith('data:image')}")
    
    # 测试numpy图片
    numpy_image = test_data['numpy_image']
    result = fetch_image(numpy_image, "base64")
    print(f"Numpy image result: {result.startswith('data:image')}")
    
    print()


def test_performance_comparison(test_data: Dict[str, Any]) -> None:
    """
    测试性能对比
    """
    print("\n" + "="*50)
    print("TEST 6: Performance Comparison")
    print("="*50)
    
    # 测试大图片
    large_file = test_data['file_images'][0]  # 2048x2048
    
    # 原始方法
    print("Testing original method...")
    start_time = time.time()
    original_result = fetch_image(large_file, "base64")
    original_time = time.time() - start_time
    original_size = len(original_result)
    
    # 优化方法
    print("Testing optimized method...")
    start_time = time.time()
    optimized_result = optimize_image_for_processing(large_file)
    optimized_time = time.time() - start_time
    optimized_size = len(optimized_result)
    
    print(f"Original method:")
    print(f"  Time: {original_time:.3f}s")
    print(f"  Size: {original_size:,} bytes")
    print(f"Optimized method:")
    print(f"  Time: {optimized_time:.3f}s")
    print(f"  Size: {optimized_size:,} bytes")
    print(f"Speed improvement: {original_time/optimized_time:.2f}x")
    print(f"Size reduction: {(original_size-optimized_size)/original_size:.1%}")
    print()


def test_cache_functionality() -> None:
    """
    测试缓存功能
    """
    print("\n" + "="*50)
    print("TEST 7: Cache Functionality")
    print("="*50)
    
    optimizer = ImageOptimizer(enable_cache=True, cache_size=5)
    
    # 创建测试图片
    test_image = Image.new('RGB', (100, 100), color='purple')
    
    # 第一次处理
    start_time = time.time()
    result1 = optimizer.optimize_image(test_image)
    first_time = time.time() - start_time
    
    # 第二次处理（应该从缓存获取）
    start_time = time.time()
    result2 = optimizer.optimize_image(test_image)
    second_time = time.time() - start_time
    
    print(f"First processing time: {first_time:.3f}s")
    print(f"Second processing time: {second_time:.3f}s")
    print(f"Cache speedup: {first_time/second_time:.2f}x")
    print(f"Results match: {result1 == result2}")
    
    # 获取缓存统计
    stats = optimizer.get_cache_stats()
    print(f"Cache stats: {stats}")
    print()


def cleanup_test_data(test_data: Dict[str, Any]) -> None:
    """
    清理测试数据
    """
    import shutil
    try:
        shutil.rmtree(test_data['temp_dir'])
        print(f"Cleaned up test directory: {test_data['temp_dir']}")
    except Exception as e:
        print(f"Warning: Failed to cleanup test directory: {e}")


def main():
    """
    运行所有测试
    """
    print("🚀 Starting Image Optimization Tests")
    print("="*80)
    
    try:
        # 创建测试数据
        test_data = create_test_images()
        
        # 运行测试
        test_url_image_handling(test_data)
        test_file_image_optimization(test_data)
        test_numpy_image_optimization(test_data)
        test_base64_image_optimization(test_data)
        test_fetch_image_integration(test_data)
        test_performance_comparison(test_data)
        test_cache_functionality()
        
        print("\n" + "="*80)
        print("✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        # 清理测试数据
        if 'test_data' in locals():
            cleanup_test_data(test_data)
    
    return 0


if __name__ == "__main__":
    exit(main())
