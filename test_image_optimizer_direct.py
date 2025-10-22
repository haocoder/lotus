#!/usr/bin/env python3
"""
直接测试图片优化器模块

不通过lotus模块导入，直接测试核心功能
"""

import sys
import time
from pathlib import Path

# 直接导入图片优化器模块
sys.path.insert(0, str(Path(__file__).parent / "lotus" / "utils"))

def test_url_detection():
    """
    测试URL图片检测功能
    """
    print("Testing URL image detection...")
    
    try:
        from image_optimizer import is_url_image
        
        # 测试URL图片
        url_images = [
            'https://example.com/image.jpg',
            'http://example.com/image.png',
            's3://bucket/image.jpg'
        ]
        
        for url in url_images:
            result = is_url_image(url)
            print(f"  {url}: {result}")
            assert result == True, f"URL {url} should be detected as URL"
        
        # 测试非URL图片
        non_url_images = [
            '/path/to/image.jpg',
            'data:image/png;base64,abc123',
            'image.jpg'
        ]
        
        for non_url in non_url_images:
            result = is_url_image(non_url)
            print(f"  {non_url}: {result}")
            assert result == False, f"Non-URL {non_url} should not be detected as URL"
        
        print("URL detection test passed!")
        return True
        
    except Exception as e:
        print(f"URL detection test failed: {e}")
        return False


def test_optimizer_initialization():
    """
    测试优化器初始化
    """
    print("\nTesting optimizer initialization...")
    
    try:
        from image_optimizer import ImageOptimizer
        
        # 测试默认参数
        optimizer1 = ImageOptimizer()
        print(f"  Default max_size: {optimizer1.max_size}")
        print(f"  Default quality: {optimizer1.quality}")
        print(f"  Default format: {optimizer1.format}")
        
        # 测试自定义参数
        optimizer2 = ImageOptimizer(
            max_size=(512, 512),
            quality=80,
            format="JPEG",
            enable_cache=True,
            cache_size=100
        )
        print(f"  Custom max_size: {optimizer2.max_size}")
        print(f"  Custom quality: {optimizer2.quality}")
        print(f"  Custom format: {optimizer2.format}")
        print(f"  Cache enabled: {optimizer2.enable_cache}")
        print(f"  Cache size: {optimizer2.cache_size}")
        
        print("Optimizer initialization test passed!")
        return True
        
    except Exception as e:
        print(f"Optimizer initialization test failed: {e}")
        return False


def test_url_image_handling():
    """
    测试URL图片处理
    """
    print("\nTesting URL image handling...")
    
    try:
        from image_optimizer import ImageOptimizer
        
        optimizer = ImageOptimizer()
        
        # 测试URL图片
        url_images = [
            'https://example.com/image1.jpg',
            'http://example.com/image2.png',
            's3://bucket/image3.jpg'
        ]
        
        for url in url_images:
            result = optimizer.optimize_image(url)
            print(f"  {url} -> {result[:50]}...")
            assert result == url, f"URL {url} should be returned unchanged"
        
        print("URL image handling test passed!")
        return True
        
    except Exception as e:
        print(f"URL image handling test failed: {e}")
        return False


def test_cache_functionality():
    """
    测试缓存功能
    """
    print("\nTesting cache functionality...")
    
    try:
        from image_optimizer import ImageOptimizer
        
        optimizer = ImageOptimizer(enable_cache=True, cache_size=5)
        
        # 测试缓存统计
        stats = optimizer.get_cache_stats()
        print(f"  Initial cache stats: {stats}")
        assert stats['enabled'] == True
        assert stats['cache_size'] == 0
        
        # 测试清空缓存
        optimizer.clear_cache()
        stats = optimizer.get_cache_stats()
        print(f"  After clear cache: {stats}")
        assert stats['cache_size'] == 0
        
        print("Cache functionality test passed!")
        return True
        
    except Exception as e:
        print(f"Cache functionality test failed: {e}")
        return False


def test_convenience_functions():
    """
    测试便捷函数
    """
    print("\nTesting convenience functions...")
    
    try:
        from image_optimizer import optimize_image_for_processing
        
        # 测试URL图片
        url_image = 'https://example.com/test.jpg'
        result = optimize_image_for_processing(url_image)
        print(f"  URL image optimization: {result}")
        assert result == url_image, "URL image should be returned unchanged"
        
        print("Convenience functions test passed!")
        return True
        
    except Exception as e:
        print(f"Convenience functions test failed: {e}")
        return False


def main():
    """
    运行所有测试
    """
    print("Direct Image Optimizer Tests")
    print("="*60)
    
    tests = [
        test_url_detection,
        test_optimizer_initialization,
        test_url_image_handling,
        test_cache_functionality,
        test_convenience_functions
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "="*60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("All tests passed!")
        return 0
    else:
        print("Some tests failed!")
        return 1


if __name__ == "__main__":
    exit(main())
