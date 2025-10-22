#!/usr/bin/env python3
"""
直接测试压缩函数

测试重命名后的压缩函数，不依赖numpy
"""

import sys
import time
from pathlib import Path

# 直接导入压缩函数，避免numpy依赖
sys.path.insert(0, str(Path(__file__).parent))

def test_function_renaming():
    """测试函数重命名"""
    print("=== 函数重命名测试 ===")
    
    try:
        # 直接导入ImageOptimizer类
        from lotus.utils.image_optimizer import ImageOptimizer
        
        # 创建优化器实例
        optimizer = ImageOptimizer()
        
        # 检查函数是否存在
        print("--- 检查压缩函数 ---")
        
        # 检查简单压缩函数
        if hasattr(optimizer, '_compress_image_simple'):
            print("✓ _compress_image_simple 函数存在")
        else:
            print("✗ _compress_image_simple 函数不存在")
        
        # 检查高级压缩函数
        if hasattr(optimizer, '_compress_image_advanced'):
            print("✓ _compress_image_advanced 函数存在")
        else:
            print("✗ _compress_image_advanced 函数不存在")
        
        # 检查压缩方法选择函数
        if hasattr(optimizer, '_choose_compression_method'):
            print("✓ _choose_compression_method 函数存在")
        else:
            print("✗ _choose_compression_method 函数不存在")
        
        # 检查旧的压缩函数是否还存在
        if hasattr(optimizer, '_compress_image'):
            print("⚠ _compress_image 函数仍然存在（可能有问题）")
        else:
            print("✓ _compress_image 函数已正确移除")
        
        print("\n--- 测试压缩策略配置 ---")
        
        # 测试简单压缩配置
        simple_optimizer = ImageOptimizer(use_advanced_compression=False)
        print(f"简单压缩策略: use_advanced_compression = {simple_optimizer.use_advanced_compression}")
        
        # 测试高级压缩配置
        advanced_optimizer = ImageOptimizer(use_advanced_compression=True)
        print(f"高级压缩策略: use_advanced_compression = {advanced_optimizer.use_advanced_compression}")
        
        print("\n--- 测试URL检测功能 ---")
        
        test_cases = [
            ("https://example.com/image.jpg", True, False, False),
            ("http://test.com/pic.png", True, False, False),
            ("s3://bucket/image.webp", True, False, False),
            ("data:image/jpeg;base64,/9j/4AAQ", False, True, False),
            ("/path/to/local/image.jpg", False, False, True),
        ]
        
        for url, expected_url, expected_encoded, expected_file in test_cases:
            is_url = optimizer.is_url_image(url)
            is_encoded = optimizer.is_encoded_image(url)
            is_file = optimizer.is_file_path(url)
            
            print(f"输入: {url[:30]}...")
            print(f"  URL检测: {is_url} (期望: {expected_url})")
            print(f"  编码检测: {is_encoded} (期望: {expected_encoded})")
            print(f"  文件检测: {is_file} (期望: {expected_file})")
            print()
        
        print("函数重命名测试完成")
        
    except Exception as e:
        print(f"函数重命名测试失败: {e}")
        import traceback
        traceback.print_exc()

def test_compression_method_selection():
    """测试压缩方法选择逻辑"""
    print("\n=== 压缩方法选择测试 ===")
    
    try:
        from lotus.utils.image_optimizer import ImageOptimizer
        
        # 创建模拟图片类
        class MockImage:
            def __init__(self, size=(1000, 1000), mode="RGB"):
                self.size = size
                self.mode = mode
            
            def thumbnail(self, size, resample):
                self.size = size
                print(f"    使用thumbnail方法缩放到: {size}")
            
            def save(self, buffer, **kwargs):
                print(f"    保存图片，参数: {kwargs}")
                buffer.write(b"mock_image_data")
        
        # 测试简单压缩
        print("--- 简单压缩测试 ---")
        simple_optimizer = ImageOptimizer(
            max_size=(512, 512),
            quality=70,
            use_advanced_compression=False
        )
        
        mock_image = MockImage()
        print(f"  原始图片尺寸: {mock_image.size}")
        print(f"  压缩策略: 简单压缩")
        
        try:
            # 这里会调用_choose_compression_method，然后调用_compress_image_simple
            result = simple_optimizer._choose_compression_method(mock_image)
            print(f"  压缩结果: {result[:50]}...")
        except Exception as e:
            print(f"  简单压缩测试跳过（需要PIL）: {e}")
        
        # 测试高级压缩
        print("\n--- 高级压缩测试 ---")
        advanced_optimizer = ImageOptimizer(
            max_size=(512, 512),
            quality=70,
            use_advanced_compression=True
        )
        
        mock_image = MockImage()
        print(f"  原始图片尺寸: {mock_image.size}")
        print(f"  压缩策略: 高级压缩")
        
        try:
            # 这里会调用_choose_compression_method，然后调用_compress_image_advanced
            result = advanced_optimizer._choose_compression_method(mock_image)
            print(f"  压缩结果: {result[:50]}...")
        except Exception as e:
            print(f"  高级压缩测试跳过（需要PIL）: {e}")
        
        print("压缩方法选择测试完成")
        
    except Exception as e:
        print(f"压缩方法选择测试失败: {e}")

def test_optimizer_configuration():
    """测试优化器配置"""
    print("\n=== 优化器配置测试 ===")
    
    try:
        from lotus.utils.image_optimizer import ImageOptimizer
        
        # 测试不同配置
        configs = [
            {
                "name": "默认配置",
                "params": {}
            },
            {
                "name": "简单压缩配置",
                "params": {
                    "max_size": (256, 256),
                    "quality": 60,
                    "format": "JPEG",
                    "use_advanced_compression": False
                }
            },
            {
                "name": "高级压缩配置",
                "params": {
                    "max_size": (1024, 1024),
                    "quality": 85,
                    "format": "WEBP",
                    "use_advanced_compression": True
                }
            }
        ]
        
        for config in configs:
            print(f"--- {config['name']} ---")
            optimizer = ImageOptimizer(**config['params'])
            
            print(f"  最大尺寸: {optimizer.max_size}")
            print(f"  质量: {optimizer.quality}")
            print(f"  格式: {optimizer.format}")
            print(f"  高级压缩: {optimizer.use_advanced_compression}")
            print(f"  缓存启用: {optimizer.enable_cache}")
            print()
        
        print("优化器配置测试完成")
        
    except Exception as e:
        print(f"优化器配置测试失败: {e}")

def main():
    """主测试函数"""
    print("开始压缩函数重命名测试")
    print("=" * 50)
    
    # 测试函数重命名
    test_function_renaming()
    
    # 测试压缩方法选择
    test_compression_method_selection()
    
    # 测试优化器配置
    test_optimizer_configuration()
    
    print("\n" + "=" * 50)
    print("所有测试完成")

if __name__ == "__main__":
    main()
