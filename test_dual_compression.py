#!/usr/bin/env python3
"""
双压缩策略测试脚本

测试简单压缩和高级压缩两种策略的差异
"""

import sys
import time
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

def test_compression_strategies():
    """测试两种压缩策略的差异"""
    print("=== 双压缩策略测试 ===")
    
    try:
        from lotus.utils.image_optimizer import ImageOptimizer
        
        # 测试URL检测功能
        print("--- URL检测测试 ---")
        optimizer = ImageOptimizer()
        
        test_urls = [
            "https://example.com/image.jpg",
            "http://test.com/pic.png", 
            "s3://bucket/image.webp",
            "data:image/jpeg;base64,/9j/4AAQ",
            "/path/to/local/image.jpg"
        ]
        
        for url in test_urls:
            is_url = optimizer.is_url_image(url)
            is_encoded = optimizer.is_encoded_image(url)
            is_file = optimizer.is_file_path(url)
            
            print(f"输入: {url[:30]}...")
            print(f"  URL: {is_url}")
            print(f"  编码: {is_encoded}")
            print(f"  文件: {is_file}")
            print()
        
        # 测试压缩策略选择
        print("--- 压缩策略选择测试 ---")
        
        # 简单压缩策略
        simple_optimizer = ImageOptimizer(
            max_size=(512, 512),
            quality=70,
            format="JPEG",
            use_advanced_compression=False
        )
        
        # 高级压缩策略
        advanced_optimizer = ImageOptimizer(
            max_size=(512, 512),
            quality=70,
            format="JPEG", 
            use_advanced_compression=True
        )
        
        print(f"简单压缩策略: use_advanced_compression = {simple_optimizer.use_advanced_compression}")
        print(f"高级压缩策略: use_advanced_compression = {advanced_optimizer.use_advanced_compression}")
        
        # 测试缓存功能
        print("--- 缓存功能测试 ---")
        
        cache_optimizer = ImageOptimizer(
            enable_cache=True,
            cache_size=5,
            use_advanced_compression=True
        )
        
        # 模拟缓存操作
        for i in range(8):  # 超过缓存大小
            key = f"image_{i}"
            cache_optimizer._cache[key] = f"compressed_{i}"
            cache_optimizer._cache_access_times[key] = time.time()
            cache_optimizer._cache_sizes[key] = 1000
            time.sleep(0.001)  # 确保时间不同
        
        print(f"缓存大小: {len(cache_optimizer._cache)}")
        print(f"缓存统计: {cache_optimizer.get_cache_stats()}")
        
        # 清理缓存
        cache_optimizer.clear_cache()
        print(f"清理后缓存大小: {len(cache_optimizer._cache)}")
        
        print("\n双压缩策略测试完成")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()

def test_compression_method_selection():
    """测试压缩方法选择逻辑"""
    print("\n=== 压缩方法选择测试 ===")
    
    try:
        from lotus.utils.image_optimizer import ImageOptimizer
        
        # 测试简单压缩
        print("--- 简单压缩测试 ---")
        simple_optimizer = ImageOptimizer(use_advanced_compression=False)
        
        # 模拟图片对象
        class MockImage:
            def __init__(self):
                self.size = (1000, 1000)
                self.mode = "RGB"
            
            def thumbnail(self, size, resample):
                self.size = size
                print(f"  使用thumbnail方法缩放到: {size}")
            
            def save(self, buffer, **kwargs):
                print(f"  保存图片，参数: {kwargs}")
                buffer.write(b"mock_image_data")
        
        mock_image = MockImage()
        
        try:
            result = simple_optimizer._choose_compression_method(mock_image)
            print(f"  简单压缩结果: {result[:50]}...")
        except Exception as e:
            print(f"  简单压缩测试跳过（需要PIL）: {e}")
        
        # 测试高级压缩
        print("--- 高级压缩测试 ---")
        advanced_optimizer = ImageOptimizer(use_advanced_compression=True)
        
        try:
            result = advanced_optimizer._choose_compression_method(mock_image)
            print(f"  高级压缩结果: {result[:50]}...")
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
    print("开始双压缩策略测试")
    print("=" * 50)
    
    # 测试压缩策略
    test_compression_strategies()
    
    # 测试压缩方法选择
    test_compression_method_selection()
    
    # 测试优化器配置
    test_optimizer_configuration()
    
    print("\n" + "=" * 50)
    print("所有测试完成")

if __name__ == "__main__":
    main()
