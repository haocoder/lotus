#!/usr/bin/env python3
"""
测试sem_filter中的图片压缩功能集成

验证图片压缩功能是否正确集成到sem_filter算子中
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

def test_image_compression_integration():
    """测试图片压缩功能集成"""
    print("=== 测试图片压缩功能集成 ===")
    
    try:
        # 测试配置管理器
        from lotus.utils.image_compression_config import (
            ImageCompressionConfig, 
            get_global_config, 
            set_global_config,
            create_config_from_sem_filter_params
        )
        
        print("--- 测试配置管理器 ---")
        
        # 测试创建配置
        config = ImageCompressionConfig(
            enable_compression=True,
            strategy="advanced",
            max_size=(512, 512),
            quality=70,
            format="JPEG"
        )
        
        print(f"配置创建成功: {config.get_config()}")
        
        # 测试从sem_filter参数创建配置
        sem_filter_config = create_config_from_sem_filter_params(
            enable_image_compression=True,
            image_compression_strategy="simple",
            image_max_size=(800, 800),
            image_quality=80,
            image_format="PNG"
        )
        
        print(f"sem_filter配置: {sem_filter_config.get_config()}")
        
        # 测试全局配置
        print("\n--- 测试全局配置 ---")
        global_config = get_global_config()
        print(f"全局配置: {global_config.get_config()}")
        
        # 测试设置全局配置
        set_global_config(
            enable_compression=True,
            strategy="advanced",
            max_size=(1024, 1024),
            quality=85
        )
        
        updated_config = get_global_config()
        print(f"更新后全局配置: {updated_config.get_config()}")
        
        print("配置管理器测试完成")
        
    except Exception as e:
        print(f"配置管理器测试失败: {e}")
        import traceback
        traceback.print_exc()

def test_sem_filter_parameters():
    """测试sem_filter参数"""
    print("\n=== 测试sem_filter参数 ===")
    
    try:
        from lotus.sem_ops.sem_filter import sem_filter
        
        # 检查函数签名
        import inspect
        sig = inspect.signature(sem_filter)
        
        # 检查图片压缩参数是否存在
        image_params = [
            'enable_image_compression',
            'image_compression_strategy', 
            'image_max_size',
            'image_quality',
            'image_format'
        ]
        
        print("--- 检查sem_filter参数 ---")
        for param in image_params:
            if param in sig.parameters:
                print(f"✓ {param} 参数存在")
                param_info = sig.parameters[param]
                print(f"  默认值: {param_info.default}")
                print(f"  类型: {param_info.annotation}")
            else:
                print(f"✗ {param} 参数不存在")
        
        print("sem_filter参数检查完成")
        
    except Exception as e:
        print(f"sem_filter参数测试失败: {e}")

def test_fetch_image_integration():
    """测试fetch_image集成"""
    print("\n=== 测试fetch_image集成 ===")
    
    try:
        from lotus.utils import fetch_image
        
        # 测试URL图片
        print("--- 测试URL图片处理 ---")
        url_image = "https://example.com/image.jpg"
        result = fetch_image(url_image, "base64")
        print(f"URL图片处理结果: {result}")
        
        # 测试本地图片路径
        print("--- 测试本地图片路径处理 ---")
        local_image = "/path/to/local/image.jpg"
        result = fetch_image(local_image, "base64")
        print(f"本地图片处理结果: {result}")
        
        # 测试base64图片
        print("--- 测试base64图片处理 ---")
        base64_image = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD"
        result = fetch_image(base64_image, "base64")
        print(f"base64图片处理结果: {result[:50]}...")
        
        print("fetch_image集成测试完成")
        
    except Exception as e:
        print(f"fetch_image集成测试失败: {e}")

def test_image_compression_workflow():
    """测试图片压缩工作流程"""
    print("\n=== 测试图片压缩工作流程 ===")
    
    try:
        # 模拟DataFrame中的图片数据
        print("--- 模拟DataFrame图片数据处理 ---")
        
        # 创建模拟的图片数据
        mock_image_data = [
            "https://example.com/image1.jpg",  # URL图片
            "data:image/jpeg;base64,/9j/4AAQ",  # base64图片
            "/path/to/local/image.jpg",  # 本地路径
        ]
        
        # 测试不同类型的图片处理
        from lotus.utils import fetch_image
        
        for i, image_data in enumerate(mock_image_data):
            print(f"处理图片 {i+1}: {image_data[:30]}...")
            result = fetch_image(image_data, "base64")
            print(f"  处理结果: {result[:50]}...")
        
        print("图片压缩工作流程测试完成")
        
    except Exception as e:
        print(f"图片压缩工作流程测试失败: {e}")

def test_compression_strategies():
    """测试压缩策略"""
    print("\n=== 测试压缩策略 ===")
    
    try:
        from lotus.utils.image_compression_config import ImageCompressionConfig
        
        # 测试简单压缩策略
        print("--- 测试简单压缩策略 ---")
        simple_config = ImageCompressionConfig(
            enable_compression=True,
            strategy="simple",
            max_size=(256, 256),
            quality=60
        )
        print(f"简单压缩配置: {simple_config.get_config()}")
        
        # 测试高级压缩策略
        print("--- 测试高级压缩策略 ---")
        advanced_config = ImageCompressionConfig(
            enable_compression=True,
            strategy="advanced",
            max_size=(1024, 1024),
            quality=85
        )
        print(f"高级压缩配置: {advanced_config.get_config()}")
        
        # 测试禁用压缩
        print("--- 测试禁用压缩 ---")
        disabled_config = ImageCompressionConfig(
            enable_compression=False
        )
        print(f"禁用压缩配置: {disabled_config.get_config()}")
        
        print("压缩策略测试完成")
        
    except Exception as e:
        print(f"压缩策略测试失败: {e}")

def main():
    """主测试函数"""
    print("开始sem_filter图片压缩功能集成测试")
    print("=" * 60)
    
    # 测试配置管理器
    test_image_compression_integration()
    
    # 测试sem_filter参数
    test_sem_filter_parameters()
    
    # 测试fetch_image集成
    test_fetch_image_integration()
    
    # 测试图片压缩工作流程
    test_image_compression_workflow()
    
    # 测试压缩策略
    test_compression_strategies()
    
    print("\n" + "=" * 60)
    print("所有测试完成")

if __name__ == "__main__":
    main()
