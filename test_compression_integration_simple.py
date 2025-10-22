#!/usr/bin/env python3
"""
简化的图片压缩集成测试

测试图片压缩功能在sem_filter中的集成，不依赖numpy
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

def test_compression_config():
    """测试压缩配置"""
    print("=== 测试压缩配置 ===")
    
    try:
        # 直接导入配置管理器
        from lotus.utils.image_compression_config import (
            ImageCompressionConfig, 
            get_global_config, 
            set_global_config
        )
        
        print("--- 测试配置创建 ---")
        
        # 测试创建配置
        config = ImageCompressionConfig(
            enable_compression=True,
            strategy="advanced",
            max_size=(512, 512),
            quality=70,
            format="JPEG"
        )
        
        print(f"配置创建成功")
        print(f"  启用压缩: {config.enable_compression}")
        print(f"  策略: {config.strategy}")
        print(f"  最大尺寸: {config.max_size}")
        print(f"  质量: {config.quality}")
        print(f"  格式: {config.format}")
        
        # 测试配置更新
        print("\n--- 测试配置更新 ---")
        config.update_config(
            strategy="simple",
            quality=60,
            max_size=(256, 256)
        )
        
        print(f"更新后配置:")
        print(f"  策略: {config.strategy}")
        print(f"  质量: {config.quality}")
        print(f"  最大尺寸: {config.max_size}")
        
        # 测试全局配置
        print("\n--- 测试全局配置 ---")
        global_config = get_global_config()
        print(f"全局配置获取成功")
        
        # 测试设置全局配置
        set_global_config(
            enable_compression=True,
            strategy="advanced",
            max_size=(1024, 1024),
            quality=85
        )
        
        updated_config = get_global_config()
        print(f"全局配置更新成功")
        print(f"  启用压缩: {updated_config.enable_compression}")
        print(f"  策略: {updated_config.strategy}")
        
        print("压缩配置测试完成")
        
    except Exception as e:
        print(f"压缩配置测试失败: {e}")
        import traceback
        traceback.print_exc()

def test_sem_filter_signature():
    """测试sem_filter函数签名"""
    print("\n=== 测试sem_filter函数签名 ===")
    
    try:
        # 直接导入sem_filter函数
        from lotus.sem_ops.sem_filter import sem_filter
        import inspect
        
        # 获取函数签名
        sig = inspect.signature(sem_filter)
        
        print("--- 检查sem_filter参数 ---")
        
        # 检查图片压缩参数
        image_params = [
            'enable_image_compression',
            'image_compression_strategy', 
            'image_max_size',
            'image_quality',
            'image_format'
        ]
        
        found_params = []
        for param in image_params:
            if param in sig.parameters:
                found_params.append(param)
                param_info = sig.parameters[param]
                print(f"✓ {param}: {param_info.annotation} = {param_info.default}")
            else:
                print(f"✗ {param} 参数不存在")
        
        print(f"\n找到 {len(found_params)}/{len(image_params)} 个图片压缩参数")
        
        # 检查所有参数
        print(f"\n--- sem_filter所有参数 ---")
        all_params = list(sig.parameters.keys())
        print(f"总参数数量: {len(all_params)}")
        
        # 显示最后几个参数（图片压缩参数应该在最后）
        print("最后几个参数:")
        for param in all_params[-10:]:
            print(f"  {param}")
        
        print("sem_filter函数签名测试完成")
        
    except Exception as e:
        print(f"sem_filter函数签名测试失败: {e}")

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
        print(f"简单压缩策略配置成功")
        print(f"  策略: {simple_config.strategy}")
        print(f"  最大尺寸: {simple_config.max_size}")
        print(f"  质量: {simple_config.quality}")
        
        # 测试高级压缩策略
        print("\n--- 测试高级压缩策略 ---")
        advanced_config = ImageCompressionConfig(
            enable_compression=True,
            strategy="advanced",
            max_size=(1024, 1024),
            quality=85
        )
        print(f"高级压缩策略配置成功")
        print(f"  策略: {advanced_config.strategy}")
        print(f"  最大尺寸: {advanced_config.max_size}")
        print(f"  质量: {advanced_config.quality}")
        
        # 测试禁用压缩
        print("\n--- 测试禁用压缩 ---")
        disabled_config = ImageCompressionConfig(
            enable_compression=False
        )
        print(f"禁用压缩配置成功")
        print(f"  启用压缩: {disabled_config.enable_compression}")
        
        print("压缩策略测试完成")
        
    except Exception as e:
        print(f"压缩策略测试失败: {e}")

def test_config_from_sem_filter():
    """测试从sem_filter参数创建配置"""
    print("\n=== 测试从sem_filter参数创建配置 ===")
    
    try:
        from lotus.utils.image_compression_config import create_config_from_sem_filter_params
        
        # 测试创建配置
        config = create_config_from_sem_filter_params(
            enable_image_compression=True,
            image_compression_strategy="advanced",
            image_max_size=(800, 800),
            image_quality=80,
            image_format="PNG"
        )
        
        print("从sem_filter参数创建配置成功")
        print(f"  启用压缩: {config.enable_compression}")
        print(f"  策略: {config.strategy}")
        print(f"  最大尺寸: {config.max_size}")
        print(f"  质量: {config.quality}")
        print(f"  格式: {config.format}")
        
        # 测试配置获取
        config_dict = config.get_config()
        print(f"\n配置字典: {config_dict}")
        
        print("从sem_filter参数创建配置测试完成")
        
    except Exception as e:
        print(f"从sem_filter参数创建配置测试失败: {e}")

def main():
    """主测试函数"""
    print("开始简化的图片压缩集成测试")
    print("=" * 50)
    
    # 测试压缩配置
    test_compression_config()
    
    # 测试sem_filter函数签名
    test_sem_filter_signature()
    
    # 测试压缩策略
    test_compression_strategies()
    
    # 测试从sem_filter参数创建配置
    test_config_from_sem_filter()
    
    print("\n" + "=" * 50)
    print("所有测试完成")

if __name__ == "__main__":
    main()
