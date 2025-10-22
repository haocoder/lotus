#!/usr/bin/env python3
"""
验证图片压缩功能集成

通过检查文件内容验证图片压缩功能是否正确集成
"""

import re
from pathlib import Path

def check_sem_filter_integration():
    """检查sem_filter中的图片压缩集成"""
    print("=== 检查sem_filter集成 ===")
    
    file_path = Path("lotus/sem_ops/sem_filter.py")
    
    if not file_path.exists():
        print("文件不存在: lotus/sem_ops/sem_filter.py")
        return
    
    # 读取文件内容
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("--- 检查图片压缩参数 ---")
    
    # 检查图片压缩参数
    image_params = [
        'enable_image_compression',
        'image_compression_strategy',
        'image_max_size',
        'image_quality',
        'image_format'
    ]
    
    for param in image_params:
        if param in content:
            print(f"{param} 参数存在")
        else:
            print(f"{param} 参数不存在")
    
    print("\n--- 检查配置设置 ---")
    
    # 检查配置设置代码
    if 'set_global_config' in content:
        print("全局配置设置代码存在")
    else:
        print("全局配置设置代码不存在")
    
    if 'image_compression_config' in content:
        print(" 图片压缩配置导入存在")
    else:
        print(" 图片压缩配置导入不存在")
    
    print("\n--- 检查函数参数传递 ---")
    
    # 检查参数传递
    if 'enable_image_compression, image_compression_strategy' in content:
        print(" 参数传递代码存在")
    else:
        print(" 参数传递代码不存在")
    
    print("sem_filter集成检查完成")

def check_utils_integration():
    """检查utils中的图片压缩集成"""
    print("\n=== 检查utils集成 ===")
    
    file_path = Path("lotus/utils.py")
    
    if not file_path.exists():
        print("文件不存在: lotus/utils.py")
        return
    
    # 读取文件内容
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("--- 检查fetch_image函数 ---")
    
    # 检查fetch_image函数
    if 'def fetch_image(' in content:
        print(" fetch_image函数存在")
    else:
        print(" fetch_image函数不存在")
    
    # 检查图片优化器导入
    if 'image_optimizer' in content:
        print(" 图片优化器导入存在")
    else:
        print(" 图片优化器导入不存在")
    
    # 检查配置管理器导入
    if 'image_compression_config' in content:
        print(" 配置管理器导入存在")
    else:
        print(" 配置管理器导入不存在")
    
    # 检查全局配置使用
    if 'get_global_config' in content:
        print(" 全局配置使用存在")
    else:
        print(" 全局配置使用不存在")
    
    print("utils集成检查完成")

def check_config_manager():
    """检查配置管理器"""
    print("\n=== 检查配置管理器 ===")
    
    file_path = Path("lotus/utils/image_compression_config.py")
    
    if not file_path.exists():
        print("文件不存在: lotus/utils/image_compression_config.py")
        return
    
    # 读取文件内容
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("--- 检查配置管理器类 ---")
    
    # 检查ImageCompressionConfig类
    if 'class ImageCompressionConfig:' in content:
        print(" ImageCompressionConfig类存在")
    else:
        print(" ImageCompressionConfig类不存在")
    
    # 检查关键方法
    methods = [
        '__init__',
        'update_config',
        'optimize_image',
        'get_config',
        'clear_cache'
    ]
    
    for method in methods:
        if f'def {method}(' in content:
            print(f" {method}方法存在")
        else:
            print(f" {method}方法不存在")
    
    print("\n--- 检查全局配置函数 ---")
    
    # 检查全局配置函数
    global_functions = [
        'get_global_config',
        'set_global_config',
        'create_config_from_sem_filter_params'
    ]
    
    for func in global_functions:
        if f'def {func}(' in content:
            print(f" {func}函数存在")
        else:
            print(f" {func}函数不存在")
    
    print("配置管理器检查完成")

def check_image_optimizer():
    """检查图片优化器"""
    print("\n=== 检查图片优化器 ===")
    
    file_path = Path("lotus/utils/image_optimizer.py")
    
    if not file_path.exists():
        print("文件不存在: lotus/utils/image_optimizer.py")
        return
    
    # 读取文件内容
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("--- 检查图片优化器类 ---")
    
    # 检查ImageOptimizer类
    if 'class ImageOptimizer:' in content:
        print(" ImageOptimizer类存在")
    else:
        print(" ImageOptimizer类不存在")
    
    # 检查压缩方法
    compression_methods = [
        '_compress_image_simple',
        '_compress_image_advanced',
        '_choose_compression_method'
    ]
    
    for method in compression_methods:
        if f'def {method}(' in content:
            print(f" {method}方法存在")
        else:
            print(f" {method}方法不存在")
    
    print("\n--- 检查配置参数 ---")
    
    # 检查配置参数
    config_params = [
        'use_advanced_compression',
        'max_size',
        'quality',
        'format',
        'enable_cache'
    ]
    
    for param in config_params:
        if param in content:
            print(f" {param}参数存在")
        else:
            print(f" {param}参数不存在")
    
    print("图片优化器检查完成")

def check_integration_completeness():
    """检查集成完整性"""
    print("\n=== 检查集成完整性 ===")
    
    # 检查关键文件是否存在
    key_files = [
        "lotus/sem_ops/sem_filter.py",
        "lotus/utils.py", 
        "lotus/utils/image_optimizer.py",
        "lotus/utils/image_compression_config.py"
    ]
    
    print("--- 检查关键文件 ---")
    for file_path in key_files:
        if Path(file_path).exists():
            print(f" {file_path} 存在")
        else:
            print(f" {file_path} 不存在")
    
    print("\n--- 检查集成流程 ---")
    
    # 检查集成流程
    integration_steps = [
        "sem_filter函数添加图片压缩参数",
        "参数传递给子函数",
        "设置全局配置",
        "fetch_image使用全局配置",
        "配置管理器管理压缩设置",
        "图片优化器执行压缩"
    ]
    
    for i, step in enumerate(integration_steps, 1):
        print(f"{i}. {step}")
    
    print("集成完整性检查完成")

def main():
    """主测试函数"""
    print("开始图片压缩功能集成验证")
    print("=" * 60)
    
    # 检查sem_filter集成
    check_sem_filter_integration()
    
    # 检查utils集成
    check_utils_integration()
    
    # 检查配置管理器
    check_config_manager()
    
    # 检查图片优化器
    check_image_optimizer()
    
    # 检查集成完整性
    check_integration_completeness()
    
    print("\n" + "=" * 60)
    print("集成验证完成")

if __name__ == "__main__":
    main()
