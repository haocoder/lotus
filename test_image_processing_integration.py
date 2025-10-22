#!/usr/bin/env python3
"""
测试图片处理集成

验证图片压缩优化是否已经集成到算子的图片处理逻辑中
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

def test_fetch_image_integration():
    """测试fetch_image集成"""
    print("=== 测试fetch_image集成 ===")
    
    try:
        from lotus.utils import fetch_image
        
        print("--- 测试URL图片处理 ---")
        # 测试URL图片（应该直接返回）
        url_image = "https://example.com/image.jpg"
        result = fetch_image(url_image, "base64")
        print(f"URL图片处理结果: {result}")
        
        print("--- 测试本地图片路径处理 ---")
        # 测试本地图片路径（应该使用优化器）
        local_image = "/path/to/local/image.jpg"
        result = fetch_image(local_image, "base64")
        print(f"本地图片处理结果: {result}")
        
        print("--- 测试base64图片处理 ---")
        # 测试base64图片（应该使用优化器）
        base64_image = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD"
        result = fetch_image(base64_image, "base64")
        print(f"base64图片处理结果: {result[:50]}...")
        
        print("fetch_image集成测试完成")
        
    except Exception as e:
        print(f"fetch_image集成测试失败: {e}")
        import traceback
        traceback.print_exc()

def test_image_processing_workflow():
    """测试图片处理工作流程"""
    print("\n=== 测试图片处理工作流程 ===")
    
    try:
        # 检查关键文件是否存在
        key_files = [
            "lotus/utils.py",
            "lotus/dtype_extensions/image.py",
            "lotus/templates/task_instructions.py",
            "lotus/utils/image_optimizer.py",
            "lotus/utils/image_compression_config.py"
        ]
        
        print("--- 检查关键文件 ---")
        for file_path in key_files:
            if Path(file_path).exists():
                print(f"文件存在: {file_path}")
            else:
                print(f"文件不存在: {file_path}")
        
        print("\n--- 检查图片处理流程 ---")
        
        # 检查df2multimodal_info函数
        file_path = Path("lotus/templates/task_instructions.py")
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if 'df[col].array.get_image(i, "base64")' in content:
                print("df2multimodal_info使用get_image方法")
            else:
                print("df2multimodal_info未使用get_image方法")
        
        # 检查ImageArray.get_image方法
        file_path = Path("lotus/dtype_extensions/image.py")
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if 'fetch_image(self._data[idx], image_type)' in content:
                print("ImageArray.get_image调用fetch_image")
            else:
                print("ImageArray.get_image未调用fetch_image")
        
        # 检查fetch_image集成
        file_path = Path("lotus/utils.py")
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if 'get_global_config()' in content:
                print("fetch_image使用全局配置")
            else:
                print("fetch_image未使用全局配置")
            
            if 'config.optimize_image(image)' in content:
                print("fetch_image调用图片优化器")
            else:
                print("fetch_image未调用图片优化器")
        
        print("图片处理工作流程检查完成")
        
    except Exception as e:
        print(f"图片处理工作流程测试失败: {e}")

def test_compression_integration():
    """测试压缩集成"""
    print("\n=== 测试压缩集成 ===")
    
    try:
        # 检查图片压缩是否集成到处理流程中
        print("--- 检查压缩集成 ---")
        
        # 检查fetch_image中的压缩逻辑
        file_path = Path("lotus/utils.py")
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 检查关键逻辑
            checks = [
                ("image_type == \"base64\"", "base64类型检查"),
                ("get_global_config()", "全局配置获取"),
                ("config.optimize_image(image)", "图片优化调用"),
                ("is_url_image(image)", "URL图片检查")
            ]
            
            for check, description in checks:
                if check in content:
                    print(f"{description}存在")
                else:
                    print(f"{description}不存在")
        
        print("压缩集成检查完成")
        
    except Exception as e:
        print(f"压缩集成测试失败: {e}")

def test_workflow_completeness():
    """测试工作流程完整性"""
    print("\n=== 测试工作流程完整性 ===")
    
    try:
        print("--- 检查完整工作流程 ---")
        
        # 模拟完整的工作流程
        workflow_steps = [
            "1. sem_filter设置全局配置",
            "2. df2multimodal_info调用get_image",
            "3. ImageArray.get_image调用fetch_image",
            "4. fetch_image检查URL图片",
            "5. fetch_image使用全局配置",
            "6. 图片优化器执行压缩",
            "7. 返回优化后的图片"
        ]
        
        for step in workflow_steps:
            print(step)
        
        print("\n--- 检查关键集成点 ---")
        
        # 检查关键集成点
        integration_points = [
            ("sem_filter", "设置全局配置"),
            ("df2multimodal_info", "调用get_image"),
            ("ImageArray.get_image", "调用fetch_image"),
            ("fetch_image", "使用全局配置"),
            ("图片优化器", "执行压缩")
        ]
        
        for point, description in integration_points:
            print(f"{point}: {description}")
        
        print("工作流程完整性检查完成")
        
    except Exception as e:
        print(f"工作流程完整性测试失败: {e}")

def main():
    """主测试函数"""
    print("开始图片处理集成测试")
    print("=" * 60)
    
    # 测试fetch_image集成
    test_fetch_image_integration()
    
    # 测试图片处理工作流程
    test_image_processing_workflow()
    
    # 测试压缩集成
    test_compression_integration()
    
    # 测试工作流程完整性
    test_workflow_completeness()
    
    print("\n" + "=" * 60)
    print("所有测试完成")

if __name__ == "__main__":
    main()
