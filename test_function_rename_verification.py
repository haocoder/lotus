#!/usr/bin/env python3
"""
验证函数重命名

直接检查文件内容，验证函数重命名是否成功
"""

import re
from pathlib import Path

def check_function_renaming():
    """检查函数重命名"""
    print("=== 函数重命名验证 ===")
    
    file_path = Path("lotus/utils/image_optimizer.py")
    
    if not file_path.exists():
        print("文件不存在: lotus/utils/image_optimizer.py")
        return
    
    # 读取文件内容
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("--- 检查函数定义 ---")
    
    # 检查简单压缩函数
    simple_pattern = r'def _compress_image_simple\('
    if re.search(simple_pattern, content):
        print("_compress_image_simple 函数已定义")
    else:
        print("_compress_image_simple 函数未找到")
    
    # 检查高级压缩函数
    advanced_pattern = r'def _compress_image_advanced\('
    if re.search(advanced_pattern, content):
        print("_compress_image_advanced 函数已定义")
    else:
        print("_compress_image_advanced 函数未找到")
    
    # 检查压缩方法选择函数
    choose_pattern = r'def _choose_compression_method\('
    if re.search(choose_pattern, content):
        print("_choose_compression_method 函数已定义")
    else:
        print("_choose_compression_method 函数未找到")
    
    # 检查旧的压缩函数是否还存在
    old_pattern = r'def _compress_image\('
    matches = re.findall(old_pattern, content)
    if len(matches) > 0:
        print(f"_compress_image 函数仍然存在 {len(matches)} 次")
    else:
        print("_compress_image 函数已完全移除")
    
    print("\n--- 检查函数调用 ---")
    
    # 检查函数调用
    simple_calls = content.count('_compress_image_simple(')
    advanced_calls = content.count('_compress_image_advanced(')
    choose_calls = content.count('_choose_compression_method(')
    old_calls = content.count('_compress_image(')
    
    print(f"_compress_image_simple 调用次数: {simple_calls}")
    print(f"_compress_image_advanced 调用次数: {advanced_calls}")
    print(f"_choose_compression_method 调用次数: {choose_calls}")
    print(f"_compress_image 调用次数: {old_calls}")
    
    print("\n--- 检查配置参数 ---")
    
    # 检查use_advanced_compression参数
    if 'use_advanced_compression' in content:
        print("use_advanced_compression 参数已添加")
    else:
        print("use_advanced_compression 参数未找到")
    
    # 检查参数使用
    if 'self.use_advanced_compression' in content:
        print("use_advanced_compression 参数已使用")
    else:
        print("use_advanced_compression 参数未使用")
    
    print("\n--- 检查文档字符串 ---")
    
    # 检查类文档
    if '简单快速压缩' in content and '智能渐进式压缩' in content:
        print("类文档已更新，包含两种压缩策略说明")
    else:
        print("类文档未更新或缺少压缩策略说明")
    
    print("\n函数重命名验证完成")

def check_function_differences():
    """检查两个压缩函数的差异"""
    print("\n=== 压缩函数差异检查 ===")
    
    file_path = Path("lotus/utils/image_optimizer.py")
    
    if not file_path.exists():
        print("文件不存在: lotus/utils/image_optimizer.py")
        return
    
    # 读取文件内容
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 查找简单压缩函数
    simple_match = re.search(r'def _compress_image_simple\(.*?\):\s*""".*?""".*?(?=def|\Z)', content, re.DOTALL)
    if simple_match:
        simple_func = simple_match.group(0)
        print("找到简单压缩函数")
        
        # 检查简单压缩函数的特点
        if 'thumbnail(' in simple_func:
            print("  - 使用thumbnail方法")
        if 'optimize=True' in simple_func:
            print("  - 启用优化")
        if 'quality=' in simple_func:
            print("  - 使用质量参数")
    else:
        print("未找到简单压缩函数")
    
    # 查找高级压缩函数
    advanced_match = re.search(r'def _compress_image_advanced\(.*?\):\s*""".*?""".*?(?=def|\Z)', content, re.DOTALL)
    if advanced_match:
        advanced_func = advanced_match.group(0)
        print("找到高级压缩函数")
        
        # 检查高级压缩函数的特点
        if 'compression_levels' in advanced_func:
            print("  - 使用渐进式压缩策略")
        if '_resize_image(' in advanced_func:
            print("  - 使用自定义resize方法")
        if '_encode_with_quality(' in advanced_func:
            print("  - 使用质量编码方法")
        if 'best_result' in advanced_func:
            print("  - 自动选择最优结果")
    else:
        print("未找到高级压缩函数")
    
    print("压缩函数差异检查完成")

def main():
    """主测试函数"""
    print("开始函数重命名验证")
    print("=" * 50)
    
    # 检查函数重命名
    check_function_renaming()
    
    # 检查函数差异
    check_function_differences()
    
    print("\n" + "=" * 50)
    print("验证完成")

if __name__ == "__main__":
    main()
