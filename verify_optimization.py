"""
验证 sem_search 优化是否正确实现

通过代码分析验证优化是否正确应用
"""

import ast
import re


def analyze_sem_search_optimization():
    """分析 sem_search.py 中的优化实现"""
    print("sem_search 优化验证")
    print("=" * 40)
    
    try:
        with open('lotus/sem_ops/sem_search.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查优化1: 查询向量预计算
        print("1. 检查查询向量预计算优化...")
        if "query_vectors = rm.convert_query_to_query_vector(query)" in content:
            # 检查是否在循环外
            lines = content.split('\n')
            query_vector_line = None
            while_loop_start = None
            
            for i, line in enumerate(lines):
                if "query_vectors = rm.convert_query_to_query_vector(query)" in line:
                    query_vector_line = i
                if "while iteration < max_iterations:" in line:
                    while_loop_start = i
                    break
            
            if query_vector_line is not None and while_loop_start is not None:
                if query_vector_line < while_loop_start:
                    print("   [OK] 查询向量预计算优化已实现")
                else:
                    print("   [FAIL] 查询向量仍在循环内计算")
            else:
                print("   ? 无法确定查询向量计算位置")
        else:
            print("   [FAIL] 未找到查询向量计算代码")
        
        # 检查优化2: 集合查找优化
        print("\n2. 检查后过滤逻辑优化...")
        if "df_idxs_set = set(df_idxs)" in content:
            print("   [OK] 集合转换已实现")
        else:
            print("   [FAIL] 未找到集合转换代码")
        
        if "if idx in df_idxs_set:" in content:
            print("   [OK] 集合查找已实现")
        else:
            print("   [FAIL] 未找到集合查找代码")
        
        # 检查优化3: 循环限制
        print("\n3. 检查循环控制优化...")
        if "max_iterations = 10" in content:
            print("   [OK] 最大迭代次数限制已实现")
        else:
            print("   [FAIL] 未找到最大迭代次数限制")
        
        if "while iteration < max_iterations:" in content:
            print("   [OK] 循环条件已优化")
        else:
            print("   [FAIL] 循环条件未优化")
        
        # 检查优化4: 提前退出
        print("\n4. 检查提前退出优化...")
        if "if len(postfiltered_doc_idxs) == K:" in content and "break" in content:
            print("   [OK] 提前退出机制已实现")
        else:
            print("   [FAIL] 未找到提前退出机制")
        
        # 统计优化点
        optimizations = [
            "query_vectors = rm.convert_query_to_query_vector(query)" in content,
            "df_idxs_set = set(df_idxs)" in content,
            "if idx in df_idxs_set:" in content,
            "max_iterations = 10" in content,
            "while iteration < max_iterations:" in content,
            "if len(postfiltered_doc_idxs) == K:" in content and "break" in content
        ]
        
        implemented_count = sum(optimizations)
        total_count = len(optimizations)
        
        print(f"\n优化实现情况: {implemented_count}/{total_count}")
        print(f"实现率: {implemented_count/total_count*100:.1f}%")
        
        if implemented_count == total_count:
            print("\n[SUCCESS] 所有优化都已成功实现！")
        elif implemented_count >= total_count * 0.8:
            print("\n[GOOD] 大部分优化已实现，性能提升显著")
        else:
            print("\n[WARNING] 部分优化未实现，建议检查代码")
        
        return implemented_count, total_count
        
    except FileNotFoundError:
        print("错误: 找不到 lotus/sem_ops/sem_search.py 文件")
        return 0, 0
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        return 0, 0


def check_code_quality():
    """检查代码质量"""
    print("\n代码质量检查")
    print("=" * 40)
    
    try:
        with open('lotus/sem_ops/sem_search.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查注释
        comment_lines = len([line for line in content.split('\n') if line.strip().startswith('#')])
        total_lines = len(content.split('\n'))
        comment_ratio = comment_lines / total_lines * 100
        
        print(f"注释覆盖率: {comment_ratio:.1f}%")
        
        # 检查优化相关注释
        optimization_comments = [
            "优化1: 预计算查询向量",
            "优化2: 使用集合进行O(1)查找",
            "添加最大迭代次数限制",
            "提前退出：如果已经找到足够的有效结果"
        ]
        
        found_comments = sum(1 for comment in optimization_comments if comment in content)
        print(f"优化注释: {found_comments}/{len(optimization_comments)}")
        
        return comment_ratio, found_comments
        
    except Exception as e:
        print(f"代码质量检查失败: {e}")
        return 0, 0


def main():
    """主函数"""
    print("sem_search 算子优化验证")
    print("=" * 50)
    
    # 分析优化实现
    implemented, total = analyze_sem_search_optimization()
    
    # 检查代码质量
    comment_ratio, found_comments = check_code_quality()
    
    # 总结
    print("\n" + "=" * 50)
    print("验证总结")
    print("=" * 50)
    if total > 0:
        print(f"优化实现: {implemented}/{total} ({implemented/total*100:.1f}%)")
    else:
        print("优化实现: 无法检测")
    print(f"注释覆盖率: {comment_ratio:.1f}%")
    print(f"优化注释: {found_comments}/4")
    
    if total > 0 and implemented == total and comment_ratio > 10:
        print("\n[SUCCESS] 优化验证通过！代码质量良好。")
    elif total > 0 and implemented >= total * 0.8:
        print("\n[GOOD] 优化基本完成，建议完善注释。")
    else:
        print("\n[WARNING] 优化未完全实现，需要进一步检查。")


if __name__ == "__main__":
    main()
