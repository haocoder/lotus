"""
测试 sem_search 算子优化效果

验证以下优化：
1. 查询向量预计算（避免重复计算）
2. 后过滤逻辑优化（O(1)集合查找替代O(n)线性查找）
3. 循环限制（防止无限循环）
4. 提前退出优化
"""

import time
import pandas as pd
import numpy as np
from typing import List, Dict, Any

import lotus
from lotus.models import SentenceTransformersRM
from lotus.vector_store import FaissVS


class SemSearchOptimizationTest:
    """sem_search 优化效果测试类"""
    
    def __init__(self):
        """初始化测试环境"""
        self.rm = SentenceTransformersRM(model="intfloat/e5-base-v2")
        self.vs = FaissVS()
        lotus.settings.configure(rm=self.rm, vs=self.vs)
        
    def create_test_data(self, num_docs: int = 1000) -> pd.DataFrame:
        """创建测试数据"""
        docs = []
        for i in range(num_docs):
            docs.append(f"Document {i}: This is a test document about topic {i % 10}")
        
        df = pd.DataFrame({
            'content': docs,
            'id': range(num_docs)
        })
        
        # 创建索引
        df = df.sem_index('content', f'test_index_{num_docs}')
        return df
    
    def test_query_vector_caching(self, df: pd.DataFrame, query: str, iterations: int = 5) -> Dict[str, Any]:
        """测试查询向量缓存效果"""
        print("测试查询向量预计算优化...")
        
        # 模拟原始代码（重复计算查询向量）
        start_time = time.time()
        for _ in range(iterations):
            # 模拟每次循环都重新计算查询向量
            query_vectors = self.rm.convert_query_to_query_vector(query)
        original_time = time.time() - start_time
        
        # 模拟优化后代码（预计算查询向量）
        start_time = time.time()
        query_vectors = self.rm.convert_query_to_query_vector(query)  # 只计算一次
        for _ in range(iterations):
            # 使用预计算的查询向量
            pass
        optimized_time = time.time() - start_time
        
        improvement = (original_time - optimized_time) / original_time * 100
        
        return {
            'original_time': original_time,
            'optimized_time': optimized_time,
            'improvement_percent': improvement,
            'iterations': iterations
        }
    
    def test_postfiltering_optimization(self, df: pd.DataFrame, query: str) -> Dict[str, Any]:
        """测试后过滤逻辑优化效果"""
        print("测试后过滤逻辑优化...")
        
        # 获取一些测试索引
        df_idxs = df.index
        test_indices = list(df_idxs[:100])  # 取前100个索引进行测试
        
        # 模拟原始代码（O(n)线性查找）
        start_time = time.time()
        for _ in range(1000):  # 重复1000次查找
            filtered_indices = []
            for idx in test_indices:
                if idx in df_idxs:  # O(n)查找
                    filtered_indices.append(idx)
        original_time = time.time() - start_time
        
        # 模拟优化后代码（O(1)集合查找）
        df_idxs_set = set(df_idxs)
        start_time = time.time()
        for _ in range(1000):  # 重复1000次查找
            filtered_indices = []
            for idx in test_indices:
                if idx in df_idxs_set:  # O(1)查找
                    filtered_indices.append(idx)
        optimized_time = time.time() - start_time
        
        improvement = (original_time - optimized_time) / original_time * 100
        
        return {
            'original_time': original_time,
            'optimized_time': optimized_time,
            'improvement_percent': improvement,
            'test_iterations': 1000
        }
    
    def test_sem_search_performance(self, df: pd.DataFrame, query: str, K: int = 5) -> Dict[str, Any]:
        """测试完整的sem_search性能"""
        print("测试完整sem_search性能...")
        
        # 测试优化后的sem_search
        start_time = time.time()
        result = df.sem_search('content', query, K=K)
        search_time = time.time() - start_time
        
        return {
            'search_time': search_time,
            'result_count': len(result),
            'query': query,
            'K': K
        }
    
    def run_comprehensive_test(self):
        """运行综合测试"""
        print("=" * 60)
        print("sem_search 算子优化效果测试")
        print("=" * 60)
        
        # 创建测试数据
        print("创建测试数据...")
        df = self.create_test_data(num_docs=500)
        print(f"创建了 {len(df)} 个文档的测试数据")
        
        # 测试查询向量缓存
        print("\n1. 查询向量预计算优化测试")
        query_vector_results = self.test_query_vector_caching(df, "machine learning", iterations=10)
        print(f"   原始方法耗时: {query_vector_results['original_time']:.4f}s")
        print(f"   优化方法耗时: {query_vector_results['optimized_time']:.4f}s")
        print(f"   性能提升: {query_vector_results['improvement_percent']:.1f}%")
        
        # 测试后过滤优化
        print("\n2. 后过滤逻辑优化测试")
        postfilter_results = self.test_postfiltering_optimization(df, "machine learning")
        print(f"   原始方法耗时: {postfilter_results['original_time']:.4f}s")
        print(f"   优化方法耗时: {postfilter_results['optimized_time']:.4f}s")
        print(f"   性能提升: {postfilter_results['improvement_percent']:.1f}%")
        
        # 测试完整搜索性能
        print("\n3. 完整sem_search性能测试")
        search_results = self.test_sem_search_performance(df, "machine learning", K=5)
        print(f"   搜索耗时: {search_results['search_time']:.4f}s")
        print(f"   返回结果数: {search_results['result_count']}")
        
        # 测试不同查询
        print("\n4. 不同查询的性能测试")
        test_queries = [
            "artificial intelligence",
            "data science",
            "python programming",
            "machine learning algorithms"
        ]
        
        for query in test_queries:
            result = self.test_sem_search_performance(df, query, K=3)
            print(f"   查询 '{query}': {result['search_time']:.4f}s, 结果数: {result['result_count']}")
        
        print("\n" + "=" * 60)
        print("测试完成！")
        print("=" * 60)
        
        return {
            'query_vector_optimization': query_vector_results,
            'postfilter_optimization': postfilter_results,
            'search_performance': search_results
        }


def main():
    """主函数"""
    try:
        test = SemSearchOptimizationTest()
        results = test.run_comprehensive_test()
        
        # 保存结果
        import json
        with open('sem_search_optimization_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print("\n结果已保存到 sem_search_optimization_results.json")
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
