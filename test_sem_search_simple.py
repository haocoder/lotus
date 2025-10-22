"""
简化的 sem_search 优化测试

测试优化前后的性能差异
"""

import time
import pandas as pd
import numpy as np

import lotus
from lotus.models import SentenceTransformersRM
from lotus.vector_store import FaissVS


def create_simple_test_data():
    """创建简单的测试数据"""
    docs = [
        "Machine learning is a subset of artificial intelligence",
        "Data science involves statistics and programming",
        "Python is a popular programming language",
        "Deep learning uses neural networks",
        "Natural language processing deals with text",
        "Computer vision processes images",
        "Reinforcement learning uses rewards",
        "Supervised learning uses labeled data",
        "Unsupervised learning finds patterns",
        "Clustering groups similar data points"
    ]
    
    df = pd.DataFrame({
        'content': docs,
        'id': range(len(docs))
    })
    
    return df


def test_optimization_effects():
    """测试优化效果"""
    print("sem_search 优化效果测试")
    print("=" * 40)
    
    # 配置模型
    rm = SentenceTransformersRM(model="intfloat/e5-base-v2")
    vs = FaissVS()
    lotus.settings.configure(rm=rm, vs=vs)
    
    # 创建测试数据
    df = create_simple_test_data()
    df = df.sem_index('content', 'test_simple_index')
    
    query = "machine learning"
    K = 3
    
    print(f"测试查询: '{query}'")
    print(f"要求返回: {K} 个结果")
    print(f"数据量: {len(df)} 个文档")
    
    # 测试优化后的sem_search
    print("\n执行优化后的sem_search...")
    start_time = time.time()
    result = df.sem_search('content', query, K=K)
    search_time = time.time() - start_time
    
    print(f"搜索耗时: {search_time:.4f}s")
    print(f"返回结果数: {len(result)}")
    print("\n搜索结果:")
    for i, row in result.iterrows():
        print(f"  {i}: {row['content']}")
    
    # 测试多次搜索的一致性
    print("\n测试多次搜索的一致性...")
    results = []
    for i in range(3):
        result = df.sem_search('content', query, K=K)
        results.append(result['content'].tolist())
    
    # 检查结果是否一致
    all_same = all(results[0] == result for result in results[1:])
    print(f"多次搜索结果一致: {all_same}")
    
    return {
        'search_time': search_time,
        'result_count': len(result),
        'consistent_results': all_same
    }


if __name__ == "__main__":
    try:
        results = test_optimization_effects()
        print(f"\n测试完成！结果: {results}")
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
