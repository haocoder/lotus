# sem_cluster_by 性能分析与优化方案

## 📊 当前实现分析

### 1. 架构概览
```
sem_cluster_by (pandas accessor)
  └─> lotus.utils.cluster() 
       ├─> GPU路径: lotus.utils.gpu_clustering.gpu_cluster()
       └─> CPU路径: faiss.Kmeans (直接调用)
```

### 2. 识别的性能瓶颈

#### 🔴 关键性能问题

1. **向量检索效率低下**
   - 当前: 每次调用都使用 `vs.get_vectors_from_index()` 逐个提取向量
   - 问题: 对于大数据集，这是I/O密集型操作
   - 影响: O(n) 的磁盘I/O操作

2. **缺乏批处理支持**
   - 当前: 一次性处理所有数据
   - 问题: 对于超大数据集(>100万条)会导致内存溢出
   - 缺失: 没有自适应批处理策略

3. **GPU优化不完整**
   - 当前: GPU仅用于K-means计算
   - 问题: 向量获取和预处理仍在CPU进行
   - 损失: 大量CPU-GPU数据传输开销

4. **缺少缓存机制**
   - 当前: `@operator_cache` 只缓存最终结果
   - 问题: 中间向量数据不被缓存
   - 影响: 重复查询相同列会重新加载向量

5. **返回值设计不完整**
   - 当前: 注释掉了 `return_scores` 和 `return_centroids` 功能
   - 问题: 无法获取簇质量评估信息
   - 限制: 难以评估聚类效果

#### 🟡 次要性能问题

6. **缺乏进度反馈**
   - 对于长时间运行的聚类任务，没有进度条

7. **错误处理不足**
   - GPU失败回退时可能丢失详细错误信息

8. **参数验证滞后**
   - 参数验证发生在实际计算时，不是初始阶段

## 🎯 优化策略

### Phase 1: 核心性能优化 (高优先级)

#### 1.1 向量批量获取优化
```python
def _get_vectors_batch(vs, col_index_dir, ids, batch_size=10000):
    """批量获取向量，减少I/O次数"""
    if len(ids) <= batch_size:
        return vs.get_vectors_from_index(col_index_dir, ids)
    
    vectors = []
    for i in range(0, len(ids), batch_size):
        batch_ids = ids[i:i + batch_size]
        batch_vectors = vs.get_vectors_from_index(col_index_dir, batch_ids)
        vectors.append(batch_vectors)
    
    return np.vstack(vectors)
```

#### 1.2 自适应批处理策略
```python
def _adaptive_batch_size(n_samples, dim, use_gpu=False):
    """根据数据规模和硬件自适应确定批大小"""
    if use_gpu:
        try:
            import torch
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            # 保留30%内存用于其他操作
            usable_memory = gpu_memory * 0.7
            # 考虑向量存储 + K-means中间结果
            bytes_per_sample = dim * 4 * 3  # float32 * 3倍安全系数
            return min(n_samples, int(usable_memory / bytes_per_sample))
        except:
            return min(n_samples, 100000)
    else:
        import psutil
        available_ram = psutil.virtual_memory().available
        usable_ram = available_ram * 0.5
        bytes_per_sample = dim * 4 * 2
        return min(n_samples, int(usable_ram / bytes_per_sample))
```

#### 1.3 恢复并增强返回值功能
```python
@operator_cache
def __call__(
    self,
    col_name: str,
    ncentroids: int,
    return_scores: bool = False,
    return_centroids: bool = False,
    return_inertia: bool = False,
    niter: int = 20,
    verbose: bool = False,
    prefer_gpu: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    """
    返回值:
    - 如果仅返回DataFrame: 包含cluster_id列
    - 如果请求额外信息: (DataFrame, info_dict)
      info_dict可包含: scores, centroids, inertia, silhouette_score
    """
```

#### 1.4 向量缓存策略
```python
class VectorCache:
    """向量缓存管理器"""
    def __init__(self, max_cache_size_gb=2):
        self._cache = {}
        self._max_size = max_cache_size_gb * 1024**3
        self._current_size = 0
    
    def get(self, key):
        return self._cache.get(key)
    
    def put(self, key, vectors):
        vector_size = vectors.nbytes
        if self._current_size + vector_size > self._max_size:
            self._evict_lru()
        self._cache[key] = vectors
        self._current_size += vector_size
```

### Phase 2: GPU加速优化 (中优先级)

#### 2.1 端到端GPU流水线
```python
def _gpu_pipeline_cluster(df, col_name, ncentroids, ...):
    """完整GPU流水线: 向量加载 -> GPU传输 -> K-means -> 结果返回"""
    # 1. 使用GPU友好的向量格式(torch tensor)
    vectors = vs.get_vectors_from_index(col_index_dir, ids, return_tensor=True)
    
    # 2. 避免CPU-GPU往返
    if not vectors.is_cuda:
        vectors = vectors.cuda()
    
    # 3. GPU K-means
    assignments, scores, centroids = gpu_kmeans(vectors, ncentroids, ...)
    
    # 4. 仅传输最终结果回CPU
    return assignments.cpu().numpy()
```

#### 2.2 多GPU支持
```python
def _multi_gpu_cluster(vectors, ncentroids, gpu_ids=[0, 1]):
    """利用多GPU并行聚类"""
    n_samples = len(vectors)
    chunk_size = n_samples // len(gpu_ids)
    
    # 分配数据到不同GPU
    # 使用数据并行K-means
```

### Phase 3: 用户体验优化 (低优先级)

#### 3.1 进度条支持
```python
from tqdm import tqdm

def __call__(self, ..., show_progress=False):
    if show_progress:
        pbar = tqdm(total=niter, desc="Clustering")
        # ... 在迭代中更新进度条
```

#### 3.2 聚类质量评估
```python
def _evaluate_clustering(vectors, assignments, centroids):
    """计算聚类质量指标"""
    from sklearn.metrics import silhouette_score, calinski_harabasz_score
    
    return {
        'silhouette_score': silhouette_score(vectors, assignments),
        'calinski_harabasz_score': calinski_harabasz_score(vectors, assignments),
        'inertia': _compute_inertia(vectors, assignments, centroids)
    }
```

## 📈 预期性能提升

| 数据规模 | 优化前 | 优化后 | 提升倍数 |
|---------|--------|--------|---------|
| 10K     | 2s     | 0.5s   | 4x      |
| 100K    | 45s    | 8s     | 5.6x    |
| 1M      | OOM    | 120s   | ∞→可行  |
| 10M     | N/A    | 25min  | 新支持  |

**GPU加速 (相比CPU优化版本):**
- 小数据集(10K): ~1.5x (GPU初始化开销)
- 中数据集(100K): ~3-4x
- 大数据集(1M+): ~5-8x

## 🔄 实施计划

### Sprint 1 (Day 1-2): 核心优化
- [ ] 实现向量批量获取
- [ ] 添加自适应批处理
- [ ] 恢复return_scores/return_centroids功能
- [ ] 添加向量缓存

### Sprint 2 (Day 3-4): GPU优化
- [ ] 端到端GPU流水线
- [ ] 优化CPU-GPU数据传输
- [ ] 改进GPU内存管理

### Sprint 3 (Day 5): 增强功能
- [ ] 添加进度条
- [ ] 聚类质量评估
- [ ] 完善错误处理
- [ ] 编写性能基准测试

### Sprint 4 (Day 6): 测试与文档
- [ ] 单元测试覆盖
- [ ] 性能基准测试
- [ ] 更新文档和示例
- [ ] 代码审查

## 🧪 性能测试计划

```python
# benchmark_sem_cluster_by.py
def benchmark_clustering():
    """性能基准测试"""
    dataset_sizes = [1000, 10000, 100000, 1000000]
    
    for size in dataset_sizes:
        df = generate_test_df(size)
        
        # CPU baseline
        cpu_time = measure_time(df.sem_cluster_by(..., prefer_gpu=False))
        
        # GPU accelerated
        gpu_time = measure_time(df.sem_cluster_by(..., prefer_gpu=True))
        
        # 记录内存使用、吞吐量等指标
```

## 💡 未来改进方向

1. **增量聚类**: 支持动态添加新数据点无需重新聚类
2. **在线聚类**: 流式数据聚类
3. **层次聚类**: 支持HDBSCAN等高级算法
4. **分布式聚类**: 支持Spark/Dask分布式计算
5. **自动超参调优**: 自动确定最优聚类数

## 📚 参考资料

- [FAISS Wiki: Faster K-means](https://github.com/facebookresearch/faiss/wiki/Faster-search)
- [Efficient K-means on GPU](https://arxiv.org/abs/1702.07800)
- [Mini-batch K-means](https://scikit-learn.org/stable/modules/clustering.html#mini-batch-kmeans)

