# sem_sim_join 优化方案 1：移除不必要的子集搜索（ids）

本文档说明 sem_sim_join 在优化方案 1 中的设计动机、接口变更、实现细节、使用建议与预期收益。

## 背景与问题

原实现每次调用都会将右表的行索引作为 `ids` 传入向量库（VS）进行“子集搜索”。在当前 VS（尤其是 GPU 实现）中，`ids` 路径通常会为该子集临时重建索引并执行搜索，这在大规模 join 和频繁调用场景下开销很大：

- 需要提取子集向量、创建临时索引、将向量批量写入索引；
- GPU 场景下还可能涉及 CPU/GPU 间数据迁移；
- 每次调用重复上述过程，时延显著。

当右表本身已有独立索引并被加载时，最优路径是直接在右表的“全量索引”上搜索，再在结果中按 `other.index` 进行后过滤；这样可避免每次重建临时索引。

## 方案概述

1. 默认改为“全量索引搜索 + 后过滤”的快速路径：
   - 直接对已加载的右表索引执行搜索；
   - 用 `other.index` 做集合过滤，仅保留在右表中的结果。
2. 若全量搜索不足以返回每条左记录的 K 个有效结果，则采用“逐步扩 K”的迭代策略（K → 2K → 4K …），最多扩展 `max_search_expansion` 次，尽可能满足 K；
3. 若用户确有“只在右表这批 ids 内搜索”的强限制需求，可通过 `restrict_to_right_ids=True` 显式启用子集搜索（不推荐，除非确有必要）。

## 接口变更

在 `DataFrame.sem_sim_join` 增加了两个参数，兼容旧调用：

- `restrict_to_right_ids: bool = False`
  - 默认 False：使用全量索引搜索 + 后过滤（推荐，快）。
  - True：强制子集搜索，可能引发临时索引构建，较慢。
- `max_search_expansion: int = 10`
  - 逐步扩 K 的最大迭代次数，平衡稳定性与性能。

## 实现要点

- 右表索引加载逻辑保持不变，确保 `vs.index_dir == other.attrs["index_dirs"][right_on]`；
- `restrict_to_right_ids=False` 路径：
  - 调用 `vs(query_vectors, search_K)`（不带 ids）；
  - 使用 `other_index_set` 过滤不在右表的结果；
  - 若部分查询的有效命中数 < K，则将 `search_K` 翻倍继续检索，直至满足或达到 `max_search_expansion`；
  - 最终将收集到的匹配拼装为临时 DataFrame，再与左右表进行一次性 merge；
- `restrict_to_right_ids=True` 路径：沿用旧行为，传入 `ids=right_ids`；保留正确性，但性能可能较差。

## 使用建议

1. 为右表单独构建并持久化索引，`sem_sim_join` 前确保加载对应索引目录；
2. 保持默认 `restrict_to_right_ids=False`，除非确有严格子集约束；
3. 对极大规模 join，可将左侧查询按批次处理（未来版本可在 VS 或算子层提供批处理参数）；
4. GPU 环境建议统一使用 float32 向量，减少显存占用；
5. 若召回不足，可适当增大 K 或允许更多的 `max_search_expansion`；
6. 建议结合近似检索（IVF/HNSW/PQ 等）与参数调优（如 nprobe/efSearch），进一步降低时延（在后续方案中提供）。

## 预期收益

- 避免每次重建临时索引（尤其 GPU 上）：join 时延显著下降，常见可达数倍加速；
- 迭代扩 K 兼顾了稳定性与性能，不影响默认易用性；
- 与 `sem_search` 的策略保持一致，后续便于统一引入近似检索能力。

## 兼容性

- 现有代码无需改动即可使用新行为；
- 若有确切子集检索需求，显式传 `restrict_to_right_ids=True` 保持以往语义；
- 该优化不改变输出格式，仅提升性能与稳定性。


