# 运行 test_sem_agg_async_optimization.py 测试指南

## 简化后的测试文件特点

简化后的 `test_sem_agg_async_optimization.py` 专注于端到端功能测试，包含以下特点：

### 1. 测试覆盖范围
- ✅ **同步和异步模式**：测试 `sem_agg` 的同步和异步处理
- ✅ **分组和不分组**：测试 DataFrame 的分组聚合和非分组聚合
- ✅ **真实数据集**：使用构造的 DataFrame 测试数据集
- ✅ **端到端功能**：测试完整的 sem_agg 工作流程

### 2. 测试数据集
- **sample_dataframe**：6个文档，包含 AI/ML 相关内容
- **large_dataframe**：20个文档，用于性能测试
- 包含多个分类字段：category, priority, author, year

### 3. 测试场景
- 基本同步/异步功能
- DataFrame 访问器测试
- 分组聚合测试
- 多列分组测试
- 大数据集异步处理
- 向后兼容性测试
- 特定列聚合测试

## 运行测试的方法

### 方法 1：使用 pytest 运行单个测试文件
```bash
# 运行整个测试文件
python -m pytest tests/test_sem_agg_async_optimization.py -v

# 运行特定测试类
python -m pytest tests/test_sem_agg_async_optimization.py::TestSemAggEndToEnd -v

# 运行特定测试方法
python -m pytest tests/test_sem_agg_async_optimization.py::TestSemAggEndToEnd::test_sem_agg_sync_mode -v
```

### 方法 2：使用 pytest 运行特定测试模式
```bash
# 只运行同步测试
python -m pytest tests/test_sem_agg_async_optimization.py -k "sync" -v

# 只运行异步测试
python -m pytest tests/test_sem_agg_async_optimization.py -k "async" -v

# 只运行 DataFrame 测试
python -m pytest tests/test_sem_agg_async_optimization.py -k "dataframe" -v
```

### 方法 3：使用 Python 直接运行
```python
# 在 Python 交互环境中
import sys
sys.path.append('.')
from tests.test_sem_agg_async_optimization import TestSemAggEndToEnd
import pytest

# 运行特定测试
pytest.main(['tests/test_sem_agg_async_optimization.py::TestSemAggEndToEnd::test_sem_agg_sync_mode', '-v'])
```

### 方法 4：在 IDE 中运行
- 在 VS Code 或 PyCharm 中打开测试文件
- 点击测试方法旁边的运行按钮
- 或者右键选择 "Run Test"

## 依赖要求

### 必需依赖
```bash
pip install pytest pytest-mock pandas
```

### 项目依赖
确保已安装项目的主要依赖：
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## 测试输出示例

成功运行后，您应该看到类似以下的输出：
```
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-8.3.3, pluggy-1.6.0
cachedir: .pytest_cache
rootdir: G:\Learning\AI\project\lotus
configfile: pytest.ini
plugins: anyio-4.9.0, mock-3.14.0
collecting ... collected 12 items

tests/test_sem_agg_async_optimization.py::TestSemAggEndToEnd::test_sem_agg_sync_mode PASSED [ 8%]
tests/test_sem_agg_async_optimization.py::TestSemAggEndToEnd::test_sem_agg_async_mode PASSED [16%]
tests/test_sem_agg_async_optimization.py::TestSemAggEndToEnd::test_sem_agg_async_direct PASSED [25%]
tests/test_sem_agg_async_optimization.py::TestSemAggEndToEnd::test_dataframe_sync_no_grouping PASSED [33%]
tests/test_sem_agg_async_optimization.py::TestSemAggEndToEnd::test_dataframe_async_no_grouping PASSED [41%]
tests/test_sem_agg_async_optimization.py::TestSemAggEndToEnd::test_dataframe_sync_with_grouping PASSED [50%]
tests/test_sem_agg_async_optimization.py::TestSemAggEndToEnd::test_dataframe_async_with_grouping PASSED [58%]
tests/test_sem_agg_async_optimization.py::TestSemAggEndToEnd::test_dataframe_multi_column_grouping PASSED [66%]
tests/test_sem_agg_async_optimization.py::TestSemAggEndToEnd::test_dataframe_large_dataset_async PASSED [75%]
tests/test_sem_agg_async_optimization.py::TestSemAggEndToEnd::test_backward_compatibility PASSED [83%]
tests/test_sem_agg_async_optimization.py::TestSemAggEndToEnd::test_column_specific_aggregation PASSED [91%]

============================= 12 passed in 2.34s =============================
```

## 故障排除

### 1. 如果遇到导入错误
```bash
# 确保在项目根目录运行
cd G:\Learning\AI\project\lotus

# 检查 Python 路径
python -c "import sys; print(sys.path)"
```

### 2. 如果遇到依赖问题
```bash
# 安装缺失的依赖
pip install pytest-mock

# 或者安装所有开发依赖
pip install -r requirements-dev.txt
```

### 3. 如果测试运行缓慢
```bash
# 使用并行运行（如果安装了 pytest-xdist）
pip install pytest-xdist
python -m pytest tests/test_sem_agg_async_optimization.py -n auto
```

## 测试验证的功能

简化后的测试文件验证了以下核心功能：

1. **基本功能**：同步和异步 sem_agg 函数
2. **DataFrame 集成**：pandas DataFrame 访问器
3. **分组处理**：单列和多列分组聚合
4. **性能优化**：异步处理和并发控制
5. **向后兼容**：原有 API 的兼容性
6. **数据处理**：不同大小和结构的数据集

这些测试确保了 sem_agg 的异步优化功能正常工作，同时保持了与原有代码的兼容性。
