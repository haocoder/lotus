# sem_agg Performance Testing Guide

This guide explains how to test and compare the performance of synchronous vs asynchronous sem_agg operations using different dataset scales.

## Overview

The sem_agg function now supports both synchronous and asynchronous processing modes. The async mode is designed to provide better performance for larger datasets by processing batches concurrently and using thread pools for CPU-intensive operations.

## Performance Testing Tools

### 1. Quick Performance Test

The `quick_performance_test.py` script provides a simple way to test performance with a single dataset size.

**Usage:**
```bash
# Test with default settings (50 documents, 3 iterations)
python examples/quick_performance_test.py

# Test with custom dataset size
python examples/quick_performance_test.py --size 100

# Test with more iterations for better accuracy
python examples/quick_performance_test.py --size 100 --iterations 5
```

**Features:**
- Tests both sync and async modes
- Configurable dataset size and iterations
- Simple performance comparison
- Success rate reporting

### 2. Comprehensive Performance Benchmark

The `sem_agg_performance_benchmark.py` script provides comprehensive testing across multiple dataset sizes and configurations.

**Usage:**
```bash
# Run full benchmark with default settings
python examples/sem_agg_performance_benchmark.py

# Customize output directory and iterations
python examples/sem_agg_performance_benchmark.py --output-dir my_results --iterations 5
```

**Features:**
- Tests multiple dataset sizes (10, 25, 50, 100, 200, 500 documents)
- Tests different concurrency configurations
- Tests different grouping strategies
- **Real-time result logging**: Results are saved immediately after each test
- **Progress tracking**: Real-time progress updates with ETA estimation
- **Live monitoring**: Monitor benchmark progress in real-time
- Generates detailed performance reports
- Saves results in JSON and CSV formats

### 3. Real-time Benchmark Monitoring

The `monitor_benchmark.py` script provides real-time monitoring of running benchmarks.

**Usage:**
```bash
# Monitor benchmark in real-time (default 10-second refresh)
python examples/monitor_benchmark.py

# Monitor with custom refresh interval
python examples/monitor_benchmark.py --refresh-interval 5

# Monitor custom benchmark directory
python examples/monitor_benchmark.py --benchmark-dir my_results
```

**Features:**
- Real-time progress updates
- Performance comparison between sync and async modes
- Dataset size performance breakdown
- Recent test results display
- Automatic completion detection
- Clean, formatted output with emojis and progress indicators

### 4. Unit Tests

The `test_sem_agg_performance.py` file contains comprehensive unit tests for performance validation.

**Usage:**
```bash
# Run all performance tests
pytest tests/test_sem_agg_performance.py -v

# Run specific test
pytest tests/test_sem_agg_performance.py::TestSemAggPerformance::test_sync_vs_async_performance -v
```

## Real-time Monitoring and Logging

### Output Files

The benchmark generates several output files for real-time monitoring:

1. **`realtime_results.csv`**: CSV format with all test results
2. **`realtime_results.json`**: JSON format with all test results
3. **`progress_log.txt`**: Human-readable progress log with timestamps
4. **`realtime_summary.json`**: Real-time performance summaries (updated every 5 tests)
5. **`final_summary.json`**: Final benchmark summary
6. **`performance_report.md`**: Comprehensive performance report
7. **`benchmark_results.json`**: Complete results in JSON format
8. **`benchmark_results.csv`**: Complete results in CSV format

### Real-time Features

- **Immediate result saving**: Each test result is saved immediately upon completion
- **Progress tracking**: Real-time progress updates with percentage and ETA
- **Error handling**: Failed tests are logged with error details
- **Memory monitoring**: Memory usage tracking for each test
- **Performance comparison**: Real-time sync vs async performance comparison
- **Resume capability**: Can resume monitoring even if benchmark is interrupted

### Monitoring Workflow

1. **Start benchmark**: Run the benchmark script in one terminal
2. **Start monitoring**: Run the monitor script in another terminal
3. **Real-time updates**: Monitor shows live progress and statistics
4. **Automatic completion**: Monitor detects when benchmark completes
5. **Final summary**: View comprehensive results and recommendations

## Dataset Scales

The performance testing tools generate datasets of different scales:

### Small Dataset (10 documents)
- Quick testing and validation
- Minimal API costs
- Good for development and debugging

### Medium Dataset (50 documents)
- Balanced testing
- Moderate API costs
- Good for regular performance monitoring

### Large Dataset (100-200 documents)
- Real-world scenario testing
- Higher API costs
- Good for performance optimization validation

### Extra Large Dataset (500+ documents)
- Stress testing
- High API costs
- Good for identifying scalability limits

## Performance Metrics

The testing tools collect the following metrics:

### Timing Metrics
- **Duration**: Total time to complete the aggregation
- **Average**: Mean duration across multiple iterations
- **Min/Max**: Best and worst case performance
- **Standard Deviation**: Performance consistency

### Memory Metrics
- **Memory Usage**: RAM consumption during processing
- **Memory Efficiency**: Memory usage per document

### Success Metrics
- **Success Rate**: Percentage of successful operations
- **Error Rate**: Percentage of failed operations
- **Result Quality**: Verification of output correctness

## Expected Performance Characteristics

### Synchronous Mode
- **Pros**: Simple, predictable, lower memory usage
- **Cons**: Sequential processing, slower for large datasets
- **Best for**: Small datasets (< 50 documents), simple use cases

### Asynchronous Mode
- **Pros**: Concurrent processing, faster for large datasets
- **Cons**: Higher memory usage, more complex configuration
- **Best for**: Large datasets (50+ documents), performance-critical applications

### Performance Improvements
- **Small datasets (10-25 docs)**: Minimal improvement, may be slower due to overhead
- **Medium datasets (50-100 docs)**: 10-30% improvement
- **Large datasets (100+ docs)**: 20-50% improvement
- **Very large datasets (500+ docs)**: 30-60% improvement

## Configuration Recommendations

### Concurrency Settings

#### Low Concurrency (1-2 batches, 2-4 workers)
- Good for: Small datasets, limited resources
- Memory usage: Low
- Performance: Moderate

#### Medium Concurrency (2-4 batches, 4-8 workers)
- Good for: Medium datasets, balanced performance
- Memory usage: Moderate
- Performance: Good

#### High Concurrency (4-8 batches, 8-16 workers)
- Good for: Large datasets, high-performance requirements
- Memory usage: High
- Performance: Excellent

### Grouping Strategies

#### No Grouping
- Fastest processing
- Single aggregated result
- Best for: Overall summaries

#### Single Column Grouping
- Moderate processing time
- One result per group
- Best for: Category-based analysis

#### Multi-Column Grouping
- Slower processing
- One result per group combination
- Best for: Detailed analysis

## API Key Requirements

All performance testing tools require an API key for the language model. Set one of the following environment variables:

```bash
# OpenRouter (recommended for testing)
export OPENROUTER_API_KEY="your_key_here"

# Or use the test key (fallback)
export LOTUS_TEST_API_KEY="your_key_here"

# Or other providers
export OPENAI_API_KEY="your_key_here"
export ANTHROPIC_API_KEY="your_key_here"
```

## Interpreting Results

### Performance Reports

The comprehensive benchmark generates a detailed markdown report with:

1. **Executive Summary**: Overall performance comparison
2. **Performance by Dataset Size**: Detailed breakdown by scale
3. **Optimal Configurations**: Best settings for each dataset size
4. **Recommendations**: Usage guidelines and best practices

### Key Metrics to Watch

1. **Duration**: Lower is better
2. **Improvement Percentage**: Higher is better (for async vs sync)
3. **Success Rate**: Should be 100% for reliable results
4. **Memory Usage**: Monitor for resource constraints

### Red Flags

- **Negative improvement**: Async mode slower than sync
- **Low success rate**: Configuration or API issues
- **High memory usage**: May need to reduce concurrency
- **Inconsistent results**: May need more iterations

## Troubleshooting

### Common Issues

1. **API Rate Limits**: Reduce concurrency or add delays
2. **Memory Issues**: Reduce max_thread_workers
3. **Timeout Errors**: Increase timeout settings
4. **Inconsistent Results**: Increase iterations for better statistics

### Performance Optimization

1. **Start with default settings**: 4 batches, 8 workers
2. **Increase concurrency gradually**: Monitor memory usage
3. **Test with your actual data**: Synthetic data may not reflect real performance
4. **Consider API costs**: Balance performance vs cost

## Best Practices

1. **Test with realistic data**: Use actual document sizes and content
2. **Run multiple iterations**: Get statistical significance
3. **Monitor resource usage**: CPU, memory, and API quotas
4. **Document your findings**: Keep track of optimal configurations
5. **Regular testing**: Performance can change with updates

## Example Results

Here's an example of expected performance improvements:

```
Dataset Size: 100 documents
Iterations: 3

Sync mode:
  Average time: 45.2s
  Min time: 43.1s
  Max time: 47.8s

Async mode:
  Average time: 32.1s
  Min time: 30.5s
  Max time: 34.2s

Performance improvement: 29.0%
✓ Async mode is faster!
```

This shows a significant performance improvement for the async mode with larger datasets.
