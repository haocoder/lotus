# Async Sem Extract Guide

This guide explains the new asynchronous functionality added to `sem_extract` for improved performance and concurrent processing.

## Overview

The async version of `sem_extract` provides several key advantages:

1. **Concurrent Batch Processing**: Multiple batches can be processed simultaneously
2. **Better I/O Utilization**: Async operations don't block on I/O wait times
3. **Improved Performance**: Significant speedup for large datasets
4. **Resource Management**: Configurable concurrency limits to prevent resource exhaustion

## Key Features

### 1. Async Function: `sem_extract_async`

The main async function that provides the same functionality as the synchronous version but with async/await support.

```python
import asyncio
import lotus
from lotus.models import LM

async def extract_example():
    model = LM(model="gpt-4o")
    docs = [{"text": "Sample document"}]
    output_cols = {"sentiment": "positive/negative/neutral"}
    
    result = await lotus.sem_extract_async(
        docs=docs,
        model=model,
        output_cols=output_cols,
        use_batch_processing=True,
        batch_size=10,
        max_concurrent_batches=3
    )
    
    return result
```

### 2. Async DataFrame Accessor: `async_extract`

The DataFrame accessor now supports async operations:

```python
import pandas as pd

async def dataframe_example():
    df = pd.DataFrame({
        'text': ['Great product!', 'Terrible service']
    })
    
    result_df = await df.sem_extract.async_extract(
        input_cols=['text'],
        output_cols={'sentiment': 'positive/negative/neutral'},
        use_batch_processing=True,
        max_concurrent_batches=2
    )
    
    return result_df
```

## Parameters

### New Async-Specific Parameters

- `max_concurrent_batches` (int, default=3): Maximum number of batches to process concurrently
- All existing parameters from the synchronous version are supported

### Batch Processing Parameters

- `batch_size` (int, default=10): Number of documents per batch
- `use_batch_processing` (bool, default=True): Enable batch processing
- `max_concurrent_batches` (int, default=3): Concurrency limit for batch processing

## Performance Benefits

### 1. Concurrent Batch Processing

The async version can process multiple batches simultaneously:

```python
# Process 100 documents in 10 batches of 10 documents each
# With max_concurrent_batches=3, up to 3 batches run concurrently
result = await lotus.sem_extract_async(
    docs=large_document_list,
    model=model,
    output_cols=output_cols,
    batch_size=10,
    max_concurrent_batches=3  # 3 batches run concurrently
)
```

### 2. I/O Wait Time Utilization

Async operations don't block on I/O operations, allowing better CPU utilization:

```python
# While one batch is waiting for API response,
# other batches can be processed
```

### 3. Resource Management

The semaphore-based concurrency control prevents resource exhaustion:

```python
# Limits concurrent API calls to prevent rate limiting
max_concurrent_batches=3  # Only 3 concurrent API calls
```

## Error Handling

### 1. Batch Failure Recovery

If some batches fail, the system automatically falls back to individual processing:

```python
# If batch processing fails for some documents,
# they are processed individually as fallback
```

### 2. Model Compatibility

The async version gracefully handles models that don't support async operations:

```python
# Falls back to synchronous calls if model doesn't support async
if hasattr(model, 'async_call'):
    result = await model.async_call(...)
else:
    result = model(...)  # Synchronous fallback
```

## Usage Examples

### Basic Async Usage

```python
import asyncio
import lotus
from lotus.models import LM

async def basic_example():
    model = LM(model="gpt-4o-mini")
    lotus.settings.configure(lm=model)
    
    docs = [
        {"text": "This product is amazing!"},
        {"text": "Terrible quality, would not recommend."},
        {"text": "It's okay, nothing special."}
    ]
    
    result = await lotus.sem_extract_async(
        docs=docs,
        model=model,
        output_cols={
            "sentiment": "positive/negative/neutral",
            "confidence": "0-1 scale"
        },
        extract_quotes=True,
        use_batch_processing=True,
        batch_size=2,
        max_concurrent_batches=2
    )
    
    print(f"Extracted {len(result.outputs)} results")
    return result

# Run the async function
result = asyncio.run(basic_example())
```

### DataFrame Async Usage

```python
import pandas as pd
import asyncio

async def dataframe_example():
    df = pd.DataFrame({
        'text': [
            'The movie was fantastic!',
            'Boring and predictable.',
            'Decent film with good acting.'
        ],
        'rating': [5, 1, 3]
    })
    
    result_df = await df.sem_extract.async_extract(
        input_cols=['text'],
        output_cols={
            'sentiment': 'positive/negative/neutral',
            'emotion': 'joy/anger/sadness'
        },
        use_batch_processing=True,
        batch_size=2,
        max_concurrent_batches=2
    )
    
    return result_df

# Run the async function
result_df = asyncio.run(dataframe_example())
```

### Performance Comparison

```python
import time
import asyncio

async def performance_comparison():
    docs = [{"text": f"Document {i}"} for i in range(50)]
    output_cols = {"sentiment": "positive/negative/neutral"}
    
    model = LM(model="gpt-4o-mini")
    lotus.settings.configure(lm=model)
    
    # Test async processing
    start_time = time.time()
    async_result = await lotus.sem_extract_async(
        docs=docs,
        model=model,
        output_cols=output_cols,
        use_batch_processing=True,
        batch_size=10,
        max_concurrent_batches=3
    )
    async_time = time.time() - start_time
    
    # Test sync processing
    start_time = time.time()
    sync_result = lotus.sem_extract(
        docs=docs,
        model=model,
        output_cols=output_cols,
        use_batch_processing=True,
        batch_size=10
    )
    sync_time = time.time() - start_time
    
    print(f"Async time: {async_time:.2f}s")
    print(f"Sync time: {sync_time:.2f}s")
    print(f"Improvement: {((sync_time - async_time) / sync_time) * 100:.1f}%")

asyncio.run(performance_comparison())
```

## Best Practices

### 1. Concurrency Tuning

Adjust `max_concurrent_batches` based on your API limits and system resources:

```python
# For high-rate APIs
max_concurrent_batches=5

# For rate-limited APIs
max_concurrent_batches=2

# For local models
max_concurrent_batches=10
```

### 2. Batch Size Optimization

Balance batch size with concurrency:

```python
# Large batches with low concurrency
batch_size=20
max_concurrent_batches=2

# Small batches with high concurrency
batch_size=5
max_concurrent_batches=5
```

### 3. Error Handling

Always handle potential errors in async operations:

```python
try:
    result = await lotus.sem_extract_async(
        docs=docs,
        model=model,
        output_cols=output_cols
    )
except Exception as e:
    print(f"Extraction failed: {e}")
    # Handle error appropriately
```

### 4. Resource Management

Monitor resource usage with large datasets:

```python
# For very large datasets, consider processing in chunks
chunk_size = 1000
for i in range(0, len(docs), chunk_size):
    chunk = docs[i:i + chunk_size]
    result = await lotus.sem_extract_async(
        docs=chunk,
        model=model,
        output_cols=output_cols,
        max_concurrent_batches=3  # Limit concurrency
    )
```

## Migration Guide

### From Synchronous to Asynchronous

1. **Function Calls**: Replace `sem_extract` with `sem_extract_async`
2. **Await Keywords**: Add `await` before async function calls
3. **Event Loop**: Use `asyncio.run()` to run async functions
4. **DataFrame Accessor**: Use `async_extract` instead of the regular accessor

### Example Migration

```python
# Before (Synchronous)
result = lotus.sem_extract(docs, model, output_cols)
df_result = df.sem_extract(['text'], output_cols)

# After (Asynchronous)
result = await lotus.sem_extract_async(docs, model, output_cols)
df_result = await df.sem_extract.async_extract(['text'], output_cols)
```

## Troubleshooting

### Common Issues

1. **Model Not Supporting Async**: The system automatically falls back to synchronous calls
2. **Rate Limiting**: Reduce `max_concurrent_batches` to avoid API rate limits
3. **Memory Issues**: Reduce `batch_size` for large documents
4. **Timeout Errors**: Increase timeout settings in your model configuration

### Debug Mode

Enable debug logging to monitor async operations:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Your async extraction code here
```

## Conclusion

The async version of `sem_extract` provides significant performance improvements for large-scale document processing while maintaining the same API and functionality as the synchronous version. The concurrent batch processing and better I/O utilization make it ideal for production environments with high-volume document processing requirements.
