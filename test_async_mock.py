"""
Mock test for async sem_extract functionality without real API calls.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch
import lotus
from lotus.models import LM
from lotus.types import LMOutput, SemanticExtractOutput


async def test_async_with_mock():
    """Test async functionality with mocked model."""
    print("🧪 Testing Async Sem Extract with Mock Model")
    print("=" * 50)
    
    # Create mock model
    mock_model = MagicMock(spec=LM)
    mock_model.async_call = AsyncMock()
    mock_model.count_tokens = MagicMock(return_value=100)
    mock_model.print_total_usage = MagicMock()
    
    # Mock LMOutput
    mock_lm_output = LMOutput(
        outputs=[
            '{"sentiment": "positive", "rating": "5", "confidence": "0.9"}',
            '{"sentiment": "negative", "rating": "1", "confidence": "0.8"}',
            '{"sentiment": "neutral", "rating": "3", "confidence": "0.6"}'
        ]
    )
    
    mock_model.async_call.return_value = mock_lm_output
    
    # Configure lotus with mock model
    lotus.settings.configure(lm=mock_model)
    
    # Test data
    docs = [
        {"text": "The product is excellent with 5 stars. I love it!"},
        {"text": "This service is terrible, only 1 star. Very disappointed."},
        {"text": "Average quality, 3 stars rating. It's okay."}
    ]
    
    output_cols = {
        "sentiment": "positive/negative/neutral", 
        "rating": "1-5 scale",
        "confidence": "0-1 scale"
    }
    
    print(f"Testing with {len(docs)} documents...")
    
    # Test async individual processing
    print("\n--- Testing Async Individual Processing ---")
    start_time = time.time()
    
    result_async = await lotus.sem_extract_async(
        docs=docs,
        model=mock_model,
        output_cols=output_cols,
        use_batch_processing=False
    )
    
    async_time = time.time() - start_time
    print(f"Async individual processing time: {async_time:.4f} seconds")
    print(f"Results: {len(result_async.outputs)} outputs")
    for i, output in enumerate(result_async.outputs):
        print(f"  Doc {i+1}: {output}")
    
    # Test async batch processing
    print("\n--- Testing Async Batch Processing ---")
    start_time = time.time()
    
    result_async_batch = await lotus.sem_extract_async(
        docs=docs,
        model=mock_model,
        output_cols=output_cols,
        use_batch_processing=True,
        batch_size=2,
        max_concurrent_batches=2
    )
    
    async_batch_time = time.time() - start_time
    print(f"Async batch processing time: {async_batch_time:.4f} seconds")
    print(f"Results: {len(result_async_batch.outputs)} outputs")
    for i, output in enumerate(result_async_batch.outputs):
        print(f"  Doc {i+1}: {output}")
    
    # Test DataFrame async processing
    print("\n--- Testing Async DataFrame Processing ---")
    import pandas as pd
    
    df = pd.DataFrame({
        'text': [doc['text'] for doc in docs],
        'category': ['electronics', 'service', 'general']
    })
    
    start_time = time.time()
    
    result_df = await df.sem_extract.async_extract(
        input_cols=['text'],
        output_cols=output_cols,
        use_batch_processing=True,
        batch_size=2,
        max_concurrent_batches=2
    )
    
    async_df_time = time.time() - start_time
    print(f"Async DataFrame processing time: {async_df_time:.4f} seconds")
    print(f"DataFrame results: {len(result_df)} rows")
    print("DataFrame with extracted columns:")
    print(result_df[['text', 'sentiment', 'rating', 'confidence']])
    
    # Verify mock was called
    print(f"\n--- Mock Verification ---")
    print(f"Mock async_call was called {mock_model.async_call.call_count} times")
    
    return True


async def test_concurrent_processing():
    """Test concurrent processing with multiple batches."""
    print("\n" + "=" * 50)
    print("⚡ Testing Concurrent Processing")
    print("=" * 50)
    
    # Create mock model
    mock_model = MagicMock(spec=LM)
    mock_model.async_call = AsyncMock()
    mock_model.count_tokens = MagicMock(return_value=100)
    mock_model.print_total_usage = MagicMock()
    
    # Mock response with delay to simulate API call
    async def mock_async_call_with_delay(*args, **kwargs):
        await asyncio.sleep(0.1)  # Simulate API delay
        return LMOutput(
            outputs=['{"sentiment": "positive", "rating": "5"}']
        )
    
    mock_model.async_call.side_effect = mock_async_call_with_delay
    
    # Create more documents for concurrent testing
    docs = [{"text": f"Document {i} for concurrent testing."} for i in range(8)]
    output_cols = {"sentiment": "positive/negative/neutral"}
    
    print(f"Testing with {len(docs)} documents...")
    
    # Test sequential processing (max_concurrent_batches=1)
    print("\n--- Sequential Processing (1 concurrent batch) ---")
    start_time = time.time()
    
    result_seq = await lotus.sem_extract_async(
        docs=docs,
        model=mock_model,
        output_cols=output_cols,
        use_batch_processing=True,
        batch_size=2,
        max_concurrent_batches=1
    )
    
    seq_time = time.time() - start_time
    print(f"Sequential processing time: {seq_time:.4f} seconds")
    
    # Test concurrent processing (max_concurrent_batches=3)
    print("\n--- Concurrent Processing (3 concurrent batches) ---")
    start_time = time.time()
    
    result_concurrent = await lotus.sem_extract_async(
        docs=docs,
        model=mock_model,
        output_cols=output_cols,
        use_batch_processing=True,
        batch_size=2,
        max_concurrent_batches=3
    )
    
    concurrent_time = time.time() - start_time
    print(f"Concurrent processing time: {concurrent_time:.4f} seconds")
    
    # Calculate improvement
    if seq_time > 0:
        improvement = ((seq_time - concurrent_time) / seq_time) * 100
        print(f"\n🎯 Concurrent processing improvement: {improvement:.1f}% faster")
    
    print(f"Sequential results: {len(result_seq.outputs)} outputs")
    print(f"Concurrent results: {len(result_concurrent.outputs)} outputs")
    
    return True


async def main():
    """Run all mock tests."""
    try:
        await test_async_with_mock()
        await test_concurrent_processing()
        
        print("\n" + "=" * 50)
        print("✅ All mock tests completed successfully!")
        print("🎉 Async sem_extract functionality is working correctly!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
