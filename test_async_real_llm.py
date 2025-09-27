"""
Real LLM test for async sem_extract functionality.
"""

import asyncio
import time
import lotus
from lotus.models import LM


async def test_real_llm_async():
    """Test async functionality with real LLM."""
    print("🚀 Testing Async Sem Extract with Real LLM")
    print("=" * 60)
    
    # Configure real model
    lotus.settings.configure(lm=LM(
        model="openrouter/google/gemini-2.5-flash", 
        base_url="https://openrouter.ai/api/v1", 
        api_key='"key"-v1-ff17cd1b6aee306e43330dfeb8e0f7f4b85525ec32126903c2c099b8a5f3eb84'
    ))
    print("✓ Real model configured")
    
    # Create test documents
    docs = [
        {"text": "The product is excellent with 5 stars. I love it!"},
        {"text": "This service is terrible, only 1 star. Very disappointed."},
        {"text": "Average quality, 3 stars rating. It's okay."},
        {"text": "Outstanding performance, 5 stars. Highly recommended!"},
        {"text": "Poor quality, 2 stars. Not worth the money."}
    ]
    
    output_cols = {
        "sentiment": "positive/negative/neutral", 
        "rating": "1-5 scale",
        "confidence": "0-1 scale"
    }
    
    print(f"Testing with {len(docs)} documents...")
    print("Documents:")
    for i, doc in enumerate(docs):
        print(f"  {i+1}. {doc['text']}")
    
    # Test sync processing
    print("\n--- Testing Sync Processing ---")
    start_time = time.time()
    
    from lotus.sem_ops.sem_extract import sem_extract
    sync_result = sem_extract(
        docs=docs,
        model=lotus.settings.lm,
        output_cols=output_cols,
        use_batch_processing=True,
        batch_size=2
    )
    
    sync_time = time.time() - start_time
    print(f"Sync processing time: {sync_time:.2f} seconds")
    print(f"Sync results: {len(sync_result.outputs)} outputs")
    
    # Test async processing
    print("\n--- Testing Async Processing ---")
    start_time = time.time()
    
    from lotus.sem_ops.sem_extract import sem_extract_async
    async_result = await sem_extract_async(
        docs=docs,
        model=lotus.settings.lm,
        output_cols=output_cols,
        use_batch_processing=True,
        batch_size=2,
        max_concurrent_batches=2
    )
    
    async_time = time.time() - start_time
    print(f"Async processing time: {async_time:.2f} seconds")
    print(f"Async results: {len(async_result.outputs)} outputs")
    
    # Calculate performance improvement
    if sync_time > 0:
        improvement = ((sync_time - async_time) / sync_time) * 100
        print(f"\n🎯 Performance improvement: {improvement:.1f}% faster with async processing")
    
    # Show results comparison
    print("\n--- Results Comparison ---")
    print("Sync results:")
    for i, output in enumerate(sync_result.outputs):
        print(f"  Doc {i+1}: {output}")
    
    print("\nAsync results:")
    for i, output in enumerate(async_result.outputs):
        print(f"  Doc {i+1}: {output}")
    
    return sync_result, async_result


async def test_async_dataframe_real():
    """Test async DataFrame processing with real LLM."""
    print("\n" + "=" * 60)
    print("📊 Testing Async DataFrame Processing with Real LLM")
    print("=" * 60)
    
    import pandas as pd
    
    # Create DataFrame
    df = pd.DataFrame({
        'text': [
            "The movie was fantastic! I loved every minute of it.",
            "Boring and predictable, waste of time.",
            "Decent film, good acting but slow pacing.",
            "Amazing cinematography and brilliant storytelling.",
            "Terrible plot, confusing and hard to follow."
        ],
        'genre': ['action', 'drama', 'comedy', 'thriller', 'horror'],
        'year': [2023, 2022, 2023, 2024, 2022]
    })
    
    print("Original DataFrame:")
    print(df)
    
    output_cols = {
        'sentiment': 'positive/negative/neutral',
        'rating': '1-5 scale',
        'recommendation': 'yes/no'
    }
    
    # Test async DataFrame processing
    print("\n--- Testing Async DataFrame Processing ---")
    start_time = time.time()
    
    result_df = await df.sem_extract.async_extract(
        input_cols=['text'],
        output_cols=output_cols,
        use_batch_processing=True,
        batch_size=2,
        max_concurrent_batches=2
    )
    
    async_df_time = time.time() - start_time
    print(f"Async DataFrame processing time: {async_df_time:.2f} seconds")
    
    print("\nDataFrame with extracted columns:")
    print(result_df[['text', 'sentiment', 'rating', 'recommendation']])
    
    return result_df


async def test_concurrent_batches_real():
    """Test concurrent batch processing with real LLM."""
    print("\n" + "=" * 60)
    print("⚡ Testing Concurrent Batch Processing with Real LLM")
    print("=" * 60)
    
    # Create more documents for concurrent testing
    docs = [{"text": f"Concurrent test document {i} with comprehensive content for testing async processing capabilities."} for i in range(8)]
    output_cols = {"sentiment": "positive/negative/neutral", "confidence": "0-1 scale"}
    
    print(f"Testing with {len(docs)} documents...")
    
    # Test with different concurrency levels
    concurrency_tests = [
        {"max_concurrent_batches": 1, "description": "Sequential (1 concurrent batch)"},
        {"max_concurrent_batches": 2, "description": "Low concurrency (2 concurrent batches)"},
        {"max_concurrent_batches": 3, "description": "High concurrency (3 concurrent batches)"}
    ]
    
    results = []
    
    for test_config in concurrency_tests:
        print(f"\n--- {test_config['description']} ---")
        start_time = time.time()
        
        from lotus.sem_ops.sem_extract import sem_extract_async
        result = await sem_extract_async(
            docs=docs,
            model=lotus.settings.lm,
            output_cols=output_cols,
            use_batch_processing=True,
            batch_size=2,
            max_concurrent_batches=test_config['max_concurrent_batches']
        )
        
        processing_time = time.time() - start_time
        print(f"Processing time: {processing_time:.2f} seconds")
        print(f"Results: {len(result.outputs)} outputs")
        
        results.append({
            'concurrency': test_config['max_concurrent_batches'],
            'time': processing_time,
            'results': len(result.outputs)
        })
    
    # Show performance comparison
    print("\n--- Performance Comparison ---")
    for result in results:
        print(f"Concurrency {result['concurrency']}: {result['time']:.2f}s")
    
    # Calculate improvement
    if len(results) >= 2:
        sequential_time = results[0]['time']
        concurrent_time = results[-1]['time']
        improvement = ((sequential_time - concurrent_time) / sequential_time) * 100
        print(f"\n🎯 Concurrent processing improvement: {improvement:.1f}% faster")
    
    return results


async def main():
    """Run all real LLM tests."""
    try:
        # Configure model once
        lotus.settings.configure(lm=LM(
            model="openrouter/google/gemini-2.5-flash", 
            base_url="https://openrouter.ai/api/v1", 
            api_key='"key"-v1-ff17cd1b6aee306e43330dfeb8e0f7f4b85525ec32126903c2c099b8a5f3eb84'
        ))
        
        # Run tests
        await test_real_llm_async()
        await test_async_dataframe_real()
        await test_concurrent_batches_real()
        
        print("\n" + "=" * 60)
        print("✅ All real LLM async tests completed successfully!")
        print("🎉 Async sem_extract is working correctly with real LLM!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
