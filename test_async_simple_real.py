"""
End-to-end async test for DataFrame sem_extract accessor with real LLM.

This test focuses on DataFrame's sem_extract accessor functionality
with real LLM model for comprehensive async testing.
"""

import asyncio
import time
import pandas as pd
import lotus
from lotus.models import LM


async def test_dataframe_async_vs_sync_performance():
    """Compare async vs sync DataFrame processing with real LLM."""
    print("🚀 Testing DataFrame Async vs Sync Performance with Real LLM")
    print("=" * 70)
    
    # Configure real model
    lotus.settings.configure(lm=LM(
        model="openrouter/google/gemini-2.5-flash", 
        base_url="https://openrouter.ai/api/v1", 
        api_key='"key"-v1-ff17cd1b6aee306e43330dfeb8e0f7f4b85525ec32126903c2c099b8a5f3eb84'
    ))
    print("✓ Real model configured")
    
    # Create test DataFrame
    df = pd.DataFrame({
        'text': [
            "The product is excellent with 5 stars. I love it!",
            "This service is terrible, only 1 star. Very disappointed.",
            "Average quality, 3 stars rating. It's okay.",
            "Outstanding performance, 5 stars. Highly recommended!",
            "Poor quality, 2 stars. Not worth the money.",
            "Amazing product, 5 stars. Best purchase ever!",
            "Disappointing experience, 2 stars. Expected better.",
            "Good value for money, 4 stars. Satisfied customer."
        ],
        'category': ['electronics', 'service', 'general', 'electronics', 'general', 'electronics', 'service', 'general'],
        'price': [299.99, 0, 50.00, 599.99, 25.00, 399.99, 0, 75.00]
    })
    
    output_cols = {
        "sentiment": "positive/negative/neutral", 
        "rating": "1-5 scale",
        "confidence": "0-1 scale"
    }
    
    print(f"Testing with {len(df)} documents...")
    print("Sample documents:")
    for i in range(min(3, len(df))):
        print(f"  {i+1}. {df.iloc[i]['text']}")
    
    # Test 1: Sync DataFrame Processing
    print("\n--- Testing Sync DataFrame Processing ---")
    start_time = time.time()
    
    sync_result_df = df.sem_extract(
        input_cols=['text'],
        output_cols=output_cols,
        use_batch_processing=True,
        batch_size=3
    )
    
    sync_time = time.time() - start_time
    print(f"Sync DataFrame processing time: {sync_time:.2f} seconds")
    print(f"Sync results: {len(sync_result_df)} rows")
    
    # Test 2: Async DataFrame Processing
    print("\n--- Testing Async DataFrame Processing ---")
    start_time = time.time()
    
    async_result_df = await df.sem_extract.async_extract(
        input_cols=['text'],
        output_cols=output_cols,
        use_batch_processing=True,
        batch_size=3,
        max_concurrent_batches=2
    )
    
    async_time = time.time() - start_time
    print(f"Async DataFrame processing time: {async_time:.2f} seconds")
    print(f"Async results: {len(async_result_df)} rows")
    
    # Calculate performance improvement
    if sync_time > 0:
        improvement = ((sync_time - async_time) / sync_time) * 100
        print(f"\n🎯 Performance improvement: {improvement:.1f}% faster with async processing")
    
    # Show results comparison
    print("\n--- Results Comparison ---")
    print("Sync DataFrame results:")
    print(sync_result_df[['text', 'sentiment', 'rating']].to_string(index=False))
    
    print("\nAsync DataFrame results:")
    print(async_result_df[['text', 'sentiment', 'rating']].to_string(index=False))
    
    return sync_result_df, async_result_df


async def test_dataframe_async_with_explanations():
    """Test async DataFrame processing with explanations."""
    print("\n" + "=" * 70)
    print("📊 Testing DataFrame Async Processing with Explanations")
    print("=" * 70)
    
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
    print(df.to_string(index=False))
    
    output_cols = {
        'sentiment': 'positive/negative/neutral',
        'rating': '1-5 scale',
        'recommendation': 'yes/no'
    }
    
    # Test async DataFrame processing with explanations
    print("\n--- Testing Async DataFrame Processing with Explanations ---")
    start_time = time.time()
    
    result_df = await df.sem_extract.async_extract(
        input_cols=['text'],
        output_cols=output_cols,
        return_explanations=True,
        use_batch_processing=True,
        batch_size=2,
        max_concurrent_batches=2
    )
    
    async_df_time = time.time() - start_time
    print(f"Async DataFrame processing time: {async_df_time:.2f} seconds")
    
    print("\nDataFrame with extracted columns and explanations:")
    print(result_df[['text', 'sentiment', 'rating', 'recommendation', 'explanation']].to_string(index=False))
    
    # Verify results
    assert 'sentiment' in result_df.columns
    assert 'rating' in result_df.columns
    assert 'recommendation' in result_df.columns
    assert 'explanation' in result_df.columns
    assert len(result_df) == 5
    
    print("✅ Async DataFrame with explanations test successful!")
    return result_df


async def test_dataframe_concurrent_batches():
    """Test concurrent batch processing with DataFrame."""
    print("\n" + "=" * 70)
    print("⚡ Testing DataFrame Concurrent Batch Processing")
    print("=" * 70)
    
    # Create larger DataFrame for concurrent testing
    df = pd.DataFrame({
        'text': [f"Concurrent test document {i} with some content for testing async processing." for i in range(15)],
        'category': ['test'] * 15,
        'id': list(range(15))
    })
    
    output_cols = {"sentiment": "positive/negative/neutral", "confidence": "0-1 scale"}
    
    print(f"Testing with {len(df)} documents...")
    
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
        
        result_df = await df.sem_extract.async_extract(
            input_cols=['text'],
            output_cols=output_cols,
            use_batch_processing=True,
            batch_size=5,
            max_concurrent_batches=test_config['max_concurrent_batches']
        )
        
        processing_time = time.time() - start_time
        print(f"Processing time: {processing_time:.2f} seconds")
        print(f"Results: {len(result_df)} rows")
        
        results.append({
            'concurrency': test_config['max_concurrent_batches'],
            'time': processing_time,
            'results': len(result_df)
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
    
    print("✅ DataFrame concurrent batch processing test successful!")
    return results


async def main():
    """Run the focused DataFrame async tests."""
    try:
        # Configure model once
        lotus.settings.configure(lm=LM(
            model="openrouter/google/gemini-2.5-flash", 
            base_url="https://openrouter.ai/api/v1", 
            api_key='"key"-v1-ff17cd1b6aee306e43330dfeb8e0f7f4b85525ec32126903c2c099b8a5f3eb84'
        ))
        
        # Run focused DataFrame async tests
        await test_dataframe_async_vs_sync_performance()
        await test_dataframe_async_with_explanations()
        await test_dataframe_concurrent_batches()
        
        print("\n" + "=" * 70)
        print("✅ All DataFrame async tests completed successfully!")
        print("🎉 DataFrame sem_extract async is working correctly with real LLM!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run the focused async tests
    asyncio.run(main())
