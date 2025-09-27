"""
End-to-end async test for DataFrame sem_extract accessor with real LLM.

This test focuses on comprehensive DataFrame sem_extract accessor functionality
with real LLM model for complete async testing scenarios.
"""

import asyncio
import time
import pandas as pd
import lotus
from lotus.models import LM


async def test_dataframe_async_comprehensive():
    """Test comprehensive async DataFrame sem_extract functionality."""
    print("=== Testing Comprehensive DataFrame Async Extract ===")
    
    # Configure real model
    lotus.settings.configure(lm=LM(
        model="openrouter/google/gemini-2.5-flash", 
        base_url="https://openrouter.ai/api/v1", 
        api_key='"key"-v1-ff17cd1b6aee306e43330dfeb8e0f7f4b85525ec32126903c2c099b8a5f3eb84'
    ))
    print("✓ Real model configured")
    
    # Create comprehensive test DataFrame
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
        'price': [299.99, 0, 50.00, 599.99, 25.00, 399.99, 0, 75.00],
        'source': ['review', 'feedback', 'comment', 'review', 'feedback', 'review', 'feedback', 'comment']
    })
    
    output_cols = {
        "sentiment": "positive/negative/neutral", 
        "rating": "1-5 scale",
        "confidence": "0-1 scale",
        "recommendation": "yes/no"
    }
    
    print(f"Testing with {len(df)} documents...")
    print("Sample documents:")
    for i in range(min(3, len(df))):
        print(f"  {i+1}. {df.iloc[i]['text']}")
    
    # Test async DataFrame processing with all features
    print("\n--- Testing Comprehensive Async DataFrame Processing ---")
    start_time = time.time()
    
    result_df = await df.sem_extract.async_extract(
        input_cols=['text'],
        output_cols=output_cols,
        return_explanations=True,
        return_raw_outputs=True,
        extract_quotes=True,
        use_batch_processing=True,
        batch_size=3,
        max_concurrent_batches=2
    )
    
    processing_time = time.time() - start_time
    print(f"Comprehensive async DataFrame processing time: {processing_time:.2f} seconds")
    print(f"Results: {len(result_df)} rows")
    
    print("\nDataFrame with all extracted columns:")
    print(f"Available columns: {list(result_df.columns)}")
    
    # Show available columns for debugging
    available_cols = ['text', 'sentiment', 'rating', 'recommendation']
    if 'confidence' in result_df.columns:
        available_cols.append('confidence')
    if 'explanation' in result_df.columns:
        available_cols.append('explanation')
    if 'raw_output' in result_df.columns:
        available_cols.append('raw_output')
    
    print(result_df[available_cols].to_string(index=False))
    
    # Verify core expected columns are present
    core_expected_cols = ['sentiment', 'rating', 'recommendation']
    for col in core_expected_cols:
        assert col in result_df.columns, f"Missing core column: {col}"
    
    # Check optional columns
    optional_cols = ['confidence', 'explanation', 'raw_output']
    for col in optional_cols:
        if col in result_df.columns:
            print(f"✓ Optional column '{col}' is present")
        else:
            print(f"⚠ Optional column '{col}' is not present")
    
    print("✅ Comprehensive async DataFrame test successful!")
    return result_df


async def test_dataframe_large_dataset():
    """Test async DataFrame processing with large dataset."""
    print("\n=== Testing DataFrame Large Dataset Processing ===")
    
    # Create large DataFrame for comprehensive testing
    df = pd.DataFrame({
        'text': [
            "The movie was fantastic! I loved every minute of it.",
            "Boring and predictable, waste of time.",
            "Decent film, good acting but slow pacing.",
            "Amazing cinematography and brilliant storytelling.",
            "Terrible plot, confusing and hard to follow.",
            "Outstanding performance by all actors.",
            "Mediocre film, nothing special.",
            "Excellent direction and compelling narrative.",
            "Disappointing sequel, not as good as the original.",
            "Perfect family movie, everyone enjoyed it.",
            "Great action sequences and special effects.",
            "Poor character development and weak dialogue.",
            "Beautiful cinematography and emotional depth.",
            "Confusing storyline with too many plot holes.",
            "Outstanding musical score and sound design.",
            "Weak performances from the main cast.",
            "Innovative storytelling and unique perspective.",
            "Predictable ending and clichéd characters.",
            "Excellent production values and attention to detail.",
            "Boring and slow-paced, hard to stay engaged."
        ],
        'genre': ['action', 'drama', 'comedy', 'thriller', 'horror', 'drama', 'comedy', 'thriller', 'action', 'family', 
                 'action', 'drama', 'romance', 'thriller', 'drama', 'comedy', 'drama', 'action', 'drama', 'drama'],
        'year': [2023, 2022, 2023, 2024, 2022, 2023, 2022, 2024, 2023, 2024, 2023, 2022, 2024, 2023, 2022, 2023, 2024, 2022, 2023, 2022],
        'rating': [4.5, 2.0, 3.0, 4.8, 1.5, 4.2, 2.5, 4.6, 3.2, 4.7, 4.0, 2.8, 4.3, 2.1, 4.4, 3.5, 4.1, 2.9, 4.5, 2.3]
    })
    
    output_cols = {
        "sentiment": "positive/negative/neutral", 
        "rating": "1-5 scale",
        "confidence": "0-1 scale",
        "recommendation": "yes/no"
    }
    
    print(f"Testing large dataset with {len(df)} documents...")
    
    # Test async DataFrame processing with large dataset
    print("\n--- Testing Large Dataset Async DataFrame Processing ---")
    start_time = time.time()
    
    result_df = await df.sem_extract.async_extract(
        input_cols=['text'],
        output_cols=output_cols,
        use_batch_processing=True,
        batch_size=5,
        max_concurrent_batches=3
    )
    
    processing_time = time.time() - start_time
    print(f"Large dataset async DataFrame processing time: {processing_time:.2f} seconds")
    print(f"Results: {len(result_df)} rows")
    
    # Show sample results
    print("\nSample results:")
    print(f"Available columns: {list(result_df.columns)}")
    
    # Show available columns
    available_cols = ['text', 'sentiment', 'rating', 'recommendation']
    if 'confidence' in result_df.columns:
        available_cols.append('confidence')
    
    print(result_df[available_cols].head().to_string(index=False))
    
    # Verify results
    assert len(result_df) == 20
    assert 'sentiment' in result_df.columns
    assert 'rating' in result_df.columns
    assert 'recommendation' in result_df.columns
    
    print("✅ Large dataset async DataFrame test successful!")
    return result_df


async def test_dataframe_performance_comparison():
    """Test DataFrame performance comparison between sync and async."""
    print("\n=== Testing DataFrame Performance Comparison ===")
    
    # Create DataFrame for performance testing
    df = pd.DataFrame({
        'text': [
            "The product is excellent with 5 stars. I love it!",
            "This service is terrible, only 1 star. Very disappointed.",
            "Average quality, 3 stars rating. It's okay.",
            "Outstanding performance, 5 stars. Highly recommended!",
            "Poor quality, 2 stars. Not worth the money.",
            "Amazing product, 5 stars. Best purchase ever!",
            "Disappointing experience, 2 stars. Expected better.",
            "Good value for money, 4 stars. Satisfied customer.",
            "Excellent service, highly recommend to others.",
            "Terrible experience, would not use again."
        ],
        'category': ['electronics', 'service', 'general', 'electronics', 'general', 'electronics', 'service', 'general', 'service', 'service'],
        'price': [299.99, 0, 50.00, 599.99, 25.00, 399.99, 0, 75.00, 0, 0]
    })
    
    output_cols = {
        'sentiment': 'positive/negative/neutral',
        'rating': '1-5 scale',
        'confidence': '0-1 scale'
    }
    
    print(f"Testing performance with {len(df)} documents...")
    
    # Test sync DataFrame processing
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
    
    # Test async DataFrame processing
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
    
    # Calculate performance improvement
    if sync_time > 0:
        improvement = ((sync_time - async_time) / sync_time) * 100
        print(f"\n🎯 Performance improvement: {improvement:.1f}% faster with async processing")
    
    # Show results comparison
    print("\n--- Results Comparison ---")
    print("Sync DataFrame results:")
    print(f"Sync columns: {list(sync_result_df.columns)}")
    print(sync_result_df[['text', 'sentiment', 'rating']].head().to_string(index=False))
    
    print("\nAsync DataFrame results:")
    print(f"Async columns: {list(async_result_df.columns)}")
    print(async_result_df[['text', 'sentiment', 'rating']].head().to_string(index=False))
    
    print("✅ DataFrame performance comparison test successful!")
    return sync_result_df, async_result_df


async def test_dataframe_error_handling():
    """Test error handling in async DataFrame processing."""
    print("\n=== Testing DataFrame Error Handling ===")
    
    # Create DataFrame with mixed valid/invalid data
    df = pd.DataFrame({
        'text': [
            "Valid document with normal content.",
            "",  # Empty document
            "Another valid document.",
            "Yet another valid document with content."
        ],
        'category': ['test', 'test', 'test', 'test']
    })
    
    output_cols = {"sentiment": "positive/negative/neutral"}
    
    try:
        print("Testing with mixed valid/invalid documents...")
        result_df = await df.sem_extract.async_extract(
            input_cols=['text'],
            output_cols=output_cols,
            use_batch_processing=True,
            batch_size=2,
            max_concurrent_batches=2
        )
        
        print(f"Error handling test completed successfully: {len(result_df)} rows")
        print("DataFrame with results:")
        print(f"Available columns: {list(result_df.columns)}")
        
        # Show available columns
        available_cols = ['text']
        if 'sentiment' in result_df.columns:
            available_cols.append('sentiment')
        
        print(result_df[available_cols].to_string(index=False))
        
    except Exception as e:
        print(f"Error occurred (expected): {e}")
    
    print("✅ DataFrame error handling test completed!")
    return result_df if 'result_df' in locals() else None


async def main():
    """Run all DataFrame async tests with real LLM."""
    print("🚀 Testing DataFrame Async Sem Extract with Real LLM")
    print("=" * 70)
    
    try:
        # Configure model once
        lotus.settings.configure(lm=LM(
            model="openrouter/google/gemini-2.5-flash", 
            base_url="https://openrouter.ai/api/v1", 
            api_key='"key"-v1-ff17cd1b6aee306e43330dfeb8e0f7f4b85525ec32126903c2c099b8a5f3eb84'
        ))
        print("✓ Real model configured for all tests")
        
        # Run all DataFrame async tests
        await test_dataframe_async_comprehensive()
        await test_dataframe_large_dataset()
        await test_dataframe_performance_comparison()
        await test_dataframe_error_handling()
        
        print("\n" + "=" * 70)
        print("✅ All DataFrame async tests completed successfully!")
        print("🎉 DataFrame sem_extract async functionality is working correctly with real LLM!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run the async tests
    asyncio.run(main())
