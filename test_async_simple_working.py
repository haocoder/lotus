"""
End-to-end async test for DataFrame sem_extract accessor functionality.
"""

import asyncio
import time
import pandas as pd
import lotus
from lotus.models import LM


async def test_dataframe_async_basic():
    """Test basic async DataFrame sem_extract functionality."""
    print("🧪 Testing DataFrame Async Sem Extract - Basic")
    print("=" * 60)
    
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
            "This product is amazing! I love it.",
            "Terrible quality, would not recommend.",
            "It's okay, nothing special.",
            "Outstanding performance, highly recommended!",
            "Poor quality, not worth the money."
        ],
        'category': ['electronics', 'service', 'general', 'electronics', 'general'],
        'price': [299.99, 0, 50.00, 599.99, 25.00]
    })
    
    output_cols = {
        'sentiment': 'positive/negative/neutral',
        'rating': '1-5 scale',
        'recommendation': 'yes/no'
    }
    
    print("Original DataFrame:")
    print(df[['text', 'category', 'price']].to_string(index=False))
    
    print("\n--- Testing Async DataFrame Processing ---")
    start_time = time.time()
    
    # Test async DataFrame processing
    result_df = await df.sem_extract.async_extract(
        input_cols=['text'],
        output_cols=output_cols,
        use_batch_processing=False
    )
    
    processing_time = time.time() - start_time
    print(f"Async DataFrame processing time: {processing_time:.2f} seconds")
    
    print("\nDataFrame with extracted columns:")
    print(result_df[['text', 'sentiment', 'rating', 'recommendation']].to_string(index=False))
    
    # Verify results
    assert 'sentiment' in result_df.columns
    assert 'rating' in result_df.columns
    assert 'recommendation' in result_df.columns
    assert len(result_df) == 5
    
    print("✅ Basic async DataFrame test successful!")
    return result_df


async def test_dataframe_async_batch_processing():
    """Test async DataFrame batch processing."""
    print("\n⚡ Testing DataFrame Async Batch Processing")
    print("=" * 60)
    
    # Create larger DataFrame for batch testing
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
            "Perfect family movie, everyone enjoyed it."
        ],
        'genre': ['action', 'drama', 'comedy', 'thriller', 'horror', 'drama', 'comedy', 'thriller', 'action', 'family'],
        'year': [2023, 2022, 2023, 2024, 2022, 2023, 2022, 2024, 2023, 2024]
    })
    
    output_cols = {
        'sentiment': 'positive/negative/neutral',
        'rating': '1-5 scale',
        'recommendation': 'yes/no'
    }
    
    print(f"Testing with {len(df)} documents...")
    print("Sample documents:")
    for i in range(min(3, len(df))):
        print(f"  {i+1}. {df.iloc[i]['text']}")
    
    # Test async batch processing
    print("\n--- Testing Async Batch Processing ---")
    start_time = time.time()
    
    result_df = await df.sem_extract.async_extract(
        input_cols=['text'],
        output_cols=output_cols,
        use_batch_processing=True,
        batch_size=3,
        max_concurrent_batches=2
    )
    
    processing_time = time.time() - start_time
    print(f"Async batch processing time: {processing_time:.2f} seconds")
    
    print("\nDataFrame with extracted columns:")
    print(result_df[['text', 'sentiment', 'rating', 'recommendation']].to_string(index=False))
    
    # Verify results
    assert 'sentiment' in result_df.columns
    assert 'rating' in result_df.columns
    assert 'recommendation' in result_df.columns
    assert len(result_df) == 10
    
    print("✅ Async batch processing test successful!")
    return result_df


async def test_dataframe_async_with_quotes():
    """Test async DataFrame processing with quote extraction."""
    print("\n📊 Testing DataFrame Async Processing with Quotes")
    print("=" * 60)
    
    # Create DataFrame with text that contains quotes
    df = pd.DataFrame({
        'text': [
            "The customer said 'This is the best product ever!' and gave it 5 stars.",
            "Review: 'Worst purchase of my life' - very disappointed.",
            "Mixed feelings: 'It's okay' but not great."
        ],
        'source': ['review', 'feedback', 'comment']
    })
    
    output_cols = {
        'sentiment': 'positive/negative/neutral',
        'confidence': '0-1 scale'
    }
    
    print("Original DataFrame:")
    print(df.to_string(index=False))
    
    # Test async DataFrame processing with quotes
    result_df = await df.sem_extract.async_extract(
        input_cols=['text'],
        output_cols=output_cols,
        extract_quotes=True,
        use_batch_processing=False
    )
    
    print("\nDataFrame with extracted columns and quotes:")
    print(result_df[['text', 'sentiment', 'confidence', 'quotes']].to_string(index=False))
    
    # Verify results
    assert 'sentiment' in result_df.columns
    assert 'confidence' in result_df.columns
    assert 'quotes' in result_df.columns
    assert len(result_df) == 3
    
    print("✅ Async DataFrame with quotes test successful!")
    return result_df


async def main():
    """Run all DataFrame async tests."""
    try:
        await test_dataframe_async_basic()
        await test_dataframe_async_batch_processing()
        await test_dataframe_async_with_quotes()
        
        print("\n" + "=" * 60)
        print("✅ All DataFrame async tests completed successfully!")
        print("🎉 DataFrame sem_extract async functionality is working correctly!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
