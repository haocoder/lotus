"""
Example demonstrating async sem_extract functionality.

This example shows how to use the new async sem_extract functionality
for concurrent batch processing and improved performance.
"""

import asyncio
import pandas as pd
from typing import Dict, List, Any

import lotus
from lotus.models import LM


async def basic_async_extract_example() -> None:
    """Basic example of async sem_extract usage."""
    print("=== Basic Async Extract Example ===")
    
    # Configure the model
    model = LM(model="gpt-4o-mini")  # Use a model that supports async
    lotus.settings.configure(lm=model)
    
    # Sample documents
    docs = [
        {"text": "This product is absolutely amazing! I love it so much."},
        {"text": "Terrible quality, would not recommend to anyone."},
        {"text": "It's okay, nothing special but gets the job done."},
        {"text": "Outstanding service and excellent customer support."},
        {"text": "Poor experience, very disappointed with the purchase."},
    ]
    
    # Output columns to extract
    output_cols = {
        "sentiment": "positive/negative/neutral",
        "confidence": "0-1 scale",
        "emotion": "joy/anger/sadness/fear/surprise/disgust"
    }
    
    # Perform async extraction
    result = await lotus.sem_extract_async(
        docs=docs,
        model=model,
        output_cols=output_cols,
        extract_quotes=True,
        use_batch_processing=True,
        batch_size=2,
        max_concurrent_batches=2
    )
    
    print(f"Extracted {len(result.outputs)} results:")
    for i, output in enumerate(result.outputs):
        print(f"Document {i+1}: {output}")


async def dataframe_async_extract_example() -> None:
    """Example of async DataFrame extraction."""
    print("\n=== DataFrame Async Extract Example ===")
    
    # Create a sample DataFrame
    df = pd.DataFrame({
        'text': [
            "The movie was fantastic! I loved every minute of it.",
            "Boring and predictable, waste of time.",
            "Decent film, good acting but slow pacing.",
            "Amazing cinematography and brilliant storytelling.",
            "Terrible plot, confusing and hard to follow."
        ],
        'rating': [5, 1, 3, 5, 2],
        'genre': ['action', 'drama', 'comedy', 'thriller', 'horror']
    })
    
    print("Original DataFrame:")
    print(df)
    
    # Perform async extraction on DataFrame
    result_df = await df.sem_extract.async_extract(
        input_cols=['text'],
        output_cols={
            'sentiment': 'positive/negative/neutral',
            'confidence': '0-1 scale',
            'emotion': 'joy/anger/sadness/fear/surprise/disgust'
        },
        extract_quotes=True,
        use_batch_processing=True,
        batch_size=2,
        max_concurrent_batches=2
    )
    
    print("\nDataFrame with extracted attributes:")
    print(result_df[['text', 'sentiment', 'confidence', 'emotion']])


async def performance_comparison_example() -> None:
    """Compare performance between sync and async processing."""
    print("\n=== Performance Comparison Example ===")
    
    # Create a larger dataset for performance testing
    docs = [{"text": f"Sample document {i} with some text content."} for i in range(20)]
    output_cols = {"sentiment": "positive/negative/neutral", "confidence": "0-1 scale"}
    
    model = LM(model="gpt-4o-mini")
    lotus.settings.configure(lm=model)
    
    # Test async processing
    print("Testing async processing...")
    start_time = asyncio.get_event_loop().time()
    
    async_result = await lotus.sem_extract_async(
        docs=docs,
        model=model,
        output_cols=output_cols,
        use_batch_processing=True,
        batch_size=5,
        max_concurrent_batches=3
    )
    
    async_time = asyncio.get_event_loop().time() - start_time
    print(f"Async processing time: {async_time:.2f} seconds")
    print(f"Processed {len(async_result.outputs)} documents")
    
    # Test sync processing for comparison
    print("\nTesting sync processing...")
    start_time = asyncio.get_event_loop().time()
    
    sync_result = lotus.sem_extract(
        docs=docs,
        model=model,
        output_cols=output_cols,
        use_batch_processing=True,
        batch_size=5
    )
    
    sync_time = asyncio.get_event_loop().time() - start_time
    print(f"Sync processing time: {sync_time:.2f} seconds")
    print(f"Processed {len(sync_result.outputs)} documents")
    
    # Calculate performance improvement
    if sync_time > 0:
        improvement = ((sync_time - async_time) / sync_time) * 100
        print(f"\nPerformance improvement: {improvement:.1f}% faster with async processing")


async def error_handling_example() -> None:
    """Example of error handling in async processing."""
    print("\n=== Error Handling Example ===")
    
    model = LM(model="gpt-4o-mini")
    lotus.settings.configure(lm=model)
    
    docs = [
        {"text": "This is a valid document."},
        {"text": "Another valid document."},
    ]
    
    output_cols = {"sentiment": "positive/negative/neutral"}
    
    try:
        # This should work normally
        result = await lotus.sem_extract_async(
            docs=docs,
            model=model,
            output_cols=output_cols,
            use_batch_processing=True,
            batch_size=1,
            max_concurrent_batches=1
        )
        
        print(f"Successfully processed {len(result.outputs)} documents")
        
    except Exception as e:
        print(f"Error occurred: {e}")


async def main() -> None:
    """Run all examples."""
    print("Async Sem Extract Examples")
    print("=" * 50)
    
    try:
        await basic_async_extract_example()
        await dataframe_async_extract_example()
        await performance_comparison_example()
        await error_handling_example()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run the async examples
    asyncio.run(main())
