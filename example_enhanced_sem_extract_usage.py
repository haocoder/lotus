"""
Example usage of enhanced sem_extract with chunking, batching, and async support.

This example demonstrates how to use the enhanced sem_extract functionality
in real-world scenarios with different document types and processing strategies.
"""

import asyncio
import pandas as pd
from typing import List, Dict, Any

import lotus
from lotus.models import LM
from lotus.sem_ops.sem_extract_enhanced import sem_extract_enhanced, sem_extract_enhanced_async


def example_basic_usage():
    """Example of basic enhanced sem_extract usage."""
    print("=== Basic Enhanced sem_extract Usage ===")
    
    # Configure lotus
    model = LM(model="gpt-4o-mini")
    lotus.settings.configure(lm=model)
    
    # Create test documents
    docs = [
        {"text": "This product is excellent! Highly recommended.", "doc_id": "review1"},
        {"text": "Great service and fast delivery.", "doc_id": "review2"},
        {"text": "Very long product review with detailed analysis " * 100, "doc_id": "long_review"},
        {"text": "Short and sweet feedback", "doc_id": "review3"}
    ]
    
    # Define output columns
    output_cols = {
        "sentiment": "positive/negative/neutral",
        "confidence": "0-1 scale",
        "topic": "main topic of the review"
    }
    
    print(f"Processing {len(docs)} documents...")
    
    try:
        # Use enhanced sem_extract with chunking and batching
        result = sem_extract_enhanced(
            docs=docs,
            model=model,
            output_cols=output_cols,
            enable_chunking=True,
            use_batch_processing=True,
            chunk_size=1000,
            chunk_overlap=100,
            chunking_strategy="token",
            aggregation_strategy="merge"
        )
        
        print(f"✓ Successfully processed {len(result.outputs)} documents")
        
        # Display results
        for i, output in enumerate(result.outputs):
            print(f"Document {i+1}: {output}")
            
    except Exception as e:
        print(f"⚠ Processing failed (expected without API key): {e}")


async def example_async_usage():
    """Example of async enhanced sem_extract usage."""
    print("\n=== Async Enhanced sem_extract Usage ===")
    
    model = LM(model="gpt-4o-mini")
    
    # Create mixed document collection
    docs = [
        {"text": "Quick feedback", "doc_id": "quick1"},
        {"text": "Another quick review", "doc_id": "quick2"},
        {"text": "Detailed analysis " * 200, "doc_id": "detailed1"},
        {"text": "Another detailed review " * 150, "doc_id": "detailed2"},
        {"text": "Short comment", "doc_id": "short1"}
    ]
    
    output_cols = {
        "sentiment": "positive/negative/neutral",
        "emotion": "joy/anger/sadness/fear/surprise/disgust",
        "urgency": "high/medium/low"
    }
    
    print(f"Async processing {len(docs)} documents...")
    
    try:
        # Use async enhanced sem_extract
        result = await sem_extract_enhanced_async(
            docs=docs,
            model=model,
            output_cols=output_cols,
            enable_chunking=True,
            use_batch_processing=True,
            max_concurrent_batches=3,
            chunk_size=800,
            chunk_overlap=80
        )
        
        print(f"✓ Async processing completed: {len(result.outputs)} results")
        
        # Display results
        for i, output in enumerate(result.outputs):
            print(f"Document {i+1}: {output}")
            
    except Exception as e:
        print(f"⚠ Async processing failed (expected without API key): {e}")


def example_dataframe_processing():
    """Example of processing DataFrame with enhanced sem_extract."""
    print("\n=== DataFrame Processing Example ===")
    
    # Create sample DataFrame
    df = pd.DataFrame({
        'text': [
            'This product is amazing!',
            'Very long product review with detailed analysis ' * 100,
            'Great service and fast delivery',
            'Another detailed review ' * 150,
            'Quick feedback',
            'Short and sweet review'
        ],
        'rating': [5, 4, 5, 3, 4, 5],
        'category': ['electronics', 'electronics', 'books', 'clothing', 'books', 'books']
    })
    
    print(f"Created DataFrame with {len(df)} rows")
    print("Sample data:")
    print(df.head())
    
    # In a real implementation, you would extend the DataFrame accessor
    # to use the enhanced sem_extract functionality
    print("\nNote: DataFrame accessor integration would be implemented as:")
    print("""
    # Example of how DataFrame accessor would work:
    result_df = df.sem_extract_enhanced(
        input_cols=['text'],
        output_cols={'sentiment': 'positive/negative/neutral'},
        enable_chunking=True,
        use_batch_processing=True,
        chunk_size=1000,
        chunk_overlap=100
    )
    """)


def example_different_strategies():
    """Example of different processing strategies."""
    print("\n=== Different Processing Strategies ===")
    
    model = LM(model="gpt-4o-mini")
    
    # Create test documents
    docs = [
        {"text": "Short document", "doc_id": "short1"},
        {"text": "Another short one", "doc_id": "short2"},
        {"text": "Very long document " * 200, "doc_id": "long1"},
        {"text": "Medium length document " * 50, "doc_id": "medium1"}
    ]
    
    output_cols = {"sentiment": "positive/negative/neutral"}
    
    strategies = [
        {
            "name": "Standard Processing",
            "config": {
                "enable_chunking": False,
                "use_batch_processing": False
            }
        },
        {
            "name": "Batch Processing Only",
            "config": {
                "enable_chunking": False,
                "use_batch_processing": True,
                "batch_size": 3
            }
        },
        {
            "name": "Chunking Only",
            "config": {
                "enable_chunking": True,
                "use_batch_processing": False,
                "chunk_size": 1000,
                "chunk_overlap": 100
            }
        },
        {
            "name": "Enhanced Processing",
            "config": {
                "enable_chunking": True,
                "use_batch_processing": True,
                "batch_size": 3,
                "chunk_size": 1000,
                "chunk_overlap": 100,
                "chunking_strategy": "token",
                "aggregation_strategy": "merge"
            }
        }
    ]
    
    for strategy in strategies:
        print(f"\nTesting {strategy['name']}...")
        
        try:
            result = sem_extract_enhanced(
                docs=docs,
                model=model,
                output_cols=output_cols,
                **strategy['config']
            )
            print(f"  ✓ {strategy['name']}: {len(result.outputs)} results")
        except Exception as e:
            print(f"  ⚠ {strategy['name']} failed (expected): {e}")


def example_error_handling():
    """Example of error handling and fallback mechanisms."""
    print("\n=== Error Handling Example ===")
    
    model = LM(model="gpt-4o-mini")
    
    # Create problematic documents
    problematic_docs = [
        {"text": "", "doc_id": "empty"},  # Empty document
        {"text": None, "doc_id": "none"},  # None text
        {"text": "Normal document", "doc_id": "normal"},
        {"text": "Very long document " * 1000, "doc_id": "very_long"}
    ]
    
    output_cols = {"sentiment": "positive/negative/neutral"}
    
    print("Testing error handling with problematic documents...")
    
    try:
        result = sem_extract_enhanced(
            docs=problematic_docs,
            model=model,
            output_cols=output_cols,
            enable_chunking=True,
            use_batch_processing=True
        )
        print(f"✓ Error handling successful: {len(result.outputs)} results")
        
        # Show how the system handles different document types
        for i, output in enumerate(result.outputs):
            print(f"  Document {i+1}: {output}")
            
    except Exception as e:
        print(f"⚠ Error handling test failed (expected): {e}")


def example_performance_comparison():
    """Example of performance comparison between strategies."""
    print("\n=== Performance Comparison Example ===")
    
    model = LM(model="gpt-4o-mini")
    
    # Create test documents
    docs = [
        {"text": f"Short document {i}", "doc_id": f"short{i}"} for i in range(5)
    ] + [
        {"text": f"Long document {i} " * 200, "doc_id": f"long{i}"} for i in range(3)
    ]
    
    output_cols = {"sentiment": "positive/negative/neutral"}
    
    print(f"Testing with {len(docs)} documents (5 short, 3 long)")
    
    # Test different strategies
    strategies = [
        ("No Chunking, No Batching", {
            "enable_chunking": False,
            "use_batch_processing": False
        }),
        ("No Chunking, With Batching", {
            "enable_chunking": False,
            "use_batch_processing": True,
            "batch_size": 4
        }),
        ("With Chunking, No Batching", {
            "enable_chunking": True,
            "use_batch_processing": False,
            "chunk_size": 1000
        }),
        ("With Chunking, With Batching", {
            "enable_chunking": True,
            "use_batch_processing": True,
            "batch_size": 4,
            "chunk_size": 1000
        })
    ]
    
    for strategy_name, config in strategies:
        print(f"\nTesting {strategy_name}...")
        
        try:
            result = sem_extract_enhanced(
                docs=docs,
                model=model,
                output_cols=output_cols,
                **config
            )
            print(f"  ✓ {strategy_name}: {len(result.outputs)} results")
        except Exception as e:
            print(f"  ⚠ {strategy_name} failed (expected): {e}")


async def main():
    """Run all examples."""
    print("Enhanced sem_extract Usage Examples")
    print("=" * 50)
    
    # Run all examples
    example_basic_usage()
    await example_async_usage()
    example_dataframe_processing()
    example_different_strategies()
    example_error_handling()
    example_performance_comparison()
    
    print("\n" + "=" * 50)
    print("All examples completed!")
    print("\nKey Features Demonstrated:")
    print("✓ Basic enhanced sem_extract usage")
    print("✓ Async processing capabilities")
    print("✓ DataFrame integration concepts")
    print("✓ Different processing strategies")
    print("✓ Error handling and fallbacks")
    print("✓ Performance comparison")
    
    print("\nNext Steps:")
    print("1. Configure your API key in lotus.settings")
    print("2. Run the examples with real model calls")
    print("3. Integrate enhanced sem_extract into your applications")
    print("4. Monitor performance and adjust parameters as needed")


if __name__ == "__main__":
    # Run all examples
    asyncio.run(main())
