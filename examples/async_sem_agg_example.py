"""
Example demonstrating the async optimization of sem_agg functionality.

This example shows how to use the optimized sem_agg with async processing
for better performance when dealing with large datasets or multiple groups.
"""

import asyncio
import time
from typing import Any

import pandas as pd

import lotus
from lotus.models import LM


def create_sample_dataframe(num_rows: int = 100) -> pd.DataFrame:
    """
    Create a sample DataFrame for testing async optimization.

    Args:
        num_rows (int): Number of rows to create. Defaults to 100.

    Returns:
        pd.DataFrame: A sample DataFrame with text data and categories.
    """
    categories = ['Technology', 'Science', 'Business', 'Health', 'Education']
    texts = [
        f"This is document {i} about {categories[i % len(categories)].lower()}. "
        f"It contains important information about various topics and concepts. "
        f"The content is designed to test the async aggregation functionality."
        for i in range(num_rows)
    ]
    
    return pd.DataFrame({
        'text': texts,
        'category': [categories[i % len(categories)] for i in range(num_rows)],
        'priority': [i % 3 for i in range(num_rows)]  # 0, 1, 2
    })


def benchmark_aggregation(
    df: pd.DataFrame, 
    instruction: str, 
    use_async: bool = False,
    max_concurrent_batches: int = 4,
    max_thread_workers: int = 8
) -> tuple[pd.DataFrame, float]:
    """
    Benchmark the aggregation performance with and without async processing.

    Args:
        df (pd.DataFrame): The DataFrame to aggregate.
        instruction (str): The aggregation instruction.
        use_async (bool): Whether to use async processing.
        max_concurrent_batches (int): Max concurrent batches for async mode.
        max_thread_workers (int): Max thread workers for async mode.

    Returns:
        tuple[pd.DataFrame, float]: The result DataFrame and execution time in seconds.
    """
    start_time = time.time()
    
    result = df.sem_agg(
        user_instruction=instruction,
        all_cols=True,
        group_by=['category'],
        use_async=use_async,
        max_concurrent_batches=max_concurrent_batches,
        max_thread_workers=max_thread_workers,
        progress_bar_desc=f"Aggregating {'(Async)' if use_async else '(Sync)'}"
    )
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    return result, execution_time


async def async_aggregation_example() -> None:
    """
    Demonstrate async aggregation with a larger dataset.
    """
    print("=== Async Semantic Aggregation Example ===\n")
    
    # Configure the language model
    # Note: Replace with your actual model configuration
    lotus.settings.configure(
        lm=LM(model="gpt-4o-mini", max_batch_size=8)
    )
    
    # Create sample data
    print("Creating sample dataset...")
    df = create_sample_dataframe(50)  # 50 documents across 5 categories
    print(f"Created DataFrame with {len(df)} rows and {len(df.columns)} columns")
    print(f"Categories: {df['category'].unique()}\n")
    
    # Test instruction
    instruction = "Summarize the key themes and insights from the documents in each category"
    
    # Benchmark sync processing
    print("Running synchronous aggregation...")
    sync_result, sync_time = benchmark_aggregation(df, instruction, use_async=False)
    print(f"Synchronous processing time: {sync_time:.2f} seconds")
    print(f"Results shape: {sync_result.shape}\n")
    
    # Benchmark async processing
    print("Running asynchronous aggregation...")
    async_result, async_time = benchmark_aggregation(
        df, instruction, use_async=True, max_concurrent_batches=4, max_thread_workers=8
    )
    print(f"Asynchronous processing time: {async_time:.2f} seconds")
    print(f"Results shape: {async_result.shape}\n")
    
    # Compare results
    print("=== Performance Comparison ===")
    print(f"Sync time: {sync_time:.2f}s")
    print(f"Async time: {async_time:.2f}s")
    if sync_time > 0:
        speedup = sync_time / async_time
        print(f"Speedup: {speedup:.2f}x")
    
    # Display results
    print("\n=== Aggregation Results ===")
    for idx, row in async_result.iterrows():
        print(f"\nCategory: {row['category']}")
        print(f"Summary: {row['_output'][:200]}...")


def direct_async_function_example() -> None:
    """
    Demonstrate using the sem_agg_async function directly.
    """
    print("\n=== Direct Async Function Example ===\n")
    
    # Configure the language model
    lotus.settings.configure(
        lm=LM(model="gpt-4o-mini", max_batch_size=8)
    )
    
    # Sample documents
    docs = [
        "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
        "Deep learning uses neural networks with multiple layers to process data.",
        "Natural language processing enables computers to understand human language.",
        "Computer vision allows machines to interpret and analyze visual information.",
    ]
    
    partition_ids = [0, 0, 1, 1]  # Two partitions
    
    async def run_async_aggregation() -> None:
        """Run the async aggregation."""
        from lotus.sem_ops.sem_agg import sem_agg_async
        
        result = await sem_agg_async(
            docs=docs,
            model=lotus.settings.lm,
            user_instruction="Summarize the key concepts in artificial intelligence",
            partition_ids=partition_ids,
            max_concurrent_batches=2,
            max_thread_workers=4,
        )
        
        print("Async aggregation result:")
        for i, output in enumerate(result.outputs):
            print(f"Partition {i}: {output}")
    
    # Run the async function
    asyncio.run(run_async_aggregation())


def configuration_tips() -> None:
    """
    Provide tips for optimal configuration of async parameters.
    """
    print("\n=== Configuration Tips ===\n")
    
    tips = [
        "max_concurrent_batches: Controls how many batches are processed concurrently at each tree level.",
        "  - Start with 4 for most use cases",
        "  - Increase to 8-16 for large datasets with many groups",
        "  - Decrease to 2-4 if you hit rate limits",
        "",
        "max_thread_workers: Controls CPU-intensive operations like token counting.",
        "  - Start with 8 for most use cases",
        "  - Increase to 16-32 for very large datasets",
        "  - Don't exceed your CPU core count",
        "",
        "When to use async processing:",
        "  - Large datasets (>100 documents)",
        "  - Multiple groups in group_by operations",
        "  - When processing time is a bottleneck",
        "  - When you have sufficient API rate limits",
        "",
        "When to stick with sync processing:",
        "  - Small datasets (<50 documents)",
        "  - Simple aggregation without grouping",
        "  - When API rate limits are restrictive",
        "  - When debugging or development",
    ]
    
    for tip in tips:
        print(tip)


if __name__ == "__main__":
    # Run the examples
    try:
        # Main async example
        asyncio.run(async_aggregation_example())
        
        # Direct async function example
        direct_async_function_example()
        
        # Configuration tips
        configuration_tips()
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure you have configured a valid language model in lotus.settings")
