#!/usr/bin/env python3
"""
Performance tests for sem_map batch processing functionality.

This script tests the performance improvements of batch processing
compared to individual processing for sem_map operations.
"""

import time
import pandas as pd
import numpy as np
from typing import Any, Dict, List
import json
import os

# Import lotus components
import lotus
from lotus.models import LM
from lotus.types import ReasoningStrategy


def create_test_data(num_docs: int = 50) -> List[Dict[str, Any]]:
    """
    Create test data for performance testing.
    
    Args:
        num_docs: Number of documents to create
        
    Returns:
        List of document dictionaries
    """
    # Sample texts for testing
    sample_texts = [
        "The weather is beautiful today with clear blue skies.",
        "I love reading books in my spare time.",
        "Technology has revolutionized the way we communicate.",
        "The restaurant served delicious food with excellent service.",
        "Learning new programming languages is challenging but rewarding.",
        "The movie had an amazing plot with great character development.",
        "Exercise is important for maintaining good health.",
        "The conference provided valuable insights into industry trends.",
        "Cooking is both an art and a science.",
        "Traveling broadens your perspective on different cultures.",
        "Music has the power to evoke strong emotions.",
        "The project was completed successfully ahead of schedule.",
        "Education is the foundation of personal growth.",
        "The team worked collaboratively to achieve their goals.",
        "Nature provides a peaceful escape from daily stress.",
        "Innovation drives progress in every field.",
        "Friendship is one of life's greatest treasures.",
        "The presentation was well-received by the audience.",
        "Reading helps expand vocabulary and knowledge.",
        "The sunset painted the sky in beautiful colors.",
    ]
    
    # Create documents by repeating and varying the sample texts
    docs = []
    for i in range(num_docs):
        text = sample_texts[i % len(sample_texts)]
        # Add some variation to make each document unique
        variation = f" (Document {i+1})"
        docs.append({"text": text + variation})
    
    return docs


def create_test_dataframe(num_rows: int = 50) -> pd.DataFrame:
    """
    Create a test DataFrame for performance testing.
    
    Args:
        num_rows: Number of rows to create
        
    Returns:
        Test DataFrame
    """
    docs = create_test_data(num_rows)
    return pd.DataFrame(docs)


def benchmark_sem_map_individual(
    df: pd.DataFrame, 
    instruction: str,
    **kwargs
) -> Dict[str, Any]:
    """
    Benchmark individual processing for sem_map using DataFrame interface.
    
    Args:
        df: DataFrame to process
        instruction: Mapping instruction
        **kwargs: Additional arguments for sem_map
        
    Returns:
        Dictionary with timing and result information
    """
    # Configure settings for individual processing
    original_use_batch = lotus.settings.use_batch_processing
    lotus.settings.use_batch_processing = False
    
    try:
        start_time = time.time()
        
        result_df = df.sem_map(instruction, **kwargs)
        print(result_df)
        
        end_time = time.time()
        duration = end_time - start_time
        
        return {
            "duration": duration,
            "num_docs": len(df),
            "result_df": result_df,
            "processing_type": "individual"
        }
    
    finally:
        # Restore original settings
        lotus.settings.use_batch_processing = original_use_batch


def benchmark_sem_map_batch(
    df: pd.DataFrame, 
    instruction: str,
    batch_size: int = 10,
    **kwargs
) -> Dict[str, Any]:
    """
    Benchmark batch processing for sem_map using DataFrame interface.
    
    Args:
        df: DataFrame to process
        instruction: Mapping instruction
        batch_size: Size of each batch
        **kwargs: Additional arguments for sem_map
        
    Returns:
        Dictionary with timing and result information
    """
    # Configure settings for batch processing
    original_use_batch = lotus.settings.use_batch_processing
    original_batch_size = lotus.settings.batch_size
    
    lotus.settings.use_batch_processing = True
    lotus.settings.batch_size = batch_size
    
    try:
        start_time = time.time()
        
        result_df = df.sem_map(instruction, **kwargs)
        print(result_df)
        
        end_time = time.time()
        duration = end_time - start_time
        
        return {
            "duration": duration,
            "num_docs": len(df),
            "batch_size": batch_size,
            "result_df": result_df,
            "processing_type": "batch"
        }
    
    finally:
        # Restore original settings
        lotus.settings.use_batch_processing = original_use_batch
        lotus.settings.batch_size = original_batch_size




def run_performance_comparison(
    test_sizes: List[int] = [10, 25, 50],
    batch_sizes: List[int] = [5, 10, 15],
    instruction: str = "Summarize the {text} in one sentence:"
) -> Dict[str, Any]:
    """
    Run comprehensive performance comparison between individual and batch processing.
    
    Args:
        test_sizes: List of document counts to test
        batch_sizes: List of batch sizes to test
        instruction: Mapping instruction to use
        
    Returns:
        Dictionary with all performance results
    """
    results = {
        "individual_processing": {},
        "batch_processing": {},
        "summary": {}
    }
    
    print("Starting sem_map batch processing performance tests...")
    print(f"Test sizes: {test_sizes}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Instruction: {instruction}")
    print("-" * 60)
    
    for size in test_sizes:
        print(f"\nTesting with {size} documents...")
        
        # Create test DataFrame
        df = create_test_dataframe(size)
        
        # Test individual processing
        print("  Testing individual processing...")
        individual_result = benchmark_sem_map_individual(df, instruction)
        results["individual_processing"][size] = individual_result
        
        # Test batch processing with different batch sizes
        results["batch_processing"][size] = {}
        for batch_size in batch_sizes:
            if batch_size <= size:  # Only test batch sizes that make sense
                print(f"  Testing batch processing (batch_size={batch_size})...")
                batch_result = benchmark_sem_map_batch(
                    df, instruction, batch_size=batch_size
                )
                results["batch_processing"][size][batch_size] = batch_result
        
        # Print results for this size
        print(f"    Individual: {individual_result['duration']:.2f}s")
        for batch_size in batch_sizes:
            if batch_size in results["batch_processing"][size]:
                batch_result = results["batch_processing"][size][batch_size]
                speedup = individual_result['duration'] / batch_result['duration']
                print(f"    Batch (size={batch_size}): {batch_result['duration']:.2f}s (speedup: {speedup:.2f}x)")
    
    # Calculate summary statistics
    results["summary"] = calculate_summary_stats(results)
    
    return results


def calculate_summary_stats(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate summary statistics from performance results.
    
    Args:
        results: Performance results dictionary
        
    Returns:
        Dictionary with summary statistics
    """
    summary = {
        "average_speedups": {},
        "best_batch_sizes": {},
        "performance_trends": {}
    }
    
    # Calculate average speedups for each batch size
    for size in results["individual_processing"]:
        individual_time = results["individual_processing"][size]["duration"]
        
        speedups = []
        for batch_size in results["batch_processing"][size]:
            batch_time = results["batch_processing"][size][batch_size]["duration"]
            speedup = individual_time / batch_time
            speedups.append((batch_size, speedup))
        
        if speedups:
            # Find best batch size
            best_batch_size, best_speedup = max(speedups, key=lambda x: x[1])
            summary["best_batch_sizes"][size] = {
                "batch_size": best_batch_size,
                "speedup": best_speedup
            }
            
            # Calculate average speedup
            avg_speedup = sum(speedup for _, speedup in speedups) / len(speedups)
            summary["average_speedups"][size] = avg_speedup
    
    return summary


def print_performance_summary(results: Dict[str, Any]) -> None:
    """
    Print a formatted summary of performance results.
    
    Args:
        results: Performance results dictionary
    """
    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)
    
    summary = results["summary"]
    
    print("\nBest Batch Sizes by Document Count:")
    print("-" * 40)
    for size, info in summary["best_batch_sizes"].items():
        print(f"  {size:3d} docs: batch_size={info['batch_size']:2d}, speedup={info['speedup']:.2f}x")
    
    print("\nAverage Speedups by Document Count:")
    print("-" * 40)
    for size, speedup in summary["average_speedups"].items():
        print(f"  {size:3d} docs: {speedup:.2f}x speedup")
    
    # Calculate overall average speedup
    if summary["average_speedups"]:
        overall_avg = sum(summary["average_speedups"].values()) / len(summary["average_speedups"])
        print(f"\nOverall Average Speedup: {overall_avg:.2f}x")
    


def save_results_to_file(results: Dict[str, Any], filename: str = "sem_map_batch_performance_results.json") -> None:
    """
    Save performance results to a JSON file.
    
    Args:
        results: Performance results dictionary
        filename: Output filename
    """
    def make_serializable(obj):
        """Recursively make objects JSON serializable."""
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(item) for item in obj]
        elif hasattr(obj, 'shape'):  # DataFrame or similar
            return {
                "type": "DataFrame",
                "shape": obj.shape,
                "columns": obj.columns.tolist() if hasattr(obj, 'columns') else None
            }
        elif hasattr(obj, '__dict__'):  # Custom objects
            return str(obj)
        else:
            return obj
    
    # Convert results to JSON-serializable format
    json_results = make_serializable(results)
    
    with open(filename, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\nResults saved to {filename}")


def main():
    """Main function to run performance tests."""
    # Configure the language model
    # Note: You'll need to set your API key or use a local model
    try:
        model = LM(
                model="openrouter/google/gemini-2.5-flash",
                max_batch_size=4,
                temperature=0.0,
                max_tokens=256,
                api_key="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",
                base_url="https://openrouter.ai/api/v1"
        ) # Use the specified model
        lotus.settings.configure(lm=model)
        print("Language model configured successfully")
    except Exception as e:
        print(f"Error configuring language model: {e}")
        print("Please ensure you have a valid API key or local model configured")
        return
    
    # Run performance tests
    test_sizes = [40]  # Start with smaller sizes for testing
    batch_sizes = [6]
    instruction = "Summarize the {text} in one sentence:"
    
    try:
        results = run_performance_comparison(
            test_sizes=test_sizes,
            batch_sizes=batch_sizes,
            instruction=instruction
        )
        
        # Print summary
        print_performance_summary(results)
        
        # Save results
        save_results_to_file(results)
        
        print("\nPerformance testing completed successfully!")
        
    except Exception as e:
        print(f"Error during performance testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
