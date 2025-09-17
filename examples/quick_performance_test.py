#!/usr/bin/env python3
"""
Quick performance test script for sem_agg async optimization.

This script provides a simple way to test and compare sync vs async performance
with different dataset sizes. It's designed for quick testing and validation.

Usage:
    python examples/quick_performance_test.py [--size SIZE] [--iterations ITERATIONS]
"""

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

import lotus
from lotus.models import LM


def create_test_dataset(size: int) -> pd.DataFrame:
    """
    Create a test dataset of specified size.
    
    Args:
        size: Number of documents to generate.
        
    Returns:
        pd.DataFrame: Test dataset with realistic content.
    """
    categories = ['Technology', 'Science', 'Business', 'Health', 'Education']
    topics = [
        'artificial intelligence', 'machine learning', 'data science', 
        'cloud computing', 'cybersecurity', 'blockchain', 'quantum computing'
    ]
    
    data = []
    for i in range(size):
        topic = topics[i % len(topics)]
        category = categories[i % len(categories)]
        
        content = (
            f"This is document {i+1} about {topic} in the {category.lower()} domain. "
            f"It discusses important concepts, methodologies, and applications related to {topic}. "
            f"The content covers various aspects including technical details, practical implementations, "
            f"and future prospects. This document provides comprehensive information for understanding "
            f"the current state and potential developments in {topic}."
        )
        
        data.append({
            'content': content,
            'category': category,
            'topic': topic,
            'document_id': f"doc_{i+1:03d}"
        })
    
    return pd.DataFrame(data)


def run_performance_test(dataset: pd.DataFrame, use_async: bool, test_name: str) -> dict:
    """
    Run a single performance test.
    
    Args:
        dataset: Dataset to test with.
        use_async: Whether to use async processing.
        test_name: Name for the test.
        
    Returns:
        dict: Test results with timing and success status.
    """
    start_time = time.time()
    
    try:
        result = dataset.sem_agg(
            user_instruction="Summarize the key themes and insights across all documents",
            all_cols=True,
            group_by=['category'],
            use_async=use_async,
            max_concurrent_batches=4,
            max_thread_workers=8,
        )
        
        duration = time.time() - start_time
        
        # Verify results
        success = (
            isinstance(result, pd.DataFrame) and
            '_output' in result.columns and
            len(result) > 0 and
            all(isinstance(output, str) and len(output) > 0 for output in result['_output'])
        )
        
        return {
            'test_name': test_name,
            'dataset_size': len(dataset),
            'use_async': use_async,
            'duration': duration,
            'success': success,
            'result_count': len(result)
        }
        
    except Exception as e:
        duration = time.time() - start_time
        return {
            'test_name': test_name,
            'dataset_size': len(dataset),
            'use_async': use_async,
            'duration': duration,
            'success': False,
            'error': str(e),
            'result_count': 0
        }


def main() -> None:
    """Main function to run quick performance tests."""
    parser = argparse.ArgumentParser(description="Quick sem_agg performance test")
    parser.add_argument(
        "--size",
        type=int,
        default=50,
        help="Dataset size to test (default: 50)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of iterations to run (default: 3)"
    )
    
    args = parser.parse_args()
    
    # Check for API key
    if not any([
        os.getenv("OPENROUTER_API_KEY"),
        os.getenv("LOTUS_TEST_API_KEY"),
        os.getenv("OPENAI_API_KEY"),
        os.getenv("ANTHROPIC_API_KEY")
    ]):
        print("Error: No API key found. Please set one of the following environment variables:")
        print("- OPENROUTER_API_KEY")
        print("- LOTUS_TEST_API_KEY") 
        print("- OPENAI_API_KEY")
        print("- ANTHROPIC_API_KEY")
        sys.exit(1)
    
    # Initialize LM
    api_key = (
        os.getenv("OPENROUTER_API_KEY") or 
        os.getenv("LOTUS_TEST_API_KEY") or
    )
    
    lm = LM(
        model="openrouter/google/gemini-2.5-flash",
        max_batch_size=4,
        temperature=0.0,
        max_tokens=256,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )
    
    lotus.settings.configure(lm=lm)
    
    # Generate test dataset
    print(f"Generating test dataset with {args.size} documents...")
    dataset = create_test_dataset(args.size)
    
    print(f"Running performance tests with {args.iterations} iterations each...")
    print("=" * 60)
    
    sync_results = []
    async_results = []
    
    # Run sync tests
    print("Testing sync mode...")
    for i in range(args.iterations):
        print(f"  Iteration {i+1}/{args.iterations}...", end=" ")
        result = run_performance_test(dataset, use_async=False, test_name=f"sync_iter_{i+1}")
        sync_results.append(result)
        print(f"{result['duration']:.2f}s {'✓' if result['success'] else '✗'}")
    
    # Run async tests
    print("\nTesting async mode...")
    for i in range(args.iterations):
        print(f"  Iteration {i+1}/{args.iterations}...", end=" ")
        result = run_performance_test(dataset, use_async=True, test_name=f"async_iter_{i+1}")
        async_results.append(result)
        print(f"{result['duration']:.2f}s {'✓' if result['success'] else '✗'}")
    
    # Calculate and display results
    print("\n" + "=" * 60)
    print("PERFORMANCE RESULTS")
    print("=" * 60)
    
    sync_durations = [r['duration'] for r in sync_results if r['success']]
    async_durations = [r['duration'] for r in async_results if r['success']]
    
    if sync_durations and async_durations:
        sync_avg = sum(sync_durations) / len(sync_durations)
        async_avg = sum(async_durations) / len(async_durations)
        
        print(f"Dataset size: {args.size} documents")
        print(f"Iterations: {args.iterations}")
        print()
        print(f"Sync mode:")
        print(f"  Average time: {sync_avg:.2f}s")
        print(f"  Min time: {min(sync_durations):.2f}s")
        print(f"  Max time: {max(sync_durations):.2f}s")
        print()
        print(f"Async mode:")
        print(f"  Average time: {async_avg:.2f}s")
        print(f"  Min time: {min(async_durations):.2f}s")
        print(f"  Max time: {max(async_durations):.2f}s")
        print()
        
        if sync_avg > 0:
            improvement = ((sync_avg - async_avg) / sync_avg) * 100
            print(f"Performance improvement: {improvement:.1f}%")
            
            if improvement > 0:
                print("✓ Async mode is faster!")
            else:
                print("✗ Sync mode is faster")
        
        print()
        print(f"Success rate:")
        print(f"  Sync: {len(sync_durations)}/{len(sync_results)} ({len(sync_durations)/len(sync_results)*100:.1f}%)")
        print(f"  Async: {len(async_durations)}/{len(async_results)} ({len(async_durations)/len(async_results)*100:.1f}%)")
    
    else:
        print("No successful tests to compare.")
        print(f"Sync successes: {len(sync_durations)}/{len(sync_results)}")
        print(f"Async successes: {len(async_durations)}/{len(async_results)}")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
