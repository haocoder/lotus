#!/usr/bin/env python3
"""
Quick test script for sem_filter sync vs async performance.

This script runs a simple performance comparison between sync and async
sem_filter implementations with a small dataset.
"""

import asyncio
import os
import sys
import time
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

import lotus
from lotus.models import LM
from lotus.sem_ops.sem_filter import sem_filter, sem_filter_async


def create_test_data(size: int = 20) -> list[dict[str, any]]:
    """Create test data for filtering."""
    topics = [
        'artificial intelligence', 'machine learning', 'data science', 'cloud computing',
        'cybersecurity', 'blockchain', 'quantum computing', 'robotics', 'biotechnology',
        'renewable energy', 'space exploration', 'climate change', 'digital transformation'
    ]
    
    data = []
    for i in range(size):
        topic = topics[i % len(topics)]
        content = (
            f"This is document {i+1} about {topic}. "
            f"It discusses important concepts, methodologies, and applications "
            f"related to {topic}. The content covers various aspects including "
            f"technical details, practical implementations, and future prospects."
        )
        
        data.append({
            'text': content,
            'topic': topic,
            'document_id': f"doc_{i+1:03d}",
            'category': 'Technology' if i % 2 == 0 else 'Science'
        })
    
    return data


def create_lm() -> LM:
    """Create LM instance for testing."""
    api_key = (
        os.getenv("OPENROUTER_API_KEY") or 
        os.getenv("LOTUS_TEST_API_KEY") or
        os.getenv("OPENAI_API_KEY") or
        os.getenv("ANTHROPIC_API_KEY")
    )
    
    if not api_key:
        print("Error: No API key found. Please set one of the following environment variables:")
        print("- OPENROUTER_API_KEY")
        print("- LOTUS_TEST_API_KEY") 
        print("- OPENAI_API_KEY")
        print("- ANTHROPIC_API_KEY")
        sys.exit(1)
    
    return LM(
        model="openrouter/google/gemini-2.5-flash",
        max_batch_size=4,
        temperature=0.0,
        max_tokens=256,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )


def test_sync_performance(docs: list[dict[str, any]], lm: LM) -> dict[str, any]:
    """Test sync performance."""
    print("Testing sync performance...")
    
    start_time = time.time()
    result = sem_filter(
        docs,
        lm,
        "Is this document about technology topics?",
        default=True,
        use_async=False
    )
    end_time = time.time()
    
    duration = end_time - start_time
    success = hasattr(result, 'outputs') and len(result.outputs) > 0
    
    return {
        'mode': 'sync',
        'duration': duration,
        'success': success,
        'result_count': len(result.outputs) if success else 0
    }


async def test_async_performance(docs: list[dict[str, any]], lm: LM) -> dict[str, any]:
    """Test async performance."""
    print("Testing async performance...")
    
    start_time = time.time()
    result = await sem_filter_async(
        docs,
        lm,
        "Is this document about technology topics?",
        default=True,
        max_concurrent_batches=4,
        max_thread_workers=8
    )
    end_time = time.time()
    
    duration = end_time - start_time
    success = hasattr(result, 'outputs') and len(result.outputs) > 0
    
    return {
        'mode': 'async',
        'duration': duration,
        'success': success,
        'result_count': len(result.outputs) if success else 0
    }


def test_sync_with_async_param(docs: list[dict[str, any]], lm: LM) -> dict[str, any]:
    """Test sync function with async=True parameter."""
    print("Testing sync function with async=True...")
    
    start_time = time.time()
    result = sem_filter(
        docs,
        lm,
        "Is this document about technology topics?",
        default=True,
        use_async=True,
        max_concurrent_batches=4,
        max_thread_workers=8
    )
    end_time = time.time()
    
    duration = end_time - start_time
    success = hasattr(result, 'outputs') and len(result.outputs) > 0
    
    return {
        'mode': 'sync_with_async_param',
        'duration': duration,
        'success': success,
        'result_count': len(result.outputs) if success else 0
    }


def main():
    """Main test function."""
    print("=" * 60)
    print("Quick sem_filter Performance Test")
    print("=" * 60)
    
    # Create test data
    print("Creating test data...")
    docs = create_test_data(20)
    print(f"Created {len(docs)} test documents")
    
    # Create LM
    print("Creating LM instance...")
    lm = create_lm()
    lotus.settings.configure(lm=lm)
    print("LM configured successfully")
    
    # Run tests
    results = []
    
    try:
        # Test 1: Sync performance
        result1 = test_sync_performance(docs, lm)
        results.append(result1)
        print(f"✓ Sync test completed: {result1['duration']:.2f}s")
        
        # Test 2: Async performance
        result2 = asyncio.run(test_async_performance(docs, lm))
        results.append(result2)
        print(f"✓ Async test completed: {result2['duration']:.2f}s")
        
        # Test 3: Sync with async parameter
        result3 = test_sync_with_async_param(docs, lm)
        results.append(result3)
        print(f"✓ Sync with async param test completed: {result3['duration']:.2f}s")
        
        # Calculate performance improvement
        sync_duration = result1['duration']
        async_duration = result2['duration']
        improvement = ((sync_duration - async_duration) / sync_duration) * 100 if sync_duration > 0 else 0
        
        print("\n" + "=" * 60)
        print("PERFORMANCE RESULTS")
        print("=" * 60)
        print(f"Sync duration:           {sync_duration:.2f}s")
        print(f"Async duration:          {async_duration:.2f}s")
        print(f"Performance improvement: {improvement:.1f}%")
        print(f"Speedup factor:          {sync_duration/async_duration:.2f}x")
        
        # Verify all tests succeeded
        all_success = all(r['success'] for r in results)
        if all_success:
            print("\n🎉 All tests passed successfully!")
        else:
            print("\n❌ Some tests failed!")
            for r in results:
                if not r['success']:
                    print(f"  - {r['mode']} test failed")
        
        print("\nNext steps:")
        print("1. Run full benchmark: python test_sem_filter_performance.py")
        print("2. Test with larger datasets")
        print("3. Test with different concurrency settings")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
