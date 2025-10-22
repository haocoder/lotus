#!/usr/bin/env python3
"""
Small-scale batch processing test with real model.

This script performs a quick test with a very small dataset to verify
that batch processing works correctly with real models.
"""

import sys
import time
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

import lotus
from lotus.models import LM
from lotus.sem_ops.sem_filter import sem_filter


def small_test():
    """Run a small test with 3 documents."""
    print("🚀 Small Batch Processing Test")
    print("="*50)
    
    # Configure model
    model = LM(
        model="openrouter/google/gemini-2.5-flash",
        max_batch_size=4,
        temperature=0.0,
        max_tokens=256,
        api_key="''-v1-acdced37337241eb1d9ceb106f1ab6c604f189bd95b7933dc50ae72b81e16807",
        base_url="https://openrouter.ai/api/v1"
    )
    lotus.settings.configure(
        lm=model,
        enable_cache=False
    )
    
    # Very small test dataset
    docs = [
        {"text": "This product is absolutely amazing!"},
        {"text": "Terrible quality, very disappointed."},
        {"text": "The product works as described."},
    ]
    
    user_instruction = "Is this a positive sentiment? Answer True for positive, False for negative or neutral."
    
    print(f"Model: {model.model}")
    print(f"Number of documents: {len(docs)}")
    print(f"User instruction: {user_instruction}")
    
    try:
        # Test individual processing
        print("\n📝 Testing individual processing...")
        model.reset_stats()
        start_time = time.time()
        
        individual_result = sem_filter(
            docs=docs,
            model=model,
            user_instruction=user_instruction,
            use_batch_processing=False,
            show_progress_bar=True,
            progress_bar_desc="Individual"
        )
        
        individual_time = time.time() - start_time
        individual_cost = model.stats.virtual_usage.total_cost
        individual_tokens = model.stats.virtual_usage.total_tokens
        
        print(f"Individual processing completed in {individual_time:.2f}s")
        print(f"Individual results: {individual_result.outputs}")
        print(f"Individual cost: ${individual_cost:.6f}")
        print(f"Individual tokens: {individual_tokens:,}")
        
        # Test batch processing
        print("\n📦 Testing batch processing...")
        model.reset_stats()
        start_time = time.time()
        
        batch_result = sem_filter(
            docs=docs,
            model=model,
            user_instruction=user_instruction,
            use_batch_processing=True,
            batch_size=3,
            show_progress_bar=True,
            progress_bar_desc="Batch"
        )
        
        batch_time = time.time() - start_time
        batch_cost = model.stats.virtual_usage.total_cost
        batch_tokens = model.stats.virtual_usage.total_tokens
        
        print(f"Batch processing completed in {batch_time:.2f}s")
        print(f"Batch results: {batch_result.outputs}")
        print(f"Batch cost: ${batch_cost:.6f}")
        print(f"Batch tokens: {batch_tokens:,}")
        
        # Compare results
        print("\n📊 Comparison:")
        print(f"Results match: {individual_result.outputs == batch_result.outputs}")
        if individual_time > 0:
            print(f"Time improvement: {((individual_time - batch_time) / individual_time * 100):.1f}%")
        if individual_cost > 0:
            print(f"Cost improvement: {((individual_cost - batch_cost) / individual_cost * 100):.1f}%")
        if individual_tokens > 0:
            print(f"Token improvement: {((individual_tokens - batch_tokens) / individual_tokens * 100):.1f}%")
        
        # Test with explanations
        print("\n🧠 Testing batch processing with explanations...")
        model.reset_stats()
        
        batch_with_explanations = sem_filter(
            docs=docs,
            model=model,
            user_instruction=user_instruction,
            use_batch_processing=True,
            batch_size=3,
            show_progress_bar=True,
            progress_bar_desc="Batch with Explanations"
        )
        
        print(f"Batch with explanations results: {batch_with_explanations.outputs}")
        print(f"Explanations: {batch_with_explanations.explanations}")
        
        print("\n✅ Small test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run the small test."""
    success = small_test()
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
