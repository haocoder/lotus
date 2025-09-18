#!/usr/bin/env python3
"""
Simple integration test for batch filter processing.

This script demonstrates the batch processing functionality
and can be used to verify that the implementation works correctly.
"""

import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

import lotus
from lotus.models import LM
from lotus.sem_ops.sem_filter import sem_filter
from lotus.types import ReasoningStrategy


def test_batch_filter_basic():
    """Test basic batch filter functionality."""
    print("Testing basic batch filter functionality...")
    
    # Create mock model (in real usage, this would be a real model)
    model = LM(
        model="openrouter/google/gemini-2.5-flash",
        max_batch_size=4,
        temperature=0.0,
        max_tokens=256,
        api_key="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",
        base_url="https://openrouter.ai/api/v1"
    )
    
    # Test documents
    docs = [
        {"text": "This is a positive review about the product"},
        {"text": "This is a negative review about the service"},
        {"text": "This is a neutral comment about the experience"},
        {"text": "I love this amazing product!"},
        {"text": "Terrible quality, would not recommend"},
    ]
    
    user_instruction = "Is this a positive sentiment?"
    
    try:
        # Test with batch processing enabled
        print("Testing with batch processing...")
        result_batch = sem_filter(
            docs=docs,
            model=model,
            user_instruction=user_instruction,
            use_batch_processing=True,
            batch_size=3,
            safe_mode=True
        )
        
        print(f"Batch processing results: {result_batch.outputs}")
        print(f"Number of results: {len(result_batch.outputs)}")
        print(f"Raw outputs: {result_batch.raw_outputs}")
        print(f"Explanations: {result_batch.explanations}")
        
        # Test with batch processing disabled
        print("\nTesting with individual processing...")
        result_individual = sem_filter(
            docs=docs,
            model=model,
            user_instruction=user_instruction,
            use_batch_processing=False,
            safe_mode=True
        )
        
        print(f"Individual processing results: {result_individual.outputs}")
        print(f"Number of results: {len(result_individual.outputs)}")
        
        # Compare results
        print(f"\nResults match: {result_batch.outputs == result_individual.outputs}")
        
        return True
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        return False


def test_batch_filter_with_examples():
    """Test batch filter with few-shot examples."""
    print("\nTesting batch filter with examples...")
    
    model = LM(
        model="openrouter/google/gemini-2.5-flash",
        max_batch_size=4,
        temperature=0.0,
        max_tokens=256,
        api_key="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",
        base_url="https://openrouter.ai/api/v1"
    )
    
    docs = [
        {"text": "This product is excellent"},
        {"text": "I hate this service"},
    ]
    
    user_instruction = "Is this a positive sentiment?"
    
    # Examples for few-shot learning
    examples_multimodal_data = [
        {"text": "Great product!"},
        {"text": "Terrible quality"},
    ]
    examples_answers = [True, False]
    
    try:
        result = sem_filter(
            docs=docs,
            model=model,
            user_instruction=user_instruction,
            examples_multimodal_data=examples_multimodal_data,
            examples_answers=examples_answers,
            use_batch_processing=True,
            batch_size=2,
            safe_mode=True
        )
        
        print(f"Results with examples: {result.outputs}")
        print(f"Explanations: {result.explanations}")
        
        return True
        
    except Exception as e:
        print(f"Test with examples failed: {e}")
        return False


def test_batch_filter_with_cot():
    """Test batch filter with chain-of-thought reasoning."""
    print("\nTesting batch filter with CoT reasoning...")
    
    model = LM(
        model="openrouter/google/gemini-2.5-flash",
        max_batch_size=4,
        temperature=0.0,
        max_tokens=256,
        api_key="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",
        base_url="https://openrouter.ai/api/v1"
    )
    
    docs = [
        {"text": "This is a wonderful experience"},
        {"text": "I am disappointed with the quality"},
    ]
    
    user_instruction = "Is this a positive sentiment?"
    
    try:
        result = sem_filter(
            docs=docs,
            model=model,
            user_instruction=user_instruction,
            strategy=ReasoningStrategy.ZS_COT,
            use_batch_processing=True,
            batch_size=2,
            safe_mode=True
        )
        
        print(f"Results with CoT: {result.outputs}")
        print(f"Explanations: {result.explanations}")
        
        return True
        
    except Exception as e:
        print(f"Test with CoT failed: {e}")
        return False


def main():
    """Run all integration tests."""
    print("Starting batch filter integration tests...\n")
    model = LM(
        model="openrouter/google/gemini-2.5-flash",
        max_batch_size=4,
        temperature=0.0,
        max_tokens=256,
        api_key="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",
        base_url="https://openrouter.ai/api/v1"
    )
    # Configure lotus settings
    lotus.settings.configure(
        lm=model,
        enable_cache=False  # Disable cache for testing
    )
    
    tests = [
        test_batch_filter_basic,
        test_batch_filter_with_examples,
        test_batch_filter_with_cot,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
                print("✓ Test passed")
            else:
                print("✗ Test failed")
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
        print("-" * 50)
    
    print(f"\nTest Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    exit(main())
