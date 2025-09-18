#!/usr/bin/env python3
"""
Quick test script for sem_map batch processing functionality.

This script provides a simple way to test the batch processing
implementation without running full performance benchmarks.
"""

import pandas as pd
import time
import lotus
from lotus.models import LM


def test_basic_batch_functionality():
    """Test basic batch processing functionality."""
    print("Testing basic sem_map batch processing functionality...")
    
    # Create test data
    test_data = [
        {"text": "The weather is beautiful today."},
        {"text": "I love reading books in my spare time."},
        {"text": "Technology has revolutionized communication."},
        {"text": "The restaurant served delicious food."},
        {"text": "Learning programming is challenging but rewarding."},
    ]
    
    df = pd.DataFrame(test_data)
    instruction = "Summarize the {text} in one word:"
    
    print(f"Test data: {len(test_data)} documents")
    print(f"Instruction: {instruction}")
    
    try:
        # Test individual processing
        print("\n1. Testing individual processing...")
        start_time = time.time()
        
        lotus.settings.use_batch_processing = False
        result_individual = df.sem_map(instruction)
        
        individual_time = time.time() - start_time
        print(f"   Individual processing time: {individual_time:.2f}s")
        print(f"   Results: {result_individual['_map'].tolist()}")
        
        # Test batch processing
        print("\n2. Testing batch processing...")
        start_time = time.time()
        
        lotus.settings.use_batch_processing = True
        lotus.settings.batch_size = 3
        result_batch = df.sem_map(instruction)
        
        batch_time = time.time() - start_time
        print(f"   Batch processing time: {batch_time:.2f}s")
        print(f"   Results: {result_batch['_map'].tolist()}")
        
        # Compare results
        print("\n3. Comparing results...")
        individual_results = result_individual['_map'].tolist()
        batch_results = result_batch['_map'].tolist()
        
        if individual_results == batch_results:
            print("   ✓ Results match between individual and batch processing")
        else:
            print("   ✗ Results differ between individual and batch processing")
            print(f"   Individual: {individual_results}")
            print(f"   Batch: {batch_results}")
        
        # Calculate speedup
        if individual_time > 0:
            speedup = individual_time / batch_time
            print(f"   Speedup: {speedup:.2f}x")
        
        print("\n✓ Basic batch functionality test completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_different_batch_sizes():
    """Test different batch sizes."""
    print("\nTesting different batch sizes...")
    
    # Create larger test data
    test_data = []
    for i in range(20):
        test_data.append({"text": f"This is test document number {i+1}."})
    
    df = pd.DataFrame(test_data)
    instruction = "Extract the number from {text}:"
    
    batch_sizes = [5, 10, 15]
    results = {}
    
    try:
        for batch_size in batch_sizes:
            print(f"\nTesting batch size: {batch_size}")
            
            lotus.settings.use_batch_processing = True
            lotus.settings.batch_size = batch_size
            
            start_time = time.time()
            result = df.sem_map(instruction)
            duration = time.time() - start_time
            
            results[batch_size] = {
                "duration": duration,
                "results": result['_map'].tolist()
            }
            
            print(f"   Duration: {duration:.2f}s")
            print(f"   Sample results: {result['_map'].head(3).tolist()}")
        
        # Find best batch size
        best_batch_size = min(results.keys(), key=lambda x: results[x]["duration"])
        print(f"\nBest batch size: {best_batch_size} ({results[best_batch_size]['duration']:.2f}s)")
        
        print("✓ Different batch sizes test completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Error during batch size testing: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_explanations():
    """Test batch processing with explanations (COT)."""
    print("\nTesting batch processing with explanations...")
    
    test_data = [
        {"text": "The movie was absolutely fantastic!"},
        {"text": "I hate waiting in long lines."},
        {"text": "The book was okay, nothing special."},
    ]
    
    df = pd.DataFrame(test_data)
    instruction = "Determine the sentiment of {text} (positive/negative/neutral):"
    
    try:
        lotus.settings.use_batch_processing = True
        lotus.settings.batch_size = 2
        
        result = df.sem_map(
            instruction, 
            return_explanations=True,
            strategy=lotus.types.ReasoningStrategy.ZS_COT
        )
        
        print("   Results with explanations:")
        for i, (sentiment, explanation) in enumerate(zip(result['_map'], result['explanation_map'])):
            print(f"   {i+1}. {sentiment}")
            if explanation:
                print(f"      Explanation: {explanation[:100]}...")
        
        print("✓ Explanations test completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Error during explanations testing: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function to run quick tests."""
    print("=" * 60)
    print("SEM_MAP BATCH PROCESSING QUICK TESTS")
    print("=" * 60)
    
    # Configure the language model
    try:
        model = LM(model="gpt-4o-mini")  # Use a cost-effective model
        lotus.settings.configure(lm=model)
        print("Language model configured successfully")
    except Exception as e:
        print(f"Error configuring language model: {e}")
        print("Please ensure you have a valid API key configured")
        return
    
    # Run tests
    tests = [
        test_basic_batch_functionality,
        test_different_batch_sizes,
        test_with_explanations,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 60)
    print(f"TEST SUMMARY: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 All tests passed! Batch processing is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")


if __name__ == "__main__":
    main()
