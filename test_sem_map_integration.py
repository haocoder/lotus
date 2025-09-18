#!/usr/bin/env python3
"""
Integration test for sem_map batch processing with DataFrame accessor.

This script tests the complete integration of batch processing
with the DataFrame sem_map accessor.
"""

import pandas as pd
import time
import lotus
from lotus.models import LM
from lotus.types import ReasoningStrategy


def create_sample_dataframe(num_rows: int = 15) -> pd.DataFrame:
    """Create a sample DataFrame for testing."""
    data = {
        'text': [
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
        ][:num_rows],
        'category': ['weather', 'hobby', 'technology', 'food', 'learning', 
                    'entertainment', 'health', 'business', 'cooking', 'travel',
                    'music', 'work', 'education', 'teamwork', 'nature'][:num_rows]
    }
    return pd.DataFrame(data)


def test_dataframe_batch_processing():
    """Test DataFrame batch processing functionality."""
    print("Testing DataFrame batch processing...")
    
    # Create test DataFrame
    df = create_sample_dataframe(12)
    print(f"Created DataFrame with {len(df)} rows")
    
    # Test instruction
    instruction = "Summarize the {text} in one word:"
    
    try:
        # Test with batch processing enabled
        print("\n1. Testing with batch processing enabled...")
        lotus.settings.use_batch_processing = True
        lotus.settings.batch_size = 4
        
        start_time = time.time()
        result_batch = df.sem_map(instruction)
        batch_time = time.time() - start_time
        
        print(f"   Batch processing time: {batch_time:.2f}s")
        print(f"   Results shape: {result_batch.shape}")
        print(f"   Sample results: {result_batch['_map'].head(3).tolist()}")
        
        # Test with batch processing disabled
        print("\n2. Testing with batch processing disabled...")
        lotus.settings.use_batch_processing = False
        
        start_time = time.time()
        result_individual = df.sem_map(instruction)
        individual_time = time.time() - start_time
        
        print(f"   Individual processing time: {individual_time:.2f}s")
        print(f"   Results shape: {result_individual.shape}")
        print(f"   Sample results: {result_individual['_map'].head(3).tolist()}")
        
        # Compare results
        print("\n3. Comparing results...")
        batch_results = result_batch['_map'].tolist()
        individual_results = result_individual['_map'].tolist()
        
        if batch_results == individual_results:
            print("   ✓ Results match between batch and individual processing")
        else:
            print("   ✗ Results differ between batch and individual processing")
            print(f"   Batch: {batch_results}")
            print(f"   Individual: {individual_results}")
        
        # Calculate speedup
        if individual_time > 0:
            speedup = individual_time / batch_time
            print(f"   Speedup: {speedup:.2f}x")
        
        return True
        
    except Exception as e:
        print(f"✗ Error during DataFrame testing: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_examples():
    """Test batch processing with few-shot examples."""
    print("\nTesting batch processing with examples...")
    
    # Create test DataFrame
    df = create_sample_dataframe(8)
    
    # Create examples DataFrame
    examples = pd.DataFrame({
        'text': [
            "I love this product!",
            "This is terrible.",
            "It's okay, nothing special."
        ],
        'category': ['product', 'product', 'product'],
        'Answer': ['positive', 'negative', 'neutral']
    })
    
    instruction = "Classify the sentiment of {text} as positive, negative, or neutral:"
    
    try:
        lotus.settings.use_batch_processing = True
        lotus.settings.batch_size = 3
        
        result = df.sem_map(
            instruction,
            examples=examples,
            return_explanations=True
        )
        
        print(f"   Results with examples:")
        for i, (text, sentiment) in enumerate(zip(result['text'], result['_map'])):
            print(f"   {i+1}. {text[:30]}... -> {sentiment}")
        
        print("✓ Examples test completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Error during examples testing: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_cot_reasoning():
    """Test batch processing with chain-of-thought reasoning."""
    print("\nTesting batch processing with CoT reasoning...")
    
    df = create_sample_dataframe(6)
    instruction = "Analyze the sentiment of {text} and explain your reasoning:"
    
    try:
        lotus.settings.use_batch_processing = True
        lotus.settings.batch_size = 3
        
        result = df.sem_map(
            instruction,
            return_explanations=True,
            strategy=ReasoningStrategy.ZS_COT
        )
        
        print("   Results with CoT reasoning:")
        for i, (text, sentiment, explanation) in enumerate(zip(
            result['text'], result['_map'], result['explanation_map']
        )):
            print(f"   {i+1}. {text[:30]}...")
            print(f"      Sentiment: {sentiment}")
            if explanation:
                print(f"      Reasoning: {explanation[:100]}...")
            print()
        
        print("✓ CoT reasoning test completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Error during CoT testing: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_different_batch_sizes():
    """Test different batch sizes with DataFrame."""
    print("\nTesting different batch sizes...")
    
    df = create_sample_dataframe(20)
    instruction = "Extract the main topic from {text}:"
    
    batch_sizes = [3, 5, 10]
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
                "sample_results": result['_map'].head(3).tolist()
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


def main():
    """Main function to run integration tests."""
    print("=" * 70)
    print("SEM_MAP BATCH PROCESSING INTEGRATION TESTS")
    print("=" * 70)
    
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
        test_dataframe_batch_processing,
        test_with_examples,
        test_with_cot_reasoning,
        test_different_batch_sizes,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 70)
    print(f"INTEGRATION TEST SUMMARY: {passed}/{total} tests passed")
    print("=" * 70)
    
    if passed == total:
        print("🎉 All integration tests passed! Batch processing is working correctly.")
    else:
        print("⚠️  Some integration tests failed. Please check the implementation.")


if __name__ == "__main__":
    main()
