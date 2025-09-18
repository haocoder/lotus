#!/usr/bin/env python3
"""
Test script for optimized batch processing prompt.
"""

import pandas as pd
import lotus
from lotus.models import LM

def test_optimized_batch():
    """Test the optimized batch processing with a small dataset."""
    
    # Configure the language model
    try:
        model = LM(
            model="openrouter/google/gemini-2.5-flash",
            max_batch_size=4,
            temperature=0.0,
            max_tokens=512,  # Increased token limit for better responses
            api_key="'''",
            base_url="https://openrouter.ai/api/v1"
        )
        lotus.settings.configure(lm=model)
        print("Language model configured successfully")
    except Exception as e:
        print(f"Error configuring language model: {e}")
        return
    
    # Create test data
    test_data = [
        {"text": "The weather is beautiful today with clear blue skies."},
        {"text": "I love reading books in my spare time."},
        {"text": "Technology has revolutionized the way we communicate."},
        {"text": "The restaurant served delicious food with excellent service."},
        {"text": "Learning new programming languages is challenging but rewarding."},
        {"text": "The movie had an amazing plot with great character development."},
        {"text": "Exercise is important for maintaining good health."},
        {"text": "The conference provided valuable insights into industry trends."},
        {"text": "Cooking is both an art and a science."},
        {"text": "Traveling broadens your perspective on different cultures."}
    ]
    
    df = pd.DataFrame(test_data)
    
    # Configure batch processing
    lotus.settings.use_batch_processing = True
    lotus.settings.batch_size = 4
    
    print(f"Testing with {len(df)} documents, batch size = 4")
    print("=" * 60)
    
    try:
        # Test batch processing
        result_df = df.sem_map("Summarize the {text} in one sentence:")
        
        print("\nResults:")
        print(result_df)
        
        # Check if we got all results
        expected_count = len(df)
        actual_count = len(result_df)
        
        print(f"\nExpected results: {expected_count}")
        print(f"Actual results: {actual_count}")
        
        if actual_count == expected_count:
            print("✅ SUCCESS: All documents processed correctly!")
        else:
            print("❌ FAILURE: Some documents were not processed")
            
        # Check for any empty results
        empty_results = result_df['_map'].isna() | (result_df['_map'] == '')
        if empty_results.any():
            print(f"⚠️  WARNING: {empty_results.sum()} empty results found")
        else:
            print("✅ All results contain content")
            
    except Exception as e:
        print(f"Error during batch processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_optimized_batch()
