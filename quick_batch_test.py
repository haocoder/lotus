#!/usr/bin/env python3
"""
Quick test for batch processing with debug output.
"""

import pandas as pd
import lotus
from lotus.models import LM

def quick_test():
    """Quick test with 4 documents to see debug output."""
    
    # Configure the language model
    try:
        model = LM(
            model="openrouter/google/gemini-2.5-flash",
            max_batch_size=4,
            temperature=0.0,
            max_tokens=512,
            api_key="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",
            base_url="https://openrouter.ai/api/v1"
        )
        lotus.settings.configure(lm=model)
        print("Language model configured successfully")
    except Exception as e:
        print(f"Error configuring language model: {e}")
        return
    
    # Create test data with exactly 4 documents
    test_data = [
        {"text": "The weather is beautiful today with clear blue skies."},
        {"text": "I love reading books in my spare time."},
        {"text": "Technology has revolutionized the way we communicate."},
        {"text": "The restaurant served delicious food with excellent service."}
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
            
    except Exception as e:
        print(f"Error during batch processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    quick_test()
