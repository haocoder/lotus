#!/usr/bin/env python3
"""
Test batch processing with non-divisible document counts.
"""

import pandas as pd
import lotus
from lotus.models import LM

def test_batch_count_fix():
    """Test batch processing with 10 documents and batch size 4."""
    print("Testing batch processing with 10 documents and batch size 4...")
    
    # Create test data
    test_data = []
    for i in range(10):
        test_data.append({"text": f"This is test document number {i+1}."})
    
    df = pd.DataFrame(test_data)
    instruction = "Summarize the {text} in one word:"
    
    print(f"Created DataFrame with {len(df)} rows")
    print(f"Expected batches: 3 (4 + 4 + 2)")
    
    try:
        # Configure model
        model = LM(
            model="openrouter/google/gemini-2.5-flash",
            max_batch_size=4,
            temperature=0.0,
            max_tokens=256,
            api_key="'''",
            base_url="https://openrouter.ai/api/v1"
        )
        lotus.settings.configure(lm=model)
        
        # Test batch processing
        lotus.settings.use_batch_processing = True
        lotus.settings.batch_size = 4
        
        print("\nRunning batch processing...")
        result = df.sem_map(instruction)
        
        print(f"Result DataFrame shape: {result.shape}")
        print(f"Expected shape: ({len(df)}, {len(df.columns) + 1})")
        
        if len(result) == len(df):
            print("✓ All documents processed successfully!")
        else:
            print(f"✗ Missing documents: expected {len(df)}, got {len(result)}")
        
        print("\nSample results:")
        for i, (text, summary) in enumerate(zip(result['text'], result['_map'])):
            print(f"{i+1:2d}. {text[:30]}... -> {summary}")
        
        return len(result) == len(df)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_batch_count_fix()
    if success:
        print("\n🎉 Test passed!")
    else:
        print("\n❌ Test failed!")
