"""
Simple end-to-end test for sem_extract batch processing.
"""

import pandas as pd
import lotus
from lotus.models import LM


def test_simple_extract():
    """Simple test for sem_extract functionality."""
    print("Setting up test environment...")
    
    # Configure model
    lotus.settings.configure(lm=LM(model="openrouter/google/gemini-2.5-flash", base_url="https://openrouter.ai/api/v1", api_key = '"key"-v1-ff17cd1b6aee306e43330dfeb8e0f7f4b85525ec32126903c2c099b8a5f3eb84'))
    print("✓ Model configured")
    
    # Create test data
    docs = [
        {"text": "The product is excellent with 5 stars. I love it!"},
        {"text": "This service is terrible, only 1 star. Very disappointed."},
        {"text": "Average quality, 3 stars rating. It's okay."},
        {"text": "Outstanding performance, 5 stars. Highly recommended!"},
        {"text": "Poor quality, 2 stars. Not worth the money."}
    ]
    
    output_cols = {
        "sentiment": "positive/negative/neutral", 
        "rating": "1-5 scale",
        "recommendation": "yes/no"
    }
    
    print(f"Testing with {len(docs)} documents...")
    print("Documents:")
    for i, doc in enumerate(docs):
        print(f"  {i+1}. {doc['text']}")
    
    # Test individual processing
    print("\n--- Testing Individual Processing ---")
    result_individual = lotus.sem_ops.sem_extract.sem_extract(
        docs=docs,
        model=lotus.settings.lm,
        output_cols=output_cols,
        use_batch_processing=False
    )
    
    print(f"Individual processing results: {len(result_individual.outputs)} outputs")
    for i, output in enumerate(result_individual.outputs):
        print(f"  Doc {i+1}: {output}")
    
    # Test batch processing
    print("\n--- Testing Batch Processing ---")
    result_batch = lotus.sem_ops.sem_extract.sem_extract(
        docs=docs,
        model=lotus.settings.lm,
        output_cols=output_cols,
        use_batch_processing=True,
        batch_size=2
    )
    
    print(f"Batch processing results: {len(result_batch.outputs)} outputs")
    for i, output in enumerate(result_batch.outputs):
        print(f"  Doc {i+1}: {output}")
    
    # Test DataFrame processing
    print("\n--- Testing DataFrame Processing ---")
    df = pd.DataFrame({
        'text': [doc['text'] for doc in docs],
        'category': ['electronics', 'service', 'general', 'electronics', 'general']
    })
    
    print("DataFrame:")
    print(df)
    
    # Test batch DataFrame processing
    result_df_batch = df.sem_extract(
        input_cols=['text'],
        output_cols=output_cols,
        use_batch_processing=True,
        batch_size=2
    )
    
    print("\nDataFrame with extracted columns (batch processing):")
    print(result_df_batch)
    
    # Test individual DataFrame processing
    result_df_individual = df.sem_extract(
        input_cols=['text'],
        output_cols=output_cols,
        use_batch_processing=False
    )
    
    print("\nDataFrame with extracted columns (individual processing):")
    print(result_df_individual)
    
    print("\n✅ All tests completed successfully!")
    # Remove return statement to fix pytest warning


if __name__ == "__main__":
    try:
        test_simple_extract()
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
