"""
Demo script for sem_extract batch processing functionality.
"""

import pandas as pd
import lotus
from lotus.models import LM


def demo_sem_extract_batch():
    """Demo the sem_extract batch processing functionality."""
    print("🚀 Sem Extract Batch Processing Demo")
    print("=" * 50)
    
    # Configure model
    print("Setting up model...")
    lotus.settings.configure(lm=LM(
        model="openrouter/google/gemini-2.5-flash", 
        base_url="https://openrouter.ai/api/v1", 
        api_key='"key"-v1-ff17cd1b6aee306e43330dfeb8e0f7f4b85525ec32126903c2c099b8a5f3eb84'
    ))
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
    
    print(f"\n📄 Testing with {len(docs)} documents:")
    for i, doc in enumerate(docs, 1):
        print(f"  {i}. {doc['text']}")
    
    # Test batch processing
    print(f"\n🔄 Testing Batch Processing (batch_size=2)")
    result_batch = lotus.sem_ops.sem_extract.sem_extract(
        docs=docs,
        model=lotus.settings.lm,
        output_cols=output_cols,
        use_batch_processing=True,
        batch_size=2
    )
    
    print(f"✅ Batch processing completed: {len(result_batch.outputs)} outputs")
    for i, output in enumerate(result_batch.outputs, 1):
        print(f"  Doc {i}: {output}")
    
    # Test DataFrame processing
    print(f"\n📊 Testing DataFrame Processing")
    df = pd.DataFrame({
        'text': [doc['text'] for doc in docs],
        'category': ['electronics', 'service', 'general', 'electronics', 'general']
    })
    
    print("Original DataFrame:")
    print(df[['text', 'category']].to_string(index=False))
    
    # Test batch DataFrame processing
    result_df = df.sem_extract(
        input_cols=['text'],
        output_cols=output_cols,
        use_batch_processing=True,
        batch_size=2
    )
    
    print(f"\nDataFrame with extracted columns:")
    print(result_df.to_string(index=False))
    
    print(f"\n🎉 Demo completed successfully!")
    print(f"📈 Performance: Batch processing processed {len(docs)} documents efficiently")


if __name__ == "__main__":
    try:
        demo_sem_extract_batch()
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
