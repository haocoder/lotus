"""
Debug test for async sem_extract functionality.
"""

import asyncio
import sys
import traceback

def test_imports():
    """Test if all imports work correctly."""
    print("Testing imports...")
    
    try:
        import lotus
        print("✓ Lotus imported successfully")
        
        from lotus.models import LM
        print("✓ LM imported successfully")
        
        from lotus.sem_ops.sem_extract import sem_extract_async
        print("✓ sem_extract_async imported successfully")
        
        import pandas as pd
        print("✓ Pandas imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        traceback.print_exc()
        return False


def test_model_configuration():
    """Test model configuration."""
    print("\nTesting model configuration...")
    
    try:
        import lotus
        from lotus.models import LM
        
        # Configure model
        lotus.settings.configure(lm=LM(
            model="openrouter/google/gemini-2.5-flash", 
            base_url="https://openrouter.ai/api/v1", 
            api_key='"key"-v1-ff17cd1b6aee306e43330dfeb8e0f7f4b85525ec32126903c2c099b8a5f3eb84'
        ))
        
        print("✓ Model configured successfully")
        print(f"Model: {lotus.settings.lm}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model configuration failed: {e}")
        traceback.print_exc()
        return False


async def test_basic_async():
    """Test basic async functionality."""
    print("\nTesting basic async functionality...")
    
    try:
        import lotus
        from lotus.sem_ops.sem_extract import sem_extract_async
        
        # Simple test data
        docs = [{"text": "This is a test document."}]
        output_cols = {"sentiment": "positive/negative/neutral"}
        
        print("Calling sem_extract_async...")
        result = await sem_extract_async(
            docs=docs,
            model=lotus.settings.lm,
            output_cols=output_cols,
            use_batch_processing=False
        )
        
        print(f"✓ Async extraction completed: {len(result.outputs)} outputs")
        print(f"Results: {result.outputs}")
        
        return True
        
    except Exception as e:
        print(f"❌ Async test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run debug tests."""
    print("🔍 Debug Test for Async Sem Extract")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        print("❌ Import test failed")
        return
    
    # Test model configuration
    if not test_model_configuration():
        print("❌ Model configuration test failed")
        return
    
    # Test basic async functionality
    try:
        result = asyncio.run(test_basic_async())
        if result:
            print("\n✅ All debug tests passed!")
        else:
            print("\n❌ Async test failed")
    except Exception as e:
        print(f"❌ Async test failed with exception: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
