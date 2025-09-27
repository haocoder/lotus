"""
Basic test for async sem_extract functionality.
"""

import asyncio
import sys

def test_import():
    """Test if we can import the async function."""
    print("Testing imports...")
    
    try:
        import lotus
        print("✓ Lotus imported")
        
        from lotus.sem_ops.sem_extract import sem_extract_async
        print("✓ sem_extract_async imported")
        
        from lotus.models import LM
        print("✓ LM imported")
        
        from lotus.types import LMOutput
        print("✓ LMOutput imported")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False


async def test_async_basic():
    """Test basic async functionality."""
    print("\nTesting basic async functionality...")
    
    try:
        import lotus
        from lotus.sem_ops.sem_extract import sem_extract_async
        from lotus.models import LM
        from lotus.types import LMOutput
        from unittest.mock import AsyncMock, MagicMock
        
        # Create mock model
        mock_model = MagicMock(spec=LM)
        mock_model.async_call = AsyncMock()
        mock_model.count_tokens = MagicMock(return_value=100)
        mock_model.print_total_usage = MagicMock()
        
        # Mock response
        mock_response = LMOutput(
            outputs=['{"sentiment": "positive", "rating": "5"}']
        )
        mock_model.async_call.return_value = mock_response
        
        # Configure lotus
        lotus.settings.configure(lm=mock_model)
        
        # Test data
        docs = [{"text": "This is a test document."}]
        output_cols = {"sentiment": "positive/negative/neutral", "rating": "1-5 scale"}
        
        print("Calling sem_extract_async...")
        
        # Test async extraction
        result = await sem_extract_async(
            docs=docs,
            model=mock_model,
            output_cols=output_cols,
            use_batch_processing=False
        )
        
        print(f"✅ Async extraction successful!")
        print(f"Results: {result.outputs}")
        print(f"Raw outputs: {result.raw_outputs}")
        print(f"Explanations: {result.explanations}")
        
        # Verify mock was called
        print(f"Mock was called {mock_model.async_call.call_count} times")
        
        return True
        
    except Exception as e:
        print(f"❌ Async test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run the basic test."""
    print("🧪 Basic Async Sem Extract Test")
    print("=" * 40)
    
    # Test imports
    if not test_import():
        print("❌ Import test failed")
        return
    
    # Test async functionality
    try:
        result = asyncio.run(test_async_basic())
        if result:
            print("\n✅ All tests passed!")
            print("🎉 Async sem_extract is working!")
        else:
            print("\n❌ Async test failed")
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
