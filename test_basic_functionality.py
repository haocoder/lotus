#!/usr/bin/env python3
"""
Basic functionality test for sem_map batch processing.

This script tests the basic functionality without requiring API calls.
"""

import pandas as pd
import lotus
from lotus.models import LM


def test_imports_and_structure():
    """Test that all necessary imports and structures are in place."""
    print("Testing imports and structure...")
    
    try:
        # Test that sem_map function exists and has batch parameters
        import lotus.sem_ops.sem_map as sem_map_module
        
        # Check that the function signature includes batch parameters
        import inspect
        sig = inspect.signature(sem_map_module.sem_map)
        params = list(sig.parameters.keys())
        
        required_params = ['batch_size', 'use_batch_processing']
        for param in required_params:
            if param in params:
                print(f"   ✓ {param} parameter found")
            else:
                print(f"   ✗ {param} parameter missing")
                return False
        
        # Test that batch functions exist
        if hasattr(sem_map_module, '_sem_map_batch'):
            print("   ✓ _sem_map_batch function found")
        else:
            print("   ✗ _sem_map_batch function missing")
            return False
        
        if hasattr(sem_map_module, '_sem_map_individual'):
            print("   ✓ _sem_map_individual function found")
        else:
            print("   ✗ _sem_map_individual function missing")
            return False
        
        # Test that settings have batch parameters
        if hasattr(lotus.settings, 'use_batch_processing'):
            print("   ✓ use_batch_processing setting found")
        else:
            print("   ✗ use_batch_processing setting missing")
            return False
        
        if hasattr(lotus.settings, 'batch_size'):
            print("   ✓ batch_size setting found")
        else:
            print("   ✗ batch_size setting missing")
            return False
        
        print("✓ All imports and structure tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Error during import/structure testing: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_settings_configuration():
    """Test settings configuration."""
    print("\nTesting settings configuration...")
    
    try:
        # Test default values
        print(f"   Default use_batch_processing: {lotus.settings.use_batch_processing}")
        print(f"   Default batch_size: {lotus.settings.batch_size}")
        
        # Test setting values
        original_use_batch = lotus.settings.use_batch_processing
        original_batch_size = lotus.settings.batch_size
        
        lotus.settings.use_batch_processing = False
        lotus.settings.batch_size = 5
        
        print(f"   After setting - use_batch_processing: {lotus.settings.use_batch_processing}")
        print(f"   After setting - batch_size: {lotus.settings.batch_size}")
        
        # Restore original values
        lotus.settings.use_batch_processing = original_use_batch
        lotus.settings.batch_size = original_batch_size
        
        print("✓ Settings configuration test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Error during settings testing: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataframe_accessor():
    """Test DataFrame accessor functionality."""
    print("\nTesting DataFrame accessor...")
    
    try:
        # Create test DataFrame
        df = pd.DataFrame({
            'text': [
                "This is a test document.",
                "Another test document here.",
                "Yet another test document."
            ]
        })
        
        # Test that sem_map accessor exists
        if hasattr(df, 'sem_map'):
            print("   ✓ sem_map accessor found")
        else:
            print("   ✗ sem_map accessor missing")
            return False
        
        # Test that we can call it (without actually running it)
        # We'll just check the signature
        import inspect
        sig = inspect.signature(df.sem_map)
        params = list(sig.parameters.keys())
        
        print(f"   sem_map accessor parameters: {params}")
        
        print("✓ DataFrame accessor test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Error during DataFrame accessor testing: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_formatter_and_parser():
    """Test that batch formatter and parser exist."""
    print("\nTesting batch formatter and parser...")
    
    try:
        # Test batch formatter
        import lotus.templates.task_instructions as task_instructions
        if hasattr(task_instructions, 'batch_map_formatter'):
            print("   ✓ batch_map_formatter found")
        else:
            print("   ✗ batch_map_formatter missing")
            return False
        
        # Test batch parser
        import lotus.sem_ops.postprocessors as postprocessors
        if hasattr(postprocessors, 'batch_map_parser'):
            print("   ✓ batch_map_parser found")
        else:
            print("   ✗ batch_map_parser missing")
            return False
        
        print("✓ Batch formatter and parser test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Error during batch formatter/parser testing: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function to run basic functionality tests."""
    print("=" * 60)
    print("SEM_MAP BATCH PROCESSING BASIC FUNCTIONALITY TESTS")
    print("=" * 60)
    
    tests = [
        test_imports_and_structure,
        test_settings_configuration,
        test_dataframe_accessor,
        test_batch_formatter_and_parser,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 60)
    print(f"BASIC FUNCTIONALITY TEST SUMMARY: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 All basic functionality tests passed!")
        print("The batch processing implementation is structurally correct.")
        print("\nNext steps:")
        print("1. Run test_sem_map_batch_quick.py to test with actual API calls")
        print("2. Run test_sem_map_integration.py for comprehensive integration testing")
        print("3. Run test_sem_map_batch_performance.py for performance benchmarking")
    else:
        print("⚠️  Some basic functionality tests failed.")
        print("Please check the implementation before proceeding with API tests.")


if __name__ == "__main__":
    main()
