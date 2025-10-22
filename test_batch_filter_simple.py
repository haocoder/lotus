#!/usr/bin/env python3
"""
Simple unit tests for batch filter functionality.

This script tests the core batch processing components without
requiring actual model calls.
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

import lotus
from lotus.models import LM
from lotus.templates.task_instructions import batch_filter_formatter
from lotus.sem_ops.postprocessors import batch_filter_parser
from lotus.types import ReasoningStrategy


def test_batch_formatter():
    """Test batch filter formatter."""
    print("Testing batch filter formatter...")
    
    model = LM(
        model="openrouter/google/gemini-2.5-flash",
        max_batch_size=4,
        temperature=0.0,
        max_tokens=256,
        api_key="''-v1-acdced37337241eb1d9ceb106f1ab6c604f189bd95b7933dc50ae72b81e16807",
        base_url="https://openrouter.ai/api/v1"
    )
    model.is_deepseek.return_value = False
    
    docs = [
        {"text": "This is a positive review"},
        {"text": "This is a negative review"},
    ]
    user_instruction = "Is this a positive sentiment?"
    
    result = batch_filter_formatter(
        model=model,
        docs=docs,
        user_instruction=user_instruction,
        batch_size=2
    )
    
    # Verify result structure
    assert len(result) == 1, f"Expected 1 batch, got {len(result)}"
    assert len(result[0]) >= 2, f"Expected at least 2 messages, got {len(result[0])}"
    
    # Check system prompt
    system_prompt = result[0][0]["content"]
    assert "2 documents" in system_prompt
    assert "JSON format" in system_prompt
    assert "document_id" in system_prompt
    
    # Check user message
    user_message = result[0][1]["content"]
    assert "Document 1:" in str(user_message)
    assert "Document 2:" in str(user_message)
    assert "Claim: Is this a positive sentiment?" in str(user_message)
    
    print("✓ Batch formatter test passed")
    return True


def test_batch_parser():
    """Test batch filter parser."""
    print("Testing batch filter parser...")
    
    model = LM(
        model="openrouter/google/gemini-2.5-flash",
        max_batch_size=4,
        temperature=0.0,
        max_tokens=256,
        api_key="''-v1-acdced37337241eb1d9ceb106f1ab6c604f189bd95b7933dc50ae72b81e16807",
        base_url="https://openrouter.ai/api/v1"
    )
    
    # Test with proper JSON format
    batch_outputs = [
        '''{
            "results": [
                {"document_id": 1, "answer": true, "reasoning": "Positive words present"},
                {"document_id": 2, "answer": false, "reasoning": "Negative words present"}
            ]
        }'''
    ]
    
    outputs, raw_outputs, explanations = batch_filter_parser(
        batch_outputs, model, default=True, expected_doc_count=2
    )
    
    assert outputs == [True, False], f"Expected [True, False], got {outputs}"
    assert len(raw_outputs) == 1, f"Expected 1 raw output, got {len(raw_outputs)}"
    assert explanations == ["Positive words present", "Negative words present"]
    
    print("✓ Batch parser test passed")
    return True


def test_batch_parser_fallback():
    """Test batch filter parser fallback."""
    print("Testing batch filter parser fallback...")
    
    model = LM(
        model="openrouter/google/gemini-2.5-flash",
        max_batch_size=4,
        temperature=0.0,
        max_tokens=256,
        api_key="''-v1-acdced37337241eb1d9ceb106f1ab6c604f189bd95b7933dc50ae72b81e16807",
        base_url="https://openrouter.ai/api/v1"
    )
    
    # Test with malformed JSON (should use fallback)
    batch_outputs = [
        "True\nFalse\nTrue"  # Fallback format
    ]
    
    outputs, raw_outputs, explanations = batch_filter_parser(
        batch_outputs, model, default=True, expected_doc_count=3
    )
    
    assert outputs == [True, False, True], f"Expected [True, False, True], got {outputs}"
    assert len(explanations) == 3, f"Expected 3 explanations, got {len(explanations)}"
    
    print("✓ Batch parser fallback test passed")
    return True


def test_multiple_batches():
    """Test multiple batch handling."""
    print("Testing multiple batch handling...")
    
    model = Mock(spec=LM)
    model.is_deepseek.return_value = False
    
    docs = [
        {"text": f"Document {i}"} for i in range(1, 6)
    ]
    user_instruction = "Is this about technology?"
    
    result = batch_filter_formatter(
        model=model,
        docs=docs,
        user_instruction=user_instruction,
        batch_size=2
    )
    
    # Should create 3 batches: [doc1, doc2], [doc3, doc4], [doc5]
    assert len(result) == 3, f"Expected 3 batches, got {len(result)}"
    
    # Check first batch
    assert "2 documents" in result[0][0]["content"]
    assert "Document 1:" in str(result[0][1]["content"])
    assert "Document 2:" in str(result[0][1]["content"])
    
    # Check second batch
    assert "2 documents" in result[1][0]["content"]
    assert "Document 3:" in str(result[1][1]["content"])
    assert "Document 4:" in str(result[1][1]["content"])
    
    # Check third batch (smaller)
    assert "1 document" in result[2][0]["content"]
    assert "Document 5:" in str(result[2][1]["content"])
    
    print("✓ Multiple batches test passed")
    return True


def test_with_examples():
    """Test batch formatter with examples."""
    print("Testing batch formatter with examples...")
    
    model = Mock(spec=LM)
    model.is_deepseek.return_value = False
    
    docs = [
        {"text": "This is a positive review"},
        {"text": "This is a negative review"},
    ]
    user_instruction = "Is this a positive sentiment?"
    
    examples_multimodal_data = [
        {"text": "Great product!"},
        {"text": "Terrible quality"},
    ]
    examples_answer = [True, False]
    
    result = batch_filter_formatter(
        model=model,
        docs=docs,
        user_instruction=user_instruction,
        examples_multimodal_data=examples_multimodal_data,
        examples_answer=examples_answer,
        batch_size=2
    )
    
    # Should have system prompt + examples + user message
    assert len(result[0]) >= 5, f"Expected at least 5 messages, got {len(result[0])}"
    
    print("✓ Examples test passed")
    return True


def test_with_cot():
    """Test batch formatter with chain-of-thought."""
    print("Testing batch formatter with CoT...")
    
    model = LM(
        model="openrouter/google/gemini-2.5-flash",
        max_batch_size=4,
        temperature=0.0,
        max_tokens=256,
        api_key="''-v1-acdced37337241eb1d9ceb106f1ab6c604f189bd95b7933dc50ae72b81e16807",
        base_url="https://openrouter.ai/api/v1"
    )
    model.is_deepseek.return_value = False
    
    docs = [{"text": "This is a positive review"}]
    user_instruction = "Is this a positive sentiment?"
    
    result = batch_filter_formatter(
        model=model,
        docs=docs,
        user_instruction=user_instruction,
        strategy=ReasoningStrategy.COT,
        reasoning_instructions="Think step by step",
        batch_size=1
    )
    
    system_prompt = result[0][0]["content"]
    assert "Think step by step" in system_prompt
    assert "reasoning" in system_prompt
    
    print("✓ CoT test passed")
    return True


def main():
    """Run all tests."""
    print("Starting batch filter unit tests...\n")
    
    tests = [
        test_batch_formatter,
        test_batch_parser,
        test_batch_parser_fallback,
        test_multiple_batches,
        test_with_examples,
        test_with_cot,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"✗ {test.__name__} failed")
        except Exception as e:
            print(f"✗ {test.__name__} failed with exception: {e}")
        print()
    
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    exit(main())
