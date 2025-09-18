#!/usr/bin/env python3
"""
Test JSON parsing fixes for batch processing.
"""

import json
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lotus.sem_ops.postprocessors import _fix_json_format, _parse_fallback_map

def test_json_fixing():
    """Test JSON fixing functionality."""
    print("Testing JSON fixing functionality...")
    
    # Test case 1: Unterminated JSON
    broken_json = '''```json
{
    "results": [
        {
            "document_id": 1,
            "answer": "The weather is beautiful with clear blue skies.",
            "reasoning": "The original text states 'The weat...'''
    
    print("Input (broken JSON):")
    print(broken_json[:100] + "...")
    
    fixed = _fix_json_format(broken_json)
    print("\nFixed JSON:")
    print(fixed[:200] + "...")
    
    try:
        parsed = json.loads(fixed)
        print("✓ JSON parsing successful!")
        print(f"Found {len(parsed.get('results', []))} results")
    except json.JSONDecodeError as e:
        print(f"✗ JSON parsing failed: {e}")
    
    print("\n" + "="*50)
    
    # Test case 2: Fallback parsing
    print("Testing fallback parsing...")
    
    fallback_text = '''```json
{
    "results": [
        {
            "document_id": 1,
            "answer": "The weather is beautiful with clear blue skies.",
            "reasoning": "The original text states 'The weat...'''
    
    outputs, explanations = _parse_fallback_map(fallback_text)
    print(f"Fallback parsing extracted {len(outputs)} outputs")
    if outputs:
        print(f"First output: {outputs[0][:50]}...")

if __name__ == "__main__":
    test_json_fixing()
