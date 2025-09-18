#!/usr/bin/env python3
"""
Detailed debug test for batch processing to see actual JSON output.
"""

import pandas as pd
import lotus
from lotus.models import LM

def test_detailed_batch_debug():
    """Test batch processing with detailed debugging."""
    print("Testing batch processing with detailed debugging...")
    
    # Create test data with unique content
    test_data = []
    for i in range(10):
        test_data.append({"text": f"Document {i+1} content: This is unique text for document {i+1}."})
    
    df = pd.DataFrame(test_data)
    instruction = "Extract the document number from {text}:"
    
    print(f"Created DataFrame with {len(df)} rows")
    
    try:
        # Configure model
        model = LM(
            model="openrouter/google/gemini-2.5-flash",
            max_batch_size=4,
            temperature=0.0,
            max_tokens=256,
            api_key="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",
            base_url="https://openrouter.ai/api/v1"
        )
        lotus.settings.configure(lm=model)
        
        # Test batch processing
        lotus.settings.use_batch_processing = True
        lotus.settings.batch_size = 4
        
        print("\nRunning batch processing...")
        
        # Get the multimodal data to see what's being sent to the model
        col_li = lotus.nl_expression.parse_cols(instruction)
        multimodal_data = lotus.templates.task_instructions.df2multimodal_info(df, col_li)
        formatted_usr_instr = lotus.nl_expression.nle2str(instruction, col_li)
        
        print(f"Multimodal data count: {len(multimodal_data)}")
        print(f"Formatted instruction: {formatted_usr_instr}")
        
        # Call sem_map directly to see the raw outputs
        result = lotus.sem_ops.sem_map.sem_map(
            multimodal_data,
            model,
            formatted_usr_instr,
            use_batch_processing=True,
            batch_size=4
        )
        
        print(f"\nResult outputs count: {len(result.outputs)}")
        print(f"Result raw_outputs count: {len(result.raw_outputs)}")
        
        print("\nRaw outputs from model:")
        for i, raw_output in enumerate(result.raw_outputs):
            print(f"\nBatch {i+1} raw output:")
            print(raw_output[:500] + "..." if len(raw_output) > 500 else raw_output)
        
        print("\nProcessed outputs:")
        for i, output in enumerate(result.outputs):
            print(f"{i+1:2d}. {output}")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_detailed_batch_debug()
