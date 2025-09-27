"""
Batch extract formatter for sem_extract operation.
"""

from typing import Any

import lotus
from lotus.models import LM
from lotus.types import ReasoningStrategy
from lotus.templates.task_instructions import (
    context_formatter,
    cot_prompt_formatter,
    deepseek_cot_formatter,
)


def batch_extract_formatter(
    model: lotus.models.LM,
    docs: list[dict[str, Any]],
    output_cols: dict[str, str | None],
    extract_quotes: bool = False,
    strategy: ReasoningStrategy | None = None,
    batch_size: int = 10,
) -> list[list[dict[str, str]]]:
    """
    Batch formatter for sem_extract operation.
    
    Args:
        model: The language model instance
        docs: List of documents to extract from
        output_cols: Mapping of output column names to descriptions
        extract_quotes: Whether to extract supporting quotes
        strategy: Reasoning strategy to use
        batch_size: Number of documents per batch
        
    Returns:
        List of batched inputs for the model
    """
    output_col_names = list(output_cols.keys())
    output_cols_with_desc: dict[str, str] = {
        col: col if desc is None else desc for col, desc in output_cols.items()
    }
    
    all_fields = output_col_names
    if extract_quotes:
        quote_fields = [f"{col}_quote" for col in output_col_names]
        all_fields += quote_fields
    
    fields_str = ", ".join(all_fields)
    
    # System instruction for batch processing
    sys_instruction = f"""You are an expert at extracting structured information from documents.

TASK: Extract specific fields from multiple documents in a single batch.

OUTPUT FORMAT: JSON with the following fields: {fields_str}
FIELD DESCRIPTIONS: {output_cols_with_desc}

IMPORTANT INSTRUCTIONS:
1. Process ALL documents in the batch
2. Extract the specified fields for each document
3. Return a JSON array with one object per document
4. Use document_id to identify each document
5. If a field cannot be extracted, use null
6. Ensure valid JSON format

EXAMPLE OUTPUT FORMAT:
[
  {{"document_id": 1, "field1": "value1", "field2": "value2"}},
  {{"document_id": 2, "field1": "value3", "field2": "value4"}}
]"""

    if strategy == ReasoningStrategy.COT:
        sys_instruction += cot_prompt_formatter(
            reasoning_instructions="Think step by step about what information to extract from each document.",
            answer_instructions="Provide your reasoning followed by the JSON extraction."
        )
    elif strategy == ReasoningStrategy.ZS_COT:
        sys_instruction += cot_prompt_formatter(
            reasoning_instructions="Think step by step about what information to extract from each document.",
            answer_instructions="Provide your reasoning followed by the JSON extraction."
        )
    
    # Create shared messages (system prompt)
    batch_shared_messages = [{"role": "system", "content": sys_instruction}]
    
    # Process documents in batches
    batched_inputs = []
    for i in range(0, len(docs), batch_size):
        batch_docs = docs[i:i + batch_size]
        actual_batch_size = len(batch_docs)
        
        # Construct batch user message
        batch_content = []
        for j, doc in enumerate(batch_docs):
            doc_text, doc_images = context_formatter(doc)
            doc_id = i + j + 1
            
            if doc_images:
                doc_content = [{"type": "text", "text": f"Document {doc_id}:\n{doc_text}"}]
                doc_content.extend(doc_images)
                batch_content.extend(doc_content)
            else:
                batch_content.append({
                    "type": "text", 
                    "text": f"Document {doc_id}:\n{doc_text}"
                })
        
        # Add instruction with clear formatting
        instruction_text = f"Extract the following fields from each document: {fields_str}"
        if strategy == ReasoningStrategy.ZS_COT and model.is_deepseek():
            instruction_text += f"\n\n{deepseek_cot_formatter()}"
        
        # Add clear instructions for processing
        processing_instructions = f"""
{instruction_text}

DOCUMENT COUNT: {actual_batch_size} documents
DOCUMENT RANGE: Document 1 to Document {actual_batch_size}

MANDATORY TASK:
1. Process ALL {actual_batch_size} documents above
2. Extract the specified fields for EACH document
3. Provide exactly {actual_batch_size} JSON objects
4. Use document_id values: 1, 2, 3, ..., {actual_batch_size}

FINAL CHECK:
- Count your responses: Must equal {actual_batch_size}
- Check document IDs: Must include 1 through {actual_batch_size}
- Ensure JSON is complete and valid

DO NOT SKIP ANY DOCUMENTS!"""
        
        batch_content.append({
            "type": "text",
            "text": processing_instructions
        })
        
        batch_messages = batch_shared_messages + [{
            "role": "user",
            "content": batch_content
        }]
        
        batched_inputs.append(batch_messages)
    
    return batched_inputs
