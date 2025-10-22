import re
from typing import Any

import pandas as pd

import lotus
from lotus.dtype_extensions import ImageDtype
from lotus.types import ReasoningStrategy, SerializationFormat


def cot_formatter(reasoning, answer):
    return f"""Reasoning:\n{reasoning}\n\nAnswer: {answer}"""


def answer_only_formatter(answer):
    return f"""Answer: {answer}"""


def deepseek_cot_formatter():
    return """Please think through your reasoning step by step, then provide your final answer.
    You must put your reasoning inside the <think></think> tags, then provide your 
    final answer after the </think> tag with the format: Answer: your answer."""


def cot_prompt_formatter(reasoning_instructions: str = "", answer_instructions: str = "") -> str:
    reasoning_instructions = f"<Your reasoning here. {reasoning_instructions}>"
    answer_instructions = f"<Your answer here. {answer_instructions}>"
    return f"""Let's think step by step. Use the following format to provide your answer:
        {cot_formatter(reasoning_instructions, answer_instructions)}
        """


def non_cot_prompt_formatter(answer_instructions: str = "") -> str:
    answer_instructions = f"<Your answer here. {answer_instructions}>"
    return f"""Use the following format to provide your answer:
            {answer_only_formatter(answer_instructions)}
            """


def context_formatter(
    multimodal_data: dict[str, Any] | str,
) -> tuple[str, list[dict[str, str]]]:
    if isinstance(multimodal_data, str):
        text = multimodal_data
        image_inputs: list[dict[str, str]] = []
    elif isinstance(multimodal_data, dict):
        image_data: dict[str, str] = multimodal_data.get("image", {})
        _image_inputs: list[tuple[dict, dict]] = [
            (
                {
                    "type": "text",
                    "text": f"[{key.capitalize()}]: \n",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": base64_image},
                },
            )
            for key, base64_image in image_data.items()
        ]
        image_inputs = [m for image_input in _image_inputs for m in image_input]
        text = multimodal_data["text"] or ""
    else:
        raise ValueError("multimodal_data must be a dictionary or a string")
    return text, image_inputs


def user_message_formatter(
    multimodal_data: dict[str, Any] | str,
    user_instruction_with_tag: str | None = None,
) -> dict[str, Any]:
    text, image_inputs = context_formatter(multimodal_data)
    if not image_inputs or len(image_inputs) == 0:
        return {
            "role": "user",
            "content": f"Context:\n{text}\n\n{user_instruction_with_tag}",
        }
    content = [{"type": "text", "text": f"Context:\n{text}"}] + image_inputs
    if user_instruction_with_tag:
        content.append({"type": "text", "text": f"\n\n{user_instruction_with_tag}"})
    return {
        "role": "user",
        "content": content,
    }


def filter_formatter(
    model: lotus.models.LM,
    multimodal_data: dict[str, Any],
    user_instruction: str,
    examples_multimodal_data: list[dict[str, Any]] | None = None,
    examples_answer: list[bool] | None = None,
    cot_reasoning: list[str] | None = None,
    strategy: ReasoningStrategy | None = None,
    reasoning_instructions: str = "",
) -> list[dict[str, str]]:
    answer_instructions = "The answer should be either True or False"

    sys_instruction = """The user will provide a claim and some relevant context.
    Your job is to determine whether the claim is true for the given context.
     """

    if strategy == ReasoningStrategy.COT:
        sys_instruction += cot_prompt_formatter(
            reasoning_instructions=reasoning_instructions, answer_instructions=answer_instructions
        )
    elif strategy == ReasoningStrategy.ZS_COT:
        sys_instruction += cot_prompt_formatter(
            reasoning_instructions=reasoning_instructions, answer_instructions=answer_instructions
        )
    else:
        sys_instruction += non_cot_prompt_formatter(answer_instructions=answer_instructions)

    messages = [
        {"role": "system", "content": sys_instruction},
    ]

    if examples_multimodal_data:
        assert examples_answer is not None
        assert isinstance(examples_multimodal_data, list) and isinstance(examples_answer, list)
        assert len(examples_multimodal_data) == len(examples_answer)

        if cot_reasoning:
            # users don't have to provide cot reasoning examples
            # but if they do, the number of examples must match
            assert isinstance(cot_reasoning, list)
            assert len(examples_multimodal_data) == len(examples_answer) == len(cot_reasoning)

        for idx in range(len(examples_multimodal_data)):
            ex_multimodal_data = examples_multimodal_data[idx]
            ex_ans = examples_answer[idx]
            content = ""

            # if cot reasoning is provided, use it. Otherwise, supply a default
            # reasoning as filler if the user wants cot reasoning
            if cot_reasoning:
                content = cot_formatter(cot_reasoning[idx], str(ex_ans))
            elif strategy == ReasoningStrategy.COT:
                content = cot_formatter("Reasoning omitted", str(ex_ans))
            else:
                content = answer_only_formatter(str(ex_ans))

            messages.extend(
                [
                    user_message_formatter(ex_multimodal_data, f"Claim: {user_instruction}"),
                    {
                        "role": "assistant",
                        "content": content,
                    },
                ]
            )
    if strategy == ReasoningStrategy.ZS_COT and model.is_deepseek():
        user_instruction = f"Claim: {user_instruction}\n\n{deepseek_cot_formatter()}"
        messages.append(user_message_formatter(multimodal_data, user_instruction))
    else:
        messages.append(user_message_formatter(multimodal_data, f"Claim: {user_instruction}"))
    return messages


def map_formatter_cot(
    multimodal_data: dict[str, Any],
    user_instruction: str,
    examples_multimodal_data: list[dict[str, Any]],
    examples_answer: list[str],
    cot_reasoning: list[str],
    system_prompt: str | None = None,
) -> list[dict[str, str]]:
    sys_instruction = system_prompt or (
        "The user will provide an instruction and some relevant context.\n"
        "Your job is to answer the user's instruction given the context."
        "You must give your reasoning and then your final answer"
    )
    messages = [
        {"role": "system", "content": sys_instruction},
    ]

    for idx in range(len(examples_multimodal_data)):
        ex_df_txt = examples_multimodal_data[idx]
        ex_ans = examples_answer[idx]
        cot = cot_reasoning[idx]
        messages.extend(
            [
                user_message_formatter(ex_df_txt, f"Instruction: {user_instruction}"),
                {
                    "role": "assistant",
                    "content": f"Reasoning:\n{cot}\n\nAnswer: {ex_ans}",
                },
            ]
        )

    messages.append(user_message_formatter(multimodal_data, f"Instruction: {user_instruction}"))
    return messages


def map_formatter_zs_cot(
    multimodal_data: dict[str, Any],
    user_instruction: str,
    system_prompt: str | None = None,
) -> list[dict[str, str]]:
    sys_instruction = system_prompt or (
        "The user will provide an instruction and some relevant context.\n"
        "Your job is to answer the user's instruction given the context."
        'First give your reasoning. Then you MUST end your output with "Answer: your answer"'
    )
    messages = [
        {"role": "system", "content": sys_instruction},
    ]

    messages.append(user_message_formatter(multimodal_data, f"Instruction: {user_instruction}"))
    return messages


def map_formatter(
    model: lotus.models.LM,
    multimodal_data: dict[str, Any],
    user_instruction: str,
    examples_multimodal_data: list[dict[str, Any]] | None = None,
    examples_answer: list[str] | None = None,
    cot_reasoning: list[str] | None = None,
    strategy: ReasoningStrategy | str | None = None,
    system_prompt: str | None = None,
) -> list[dict[str, str]]:
    sys_instruction = system_prompt or (
        "The user will provide an instruction and some relevant context.\n"
        "Your job is to answer the user's instruction given the context."
    )
    if cot_reasoning:
        assert examples_multimodal_data is not None and examples_answer is not None
        return map_formatter_cot(
            multimodal_data, user_instruction, examples_multimodal_data, examples_answer, cot_reasoning, system_prompt
        )
    elif strategy == ReasoningStrategy.ZS_COT:
        return map_formatter_zs_cot(multimodal_data, user_instruction, system_prompt)

    messages = [
        {"role": "system", "content": sys_instruction},
    ]

    if examples_multimodal_data:
        assert examples_answer is not None
        for ex_df_txt, ex_ans in zip(examples_multimodal_data, examples_answer):
            messages.extend(
                [
                    user_message_formatter(ex_df_txt, f"Instruction: {user_instruction}"),
                    {"role": "assistant", "content": str(ex_ans)},
                ]
            )

    if strategy == ReasoningStrategy.ZS_COT and model.is_deepseek():
        user_intructions = f"Instruction: {user_instruction}\n\n{deepseek_cot_formatter()}"
        messages.append(user_message_formatter(multimodal_data, user_intructions))
    else:
        messages.append(user_message_formatter(multimodal_data, f"Instruction: {user_instruction}"))
    return messages


def extract_formatter(
    model: lotus.models.LM,
    multimodal_data: dict[str, Any],
    output_cols: dict[str, str | None],
    extract_quotes: bool = True,
    strategy: ReasoningStrategy | None = None,
) -> list[dict[str, str]]:
    output_col_names = list(output_cols.keys())
    # Set the description to be the key if no value is provided
    output_cols_with_desc: dict[str, str] = {col: col if desc is None else desc for col, desc in output_cols.items()}

    all_fields = output_col_names
    if extract_quotes:
        quote_fields = [f"{col}_quote" for col in output_col_names]
        all_fields += quote_fields

    fields_str = ", ".join(all_fields)

    sys_instruction = (
        "The user will provide the columns that need to be extracted and some relevant context.\n"
        f"Your job is to extract these columns and provide only a concise value for each field "
        f"and the corresponding full quote for each field in the '{', '.join([f'{col}_quote' for col in output_col_names])}' fields.\n"
        f"Here is a description of each field: {output_cols_with_desc}\n"
        f"The response should be valid JSON format with the following fields: {fields_str}.\n"
    )

    messages = [
        {"role": "system", "content": sys_instruction},
        user_message_formatter(multimodal_data),
    ]

    if strategy == ReasoningStrategy.ZS_COT and model.is_deepseek():
        user_intructions = f"Instruction: {deepseek_cot_formatter()}"
        messages.append(user_message_formatter(multimodal_data, user_intructions))

    return messages


# returns a list of strings corresponding to df rows
def df2text(df: pd.DataFrame, cols: list[str]) -> list[str]:
    """Formats the given DataFrame into a string containing info from cols."""

    def custom_format_row(x: pd.Series, cols: list[str]) -> str:
        return "".join([f"[{cols[i].capitalize()}]: «{x[cols[i]]}»\n" for i in range(len(cols))])

    def clean_and_escape_column_name(column_name: str) -> str:
        clean_name = re.sub(r"[^\w]", "", column_name)  # Remove spaces and special characters
        return clean_name

    # take cols that are in df
    cols = [col for col in cols if col in df.columns]
    if len(cols) == 0:
        return [""] * len(df)

    projected_df = df[cols]
    formatted_rows: list[str] = []

    if lotus.settings.serialization_format == SerializationFormat.DEFAULT:
        formatted_rows = projected_df.apply(lambda x: custom_format_row(x, cols), axis=1).tolist()
    elif lotus.settings.serialization_format == SerializationFormat.JSON:
        formatted_rows = projected_df.to_json(orient="records", lines=True).splitlines()
    elif lotus.settings.serialization_format == SerializationFormat.XML:
        try:
            import xml.etree.ElementTree as ET
        except ImportError:
            raise ImportError(
                "The 'lxml' library is required for XML serialization. "
                "You can install it with the following command:\n\n"
                "    pip install 'lotus-ai[xml]'"
            )
        projected_df = projected_df.rename(columns=lambda x: clean_and_escape_column_name(x))
        full_xml = projected_df.to_xml(root_name="data", row_name="row", pretty_print=False, index=False)
        root = ET.fromstring(full_xml)
        formatted_rows = [ET.tostring(row, encoding="unicode", method="xml") for row in root.findall("row")]

    return formatted_rows


def df2multimodal_info(df: pd.DataFrame, cols: list[str]) -> list[dict[str, Any]]:
    """
    Formats the given DataFrame into a string containing info from cols.
    Return a list of dictionaries, each containing text and image data.
    """
    image_cols = [col for col in cols if isinstance(df[col].dtype, ImageDtype)]
    text_cols = [col for col in cols if col not in image_cols]
    text_rows = df2text(df, text_cols)
    multimodal_data = [
        {
            "text": text_rows[i],
            "image": {col.capitalize(): df[col].array.get_image(i, "base64") for col in image_cols},
        }
        for i in range(len(df))
    ]
    return multimodal_data


def merge_multimodal_info(first: list[dict[str, Any]], second: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Merges two multimodal info lists into one. Each row of first is merged with each row of second.

    Args:
        first: list of multimodal info dictionaries
        second: list of multimodal info dictionaries

    Returns:
        list of merged multimodal info dictionaries
    """
    return [
        {
            "text": f"{first[i]['text']}\n{second[j]['text']}"
            if first[i]["text"] != "" and second[j]["text"] != ""
            else first[i]["text"] + second[j]["text"],
            "image": {**first[i]["image"], **second[j]["image"]},
        }
        for i in range(len(first))
        for j in range(len(second))
    ]


def li2text(li: list[str], name: str) -> str:
    return "".join([f"[{name}] {li[i]}\n" for i in range(len(li))])


def batch_filter_formatter(
    model: lotus.models.LM,
    docs: list[dict[str, Any]],
    user_instruction: str,
    examples_multimodal_data: list[dict[str, Any]] | None = None,
    examples_answer: list[bool] | None = None,
    cot_reasoning: list[str] | None = None,
    strategy: ReasoningStrategy | None = None,
    reasoning_instructions: str = "",
    batch_size: int = 10,
) -> list[list[dict[str, str]]]:
    """
    Batch formatter for filter operations that shares system prompt and examples across documents.
    
    This formatter creates batch prompts with dynamically adjusted batch sizes to handle
    cases where the total number of documents doesn't divide evenly by the batch size.
    
    Args:
        model: Language model instance
        docs: List of documents to process
        user_instruction: The filter instruction/claim
        examples_multimodal_data: Example documents for few-shot learning
        examples_answer: Expected boolean outputs for examples
        cot_reasoning: Chain-of-thought reasoning for examples
        strategy: Reasoning strategy to use
        reasoning_instructions: Additional reasoning instructions
        batch_size: Maximum number of documents per batch
        
    Returns:
        List of message lists for batch processing
    """
    answer_instructions = "The answer should be either True or False"
    
    # Add examples if provided
    example_messages = []
    if examples_multimodal_data:
        assert examples_answer is not None
        assert isinstance(examples_multimodal_data, list) and isinstance(examples_answer, list)
        assert len(examples_multimodal_data) == len(examples_answer)

        if cot_reasoning:
            assert isinstance(cot_reasoning, list)
            assert len(examples_multimodal_data) == len(examples_answer) == len(cot_reasoning)

        for idx in range(len(examples_multimodal_data)):
            ex_multimodal_data = examples_multimodal_data[idx]
            ex_ans = examples_answer[idx]
            content = ""

            if cot_reasoning:
                content = cot_formatter(cot_reasoning[idx], str(ex_ans))
            elif strategy == ReasoningStrategy.COT:
                content = cot_formatter("Reasoning omitted", str(ex_ans))
            else:
                content = answer_only_formatter(str(ex_ans))

            example_messages.extend([
                user_message_formatter(ex_multimodal_data, f"Claim: {user_instruction}"),
                {"role": "assistant", "content": content},
            ])
    
    # Process documents in batches
    batched_inputs = []
    for i in range(0, len(docs), batch_size):
        batch_docs = docs[i:i + batch_size]
        actual_batch_size = len(batch_docs)
        
        # Construct system prompt with ACTUAL batch size (Problem 3 fix)
        # This ensures the model knows exactly how many documents to process
        sys_instruction = f"""You will receive EXACTLY {actual_batch_size} document{"s" if actual_batch_size > 1 else ""} and a claim.

YOUR TASK:
- Evaluate whether the claim is true for EACH of the {actual_batch_size} document{"s" if actual_batch_size > 1 else ""}
- Provide EXACTLY {actual_batch_size} result{"s" if actual_batch_size > 1 else ""}
- Use document_id values: 1, 2, 3, ..., {actual_batch_size}

REQUIRED JSON FORMAT (you must follow this exactly):
{{
    "results": [
        {{"document_id": 1, "answer": true, "reasoning": "your reasoning for doc 1"}},
        {{"document_id": 2, "answer": false, "reasoning": "your reasoning for doc 2"}},
        ...
        {{"document_id": {actual_batch_size}, "answer": true/false, "reasoning": "your reasoning for doc {actual_batch_size}"}}
    ]
}}

CRITICAL REQUIREMENTS:
✓ Return exactly {actual_batch_size} results (one per document)
✓ Use document_id from 1 to {actual_batch_size} (inclusive)
✓ Answer must be true or false (lowercase, no quotes)
✓ Provide reasoning for each document
✓ Return valid JSON (no markdown code blocks, no missing brackets)
✓ Do NOT skip any documents

BEFORE SUBMITTING YOUR RESPONSE:
1. Count your results: Must be exactly {actual_batch_size}
2. Check document IDs: Must include all from 1 to {actual_batch_size}
3. Validate JSON format: Must be complete and parseable
4. Verify no documents were skipped or duplicated"""

        # Add strategy-specific instructions
        if strategy == ReasoningStrategy.COT:
            sys_instruction += "\n\n" + cot_prompt_formatter(
                reasoning_instructions=reasoning_instructions, 
                answer_instructions=answer_instructions
            )
        elif strategy == ReasoningStrategy.ZS_COT:
            sys_instruction += "\n\n" + cot_prompt_formatter(
                reasoning_instructions=reasoning_instructions, 
                answer_instructions=answer_instructions
            )
        else:
            sys_instruction += "\n\n" + non_cot_prompt_formatter(answer_instructions=answer_instructions)

        # Construct messages for this batch
        batch_messages = [{"role": "system", "content": sys_instruction}]
        
        # Add example messages if provided
        batch_messages.extend(example_messages)
        
        # Construct batch user message with documents
        batch_content = []
        for j, doc in enumerate(batch_docs):
            doc_text, doc_images = context_formatter(doc)
            doc_id = j + 1  # Within-batch ID (always starts from 1)
            
            if doc_images:
                # Handle multimodal content
                doc_content = [{"type": "text", "text": f"Document {doc_id}:\n{doc_text}"}]
                doc_content.extend(doc_images)
                batch_content.extend(doc_content)
            else:
                batch_content.append({
                    "type": "text", 
                    "text": f"Document {doc_id}:\n{doc_text}"
                })
        
        # Add claim instruction with explicit batch information
        claim_text = f"\n{'=' * 60}\nCLAIM: {user_instruction}\n{'=' * 60}\n"
        claim_text += f"\nBATCH SUMMARY:\n"
        claim_text += f"- Total documents in this batch: {actual_batch_size}\n"
        claim_text += f"- Document IDs to process: 1 through {actual_batch_size}\n"
        claim_text += f"- Required number of results: {actual_batch_size}\n"
        
        if strategy == ReasoningStrategy.ZS_COT and model.is_deepseek():
            claim_text += f"\n{deepseek_cot_formatter()}\n"
        
        claim_text += f"\nNow evaluate the claim for EACH of the {actual_batch_size} documents above."
        claim_text += f"\nRemember: You MUST provide exactly {actual_batch_size} results in valid JSON format."
        
        batch_content.append({
            "type": "text",
            "text": claim_text
        })
        
        batch_messages.append({
            "role": "user",
            "content": batch_content
        })
        
        batched_inputs.append(batch_messages)
    
    return batched_inputs


def batch_map_formatter(
    model: lotus.models.LM,
    docs: list[dict[str, Any]],
    user_instruction: str,
    examples_multimodal_data: list[dict[str, Any]] | None = None,
    examples_answer: list[str] | None = None,
    cot_reasoning: list[str] | None = None,
    strategy: ReasoningStrategy | None = None,
    system_prompt: str | None = None,
    batch_size: int = 10,
) -> list[list[dict[str, str]]]:
    """
    Batch formatter for map operations that shares system prompt and examples across documents.
    
    Args:
        model: Language model instance
        docs: List of documents to process
        user_instruction: The mapping instruction
        examples_multimodal_data: Example documents for few-shot learning
        examples_answer: Expected outputs for examples
        cot_reasoning: Chain-of-thought reasoning for examples
        strategy: Reasoning strategy to use
        system_prompt: Custom system prompt
        batch_size: Number of documents per batch
        
    Returns:
        List of message lists for batch processing
    """
    # Construct shared system prompt
    sys_instruction = system_prompt or f"""You are processing {batch_size} documents. You MUST provide exactly {batch_size} responses - one for each document.

MANDATORY REQUIREMENTS:
1. Count the documents: You will receive exactly {batch_size} documents
2. Process ALL documents: Every document must have a response
3. Use correct document IDs: Document IDs must be 1, 2, 3, ..., {batch_size}
4. Return valid JSON: Use the exact format below

REQUIRED JSON FORMAT:
{{
    "results": [
        {{"document_id": 1, "answer": "response for document 1"}},
        {{"document_id": 2, "answer": "response for document 2"}},
        {{"document_id": 3, "answer": "response for document 3"}},
        {{"document_id": 4, "answer": "response for document 4"}}
    ]
}}

CRITICAL RULES:
- NO markdown code blocks (no ```json or ```)
- NO incomplete responses
- NO missing document IDs
- ALL {batch_size} documents must be processed
- JSON must be complete and valid

BEFORE SUBMITTING:
1. Count your responses: Must be exactly {batch_size}
2. Check document IDs: Must include 1, 2, 3, ..., {batch_size}
3. Validate JSON: Must be properly formatted
4. Ensure completeness: No missing or empty responses"""

    if strategy == ReasoningStrategy.COT:
        sys_instruction += " You must provide reasoning for each answer."
    elif strategy == ReasoningStrategy.ZS_COT:
        sys_instruction += " You must provide reasoning for each answer."

    # Construct shared messages (system prompt + examples)
    shared_messages = [{"role": "system", "content": sys_instruction}]
    
    # Add examples if provided
    if examples_multimodal_data:
        assert examples_answer is not None
        for ex_df_txt, ex_ans in zip(examples_multimodal_data, examples_answer):
            if cot_reasoning:
                content = f"Reasoning:\n{cot_reasoning[examples_multimodal_data.index(ex_df_txt)]}\n\nAnswer: {ex_ans}"
            else:
                content = str(ex_ans)
            
            shared_messages.extend([
                user_message_formatter(ex_df_txt, f"Instruction: {user_instruction}"),
                {"role": "assistant", "content": content},
            ])
    
    # Process documents in batches
    batched_inputs = []
    for i in range(0, len(docs), batch_size):
        batch_docs = docs[i:i + batch_size]
        actual_batch_size = len(batch_docs)
        
        # Update system prompt for actual batch size
        if actual_batch_size != batch_size:
            batch_sys_instruction = sys_instruction.replace(
                f"{batch_size} documents", f"{actual_batch_size} documents"
            )
            batch_shared_messages = [{"role": "system", "content": batch_sys_instruction}]
            # Add examples to batch messages
            if len(shared_messages) > 1:
                batch_shared_messages.extend(shared_messages[1:])
        else:
            batch_shared_messages = shared_messages.copy()
        
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
        instruction_text = f"INSTRUCTION: {user_instruction}"
        if strategy == ReasoningStrategy.ZS_COT and model.is_deepseek():
            instruction_text += f"\n\n{deepseek_cot_formatter()}"
        
        # Add clear instructions for processing
        processing_instructions = f"""
{instruction_text}

DOCUMENT COUNT: {actual_batch_size} documents
DOCUMENT RANGE: Document 1 to Document {actual_batch_size}

MANDATORY TASK:
1. Process ALL {actual_batch_size} documents above
2. Apply the instruction to EACH document
3. Provide exactly {actual_batch_size} responses
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