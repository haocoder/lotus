import json
from typing import Any, Callable

import lotus
from lotus.types import (
    SemanticExtractPostprocessOutput,
    SemanticFilterPostprocessOutput,
    SemanticMapPostprocessOutput,
)


def cot_postprocessor(llm_answers: list[str]):
    outputs: list[str | None] = []
    explanations: list[str | None] = []
    for llm_answer in llm_answers:
        reasoning_idx = llm_answer.find("Reasoning:\n")
        if reasoning_idx == -1:
            reasoning_idx = 0
        else:
            reasoning_idx += len("Reasoning:\n")

        answer_idx = llm_answer.find("Answer:")
        reasoning = llm_answer[reasoning_idx:answer_idx].rstrip("\n").lstrip("\n")
        answer = llm_answer[answer_idx + len("Answer:") :]

        explanations.append(reasoning)
        outputs.append(answer)

    return outputs, explanations


def deepseek_cot_postprocessor(llm_answers: list[str], for_extract: bool = False):
    """
    Postprocess outputs from DeepSeek models with CoT reasoning.

    Args:
        llm_answers (list[str]): The list of llm answers from DeepSeek.

    Returns:
        Tuple: (outputs, explanations)
    """
    outputs: list[str | None] = []
    explanations: list[str | None] = []

    for llm_answer in llm_answers:
        think_start = llm_answer.find("<think>")
        think_end = llm_answer.find("</think>")

        answer_start = llm_answer.find("Answer:")

        if think_start != -1 and think_end != -1:
            # Extract the reasoning between the <think> tags
            reasoning = llm_answer[think_start + len("<think>") : think_end].strip()
            answer = llm_answer[answer_start + len("Answer:") :].strip()

            answer = answer.strip()

            # If ther is nothing after </think> tag, check if the answer is at the beginning
            if not answer and think_start > 0:
                answer = llm_answer[:think_start].strip()

        else:
            reasoning = ""
            answer = llm_answer.strip()

        explanations.append(reasoning)

        if for_extract:
            try:
                json_obj = json.loads(llm_answer)
            except json.JSONDecodeError:
                lotus.logger.info(f"\t Failed to parse: {llm_answer}")
                json_obj = {}
            json_obj = {key: str(value) for key, value in json_obj.items()}
            outputs.append(json_obj)
        else:
            outputs.append(answer)

    return outputs, explanations


COT_POSTPROCESSORS = {
    "deepseek-r1": deepseek_cot_postprocessor,
    # Add more model-specific postprocessors here
}


def get_cot_postprocessor(model: lotus.models.LM, for_extract: bool = False) -> Callable:
    """
    Returns the appropriate CoT postprocessor for the given model.
    Falls back to standard postprocessor if no specific one is defined.

    Args:
        model (lotus.models.LM): The language model.
        for_extract (bool): Whether to process for extraction (convert to JSON).

    Returns:
        Callable: The appropriate postprocessor function.
    """
    model_name = model.get_model_name()
    for processor_key in COT_POSTPROCESSORS:
        if model_name.startswith(processor_key):
            base_processor = COT_POSTPROCESSORS[processor_key]
            return lambda llm_answers: base_processor(llm_answers, for_extract=for_extract)

    return cot_postprocessor


def map_postprocess(
    llm_answers: list[str],
    model: lotus.models.LM,
    cot_reasoning: bool = False,
) -> SemanticMapPostprocessOutput:
    """
    Postprocess the output of the map operator.

    Args:
        llm_answers (list[str]): The list of llm answers.
        cot_reasoning (bool): Whether there is CoT reasoning.

    Returns:
        SemanticMapPostprocessOutput
    """

    if cot_reasoning:
        postprocessor = get_cot_postprocessor(model)
        outputs, explanations = postprocessor(llm_answers)
    else:
        outputs = llm_answers
        explanations = [None] * len(llm_answers)

    return SemanticMapPostprocessOutput(raw_outputs=llm_answers, outputs=outputs, explanations=explanations)


def extract_postprocess(
    llm_answers: list[str], model: lotus.models.LM, cot_reasoning: bool = False
) -> SemanticExtractPostprocessOutput:
    """
    Postprocess the output of the extract operator to extract the schema.

    Args:
        llm_answers (list[str]): The list of llm answers containging the extract.

    Returns:
        SemanticExtractPostprocessOutput
    """

    if cot_reasoning:
        postprocessor = get_cot_postprocessor(model, for_extract=True)
        extract_data, explanations = postprocessor(llm_answers)
    else:
        extract_data = []
        explanations = [None] * len(llm_answers)

    for llm_answer in llm_answers:
        try:
            output = json.loads(llm_answer)
        except json.JSONDecodeError:
            lotus.logger.info(f"\t Failed to parse: {llm_answer}")
            output = {}

        output = {key: str(value) for key, value in output.items()}
        extract_data.append(output)

    return SemanticExtractPostprocessOutput(raw_outputs=llm_answers, outputs=extract_data, explanations=explanations)


def filter_postprocess(
    llm_answers: list[str],
    model: lotus.models.LM,
    default: bool = True,
) -> SemanticFilterPostprocessOutput:
    """
    Postprocess the output of the filter operator.

    Args:
        llm_answers (list[str]): The list of llm answers.
        default (bool): The default value to use if we fail to parse the answer.
        cot_reasoning (bool): Whether there is CoT reasoning.

    Returns:
        SemanticFilterPostprocessOutput

    """

    def process_outputs(answer):
        if answer is None:
            lotus.logger.info(f"\t Failed to parse {answer}: defaulting to {default}")
            return default

        if "True" in answer:
            return True
        elif "False" in answer:
            return False
        else:
            lotus.logger.info(f"\t Failed to parse {answer}: defaulting to {default}")
            return default

    postprocessor = get_cot_postprocessor(model)
    outputs, explanations = postprocessor(llm_answers)

    boolean_outputs = [process_outputs(answer) for answer in outputs]

    return SemanticFilterPostprocessOutput(raw_outputs=llm_answers, outputs=boolean_outputs, explanations=explanations)


def batch_filter_parser(
    batch_outputs: list[str], 
    model: lotus.models.LM, 
    default: bool = True,
    expected_doc_count: int | None = None
) -> tuple[list[bool], list[str], list[str]]:
    """
    Parse batch filter responses from the model with robust error handling.
    
    This parser implements a multi-level fallback strategy to handle various
    JSON format issues including incomplete JSON, missing delimiters, and
    malformed responses.
    
    Args:
        batch_outputs: List of batch responses from the model
        model: Language model instance
        default: Default value to use if parsing fails
        expected_doc_count: Expected total number of documents across all batches
        
    Returns:
        tuple: (outputs, raw_outputs, explanations)
            - outputs: List of boolean filter results
            - raw_outputs: List of raw response strings (one per batch)
            - explanations: List of reasoning strings
    """
    all_outputs = []
    all_raw_outputs = []
    all_explanations = []
    
    for batch_idx, batch_output in enumerate(batch_outputs):
        # Use enhanced parsing with multi-level fallback (Problem 2 fix)
        parsed_results = _parse_single_batch_filter_output(batch_output, default)
        
        lotus.logger.debug(
            f"Batch {batch_idx}: Parsed {len(parsed_results)} results from output"
        )
        
        # Extract outputs and explanations from parsed results
        for result in parsed_results:
            all_outputs.append(result["answer"])
            all_explanations.append(result["reasoning"])
        
        all_raw_outputs.append(batch_output)
    
    # Validate total result count if expected
    if expected_doc_count is not None and len(all_outputs) != expected_doc_count:
        lotus.logger.warning(
            f"Expected {expected_doc_count} outputs but got {len(all_outputs)}. "
            f"Adjusting results..."
        )
        all_outputs = _pad_or_truncate_results(all_outputs, expected_doc_count, default)
        all_explanations = _pad_or_truncate_results(all_explanations, expected_doc_count, "")
    
    return all_outputs, all_raw_outputs, all_explanations


def _parse_single_batch_filter_output(
    batch_output: str, 
    default: bool = True
) -> list[dict[str, Any]]:
    """
    Parse a single batch output using multi-level fallback strategy.
    
    This function implements a robust parsing strategy with 5 levels of fallback:
    1. Standard JSON parsing
    2. Auto-repair JSON format and retry
    3. Regex extraction of structured data
    4. Line-by-line boolean detection
    5. Return empty list if all methods fail
    
    Args:
        batch_output: Raw output string from the model for one batch
        default: Default boolean value to use when parsing is ambiguous
        
    Returns:
        List of dicts with keys: "document_id", "answer", "reasoning"
        Sorted by document_id
    """
    
    # === Level 1: Standard JSON parsing ===
    try:
        cleaned = _clean_markdown_wrapper(batch_output)
        parsed = json.loads(cleaned)
        
        if "results" in parsed and isinstance(parsed["results"], list):
            return _validate_and_sort_filter_results(parsed["results"], default)
        elif isinstance(parsed, list):
            return _validate_and_sort_filter_results(parsed, default)
    except json.JSONDecodeError:
        pass
    
    # === Level 2: Auto-repair JSON format ===
    try:
        fixed_json = _fix_filter_json_format(batch_output)
        parsed = json.loads(fixed_json)
        
        if "results" in parsed and isinstance(parsed["results"], list):
            return _validate_and_sort_filter_results(parsed["results"], default)
        elif isinstance(parsed, list):
            return _validate_and_sort_filter_results(parsed, default)
    except json.JSONDecodeError:
        lotus.logger.debug("JSON repair failed, trying regex extraction")
    
    # === Level 3: Regex extraction ===
    results = _extract_filter_results_with_regex(batch_output, default)
    if results:
        return results
    
    # === Level 4: Line-by-line parsing ===
    results = _parse_filter_line_by_line(batch_output, default)
    if results:
        return results
    
    # === Level 5: Complete failure ===
    lotus.logger.error(
        f"All parsing methods failed for batch output. "
        f"Output preview: {batch_output[:200]}..."
    )
    return []


def _fix_filter_json_format(text: str) -> str:
    """
    Intelligently repair malformed JSON from filter batch responses.
    
    This function handles common JSON formatting issues:
    - Missing closing brackets/braces (}, ])
    - Missing commas between objects
    - Incomplete last result object
    - Extra text before/after JSON
    - Markdown code block wrappers
    
    Strategy:
    1. Extract complete result objects
    2. Extract partial objects and complete them
    3. Balance brackets/braces
    4. Reconstruct valid JSON from fragments
    
    Args:
        text: Raw text that should contain JSON
        
    Returns:
        Repaired JSON string
    """
    import re
    
    cleaned = _clean_markdown_wrapper(text)
    
    # === Strategy 1: Extract complete result objects ===
    # Match: {"document_id": N, "answer": true/false, "reasoning": "..."}
    complete_pattern = (
        r'\{\s*"document_id"\s*:\s*(\d+)\s*,\s*'
        r'"answer"\s*:\s*(true|false)\s*'
        r'(?:,\s*"reasoning"\s*:\s*"([^"]*)")?\s*\}'
    )
    
    matches = re.findall(complete_pattern, cleaned, re.IGNORECASE | re.DOTALL)
    
    if matches:
        results = [
            {
                "document_id": int(doc_id),
                "answer": answer.lower() == "true",
                "reasoning": reasoning or ""
            }
            for doc_id, answer, reasoning in matches
        ]
        return json.dumps({"results": results})
    
    # === Strategy 2: Extract partial objects and complete them ===
    # Match: {"document_id": N, "answer": true/false (may be incomplete after this)
    partial_pattern = r'\{\s*"document_id"\s*:\s*(\d+)\s*,\s*"answer"\s*:\s*(true|false)'
    
    partial_matches = re.findall(partial_pattern, cleaned, re.IGNORECASE)
    
    if partial_matches:
        results = []
        for doc_id, answer in partial_matches:
            # Try to find corresponding reasoning for this doc_id
            reasoning_pattern = (
                rf'"document_id"\s*:\s*{doc_id}.*?'
                r'"reasoning"\s*:\s*"([^"]*)"'
            )
            reasoning_match = re.search(
                reasoning_pattern, 
                cleaned, 
                re.IGNORECASE | re.DOTALL
            )
            reasoning = reasoning_match.group(1) if reasoning_match else ""
            
            results.append({
                "document_id": int(doc_id),
                "answer": answer.lower() == "true",
                "reasoning": reasoning
            })
        
        return json.dumps({"results": results})
    
    # === Strategy 3: Balance brackets and braces ===
    fixed = cleaned
    
    # Count and balance brackets/braces
    open_braces = fixed.count('{')
    close_braces = fixed.count('}')
    open_brackets = fixed.count('[')
    close_brackets = fixed.count(']')
    
    if open_braces > close_braces:
        fixed += '}' * (open_braces - close_braces)
    
    if open_brackets > close_brackets:
        fixed += ']' * (open_brackets - close_brackets)
    
    # Fix missing commas between objects: } { -> }, {
    fixed = re.sub(r'\}\s*\{', '}, {', fixed)
    
    # Try parsing the balanced version
    try:
        json.loads(fixed)
        return fixed
    except json.JSONDecodeError:
        pass
    
    # === Strategy 4: Extract from "results" array ===
    results_match = re.search(r'"results"\s*:\s*\[(.*)', cleaned, re.DOTALL)
    if results_match:
        results_content = results_match.group(1)
        
        # Extract all document_id and answer pairs from the content
        doc_answer_pattern = (
            r'"document_id"\s*:\s*(\d+).*?'
            r'"answer"\s*:\s*(true|false)'
        )
        doc_answers = re.findall(
            doc_answer_pattern, 
            results_content, 
            re.IGNORECASE | re.DOTALL
        )
        
        if doc_answers:
            results = [
                {
                    "document_id": int(doc_id),
                    "answer": answer.lower() == "true",
                    "reasoning": ""
                }
                for doc_id, answer in doc_answers
            ]
            return json.dumps({"results": results})
    
    # If all strategies fail, return the balanced version
    return fixed


def _extract_filter_results_with_regex(
    text: str, 
    default: bool = True
) -> list[dict[str, Any]]:
    """
    Extract filter results using regex patterns even if JSON is invalid.
    
    This function tries multiple regex patterns to extract document_id and
    answer pairs, supporting various format variations the model might produce.
    
    Args:
        text: Raw text to extract from
        default: Default boolean value
        
    Returns:
        List of result dicts, or empty list if extraction fails
    """
    import re
    
    results = []
    
    # Try multiple patterns in order of specificity
    patterns = [
        # Pattern 1: Standard JSON-like format
        (r'"document_id"\s*:\s*(\d+).*?"answer"\s*:\s*(true|false)', re.IGNORECASE | re.DOTALL),
        # Pattern 2: Simplified format without quotes
        (r'document_id\s*:\s*(\d+).*?answer\s*:\s*(true|false)', re.IGNORECASE | re.DOTALL),
        # Pattern 3: Document N: true/false format
        (r'[Dd]ocument\s+(\d+)\s*:?\s*(true|false)', re.IGNORECASE),
        # Pattern 4: Doc N: true/false format
        (r'[Dd]oc\s+(\d+)\s*:?\s*(true|false)', re.IGNORECASE),
    ]
    
    for pattern, flags in patterns:
        matches = re.findall(pattern, text, flags)
        if matches:
            results = [
                {
                    "document_id": int(doc_id),
                    "answer": answer.lower() == "true",
                    "reasoning": ""
                }
                for doc_id, answer in matches
            ]
            break
    
    return _validate_and_sort_filter_results(results, default) if results else []


def _parse_filter_line_by_line(
    text: str, 
    default: bool = True
) -> list[dict[str, Any]]:
    """
    Parse filter results line-by-line as last resort fallback.
    
    This method looks for boolean values (true/false) in each line and
    assigns sequential document IDs. It's the most lenient parsing method.
    
    Args:
        text: Raw text to parse
        default: Default boolean value
        
    Returns:
        List of result dicts
    """
    results = []
    lines = text.split('\n')
    doc_id = 1
    
    for line in lines:
        line_lower = line.lower().strip()
        
        # Skip empty lines and JSON structure lines
        if not line_lower or line_lower in ['{', '}', '[', ']', '"results":', '}}', ']}', ',']:
            continue
        
        # Look for boolean values
        has_true = 'true' in line_lower
        has_false = 'false' in line_lower
        
        if has_true and not has_false:
            results.append({
                "document_id": doc_id,
                "answer": True,
                "reasoning": line.strip()
            })
            doc_id += 1
        elif has_false and not has_true:
            results.append({
                "document_id": doc_id,
                "answer": False,
                "reasoning": line.strip()
            })
            doc_id += 1
    
    return results


def _validate_and_sort_filter_results(
    results: list[dict[str, Any]], 
    default: bool = True
) -> list[dict[str, Any]]:
    """
    Validate and sort filter results to ensure consistency.
    
    Operations performed:
    - Filter out results without document_id
    - Parse answer field to boolean
    - Sort by document_id
    - Deduplicate (keep first occurrence for duplicate IDs)
    - Ensure all required fields exist
    
    Args:
        results: Raw list of result dicts
        default: Default boolean value for unparseable answers
        
    Returns:
        Cleaned and sorted list of result dicts
    """
    # Filter and validate results
    valid_results = []
    for r in results:
        if "document_id" not in r:
            continue
        
        # Ensure document_id is int
        try:
            doc_id = int(r["document_id"])
        except (ValueError, TypeError):
            continue
        
        # Parse answer to boolean
        answer_value = r.get("answer", default)
        if isinstance(answer_value, bool):
            answer = answer_value
        elif isinstance(answer_value, str):
            answer = answer_value.lower() in ["true", "yes", "1"]
        elif isinstance(answer_value, (int, float)):
            answer = bool(answer_value)
        else:
            answer = default
        
        # Ensure reasoning exists
        reasoning = r.get("reasoning", "")
        if reasoning is None:
            reasoning = ""
        
        valid_results.append({
            "document_id": doc_id,
            "answer": answer,
            "reasoning": str(reasoning)
        })
    
    # Sort by document_id
    valid_results.sort(key=lambda x: x["document_id"])
    
    # Deduplicate by document_id (keep first occurrence)
    seen_ids = set()
    deduped_results = []
    for r in valid_results:
        doc_id = r["document_id"]
        if doc_id not in seen_ids:
            seen_ids.add(doc_id)
            deduped_results.append(r)
        else:
            lotus.logger.warning(
                f"Duplicate document_id {doc_id} found, keeping first occurrence"
            )
    
    return deduped_results


def _clean_markdown_wrapper(text: str) -> str:
    """
    Remove markdown code block wrappers from text.
    
    Handles patterns like:
    - ```json ... ```
    - ``` ... ```
    - ```JSON ... ```
    
    Args:
        text: Raw text potentially wrapped in markdown
        
    Returns:
        Cleaned text without markdown wrappers
    """
    import re
    
    cleaned = text.strip()
    
    # Remove opening markdown code blocks
    patterns = [
        (r'^```json\s*', ''),
        (r'^```JSON\s*', ''),
        (r'^```\s*', ''),
    ]
    
    for pattern, replacement in patterns:
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
    
    # Remove closing markdown code blocks
    cleaned = re.sub(r'\s*```$', '', cleaned)
    
    return cleaned.strip()


def batch_map_parser(
    batch_outputs: list[str], 
    model: lotus.models.LM,
    expected_doc_count: int | None = None
) -> tuple[list[str], list[str], list[str]]:
    """
    Parse batch map responses from the model.
    
    Args:
        batch_outputs: List of batch responses from the model
        model: Language model instance
        expected_doc_count: Expected total number of documents
        
    Returns:
        tuple: (outputs, raw_outputs, explanations)
    """
    all_outputs = []
    all_raw_outputs = []
    all_explanations = []
    
    for batch_idx, batch_output in enumerate(batch_outputs):
        print(f"\n=== DEBUG: Batch {batch_idx + 1} Raw Output ===")
        print(f"Length: {len(batch_output)} characters")
        print(f"Content: {batch_output}")
        print("=" * 50)
        
        try:
            # Clean the output - remove markdown code blocks if present
            cleaned_output = batch_output.strip()
            if cleaned_output.startswith("```json"):
                cleaned_output = cleaned_output[7:]  # Remove ```json
            if cleaned_output.endswith("```"):
                cleaned_output = cleaned_output[:-3]  # Remove ```
            cleaned_output = cleaned_output.strip()
            
            print(f"Cleaned output: {cleaned_output}")
            
            # Try to fix common JSON issues
            cleaned_output = _fix_json_format(cleaned_output)
            
            print(f"Fixed output: {cleaned_output}")
            
            # Try to parse JSON format
            parsed = json.loads(cleaned_output)
            if "results" in parsed and isinstance(parsed["results"], list):
                batch_results = parsed["results"]
                
                # Sort results by document_id to ensure correct order
                batch_results.sort(key=lambda x: x.get("document_id", 0))
                
                print(f"Batch {batch_idx + 1}: Successfully parsed {len(batch_results)} results")
                for i, result in enumerate(batch_results):
                    print(f"  Result {i+1}: doc_id={result.get('document_id', 'N/A')}, answer='{result.get('answer', 'N/A')[:50]}...'")
                
                # Extract results for each document
                for result in batch_results:
                    if "answer" in result:
                        all_outputs.append(str(result["answer"]))
                        reasoning = result.get("reasoning", "")
                        all_explanations.append(reasoning)
                    else:
                        all_outputs.append("")
                        all_explanations.append("")
            else:
                # Not expected JSON format, use fallback parsing
                print(f"Batch {batch_idx + 1}: Unexpected JSON format, using fallback")
                outputs, explanations = _parse_fallback_map(batch_output)
                all_outputs.extend(outputs)
                all_explanations.extend(explanations)
                
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            # JSON parsing failed, use fallback method
            print(f"Batch {batch_idx + 1}: JSON parsing failed: {e}")
            print(f"Raw output: {batch_output[:200]}...")
            outputs, explanations = _parse_fallback_map(batch_output)
            all_outputs.extend(outputs)
            all_explanations.extend(explanations)
        
        all_raw_outputs.append(batch_output)
    
    # Validate total result count if expected
    if expected_doc_count is not None and len(all_outputs) != expected_doc_count:
        print(f"Warning: Expected {expected_doc_count} outputs, got {len(all_outputs)}")
        all_outputs = _pad_or_truncate_results(all_outputs, expected_doc_count, "")
        all_explanations = _pad_or_truncate_results(all_explanations, expected_doc_count, "")
    
    return all_outputs, all_raw_outputs, all_explanations


def _parse_fallback_filter(text: str, default: bool) -> tuple[list[bool], list[str]]:
    """Fallback parsing method for filter when JSON parsing fails"""
    outputs = []
    explanations = []
    
    # Try to extract boolean values from text
    lines = text.split('\n')
    for line in lines:
        if 'true' in line.lower() or 'false' in line.lower():
            outputs.append('true' in line.lower())
            explanations.append(line.strip())
        elif 'yes' in line.lower() or 'no' in line.lower():
            outputs.append('yes' in line.lower())
            explanations.append(line.strip())
    
    if not outputs:
        # If can't parse, return default value
        outputs = [default]
        explanations = [text]
    
    return outputs, explanations


def _fix_json_format(text: str) -> str:
    """Fix common JSON formatting issues with improved robustness."""
    import re
    
    # First, clean markdown code blocks
    cleaned_text = text.strip()
    if cleaned_text.startswith("```json"):
        cleaned_text = cleaned_text[7:]
    if cleaned_text.endswith("```"):
        cleaned_text = cleaned_text[:-3]
    cleaned_text = cleaned_text.strip()
    
    # Look for complete result objects with proper structure
    # This pattern looks for complete JSON objects with document_id and answer
    result_pattern = r'\{\s*"document_id":\s*\d+,\s*"answer":\s*"[^"]*",\s*"reasoning":\s*"[^"]*"\s*\}'
    matches = re.findall(result_pattern, cleaned_text, re.DOTALL)
    
    if matches:
        # Reconstruct JSON with only complete results
        fixed_json = '{\n    "results": [\n'
        for i, match in enumerate(matches):
            if i > 0:
                fixed_json += ',\n'
            fixed_json += '        ' + match
        fixed_json += '\n    ]\n}'
        return fixed_json
    
    # Try to find document_id patterns and match them with answers
    doc_id_pattern = r'"document_id":\s*(\d+)'
    answer_pattern = r'"answer":\s*"([^"]*)"'
    
    doc_ids = re.findall(doc_id_pattern, cleaned_text)
    answers = re.findall(answer_pattern, cleaned_text)
    
    if doc_ids and answers:
        # Match document IDs with answers, handling cases where counts might differ
        fixed_json = '{\n    "results": [\n'
        
        # Create a mapping of doc_id to answer
        doc_answer_map = {}
        for i, doc_id in enumerate(doc_ids):
            if i < len(answers):
                doc_answer_map[int(doc_id)] = answers[i]
        
        # Also handle cases where we have answers without explicit doc_ids
        for i, answer in enumerate(answers):
            doc_id = i + 1
            if doc_id not in doc_answer_map:
                doc_answer_map[doc_id] = answer
        
        # Sort by document ID and create JSON
        for i, doc_id in enumerate(sorted(doc_answer_map.keys())):
            if i > 0:
                fixed_json += ',\n'
            answer = doc_answer_map[doc_id]
            fixed_json += f'        {{"document_id": {doc_id}, "answer": "{answer}", "reasoning": ""}}'
        
        fixed_json += '\n    ]\n}'
        return fixed_json
    
    # If no complete results found, try to extract partial results and fix them
    # Look for answer fields even in incomplete JSON
    answer_pattern = r'"answer":\s*"([^"]*)"'
    answers = re.findall(answer_pattern, cleaned_text)
    
    if answers:
        # Reconstruct JSON with extracted answers
        fixed_json = '{\n    "results": [\n'
        for i, answer in enumerate(answers):
            if i > 0:
                fixed_json += ',\n'
            fixed_json += f'        {{"document_id": {i+1}, "answer": "{answer}", "reasoning": ""}}'
        fixed_json += '\n    ]\n}'
        return fixed_json
    
    # Try to find any text that looks like responses (fallback)
    # Look for lines that might contain answers
    lines = cleaned_text.split('\n')
    potential_answers = []
    
    for line in lines:
        line = line.strip()
        # Skip JSON structure lines
        if line.startswith('{') or line.startswith('}') or line.startswith('[') or line.startswith(']') or '"results":' in line:
            continue
        # Look for lines that might be answers (not empty, not just punctuation)
        if line and len(line) > 3 and not line.startswith('"') and not line.endswith('"'):
            potential_answers.append(line)
    
    if potential_answers:
        # Use potential answers as responses
        fixed_json = '{\n    "results": [\n'
        for i, answer in enumerate(potential_answers):
            if i > 0:
                fixed_json += ',\n'
            # Escape quotes in the answer
            escaped_answer = answer.replace('"', '\\"')
            fixed_json += f'        {{"document_id": {i+1}, "answer": "{escaped_answer}", "reasoning": ""}}'
        fixed_json += '\n    ]\n}'
        return fixed_json
    
    # If still no results, try to close incomplete JSON structure
    if '"results":' in cleaned_text and not cleaned_text.strip().endswith('}'):
        lines = cleaned_text.split('\n')
        fixed_lines = []
        in_results = False
        result_count = 0
        
        for line in lines:
            if '"results":' in line:
                in_results = True
                fixed_lines.append(line)
            elif in_results and line.strip().startswith('{'):
                # Check if this line contains an answer
                if '"answer"' in line:
                    result_count += 1
                    # Fix the line if it's incomplete
                    if not line.strip().endswith('}'):
                        # Try to complete the line
                        if '"reasoning"' not in line:
                            line = line.rstrip(',') + ', "reasoning": ""}'
                        else:
                            line = line.rstrip(',') + '}'
                    fixed_lines.append(line)
            elif in_results and line.strip() == '}':
                # End of results array
                fixed_lines.append(line)
                break
            elif in_results:
                fixed_lines.append(line)
            else:
                fixed_lines.append(line)
        
        # Close the JSON structure
        if not fixed_lines[-1].strip().endswith('}'):
            fixed_lines.append('    ]')
            fixed_lines.append('}')
        
        return '\n'.join(fixed_lines)
    
    return cleaned_text


def _parse_fallback_map(text: str) -> tuple[list[str], list[str]]:
    """Fallback parsing method for map when JSON parsing fails"""
    outputs = []
    explanations = []
    
    # Try to extract JSON-like structure even if not valid JSON
    import re
    
    # Look for "answer" fields in the text
    answer_pattern = r'"answer":\s*"([^"]*)"'
    answers = re.findall(answer_pattern, text)
    
    if answers:
        # Found answer fields, use them
        for answer in answers:
            outputs.append(answer)
            explanations.append("")
    else:
        # Fallback to line-by-line parsing
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith('{') and not line.startswith('}') and not line.startswith('[') and not line.startswith(']'):
                outputs.append(line)
                explanations.append("")
    
    if not outputs:
        outputs = [text]
        explanations = [""]
    
    return outputs, explanations


def _pad_or_truncate_results(results: list, expected_count: int, default_value) -> list:
    """Pad or truncate results to match expected count"""
    if len(results) < expected_count:
        # Pad with default values
        results.extend([default_value] * (expected_count - len(results)))
    elif len(results) > expected_count:
        # Truncate
        results = results[:expected_count]
    
    return results


def batch_extract_parser(
    batch_outputs: list[str], 
    model: lotus.models.LM, 
    expected_doc_count: int | None = None
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """
    Parse batch extract responses from the model.
    
    Args:
        batch_outputs: List of batch responses from the model
        model: Language model instance
        expected_doc_count: Expected total number of documents
        
    Returns:
        tuple: (outputs, raw_outputs, explanations)
    """
    all_outputs = []
    all_raw_outputs = []
    all_explanations = []
    
    for batch_idx, batch_output in enumerate(batch_outputs):
        try:
            # Clean the output - remove markdown code blocks if present
            cleaned_output = batch_output.strip()
            if cleaned_output.startswith("```json"):
                cleaned_output = cleaned_output[7:]  # Remove ```json
            if cleaned_output.endswith("```"):
                cleaned_output = cleaned_output[:-3]  # Remove ```
            cleaned_output = cleaned_output.strip()
            
            # Try to fix common JSON issues
            cleaned_output = _fix_extract_json_format(cleaned_output)
            
            # Try to parse JSON format
            parsed = json.loads(cleaned_output)
            
            if isinstance(parsed, list):
                # Direct array format
                batch_results = parsed
            elif "results" in parsed and isinstance(parsed["results"], list):
                # Results wrapper format
                batch_results = parsed["results"]
            else:
                # Not expected JSON format, use fallback parsing
                outputs, explanations = _parse_fallback_extract(batch_output)
                all_outputs.extend(outputs)
                all_explanations.extend(explanations)
                all_raw_outputs.append(batch_output)
                continue
            
            # Sort results by document_id to ensure correct order
            batch_results.sort(key=lambda x: x.get("document_id", 0))
            
            # Extract results for each document
            for result in batch_results:
                if isinstance(result, dict):
                    # Remove document_id from the final output
                    output_dict = {k: v for k, v in result.items() if k != "document_id"}
                    all_outputs.append(output_dict)
                    
                    # Extract reasoning if available
                    reasoning = result.get("reasoning", "")
                    all_explanations.append(reasoning)
                else:
                    # Handle non-dict results
                    all_outputs.append({})
                    all_explanations.append("")
                
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            # JSON parsing failed, use fallback method
            outputs, explanations = _parse_fallback_extract(batch_output)
            all_outputs.extend(outputs)
            all_explanations.extend(explanations)
        
        all_raw_outputs.append(batch_output)
    
    # Validate total result count if expected
    if expected_doc_count is not None and len(all_outputs) != expected_doc_count:
        all_outputs = _pad_or_truncate_results(all_outputs, expected_doc_count, {})
        all_explanations = _pad_or_truncate_results(all_explanations, expected_doc_count, "")
    
    return all_outputs, all_raw_outputs, all_explanations


def _fix_extract_json_format(text: str) -> str:
    """Fix common JSON formatting issues for extract operations."""
    import re
    
    # First, clean markdown code blocks
    cleaned_text = text.strip()
    if cleaned_text.startswith("```json"):
        cleaned_text = cleaned_text[7:]
    if cleaned_text.endswith("```"):
        cleaned_text = cleaned_text[:-3]
    cleaned_text = cleaned_text.strip()
    
    # Look for complete result objects with proper structure
    # This pattern looks for complete JSON objects with document_id
    result_pattern = r'\{\s*"document_id":\s*\d+,[^}]*\}'
    matches = re.findall(result_pattern, cleaned_text, re.DOTALL)
    
    if matches:
        # Reconstruct JSON with only complete results
        fixed_json = '[\n'
        for i, match in enumerate(matches):
            if i > 0:
                fixed_json += ',\n'
            fixed_json += '    ' + match
        fixed_json += '\n]'
        return fixed_json
    
    # Try to find document_id patterns and match them with field data
    doc_id_pattern = r'"document_id":\s*(\d+)'
    doc_ids = re.findall(doc_id_pattern, cleaned_text)
    
    if doc_ids:
        # Try to extract field data for each document
        # Look for patterns like "field_name": "value"
        field_pattern = r'"([^"]+)":\s*"([^"]*)"'
        field_matches = re.findall(field_pattern, cleaned_text)
        
        if field_matches:
            # Group fields by document (this is a simplified approach)
            fixed_json = '[\n'
            for i, doc_id in enumerate(doc_ids):
                if i > 0:
                    fixed_json += ',\n'
                fixed_json += f'    {{"document_id": {doc_id}'
                
                # Add available fields
                for field_name, field_value in field_matches:
                    if field_name != "document_id":
                        fixed_json += f', "{field_name}": "{field_value}"'
                
                fixed_json += '}'
            fixed_json += '\n]'
            return fixed_json
    
    return cleaned_text


def _parse_fallback_extract(text: str) -> tuple[list[dict[str, Any]], list[str]]:
    """Fallback parsing method for extract when JSON parsing fails"""
    outputs = []
    explanations = []
    
    # Try to extract JSON-like structure even if not valid JSON
    import re
    
    # Look for field patterns in the text
    field_pattern = r'"([^"]+)":\s*"([^"]*)"'
    field_matches = re.findall(field_pattern, text)
    
    if field_matches:
        # Group fields into a single result
        result_dict = {}
        for field_name, field_value in field_matches:
            if field_name != "document_id":
                result_dict[field_name] = field_value
        
        if result_dict:
            outputs.append(result_dict)
            explanations.append("")
        else:
            outputs.append({})
            explanations.append("")
    else:
        # Fallback to empty result
        outputs.append({})
        explanations.append("")
    
    return outputs, explanations