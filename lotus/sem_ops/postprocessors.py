import json
from typing import Callable

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
    Parse batch filter responses from the model.
    
    Args:
        batch_outputs: List of batch responses from the model
        model: Language model instance
        default: Default value to use if parsing fails
        expected_doc_count: Expected total number of documents
        
    Returns:
        tuple: (outputs, raw_outputs, explanations)
    """
    all_outputs = []
    all_raw_outputs = []
    all_explanations = []
    
    for batch_output in batch_outputs:
        try:
            # Try to parse JSON format
            parsed = json.loads(batch_output)
            if "results" in parsed and isinstance(parsed["results"], list):
                batch_results = parsed["results"]
                
                # Extract results for each document
                for result in batch_results:
                    if "answer" in result:
                        # Parse boolean value
                        answer = result["answer"]
                        if isinstance(answer, bool):
                            all_outputs.append(answer)
                        elif isinstance(answer, str):
                            all_outputs.append(answer.lower() in ["true", "yes", "1"])
                        else:
                            all_outputs.append(default)
                        
                        # Extract reasoning
                        reasoning = result.get("reasoning", "")
                        all_explanations.append(reasoning)
                    else:
                        all_outputs.append(default)
                        all_explanations.append("")
            else:
                # Not expected JSON format, use fallback parsing
                outputs, explanations = _parse_fallback_filter(batch_output, default)
                all_outputs.extend(outputs)
                all_explanations.extend(explanations)
                
        except (json.JSONDecodeError, KeyError, TypeError):
            # JSON parsing failed, use fallback method
            outputs, explanations = _parse_fallback_filter(batch_output, default)
            all_outputs.extend(outputs)
            all_explanations.extend(explanations)
        
        all_raw_outputs.append(batch_output)
    
    # Validate total result count if expected
    if expected_doc_count is not None and len(all_outputs) != expected_doc_count:
        all_outputs = _pad_or_truncate_results(all_outputs, expected_doc_count, default)
        all_explanations = _pad_or_truncate_results(all_explanations, expected_doc_count, "")
    
    return all_outputs, all_raw_outputs, all_explanations


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