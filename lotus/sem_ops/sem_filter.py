import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

import lotus
from lotus.cache import operator_cache
from lotus.templates import task_instructions
from lotus.types import (
    CascadeArgs,
    LMOutput,
    LogprobsForFilterCascade,
    ProxyModel,
    ReasoningStrategy,
    SemanticFilterOutput,
)
from lotus.utils import show_safe_mode

from .cascade_utils import calibrate_llm_logprobs, importance_sampling, learn_cascade_thresholds
from .postprocessors import filter_postprocess


def sem_filter(
    docs: list[dict[str, Any]],
    model: lotus.models.LM,
    user_instruction: str,
    default: bool = True,
    examples_multimodal_data: list[dict[str, Any]] | None = None,
    examples_answers: list[bool] | None = None,
    cot_reasoning: list[str] | None = None,
    strategy: ReasoningStrategy | None = None,
    logprobs: bool = False,
    safe_mode: bool = False,
    show_progress_bar: bool = True,
    progress_bar_desc: str = "Filtering",
    additional_cot_instructions: str = "",
    use_async: bool = False,
    max_concurrent_batches: int = 4,
    max_thread_workers: int = 8,
    # New batch processing parameters
    batch_size: int = 10,
    use_batch_processing: bool = True,
    # Image compression parameters
    enable_image_compression: bool = True,
    image_compression_strategy: str = "advanced",  # "simple" or "advanced"
    image_max_size: tuple[int, int] = (1024, 1024),
    image_quality: int = 85,
    image_format: str = "JPEG",
) -> SemanticFilterOutput:
    """
    Filters a list of documents based on a natural language instruction using a language model.

    This function applies a natural language filter condition to each document in the
    input list, returning boolean values indicating whether each document passes the filter.
    It supports few-shot learning through examples and various reasoning strategies.
    Can use async processing and batch processing for better performance with large datasets.

    Args:
        docs (list[dict[str, Any]]): The list of documents to filter. Each document
            should be a dictionary containing multimodal information (text, images, etc.).
        model (lotus.models.LM): The language model instance to use for filtering.
            Must be properly configured with appropriate API keys and settings.
        user_instruction (str): The natural language instruction that defines the
            filter condition. Should describe what criteria documents must meet.
        default (bool, optional): The default value to use when the model output
            cannot be parsed as a boolean. Defaults to True.
        examples_multimodal_data (list[dict[str, Any]] | None, optional): Example
            documents for few-shot learning. Each example should have the same
            structure as the input docs. Defaults to None.
        examples_answers (list[bool] | None, optional): Expected boolean outputs for
            the example documents. Should have the same length as examples_multimodal_data.
            Defaults to None.
        cot_reasoning (list[str] | None, optional): Chain-of-thought reasoning
            for the example documents. Used when strategy includes COT reasoning.
            Defaults to None.
        strategy (ReasoningStrategy | None, optional): The reasoning strategy to use.
            Can be None, COT, or ZS_COT. Defaults to None.
        logprobs (bool, optional): Whether to return log probabilities for the
            model outputs. Useful for confidence estimation. Defaults to False.
        safe_mode (bool, optional): Whether to enable safe mode with cost estimation.
            Defaults to False.
        show_progress_bar (bool, optional): Whether to show a progress bar during
            processing. Defaults to True.
        progress_bar_desc (str, optional): Description for the progress bar.
            Defaults to "Filtering".
        additional_cot_instructions (str, optional): Additional instructions for
            chain-of-thought reasoning. Defaults to "".
        use_async (bool, optional): Whether to use async processing for better
            performance. Defaults to False.
        max_concurrent_batches (int, optional): Maximum number of concurrent batches
            to process when using async. Defaults to 4.
        max_thread_workers (int, optional): Maximum number of threads for CPU-intensive
            operations when using async. Defaults to 8.
        batch_size (int, optional): Number of documents to process in each batch
            when using batch processing. Defaults to 10.
        use_batch_processing (bool, optional): Whether to use batch processing to
            share system prompts and examples across documents. Defaults to True.
        enable_image_compression (bool, optional): Whether to enable image compression
            for multimodal documents. Defaults to True.
        image_compression_strategy (str, optional): Compression strategy to use.
            Can be "simple" for fast processing or "advanced" for better compression.
            Defaults to "advanced".
        image_max_size (tuple[int, int], optional): Maximum image dimensions
            for compression. Defaults to (1024, 1024).
        image_quality (int, optional): JPEG quality for compression (1-100).
            Defaults to 85.
        image_format (str, optional): Output format for compressed images.
            Can be "JPEG", "PNG", or "WEBP". Defaults to "JPEG".

    Returns:
        SemanticFilterOutput: An object containing the boolean filter outputs, raw
            outputs, explanations (if applicable), and log probabilities (if requested).

    Raises:
        ValueError: If the model is not properly configured or if there are
            issues with the input parameters.

    Example:
        >>> docs = [{"text": "Positive review"}, {"text": "Negative review"}]
        >>> model = LM(model="gpt-4o")
        >>> result = sem_filter(docs, model, "Is this a positive sentiment?")
        >>> print(result.outputs)  # [True, False]
        
        # Using batch processing for better performance
        >>> result = sem_filter(docs, model, "Is this a positive sentiment?", 
        ...                     use_batch_processing=True, batch_size=5)
        >>> print(result.outputs)  # [True, False]
    """
    # 设置图片压缩配置
    try:
        from lotus.utils.image_compression_config import set_global_config
        set_global_config(
            enable_compression=enable_image_compression,
            strategy=image_compression_strategy,
            max_size=image_max_size,
            quality=image_quality,
            format=image_format
        )
    except ImportError:
        # 如果配置管理器不可用，跳过配置
        pass
    
    # Choose between sync and async processing
    if use_async:
        return asyncio.run(sem_filter_async(
            docs, model, user_instruction, default, examples_multimodal_data,
            examples_answers, cot_reasoning, strategy, logprobs, safe_mode,
            show_progress_bar, progress_bar_desc, additional_cot_instructions,
            max_concurrent_batches, max_thread_workers,
            # Image compression parameters
            enable_image_compression, image_compression_strategy, image_max_size,
            image_quality, image_format
        ))

    # Choose between batch and individual processing
    if use_batch_processing and len(docs) > 1:
        return _sem_filter_batch(
            docs, model, user_instruction, default, examples_multimodal_data,
            examples_answers, cot_reasoning, strategy, logprobs, safe_mode,
            show_progress_bar, progress_bar_desc, additional_cot_instructions,
            batch_size,
            # Image compression parameters
            enable_image_compression, image_compression_strategy, image_max_size,
            image_quality, image_format
        )
    else:
        return _sem_filter_individual(
            docs, model, user_instruction, default, examples_multimodal_data,
            examples_answers, cot_reasoning, strategy, logprobs, safe_mode,
            show_progress_bar, progress_bar_desc, additional_cot_instructions,
            # Image compression parameters
            enable_image_compression, image_compression_strategy, image_max_size,
            image_quality, image_format
        )


def _sem_filter_batch(
    docs: list[dict[str, Any]],
    model: lotus.models.LM,
    user_instruction: str,
    default: bool = True,
    examples_multimodal_data: list[dict[str, Any]] | None = None,
    examples_answers: list[bool] | None = None,
    cot_reasoning: list[str] | None = None,
    strategy: ReasoningStrategy | None = None,
    logprobs: bool = False,
    safe_mode: bool = False,
    show_progress_bar: bool = True,
    progress_bar_desc: str = "Filtering",
    additional_cot_instructions: str = "",
    batch_size: int = 10,
    max_retries: int = 2,
    # Image compression parameters
    enable_image_compression: bool = True,
    image_compression_strategy: str = "advanced",
    image_max_size: tuple[int, int] = (1024, 1024),
    image_quality: int = 85,
    image_format: str = "JPEG",
) -> SemanticFilterOutput:
    """
    Batch processing implementation for sem_filter with robust error handling.
    
    This implementation includes:
    - Document mapping to track batch-local IDs to original indices
    - Missing document detection and retry mechanism
    - Robust JSON parsing with multiple fallback strategies
    - Final validation and result ordering
    
    Args:
        docs: List of documents to filter
        model: Language model instance
        user_instruction: Filter instruction/claim
        default: Default value for unparseable results
        examples_multimodal_data: Example documents for few-shot learning
        examples_answers: Expected boolean outputs for examples
        cot_reasoning: Chain-of-thought reasoning for examples
        strategy: Reasoning strategy to use
        logprobs: Whether to return log probabilities
        safe_mode: Whether to enable safe mode with cost estimation
        show_progress_bar: Whether to show progress bar
        progress_bar_desc: Description for progress bar
        additional_cot_instructions: Additional CoT instructions
        batch_size: Number of documents per batch
        max_retries: Maximum number of retry attempts for missing documents
        
    Returns:
        SemanticFilterOutput with results in original document order
    """
    from .postprocessors import batch_filter_parser, _parse_single_batch_filter_output
    
    try:
        total_docs = len(docs)
        
        # Step 1: Build document mapping (Problem 1 fix)
        # This tracks the relationship between (batch_idx, doc_id_in_batch) -> original_doc_idx
        mapping, expected_doc_ids_per_batch = _build_document_mapping(total_docs, batch_size)
        
        lotus.logger.debug(
            f"Processing {total_docs} documents in {len(expected_doc_ids_per_batch)} batches "
            f"with batch_size={batch_size}"
        )
        
        # Step 2: Format batch inputs
        batched_inputs = lotus.templates.task_instructions.batch_filter_formatter(
            model, docs, user_instruction, examples_multimodal_data,
            examples_answers, cot_reasoning, strategy,
            reasoning_instructions=additional_cot_instructions,
            batch_size=batch_size
        )
        
        lotus.logger.debug(f"Generated {len(batched_inputs)} batch inputs")
        
        kwargs: dict[str, Any] = {"logprobs": logprobs}
        
        if safe_mode:
            estimated_total_calls = len(batched_inputs)
            estimated_total_cost = sum(model.count_tokens(input) for input in batched_inputs)
            show_safe_mode(estimated_total_cost, estimated_total_calls)
        
        # Step 3: Call model with batch inputs
        lm_output: LMOutput = model(
            batched_inputs, 
            show_progress_bar=show_progress_bar, 
            progress_bar_desc=progress_bar_desc, 
            **kwargs
        )
        
        lotus.logger.debug(f"Received {len(lm_output.outputs)} batch outputs")
        
        # Step 4: Parse batch responses with mapping
        outputs, raw_outputs, explanations, missing_docs = _parse_batch_with_mapping(
            lm_output.outputs,
            mapping,
            expected_doc_ids_per_batch,
            total_docs,
            default
        )
        
        # Step 5: Handle missing documents with retry mechanism
        retry_count = 0
        while missing_docs and retry_count < max_retries:
            retry_count += 1
            total_missing = sum(len(doc_ids) for doc_ids in missing_docs.values())
            
            lotus.logger.warning(
                f"Retry {retry_count}/{max_retries}: Processing {total_missing} missing documents"
            )
            
            # Retry missing documents individually (more reliable)
            retry_results = _retry_missing_documents(
                docs=docs,
                missing_docs=missing_docs,
                mapping=mapping,
                model=model,
                user_instruction=user_instruction,
                default=default,
                examples_multimodal_data=examples_multimodal_data,
                examples_answers=examples_answers,
                cot_reasoning=cot_reasoning,
                strategy=strategy,
                show_progress_bar=show_progress_bar,
                progress_bar_desc=f"Retry {retry_count}",
                additional_cot_instructions=additional_cot_instructions,
            )
            
            # Merge retry results back
            for original_idx, result in retry_results.items():
                outputs[original_idx] = result["output"]
                raw_outputs[original_idx] = result["raw_output"]
                explanations[original_idx] = result["explanation"]
            
            # Check if there are still missing documents after retry
            # (This would happen if the retry also failed)
            missing_docs = {}  # Clear for next iteration
            for idx, output in enumerate(outputs):
                if output is None:
                    # Find which batch this belongs to
                    for (batch_idx, doc_id_in_batch), orig_idx in mapping.items():
                        if orig_idx == idx:
                            if batch_idx not in missing_docs:
                                missing_docs[batch_idx] = set()
                            missing_docs[batch_idx].add(doc_id_in_batch)
                            break
        
        # Step 6: Final validation and fallback
        # Replace any remaining None values with default
        for i in range(total_docs):
            if outputs[i] is None:
                lotus.logger.error(f"Document {i} still missing after {max_retries} retries, using default")
                outputs[i] = default
                raw_outputs[i] = raw_outputs[i] or ""
                explanations[i] = explanations[i] or ""
        
        # Verify output counts
        assert len(outputs) == total_docs, f"Output count mismatch: {len(outputs)} != {total_docs}"
        assert len(raw_outputs) == total_docs, f"Raw output count mismatch: {len(raw_outputs)} != {total_docs}"
        assert len(explanations) == total_docs, f"Explanation count mismatch: {len(explanations)} != {total_docs}"
        
        lotus.logger.debug(f"Final outputs: {outputs}")
        lotus.logger.debug(f"Final explanations count: {len(explanations)}")
        
        if safe_mode:
            model.print_total_usage()
        
        return SemanticFilterOutput(
            raw_outputs=raw_outputs,
            outputs=outputs,
            explanations=explanations,
            logprobs=lm_output.logprobs if logprobs else None,
        )
        
    except Exception as e:
        # Batch processing failed completely, fall back to individual processing
        lotus.logger.warning(f"Batch processing failed: {e}. Falling back to individual processing.")
        return _sem_filter_individual(
            docs, model, user_instruction, default, examples_multimodal_data,
            examples_answers, cot_reasoning, strategy, logprobs, safe_mode,
            show_progress_bar, progress_bar_desc, additional_cot_instructions
        )


def _sem_filter_individual(
    docs: list[dict[str, Any]],
    model: lotus.models.LM,
    user_instruction: str,
    default: bool = True,
    examples_multimodal_data: list[dict[str, Any]] | None = None,
    examples_answers: list[bool] | None = None,
    cot_reasoning: list[str] | None = None,
    strategy: ReasoningStrategy | None = None,
    logprobs: bool = False,
    safe_mode: bool = False,
    show_progress_bar: bool = True,
    progress_bar_desc: str = "Filtering",
    additional_cot_instructions: str = "",
    # Image compression parameters
    enable_image_compression: bool = True,
    image_compression_strategy: str = "advanced",
    image_max_size: tuple[int, int] = (1024, 1024),
    image_quality: int = 85,
    image_format: str = "JPEG",
) -> SemanticFilterOutput:
    """Individual processing implementation for sem_filter (original logic)."""
    inputs = []
    for doc in docs:
        prompt = lotus.templates.task_instructions.filter_formatter(
            model,
            doc,
            user_instruction,
            examples_multimodal_data,
            examples_answers,
            cot_reasoning,
            strategy,
            reasoning_instructions=additional_cot_instructions,
        )
        lotus.logger.debug(f"input to model: {prompt}")
        inputs.append(prompt)
    kwargs: dict[str, Any] = {"logprobs": logprobs}

    if safe_mode:
        estimated_total_calls = len(docs)
        estimated_total_cost = sum(model.count_tokens(input) for input in inputs)
        show_safe_mode(estimated_total_cost, estimated_total_calls)

    lm_output: LMOutput = model(
        inputs, show_progress_bar=show_progress_bar, progress_bar_desc=progress_bar_desc, **kwargs
    )

    postprocess_output = filter_postprocess(lm_output.outputs, model, default)
    lotus.logger.debug(f"outputs: {postprocess_output.outputs}")
    lotus.logger.debug(f"raw_outputs: {postprocess_output.raw_outputs}")
    lotus.logger.debug(f"explanations: {postprocess_output.explanations}")

    if safe_mode:
        model.print_total_usage()

    return SemanticFilterOutput(
        raw_outputs=postprocess_output.raw_outputs,
        outputs=postprocess_output.outputs,
        explanations=postprocess_output.explanations,
        logprobs=lm_output.logprobs if logprobs else None,
    )


async def sem_filter_async(
    docs: list[dict[str, Any]],
    model: lotus.models.LM,
    user_instruction: str,
    default: bool = True,
    examples_multimodal_data: list[dict[str, Any]] | None = None,
    examples_answers: list[bool] | None = None,
    cot_reasoning: list[str] | None = None,
    strategy: ReasoningStrategy | None = None,
    logprobs: bool = False,
    safe_mode: bool = False,
    show_progress_bar: bool = True,
    progress_bar_desc: str = "Filtering",
    additional_cot_instructions: str = "",
    max_concurrent_batches: int = 4,
    max_thread_workers: int = 8,
    # Image compression parameters
    enable_image_compression: bool = True,
    image_compression_strategy: str = "advanced",
    image_max_size: tuple[int, int] = (1024, 1024),
    image_quality: int = 85,
    image_format: str = "JPEG",
) -> SemanticFilterOutput:
    """
    Asynchronous version of semantic filtering with optimized concurrent processing.

    This function applies a natural language filter condition to each document in the
    input list, returning boolean values indicating whether each document passes the filter.
    The async version uses asyncio and thread pools for concurrent processing of batches
    and CPU-intensive operations.

    Args:
        docs (list[dict[str, Any]]): The list of documents to filter. Each document
            should be a dictionary containing multimodal information (text, images, etc.).
        model (lotus.models.LM): The language model instance to use for filtering.
            Must be properly configured with appropriate API keys and settings.
        user_instruction (str): The natural language instruction that defines the
            filter condition. Should describe what criteria documents must meet.
        default (bool, optional): The default value to use when the model output
            cannot be parsed as a boolean. Defaults to True.
        examples_multimodal_data (list[dict[str, Any]] | None, optional): Example
            documents for few-shot learning. Each example should have the same
            structure as the input docs. Defaults to None.
        examples_answers (list[bool] | None, optional): Expected boolean outputs for
            the example documents. Should have the same length as examples_multimodal_data.
            Defaults to None.
        cot_reasoning (list[str] | None, optional): Chain-of-thought reasoning
            for the example documents. Used when strategy includes COT reasoning.
            Defaults to None.
        strategy (ReasoningStrategy | None, optional): The reasoning strategy to use.
            Can be None, COT, or ZS_COT. Defaults to None.
        logprobs (bool, optional): Whether to return log probabilities for the
            model outputs. Useful for confidence estimation. Defaults to False.
        safe_mode (bool, optional): Whether to enable safe mode with cost estimation.
            Defaults to False.
        show_progress_bar (bool, optional): Whether to show a progress bar during
            processing. Defaults to True.
        progress_bar_desc (str, optional): Description for the progress bar.
            Defaults to "Filtering".
        additional_cot_instructions (str, optional): Additional instructions for
            chain-of-thought reasoning. Defaults to "".
        max_concurrent_batches (int, optional): Maximum number of concurrent batches
            to process. Defaults to 4.
        max_thread_workers (int, optional): Maximum number of threads for CPU-intensive
            operations. Defaults to 8.

    Returns:
        SemanticFilterOutput: An object containing the boolean filter outputs, raw
            outputs, explanations (if applicable), and log probabilities (if requested).

    Raises:
        ValueError: If the model is not properly configured or if there are
            issues with the input parameters.

    Example:
        >>> import asyncio
        >>> docs = [{"text": "Positive review"}, {"text": "Negative review"}]
        >>> model = LM(model="gpt-4o")
        >>> result = await sem_filter_async(docs, model, "Is this a positive sentiment?")
        >>> print(result.outputs)  # [True, False]
    """
    
    async def process_batch_async(batch: list[list[dict[str, str]]], semaphore: asyncio.Semaphore) -> LMOutput:
        """
        Process a batch of messages asynchronously with semaphore control.

        Args:
            batch (list[list[dict[str, str]]]): The batch of messages to process.
            semaphore (asyncio.Semaphore): Semaphore to control concurrent execution.

        Returns:
            LMOutput: The output from the language model.
        """
        async with semaphore:
            # Run the synchronous model call in a thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor(max_workers=1) as executor:
                return await loop.run_in_executor(
                    executor, 
                    lambda: model(batch, show_progress_bar=show_progress_bar, progress_bar_desc=progress_bar_desc, **kwargs)
                )

    async def count_tokens_async(text: str) -> int:
        """
        Count tokens asynchronously using thread pool.

        Args:
            text (str): The text to count tokens for.

        Returns:
            int: The number of tokens in the text.
        """
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            return await loop.run_in_executor(executor, model.count_tokens, text)

    # Prepare inputs
    inputs = []
    for doc in docs:
        prompt = lotus.templates.task_instructions.filter_formatter(
            model,
            doc,
            user_instruction,
            examples_multimodal_data,
            examples_answers,
            cot_reasoning,
            strategy,
            reasoning_instructions=additional_cot_instructions,
        )
        lotus.logger.debug(f"input to model: {prompt}")
        inputs.append(prompt)
    
    kwargs: dict[str, Any] = {"logprobs": logprobs}

    if safe_mode:
        # Count tokens asynchronously for cost estimation
        estimated_total_calls = len(docs)
        token_counts = await asyncio.gather(*[count_tokens_async(input_text) for input_text in inputs])
        estimated_total_cost = sum(token_counts)
        show_safe_mode(estimated_total_cost, estimated_total_calls)

    # Create semaphore to limit concurrent batches
    semaphore = asyncio.Semaphore(max_concurrent_batches)
    
    # Process inputs in batches
    batch_size = min(model.max_batch_size, len(inputs))
    batches = [inputs[i:i + batch_size] for i in range(0, len(inputs), batch_size)]
    
    # Process all batches concurrently with semaphore control
    if batches:
        # Process batches concurrently
        tasks = [process_batch_async(batch, semaphore) for batch in batches]
        lm_outputs = await asyncio.gather(*tasks)
        
        # Combine outputs from all batches
        all_outputs = []
        all_logprobs = []
        for lm_output in lm_outputs:
            all_outputs.extend(lm_output.outputs)
            if lm_output.logprobs:
                all_logprobs.extend(lm_output.logprobs)
        
        lm_output = LMOutput(outputs=all_outputs, logprobs=all_logprobs if logprobs else None)
    else:
        lm_output = LMOutput(outputs=[], logprobs=None)

    # Postprocess outputs
    postprocess_output = filter_postprocess(lm_output.outputs, model, default)
    lotus.logger.debug(f"outputs: {postprocess_output.outputs}")
    lotus.logger.debug(f"raw_outputs: {postprocess_output.raw_outputs}")
    lotus.logger.debug(f"explanations: {postprocess_output.explanations}")

    if safe_mode:
        model.print_total_usage()

    return SemanticFilterOutput(
        raw_outputs=postprocess_output.raw_outputs,
        outputs=postprocess_output.outputs,
        explanations=postprocess_output.explanations,
        logprobs=lm_output.logprobs if logprobs else None,
    )


def learn_filter_cascade_thresholds(
    sample_multimodal_data: list[dict[str, Any]],
    lm: lotus.models.LM,
    formatted_usr_instr: str,
    default: bool,
    cascade_args: CascadeArgs,
    proxy_scores: list[float],
    sample_correction_factors: NDArray[np.float64],
    examples_multimodal_data: list[dict[str, Any]] | None = None,
    examples_answers: list[bool] | None = None,
    cot_reasoning: list[str] | None = None,
    strategy: ReasoningStrategy | None = None,
    additional_cot_instructions: str = "",
) -> tuple[float, float]:
    """
    Automatically learns optimal cascade thresholds for filter operations.

    This function uses a sample of data to determine the best threshold values
    for cascade filtering, which combines a fast proxy model with a more accurate
    but slower language model. It searches across different threshold combinations
    to find the one that gives the best accuracy.

    Args:
        sample_multimodal_data (list[dict[str, Any]]): Sample documents to use
            for threshold learning. Should be representative of the full dataset.
        lm (lotus.models.LM): The language model to use as the oracle for
            determining ground truth labels.
        formatted_usr_instr (str): The formatted user instruction for filtering.
        default (bool): The default value to use when parsing fails.
        cascade_args (CascadeArgs): Configuration arguments for the cascade
            including recall target, precision target, sampling percentage, etc.
        proxy_scores (list[float]): Scores from the proxy model for each sample.
            Should have the same length as sample_multimodal_data.
        sample_correction_factors (NDArray[np.float64]): Correction factors for
            importance sampling. Should have the same length as sample_multimodal_data.
        examples_multimodal_data (list[dict[str, Any]] | None, optional): Example
            documents for few-shot learning. Defaults to None.
        examples_answers (list[bool] | None, optional): Expected boolean outputs
            for the example documents. Defaults to None.
        cot_reasoning (list[str] | None, optional): Chain-of-thought reasoning
            for the example documents. Defaults to None.
        strategy (ReasoningStrategy | None, optional): The reasoning strategy to use.
            Defaults to None.
        additional_cot_instructions (str, optional): Additional instructions for
            chain-of-thought reasoning. Defaults to "".

    Returns:
        tuple[float, float]: A tuple containing the learned low and high thresholds
            for the cascade filter.

    Raises:
        Exception: If there's an error during the threshold learning process.

    Example:
        >>> sample_data = [{"text": "doc1"}, {"text": "doc2"}]
        >>> proxy_scores = [0.8, 0.3]
        >>> thresholds = learn_filter_cascade_thresholds(
        ...     sample_data, model, "Is positive?", True, cascade_args,
        ...     proxy_scores, correction_factors
        ... )
        >>> print(thresholds)  # (0.3, 0.8)
    """

    try:
        large_outputs = sem_filter(
            sample_multimodal_data,
            lm,
            formatted_usr_instr,
            default=default,
            examples_multimodal_data=examples_multimodal_data,
            examples_answers=examples_answers,
            cot_reasoning=cot_reasoning,
            strategy=strategy,
            safe_mode=False,
            progress_bar_desc="Running oracle for threshold learning",
            additional_cot_instructions=additional_cot_instructions,
        ).outputs

        best_combination, _ = learn_cascade_thresholds(
            proxy_scores=proxy_scores,
            oracle_outputs=large_outputs,
            sample_correction_factors=sample_correction_factors,
            cascade_args=cascade_args,
        )

        lotus.logger.info(f"Learned cascade thresholds: {best_combination}")
        return best_combination

    except Exception as e:
        lotus.logger.error(f"Error while learning filter cascade thresholds: {e}")
        raise e


def _build_document_mapping(
    total_docs: int, 
    batch_size: int
) -> tuple[dict[tuple[int, int], int], list[set[int]]]:
    """
    Build mapping from (batch_idx, doc_id_in_batch) to original document index.
    
    This mapping is critical for handling variable batch sizes and tracking
    which documents should be in which batch. Each batch uses document IDs
    starting from 1, but we need to map them back to the original list indices.
    
    Example:
        10 documents, batch_size=4:
        - Batch 0: docs[0:4]  -> doc_ids 1,2,3,4   -> mapping: {(0,1):0, (0,2):1, (0,3):2, (0,4):3}
        - Batch 1: docs[4:8]  -> doc_ids 1,2,3,4   -> mapping: {(1,1):4, (1,2):5, (1,3):6, (1,4):7}
        - Batch 2: docs[8:10] -> doc_ids 1,2       -> mapping: {(2,1):8, (2,2):9}
    
    Args:
        total_docs: Total number of documents to process
        batch_size: Maximum number of documents per batch
        
    Returns:
        tuple containing:
            - mapping: Dict mapping (batch_idx, doc_id_in_batch) to original_doc_idx
            - expected_doc_ids_per_batch: List of sets, each set contains expected
              document IDs for that batch (always starting from 1)
    """
    mapping: dict[tuple[int, int], int] = {}
    expected_doc_ids_per_batch: list[set[int]] = []
    
    batch_idx = 0
    for batch_start in range(0, total_docs, batch_size):
        batch_end = min(batch_start + batch_size, total_docs)
        actual_batch_size = batch_end - batch_start
        
        # Expected document IDs for this batch (always 1 to actual_batch_size)
        expected_doc_ids = set(range(1, actual_batch_size + 1))
        expected_doc_ids_per_batch.append(expected_doc_ids)
        
        # Build mapping for this batch
        for doc_id_in_batch in range(1, actual_batch_size + 1):
            original_idx = batch_start + (doc_id_in_batch - 1)
            mapping[(batch_idx, doc_id_in_batch)] = original_idx
        
        batch_idx += 1
    
    return mapping, expected_doc_ids_per_batch


def _parse_batch_with_mapping(
    batch_outputs: list[str],
    mapping: dict[tuple[int, int], int],
    expected_doc_ids_per_batch: list[set[int]],
    total_docs: int,
    default: bool = True,
) -> tuple[list[bool | None], list[str], list[str | None], dict[int, set[int]]]:
    """
    Parse batch outputs using document mapping to ensure correct ordering.
    
    This function:
    1. Parses each batch output using robust JSON parsing
    2. Maps batch-local document IDs back to original indices
    3. Detects missing documents (documents that should be in a batch but aren't in results)
    4. Maintains original document order in the output
    
    Args:
        batch_outputs: List of raw output strings from the model (one per batch)
        mapping: Mapping from (batch_idx, doc_id_in_batch) to original_doc_idx
        expected_doc_ids_per_batch: Expected document IDs for each batch
        total_docs: Total number of documents
        default: Default boolean value for missing/unparseable results
        
    Returns:
        tuple containing:
            - outputs: List of boolean results in original order (may contain None for missing)
            - raw_outputs: List of raw output strings in original order
            - explanations: List of explanation strings in original order (may contain None)
            - missing_docs: Dict mapping batch_idx to set of missing doc_ids in that batch
    """
    from .postprocessors import _parse_single_batch_filter_output
    
    # Initialize result arrays with None (will be filled in)
    outputs: list[bool | None] = [None] * total_docs
    raw_outputs: list[str] = [""] * total_docs
    explanations: list[str | None] = [None] * total_docs
    missing_docs: dict[int, set[int]] = {}
    
    for batch_idx, batch_output in enumerate(batch_outputs):
        try:
            # Parse this batch's output using robust multi-level parsing
            parsed_results = _parse_single_batch_filter_output(batch_output, default)
            
            # Get expected document IDs for this batch
            expected_doc_ids = expected_doc_ids_per_batch[batch_idx]
            
            # Track which document IDs were actually returned
            returned_doc_ids = {r["document_id"] for r in parsed_results}
            
            # Detect missing documents
            missing_in_batch = expected_doc_ids - returned_doc_ids
            if missing_in_batch:
                missing_docs[batch_idx] = missing_in_batch
                lotus.logger.warning(
                    f"Batch {batch_idx}: Missing {len(missing_in_batch)} documents "
                    f"(IDs: {sorted(missing_in_batch)}). Expected {len(expected_doc_ids)}, "
                    f"got {len(returned_doc_ids)}."
                )
            
            # Map results to original positions
            for result in parsed_results:
                doc_id_in_batch = result["document_id"]
                
                # Look up original index
                original_idx = mapping.get((batch_idx, doc_id_in_batch))
                
                if original_idx is None:
                    # Unexpected document ID (not in our mapping)
                    lotus.logger.warning(
                        f"Batch {batch_idx}: Unexpected document_id {doc_id_in_batch}. "
                        f"Expected IDs: {sorted(expected_doc_ids)}"
                    )
                    continue
                
                # Store result at correct position
                outputs[original_idx] = result["answer"]
                raw_outputs[original_idx] = batch_output
                explanations[original_idx] = result["reasoning"]
        
        except Exception as e:
            # If parsing fails for entire batch, mark all documents as missing
            lotus.logger.error(f"Failed to parse batch {batch_idx}: {e}")
            missing_docs[batch_idx] = expected_doc_ids_per_batch[batch_idx]
    
    return outputs, raw_outputs, explanations, missing_docs


def _retry_missing_documents(
    docs: list[dict[str, Any]],
    missing_docs: dict[int, set[int]],
    mapping: dict[tuple[int, int], int],
    model: lotus.models.LM,
    user_instruction: str,
    default: bool = True,
    examples_multimodal_data: list[dict[str, Any]] | None = None,
    examples_answers: list[bool] | None = None,
    cot_reasoning: list[str] | None = None,
    strategy: ReasoningStrategy | None = None,
    show_progress_bar: bool = True,
    progress_bar_desc: str = "Retrying",
    additional_cot_instructions: str = "",
) -> dict[int, dict[str, Any]]:
    """
    Retry missing documents using individual processing.
    
    This function processes documents that were missing from batch responses.
    It uses individual processing (not batch) for better reliability when
    retrying failed documents.
    
    Args:
        docs: Original list of all documents
        missing_docs: Dict mapping batch_idx to set of missing doc_ids in that batch
        mapping: Mapping from (batch_idx, doc_id_in_batch) to original_doc_idx
        model: Language model instance
        user_instruction: Filter instruction/claim
        default: Default boolean value
        examples_multimodal_data: Example documents for few-shot learning
        examples_answers: Expected boolean outputs for examples
        cot_reasoning: Chain-of-thought reasoning for examples
        strategy: Reasoning strategy to use
        show_progress_bar: Whether to show progress bar
        progress_bar_desc: Description for progress bar
        additional_cot_instructions: Additional CoT instructions
        
    Returns:
        Dict mapping original_doc_idx to result dict with keys:
            - "output": boolean result
            - "raw_output": raw output string
            - "explanation": explanation string
    """
    # Collect all missing document indices
    missing_doc_indices: list[int] = []
    for batch_idx, missing_doc_ids in missing_docs.items():
        for doc_id in sorted(missing_doc_ids):
            original_idx = mapping[(batch_idx, doc_id)]
            missing_doc_indices.append(original_idx)
    
    if not missing_doc_indices:
        return {}
    
    lotus.logger.info(
        f"Retrying {len(missing_doc_indices)} missing documents individually: "
        f"indices {missing_doc_indices}"
    )
    
    # Extract missing documents
    missing_docs_list = [docs[idx] for idx in missing_doc_indices]
    
    # Process using individual method (more reliable for retries)
    retry_output = _sem_filter_individual(
        docs=missing_docs_list,
        model=model,
        user_instruction=user_instruction,
        default=default,
        examples_multimodal_data=examples_multimodal_data,
        examples_answers=examples_answers,
        cot_reasoning=cot_reasoning,
        strategy=strategy,
        logprobs=False,  # Don't need logprobs for retries
        safe_mode=False,  # Don't show cost for retries
        show_progress_bar=show_progress_bar,
        progress_bar_desc=progress_bar_desc,
        additional_cot_instructions=additional_cot_instructions,
    )
    
    # Build result mapping
    retry_results: dict[int, dict[str, Any]] = {}
    for i, original_idx in enumerate(missing_doc_indices):
        retry_results[original_idx] = {
            "output": retry_output.outputs[i],
            "raw_output": retry_output.raw_outputs[i],
            "explanation": retry_output.explanations[i],
        }
    
    return retry_results


@pd.api.extensions.register_dataframe_accessor("sem_filter")
class SemFilterDataframe:
    """
    Apply semantic filtering over a DataFrame.

    This method performs semantic filtering on the DataFrame content using
    a natural language instruction. It can process specific columns identified
    in the instruction and supports few-shot learning through examples.
    Can use async processing for better performance with large datasets.

    Args:
        user_instruction (str): The natural language instruction that defines
            the filter condition. Should describe what criteria rows must meet.
        return_raw_outputs (bool, optional): Whether to include raw model
            outputs in the output DataFrame. Useful for debugging.
            Defaults to False.
        return_explanations (bool, optional): Whether to include explanations
            in the output DataFrame. Useful for debugging and understanding
            model reasoning, when using chain-of-thought. Defaults to False.
        return_all (bool, optional): Whether to return all rows (including
            filtered out ones) or only the rows that pass the filter.
            Defaults to False.
        default (bool, optional): The default value to use when the model
            output cannot be parsed as a boolean. Defaults to True.
        suffix (str, optional): The suffix for the output column names.
            Defaults to "_filter".
        examples (pd.DataFrame | None, optional): Example DataFrame for
            few-shot learning. Should have the same column structure as the
            input DataFrame plus an "Answer" column. Defaults to None.
        helper_examples (pd.DataFrame | None, optional): Additional helper
            examples for cascade filtering. Defaults to None.
        strategy (ReasoningStrategy | None, optional): The reasoning strategy
            to use. Can be None, COT, or ZS_COT. Defaults to None.
        cascade_args (CascadeArgs | None, optional): Configuration for cascade
            filtering. Includes parameters like recall_target, precision_target,
            sampling_percentage, and failure_probability. Defaults to None.
        return_stats (bool, optional): Whether to return filtering statistics
            along with the filtered DataFrame. Defaults to False.
        safe_mode (bool, optional): Whether to enable safe mode with cost
            estimation. Defaults to False.
        progress_bar_desc (str, optional): Description for the progress bar.
            Defaults to "Filtering".
        additional_cot_instructions (str, optional): Additional instructions
            for chain-of-thought reasoning. Defaults to "".
        use_async (bool, optional): Whether to use async processing for better
            performance. Defaults to False.
        max_concurrent_batches (int, optional): Maximum number of concurrent batches
            to process when using async. Defaults to 4.
        max_thread_workers (int, optional): Maximum number of threads for CPU-intensive
            operations when using async. Defaults to 8.

    Returns:
        pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]: A DataFrame
            containing the original data plus the filter results, or a tuple
            containing the DataFrame and statistics if return_stats is True.

    Raises:
        ValueError: If the language model is not configured, if specified
            columns don't exist in the DataFrame, or if the examples DataFrame
            doesn't have the required "Answer" column.

    Example:
        >>> import pandas as pd
        >>> import lotus
        >>> from lotus.models import LM
        >>> lotus.settings.configure(lm=LM(model="gpt-4o-mini"))

        >>> df = pd.DataFrame({
                'text': ['Great product!', 'Terrible service'],
                'rating': [5, 1]
            })

        # Example 1: simple filtering
        >>> df.sem_filter("The review {text} and {rating} reflect's a positive sentiment ")
        Filtering: 100%|██████████████████████████████████████████████████████████████████ 2/2 LM calls [00:00<00:00,  2.06it/s]
                    text  rating
        0  Great product!      5

        # Example 2: with zero-shot chain-of-thought (ZS-COT) reasoning
        >>> from lotus.types import ReasoningStrategy
        >>> df.sem_filter("The review {text} and {rating} reflect's a positive sentiment ", strategy=ReasoningStrategy.ZS_COT, return_explanations=True, return_all=True)
        Filtering: 100%|██████████████████████████████████████████████████████████████████ 4/4 LM calls [00:01<00:00,  3.66it/s]
                                                        Text  filter_label explanation_filter
        0             I had two apples, then I gave away one          True
        1                         My friend gave me an apple          True
        2                      I gave away both of my apples         False
        3  I gave away my apple, then a friend gave me hi...         False

        # Example 3: with async processing for better performance
        >>> df.sem_filter("The review {text} and {rating} reflect's a positive sentiment ", use_async=True)
        Filtering: 100%|██████████████████████████████████████████████████████████████████ 2/2 LM calls [00:00<00:00,  2.06it/s]
                    text  rating
        0  Great product!      5

    """

    def __init__(self, pandas_obj: Any):
        """
        Initialize the semantic filtering accessor.

        Args:
            pandas_obj (Any): The pandas DataFrame object to attach the accessor to.
        """
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj: Any) -> None:
        """
        Validate that the object is a pandas DataFrame.

        Args:
            obj (Any): The object to validate.

        Raises:
            AttributeError: If the object is not a pandas DataFrame.
        """
        # verify that the Series has the correct type
        if not isinstance(obj, pd.DataFrame):
            raise AttributeError("Must be a DataFrame")

    @operator_cache
    def __call__(
        self,
        user_instruction: str,
        return_raw_outputs: bool = False,
        return_explanations: bool = False,
        return_all: bool = False,
        default: bool = True,
        suffix: str = "_filter",
        examples: pd.DataFrame | None = None,
        helper_examples: pd.DataFrame | None = None,
        strategy: ReasoningStrategy | None = None,
        cascade_args: CascadeArgs | None = None,
        return_stats: bool = False,
        safe_mode: bool = False,
        progress_bar_desc: str = "Filtering",
        additional_cot_instructions: str = "",
        use_async: bool = False,
        max_concurrent_batches: int = 4,
        max_thread_workers: int = 8,
    ) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
        if lotus.settings.lm is None:
            raise ValueError(
                "The language model must be an instance of LM. Please configure a valid language model using lotus.settings.configure()"
            )

        stats: dict[str, float] = {}
        lotus.logger.debug(user_instruction)
        col_li = lotus.nl_expression.parse_cols(user_instruction)
        lotus.logger.debug(col_li)
        helper_strategy = strategy

        # check that column exists
        for column in col_li:
            if column not in self._obj.columns:
                raise ValueError(f"Column {column} not found in DataFrame")

        multimodal_data = task_instructions.df2multimodal_info(self._obj, col_li)
        lotus.logger.debug(multimodal_data)
        formatted_usr_instr = lotus.nl_expression.nle2str(user_instruction, col_li)

        examples_multimodal_data = None
        examples_answers = None
        cot_reasoning = None
        if examples is not None:
            assert "Answer" in examples.columns, "Answer must be a column in examples dataframe"
            examples_multimodal_data = task_instructions.df2multimodal_info(examples, col_li)
            examples_answers = examples["Answer"].tolist()

            if strategy == ReasoningStrategy.COT and "Reasoning" in examples.columns:
                cot_reasoning = examples["Reasoning"].tolist()

        pos_cascade_threshold, neg_cascade_threshold = None, None
        if cascade_args is not None:
            # Get few-shot examples for small LM
            helper_examples_multimodal_data = None
            helper_examples_answers = None
            helper_cot_reasoning = None
            if helper_examples is not None:
                assert "Answer" in helper_examples.columns, "Answer must be a column in examples dataframe"
                helper_examples_multimodal_data = task_instructions.df2multimodal_info(helper_examples, col_li)
                helper_examples_answers = helper_examples["Answer"].tolist()
                if helper_strategy == ReasoningStrategy.COT and "Reasoning" in helper_examples.columns:
                    helper_cot_reasoning = helper_examples["Reasoning"].tolist()

        if cascade_args:
            proxy_model = cascade_args.proxy_model
            if (
                cascade_args.recall_target is None
                or cascade_args.precision_target is None
                or cascade_args.failure_probability is None
            ):
                raise ValueError(
                    "Recall target, precision target, and confidence need to be specified for learned thresholds."
                )

            # Get the proxy scores
            if proxy_model == ProxyModel.HELPER_LM:
                if not lotus.settings.helper_lm:
                    raise ValueError("Helper LM must be set in settings")

                if helper_strategy == ReasoningStrategy.COT:
                    raise ValueError("CoT not supported for helper models in cascades.")

                # Run small LM and get logits
                helper_output = sem_filter(
                    multimodal_data,
                    lotus.settings.helper_lm,
                    formatted_usr_instr,
                    default=default,
                    examples_multimodal_data=helper_examples_multimodal_data,
                    examples_answers=helper_examples_answers,
                    cot_reasoning=helper_cot_reasoning,
                    logprobs=True,
                    strategy=helper_strategy,
                    safe_mode=safe_mode,
                    show_progress_bar=True,
                    progress_bar_desc="Running helper LM",
                    use_async=use_async,
                    max_concurrent_batches=max_concurrent_batches,
                    max_thread_workers=max_thread_workers,
                )
                _, helper_logprobs = helper_output.outputs, helper_output.logprobs
                assert helper_logprobs is not None
                formatted_helper_logprobs: LogprobsForFilterCascade = (
                    lotus.settings.helper_lm.format_logprobs_for_filter_cascade(helper_logprobs)
                )
                proxy_scores = calibrate_llm_logprobs(formatted_helper_logprobs.true_probs, cascade_args)
            elif proxy_model == ProxyModel.EMBEDDING_MODEL:
                if not lotus.settings.rm:
                    raise ValueError("RM must be set in settings")

                # TODO: How to handle multiple columns?
                search_df = self._obj.sem_search(col_li[0], formatted_usr_instr, K=len(self._obj), return_scores=True)
                proxy_scores = search_df["vec_scores_sim_score"].tolist()

            sample_indices, correction_factors = importance_sampling(proxy_scores, cascade_args)
            sample_df = self._obj.loc[sample_indices]
            sample_multimodal_data = task_instructions.df2multimodal_info(sample_df, col_li)
            sample_proxy_scores = [proxy_scores[i] for i in sample_indices]
            sample_correction_factors = correction_factors[sample_indices]

            pos_cascade_threshold, neg_cascade_threshold = learn_filter_cascade_thresholds(
                sample_multimodal_data=sample_multimodal_data,
                lm=lotus.settings.lm,
                formatted_usr_instr=formatted_usr_instr,
                default=default,
                cascade_args=cascade_args,
                proxy_scores=sample_proxy_scores,
                sample_correction_factors=sample_correction_factors,
                examples_multimodal_data=examples_multimodal_data,
                examples_answers=examples_answers,
                cot_reasoning=cot_reasoning,
                strategy=strategy,
                additional_cot_instructions=additional_cot_instructions,
            )

            stats["pos_cascade_threshold"] = pos_cascade_threshold
            stats["neg_cascade_threshold"] = neg_cascade_threshold

        if pos_cascade_threshold is not None and neg_cascade_threshold is not None:
            stats["filters_resolved_by_helper_model"] = 0
            stats["filters_resolved_by_large_model"] = 0

            high_conf_idxs = set()
            proxy_outputs = [False] * len(multimodal_data)

            # Set proxy_outputs where confidence is high
            for idx_i in range(len(proxy_scores)):
                true_prob = proxy_scores[idx_i]
                if true_prob >= pos_cascade_threshold or true_prob <= neg_cascade_threshold:
                    high_conf_idxs.add(idx_i)
                    proxy_outputs[idx_i] = (
                        True
                        if true_prob >= pos_cascade_threshold
                        else False
                        if true_prob <= neg_cascade_threshold
                        else proxy_outputs[idx_i]
                    )

            lotus.logger.info(f"Num routed to smaller model: {len(high_conf_idxs)}")
            stats["num_routed_to_helper_model"] = len(high_conf_idxs)

            outputs: list[bool] = [False] * len(multimodal_data)
            raw_outputs: list[str] = [""] * len(multimodal_data)
            explanations: list[str | None] = [None] * len(multimodal_data)

            for idx in high_conf_idxs:
                outputs[idx] = proxy_outputs[idx]

            # If using helper LM, get raw outputs and explanations
            if proxy_model == ProxyModel.HELPER_LM:
                assert all(isinstance(x, str) for x in helper_output.explanations) or all(
                    x is None for x in helper_output.explanations
                )
                for idx in high_conf_idxs:
                    raw_outputs[idx] = helper_output.raw_outputs[idx]
                    explanations[idx] = helper_output.explanations[idx]

            # Send low confidence samples to large LM if any
            low_conf_idxs = sorted([i for i in range(len(proxy_outputs)) if i not in high_conf_idxs])
            low_conf_multimodal_data = [multimodal_data[idx] for idx in low_conf_idxs]
            if low_conf_idxs:
                large_output = sem_filter(
                    low_conf_multimodal_data,
                    lotus.settings.lm,
                    formatted_usr_instr,
                    default=default,
                    examples_multimodal_data=examples_multimodal_data,
                    examples_answers=examples_answers,
                    cot_reasoning=cot_reasoning,
                    strategy=strategy,
                    safe_mode=safe_mode,
                    progress_bar_desc="Running predicate evals with oracle LM",
                    additional_cot_instructions=additional_cot_instructions,
                    use_async=use_async,
                    max_concurrent_batches=max_concurrent_batches,
                    max_thread_workers=max_thread_workers,
                )

                for idx, large_idx in enumerate(low_conf_idxs):
                    outputs[large_idx] = large_output.outputs[idx]
                    raw_outputs[large_idx] = large_output.raw_outputs[idx]
                    explanations[large_idx] = large_output.explanations[idx]

            stats["filters_resolved_by_helper_model"] += len(high_conf_idxs)
            stats["filters_resolved_by_large_model"] += len(low_conf_idxs)

        else:
            output = sem_filter(
                multimodal_data,
                lotus.settings.lm,
                formatted_usr_instr,
                default=default,
                examples_multimodal_data=examples_multimodal_data,
                examples_answers=examples_answers,
                cot_reasoning=cot_reasoning,
                strategy=strategy,
                safe_mode=safe_mode,
                show_progress_bar=True,
                progress_bar_desc=progress_bar_desc,
                additional_cot_instructions=additional_cot_instructions,
                use_async=use_async,
                max_concurrent_batches=max_concurrent_batches,
                max_thread_workers=max_thread_workers,
            )
            outputs = output.outputs
            raw_outputs = output.raw_outputs
            explanations = output.explanations

        if not return_all:
            # find indices where output is True
            ids = [i for i, x in enumerate(outputs) if x]
            idx_ids = [self._obj.index[i] for i, x in enumerate(outputs) if x]
            lotus.logger.debug(f"ids: {ids}")
            lotus.logger.debug(f"idx_ids: {idx_ids}")

            [outputs[i] for i in ids]
            filtered_explanations = [explanations[i] for i in ids]
            filtered_raw_outputs = [raw_outputs[i] for i in ids]
            lotus.logger.debug(f"filtered_raw_outputs: {filtered_raw_outputs}")

            new_df = self._obj.iloc[ids]
            new_df.attrs["index_dirs"] = self._obj.attrs.get("index_dirs", None)
        else:

            def get_out_col_name(df, col_name):
                if col_name in df.columns:
                    i = 1
                    while f"{col_name}_{i}" in new_df.columns:
                        i += 1
                    return f"{col_name}_{i}"
                else:
                    return col_name

            new_df = self._obj.copy()
            new_df[get_out_col_name(new_df, "filter_label")] = outputs
            filtered_explanations = explanations
            filtered_raw_outputs = raw_outputs

        # return rows where output is True
        if return_explanations and return_raw_outputs:
            new_df["explanation" + suffix] = filtered_explanations
            new_df["raw_output" + suffix] = filtered_raw_outputs
        elif return_explanations:
            new_df["explanation" + suffix] = filtered_explanations
        elif return_raw_outputs:
            new_df["raw_output" + suffix] = filtered_raw_outputs

        if return_stats:
            return new_df, stats

        return new_df
