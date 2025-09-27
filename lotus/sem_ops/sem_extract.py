import asyncio
from typing import Any, Callable

import pandas as pd

import lotus
from lotus.cache import operator_cache
from lotus.models import LM
from lotus.templates import task_instructions
from lotus.types import LMOutput, ReasoningStrategy, SemanticExtractOutput, SemanticExtractPostprocessOutput
from lotus.utils import show_safe_mode

from .postprocessors import extract_postprocess


def sem_extract(
    docs: list[dict[str, Any]],
    model: LM,
    output_cols: dict[str, str | None],
    extract_quotes: bool = False,
    postprocessor: Callable[[list[str], lotus.models.LM, bool], SemanticExtractPostprocessOutput] = extract_postprocess,
    safe_mode: bool = False,
    progress_bar_desc: str = "Extracting",
    return_explanations: bool = False,
    strategy: ReasoningStrategy | None = None,
    # New batch processing parameters
    batch_size: int = 10,
    use_batch_processing: bool = True,
) -> SemanticExtractOutput:
    """
    Extracts structured attributes and values from a list of documents using a language model.

    This function uses a language model to extract specific information from documents
    and return it in a structured format. It can extract multiple attributes at once
    and optionally include quotes from the source text.

    Args:
        docs (list[dict[str, Any]]): The list of documents to extract from. Each
            document should be a dictionary containing multimodal information
            (text, images, etc.).
        model (lotus.models.LM): The language model instance to use for extraction.
            Must be properly configured with appropriate API keys and settings.
        output_cols (dict[str, str | None]): A mapping from desired output column
            names to optional descriptions. The descriptions help guide the model
            on what to extract. For example: {"sentiment": "positive/negative/neutral",
            "confidence": "0-1 scale"}.
        extract_quotes (bool, optional): Whether to extract supporting quotes from
            the source text for each extracted value. Defaults to False.
        postprocessor (Callable, optional): A function to post-process the model
            outputs. Should take (outputs, model, return_explanations) and return
            SemanticExtractPostprocessOutput. Defaults to extract_postprocess.
        safe_mode (bool, optional): Whether to enable safe mode with cost estimation.
            Defaults to False.
        progress_bar_desc (str, optional): Description for the progress bar.
            Defaults to "Extracting".
        return_explanations (bool, optional): Whether to return explanations for
            the extraction decisions. Useful for debugging and understanding
            model reasoning. Defaults to False.
        strategy (ReasoningStrategy | None, optional): The reasoning strategy to use.
            Can be None, COT, or ZS_COT. Defaults to None.
        batch_size (int, optional): Number of documents to process in each batch
            when using batch processing. Defaults to 10.
        use_batch_processing (bool, optional): Whether to use batch processing to
            share system prompts and examples across documents. Defaults to True.

    Returns:
        SemanticExtractOutput: An object containing the extracted outputs, raw
            outputs, and explanations (if requested).

    Raises:
        ValueError: If the model is not properly configured or if there are
            issues with the input parameters.

    Example:
        >>> docs = [{"text": "The product is excellent with 5 stars"}]
        >>> model = LM(model="gpt-4o")
        >>> output_cols = {"sentiment": "positive/negative/neutral", "rating": "1-5 scale"}
        >>> result = sem_extract(docs, model, output_cols)
        >>> print(result.outputs)  # [{"sentiment": "positive", "rating": "5"}]
    """
    # Choose between batch and individual processing
    if use_batch_processing and len(docs) > 1:
        return _sem_extract_batch(
            docs, model, output_cols, extract_quotes, postprocessor,
            safe_mode, progress_bar_desc, return_explanations, strategy, batch_size
        )
    else:
        return _sem_extract_individual(
            docs, model, output_cols, extract_quotes, postprocessor,
            safe_mode, progress_bar_desc, return_explanations, strategy
        )


def _sem_extract_batch(
    docs: list[dict[str, Any]],
    model: LM,
    output_cols: dict[str, str | None],
    extract_quotes: bool = False,
    postprocessor: Callable[[list[str], lotus.models.LM, bool], SemanticExtractPostprocessOutput] = extract_postprocess,
    safe_mode: bool = False,
    progress_bar_desc: str = "Extracting",
    return_explanations: bool = False,
    strategy: ReasoningStrategy | None = None,
    batch_size: int = 10,
) -> SemanticExtractOutput:
    """Batch processing implementation for sem_extract."""
    from .postprocessors import batch_extract_parser
    from lotus.templates.batch_extract_formatter import batch_extract_formatter
    
    try:
        # Use batch formatter
        batched_inputs = batch_extract_formatter(
            model, docs, output_cols, extract_quotes, strategy, batch_size
        )
        
        lotus.logger.debug(f"batch inputs count: {len(batched_inputs)}")
        
        if safe_mode:
            estimated_total_calls = len(batched_inputs)
            estimated_total_cost = sum(model.count_tokens(input) for input in batched_inputs)
            show_safe_mode(estimated_total_cost, estimated_total_calls)
        
        # Call model with batch inputs
        lm_output: LMOutput = model(
            batched_inputs, 
            response_format={"type": "json_object"},
            progress_bar_desc=progress_bar_desc
        )
        
        # Parse batch responses
        outputs, raw_outputs, explanations = batch_extract_parser(
            lm_output.outputs, model, len(docs)
        )
        
        lotus.logger.debug(f"batch outputs: {outputs}")
        lotus.logger.debug(f"batch raw_outputs: {raw_outputs}")
        lotus.logger.debug(f"batch explanations: {explanations}")
        
        if safe_mode:
            model.print_total_usage()
        
        return SemanticExtractOutput(
            raw_outputs=raw_outputs,
            outputs=outputs,
            explanations=explanations,
        )
        
    except Exception as e:
        lotus.logger.warning(f"Batch processing failed: {e}, falling back to individual processing")
        return _sem_extract_individual(
            docs, model, output_cols, extract_quotes, postprocessor,
            safe_mode, progress_bar_desc, return_explanations, strategy
        )


def _sem_extract_individual(
    docs: list[dict[str, Any]],
    model: LM,
    output_cols: dict[str, str | None],
    extract_quotes: bool = False,
    postprocessor: Callable[[list[str], lotus.models.LM, bool], SemanticExtractPostprocessOutput] = extract_postprocess,
    safe_mode: bool = False,
    progress_bar_desc: str = "Extracting",
    return_explanations: bool = False,
    strategy: ReasoningStrategy | None = None,
) -> SemanticExtractOutput:
    """Individual processing implementation for sem_extract."""
    # prepare model inputs
    inputs = []
    for doc in docs:
        prompt = task_instructions.extract_formatter(model, doc, output_cols, extract_quotes, strategy)
        lotus.logger.debug(f"input to model: {prompt}")
        lotus.logger.debug(f"inputs content to model: {[x.get('content') for x in prompt]}")
        inputs.append(prompt)

    # check if safe_mode is enabled
    if safe_mode:
        estimated_cost = sum(model.count_tokens(input) for input in inputs)
        estimated_LM_calls = len(docs)
        show_safe_mode(estimated_cost, estimated_LM_calls)

    # call model
    lm_output: LMOutput = model(inputs, response_format={"type": "json_object"}, progress_bar_desc=progress_bar_desc)

    # post process results
    postprocess_output = postprocessor(lm_output.outputs, model, return_explanations)
    lotus.logger.debug(f"raw_outputs: {lm_output.outputs}")
    lotus.logger.debug(f"outputs: {postprocess_output.outputs}")
    lotus.logger.debug(f"explanations: {postprocess_output.explanations}")
    if safe_mode:
        model.print_total_usage()

    return SemanticExtractOutput(
        raw_outputs=postprocess_output.raw_outputs,
        outputs=postprocess_output.outputs,
        explanations=postprocess_output.explanations,
    )


async def sem_extract_async(
    docs: list[dict[str, Any]],
    model: LM,
    output_cols: dict[str, str | None],
    extract_quotes: bool = False,
    postprocessor: Callable[[list[str], lotus.models.LM, bool], SemanticExtractPostprocessOutput] = extract_postprocess,
    safe_mode: bool = False,
    progress_bar_desc: str = "Extracting",
    return_explanations: bool = False,
    strategy: ReasoningStrategy | None = None,
    # New batch processing parameters
    batch_size: int = 10,
    use_batch_processing: bool = True,
    # Async-specific parameters
    max_concurrent_batches: int = 3,
) -> SemanticExtractOutput:
    """
    Asynchronous version of sem_extract that supports concurrent batch processing.

    This function provides the same functionality as sem_extract but with async/await
    support, allowing for concurrent processing of multiple batches and better
    utilization of I/O wait times.

    Args:
        docs (list[dict[str, Any]]): The list of documents to extract from. Each
            document should be a dictionary containing multimodal information
            (text, images, etc.).
        model (lotus.models.LM): The language model instance to use for extraction.
            Must be properly configured with appropriate API keys and settings.
        output_cols (dict[str, str | None]): A mapping from desired output column
            names to optional descriptions. The descriptions help guide the model
            on what to extract. For example: {"sentiment": "positive/negative/neutral",
            "confidence": "0-1 scale"}.
        extract_quotes (bool, optional): Whether to extract supporting quotes from
            the source text for each extracted value. Defaults to False.
        postprocessor (Callable, optional): A function to post-process the model
            outputs. Should take (outputs, model, return_explanations) and return
            SemanticExtractPostprocessOutput. Defaults to extract_postprocess.
        safe_mode (bool, optional): Whether to enable safe mode with cost estimation.
            Defaults to False.
        progress_bar_desc (str, optional): Description for the progress bar.
            Defaults to "Extracting".
        return_explanations (bool, optional): Whether to return explanations for
            the extraction decisions. Useful for debugging and understanding
            model reasoning. Defaults to False.
        strategy (ReasoningStrategy | None, optional): The reasoning strategy to use.
            Can be None, COT, or ZS_COT. Defaults to None.
        batch_size (int, optional): Number of documents to process in each batch
            when using batch processing. Defaults to 10.
        use_batch_processing (bool, optional): Whether to use batch processing to
            share system prompts and examples across documents. Defaults to True.
        max_concurrent_batches (int, optional): Maximum number of batches to process
            concurrently. Defaults to 3.

    Returns:
        SemanticExtractOutput: An object containing the extracted outputs, raw
            outputs, and explanations (if requested).

    Raises:
        ValueError: If the model is not properly configured or if there are
            issues with the input parameters.

    Example:
        >>> import asyncio
        >>> docs = [{"text": "The product is excellent with 5 stars"}]
        >>> model = LM(model="gpt-4o")
        >>> output_cols = {"sentiment": "positive/negative/neutral", "rating": "1-5 scale"}
        >>> result = await sem_extract_async(docs, model, output_cols)
        >>> print(result.outputs)  # [{"sentiment": "positive", "rating": "5"}]
    """
    # Choose between async batch and individual processing
    if use_batch_processing and len(docs) > 1:
        return await _sem_extract_async_batch(
            docs, model, output_cols, extract_quotes, postprocessor,
            safe_mode, progress_bar_desc, return_explanations, strategy, 
            batch_size, max_concurrent_batches
        )
    else:
        return await _sem_extract_async_individual(
            docs, model, output_cols, extract_quotes, postprocessor,
            safe_mode, progress_bar_desc, return_explanations, strategy
        )


async def _sem_extract_async_batch(
    docs: list[dict[str, Any]],
    model: LM,
    output_cols: dict[str, str | None],
    extract_quotes: bool = False,
    postprocessor: Callable[[list[str], lotus.models.LM, bool], SemanticExtractPostprocessOutput] = extract_postprocess,
    safe_mode: bool = False,
    progress_bar_desc: str = "Extracting",
    return_explanations: bool = False,
    strategy: ReasoningStrategy | None = None,
    batch_size: int = 10,
    max_concurrent_batches: int = 3,
) -> SemanticExtractOutput:
    """Async batch processing implementation for sem_extract with concurrent batch execution."""
    from .postprocessors import batch_extract_parser
    from lotus.templates.batch_extract_formatter import batch_extract_formatter
    
    try:
        # Use batch formatter
        batched_inputs = batch_extract_formatter(
            model, docs, output_cols, extract_quotes, strategy, batch_size
        )
        
        lotus.logger.debug(f"batch inputs count: {len(batched_inputs)}")
        
        if safe_mode:
            estimated_total_calls = len(batched_inputs)
            estimated_total_cost = sum(model.count_tokens(input) for input in batched_inputs)
            show_safe_mode(estimated_total_cost, estimated_total_calls)
        
        # Process batches concurrently with semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent_batches)
        
        async def process_batch(batch_input: list[dict[str, str]]) -> LMOutput:
            """Process a single batch asynchronously."""
            async with semaphore:
                # Create a copy of the model for this batch to avoid conflicts
                batch_model = model
                if hasattr(batch_model, 'async_call'):
                    return await batch_model.async_call(
                        batch_input, 
                        response_format={"type": "json_object"},
                        progress_bar_desc=progress_bar_desc
                    )
                else:
                    # Fallback to synchronous call if async not supported
                    return batch_model(
                        batch_input, 
                        response_format={"type": "json_object"},
                        progress_bar_desc=progress_bar_desc
                    )
        
        # Execute all batches concurrently
        batch_tasks = [process_batch(batch_input) for batch_input in batched_inputs]
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        
        # Check for exceptions in batch results
        failed_batches = []
        successful_outputs = []
        
        for i, result in enumerate(batch_results):
            if isinstance(result, Exception):
                lotus.logger.warning(f"Batch {i+1} failed: {result}")
                failed_batches.append(i)
            else:
                successful_outputs.extend(result.outputs)
        
        # If some batches failed, fall back to individual processing for failed documents
        if failed_batches:
            lotus.logger.warning(f"Failed batches: {failed_batches}, falling back to individual processing")
            # Calculate which documents were in failed batches
            failed_docs = []
            for batch_idx in failed_batches:
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(docs))
                failed_docs.extend(docs[start_idx:end_idx])
            
            # Process failed documents individually
            if failed_docs:
                individual_result = await _sem_extract_async_individual(
                    failed_docs, model, output_cols, extract_quotes, postprocessor,
                    safe_mode, progress_bar_desc, return_explanations, strategy
                )
                successful_outputs.extend(individual_result.raw_outputs)
        
        # Parse batch responses
        outputs, raw_outputs, explanations = batch_extract_parser(
            successful_outputs, model, len(docs)
        )
        
        lotus.logger.debug(f"async batch outputs: {outputs}")
        lotus.logger.debug(f"async batch raw_outputs: {raw_outputs}")
        lotus.logger.debug(f"async batch explanations: {explanations}")
        
        if safe_mode:
            model.print_total_usage()
        
        return SemanticExtractOutput(
            raw_outputs=raw_outputs,
            outputs=outputs,
            explanations=explanations,
        )
        
    except Exception as e:
        lotus.logger.warning(f"Async batch processing failed: {e}, falling back to individual processing")
        return await _sem_extract_async_individual(
            docs, model, output_cols, extract_quotes, postprocessor,
            safe_mode, progress_bar_desc, return_explanations, strategy
        )


async def _sem_extract_async_individual(
    docs: list[dict[str, Any]],
    model: LM,
    output_cols: dict[str, str | None],
    extract_quotes: bool = False,
    postprocessor: Callable[[list[str], lotus.models.LM, bool], SemanticExtractPostprocessOutput] = extract_postprocess,
    safe_mode: bool = False,
    progress_bar_desc: str = "Extracting",
    return_explanations: bool = False,
    strategy: ReasoningStrategy | None = None,
) -> SemanticExtractOutput:
    """Async individual processing implementation for sem_extract."""
    # prepare model inputs
    inputs = []
    for doc in docs:
        prompt = task_instructions.extract_formatter(model, doc, output_cols, extract_quotes, strategy)
        lotus.logger.debug(f"input to model: {prompt}")
        lotus.logger.debug(f"inputs content to model: {[x.get('content') for x in prompt]}")
        inputs.append(prompt)

    # check if safe_mode is enabled
    if safe_mode:
        estimated_cost = sum(model.count_tokens(input) for input in inputs)
        estimated_LM_calls = len(docs)
        show_safe_mode(estimated_cost, estimated_LM_calls)

    # call model asynchronously
    if hasattr(model, 'async_call'):
        lm_output: LMOutput = await model.async_call(
            inputs, 
            response_format={"type": "json_object"}, 
            progress_bar_desc=progress_bar_desc
        )
    else:
        # Fallback to synchronous call if async not supported
        lm_output: LMOutput = model(
            inputs, 
            response_format={"type": "json_object"}, 
            progress_bar_desc=progress_bar_desc
        )

    # post process results
    postprocess_output = postprocessor(lm_output.outputs, model, return_explanations)
    lotus.logger.debug(f"raw_outputs: {lm_output.outputs}")
    lotus.logger.debug(f"outputs: {postprocess_output.outputs}")
    lotus.logger.debug(f"explanations: {postprocess_output.explanations}")
    if safe_mode:
        model.print_total_usage()

    return SemanticExtractOutput(
        raw_outputs=postprocess_output.raw_outputs,
        outputs=postprocess_output.outputs,
        explanations=postprocess_output.explanations,
    )


@pd.api.extensions.register_dataframe_accessor("sem_extract")
class SemExtractDataFrame:
    """
    Extract structured attributes and values from a DataFrame.

    This method performs structured information extraction on the DataFrame
    content using specified input columns and output column definitions.
    It can extract multiple attributes simultaneously and add them as new
    columns to the DataFrame.

    Args:
        input_cols (list[str]): The columns that the model should extract
            information from. These columns will be used as input to the
            language model.
        output_cols (dict[str, str | None]): A mapping from desired output
            column names to optional descriptions. The descriptions help guide
            the model on what to extract. For example: {"sentiment": "positive/negative/neutral",
            "confidence": "0-1 scale"}.
        extract_quotes (bool, optional): Whether to extract supporting quotes
            from the source text for each extracted value. Defaults to False.
        postprocessor (Callable, optional): A function to post-process the model
            outputs. Should take (outputs, model, return_explanations) and return
            SemanticExtractPostprocessOutput. Defaults to extract_postprocess.
        return_raw_outputs (bool, optional): Whether to include raw model
            outputs in the output DataFrame. Useful for debugging.
            Defaults to False.
        safe_mode (bool, optional): Whether to enable safe mode with cost
            estimation. Defaults to False.
        progress_bar_desc (str, optional): Description for the progress bar.
            Defaults to "Extracting".
        return_explanations (bool, optional): Whether to include explanations
            in the output DataFrame. Useful for debugging and understanding
            model reasoning. Defaults to False.
        strategy (ReasoningStrategy | None, optional): The reasoning strategy
            to use. Can be None, COT, or ZS_COT. Defaults to None.
        batch_size (int, optional): Number of documents to process in each batch
            when using batch processing. Defaults to 10.
        use_batch_processing (bool, optional): Whether to use batch processing to
            share system prompts and examples across documents. Defaults to True.

    Returns:
        pd.DataFrame: A DataFrame containing the original data plus the
            extracted attributes as new columns.

    Raises:
        ValueError: If the language model is not configured, if specified
            input columns don't exist in the DataFrame, or if there are
            other configuration issues.

    Example:
        >>> import pandas as pd
        >>> import lotus
        >>> from lotus.models import LM
        >>> lotus.settings.configure(lm=LM(model="gpt-4o-mini"))

        >>> df = pd.DataFrame({
        ...     'text': ['Great product!', 'Terrible service'],
        ...     'rating': [5, 1]
        ... })

        >>> df.sem_extract(
        ...     ['text'],
        ...     {'sentiment': 'positive/negative/neutral', 'emotion': 'joy/anger/sadness'}
        ... )
        Extracting: 100%|█████████████████████████████████████████████████████████████████ 2/2 LM calls [00:00<00:00,  2.20it/s]
                    text  rating sentiment emotion
        0    Great product!    5  positive     joy
        1  Terrible service    1  negative   anger
    """

    def __init__(self, pandas_obj: pd.DataFrame):
        """
        Initialize the semantic extraction accessor.

        Args:
            pandas_obj (pd.DataFrame): The pandas DataFrame object to attach the accessor to.
        """
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj: pd.DataFrame) -> None:
        """
        Validate that the object is a pandas DataFrame.

        Args:
            obj (pd.DataFrame): The object to validate.

        Raises:
            AttributeError: If the object is not a pandas DataFrame.
        """
        if not isinstance(obj, pd.DataFrame):
            raise AttributeError("Must be a DataFrame")

    @operator_cache
    def __call__(
        self,
        input_cols: list[str],
        output_cols: dict[str, str | None],
        extract_quotes: bool = False,
        postprocessor: Callable[
            [list[str], lotus.models.LM, bool], SemanticExtractPostprocessOutput
        ] = extract_postprocess,
        return_raw_outputs: bool = False,
        safe_mode: bool = False,
        progress_bar_desc: str = "Extracting",
        return_explanations: bool = False,
        strategy: ReasoningStrategy | None = None,
        # New batch processing parameters
        batch_size: int = 10,
        use_batch_processing: bool = True,
    ) -> pd.DataFrame:
        if lotus.settings.lm is None:
            raise ValueError(
                "The language model must be an instance of LM. Please configure a valid language model using lotus.settings.configure()"
            )

        # check that column exists
        for column in input_cols:
            if column not in self._obj.columns:
                raise ValueError(f"Column {column} not found in DataFrame")

        multimodal_data = task_instructions.df2multimodal_info(self._obj, input_cols)

        out = sem_extract(
            docs=multimodal_data,
            model=lotus.settings.lm,
            output_cols=output_cols,
            extract_quotes=extract_quotes,
            postprocessor=postprocessor,
            safe_mode=safe_mode,
            progress_bar_desc=progress_bar_desc,
            return_explanations=return_explanations,
            strategy=strategy,
            batch_size=batch_size,
            use_batch_processing=use_batch_processing,
        )

        new_df = self._obj.copy()
        indices = new_df.index.to_list()
        for i, output_dict in enumerate(out.outputs):
            if i >= len(indices):
                break
            for key, value in output_dict.items():
                if key not in new_df.columns:
                    new_df[key] = None
                new_df.loc[indices[i], key] = value

        if return_raw_outputs:
            new_df["raw_output"] = out.raw_outputs

        if return_explanations:
            new_df["explanation"] = out.explanations

        return new_df

    @operator_cache
    async def async_extract(
        self,
        input_cols: list[str],
        output_cols: dict[str, str | None],
        extract_quotes: bool = False,
        postprocessor: Callable[
            [list[str], lotus.models.LM, bool], SemanticExtractPostprocessOutput
        ] = extract_postprocess,
        return_raw_outputs: bool = False,
        safe_mode: bool = False,
        progress_bar_desc: str = "Extracting",
        return_explanations: bool = False,
        strategy: ReasoningStrategy | None = None,
        # New batch processing parameters
        batch_size: int = 10,
        use_batch_processing: bool = True,
        # Async-specific parameters
        max_concurrent_batches: int = 3,
    ) -> pd.DataFrame:
        """
        Asynchronous version of the sem_extract DataFrame accessor.

        This method provides the same functionality as the synchronous version
        but with async/await support, allowing for concurrent processing of
        multiple batches and better utilization of I/O wait times.

        Args:
            input_cols (list[str]): The columns that the model should extract
                information from. These columns will be used as input to the
                language model.
            output_cols (dict[str, str | None]): A mapping from desired output
                column names to optional descriptions. The descriptions help guide
                the model on what to extract. For example: {"sentiment": "positive/negative/neutral",
                "confidence": "0-1 scale"}.
            extract_quotes (bool, optional): Whether to extract supporting quotes
                from the source text for each extracted value. Defaults to False.
            postprocessor (Callable, optional): A function to post-process the model
                outputs. Should take (outputs, model, return_explanations) and return
                SemanticExtractPostprocessOutput. Defaults to extract_postprocess.
            return_raw_outputs (bool, optional): Whether to include raw model
                outputs in the output DataFrame. Useful for debugging.
                Defaults to False.
            safe_mode (bool, optional): Whether to enable safe mode with cost
                estimation. Defaults to False.
            progress_bar_desc (str, optional): Description for the progress bar.
                Defaults to "Extracting".
            return_explanations (bool, optional): Whether to include explanations
                in the output DataFrame. Useful for debugging and understanding
                model reasoning. Defaults to False.
            strategy (ReasoningStrategy | None, optional): The reasoning strategy
                to use. Can be None, COT, or ZS_COT. Defaults to None.
            batch_size (int, optional): Number of documents to process in each batch
                when using batch processing. Defaults to 10.
            use_batch_processing (bool, optional): Whether to use batch processing to
                share system prompts and examples across documents. Defaults to True.
            max_concurrent_batches (int, optional): Maximum number of batches to process
                concurrently. Defaults to 3.

        Returns:
            pd.DataFrame: A DataFrame containing the original data plus the
                extracted attributes as new columns.

        Raises:
            ValueError: If the language model is not configured, if specified
                input columns don't exist in the DataFrame, or if there are
                other configuration issues.

        Example:
            >>> import asyncio
            >>> import pandas as pd
            >>> import lotus
            >>> from lotus.models import LM
            >>> lotus.settings.configure(lm=LM(model="gpt-4o-mini"))

            >>> df = pd.DataFrame({
            ...     'text': ['Great product!', 'Terrible service'],
            ...     'rating': [5, 1]
            ... })

            >>> result_df = await df.sem_extract.async_extract(
            ...     ['text'],
            ...     {'sentiment': 'positive/negative/neutral', 'emotion': 'joy/anger/sadness'}
            ... )
        """
        if lotus.settings.lm is None:
            raise ValueError(
                "The language model must be an instance of LM. Please configure a valid language model using lotus.settings.configure()"
            )

        # check that column exists
        for column in input_cols:
            if column not in self._obj.columns:
                raise ValueError(f"Column {column} not found in DataFrame")

        multimodal_data = task_instructions.df2multimodal_info(self._obj, input_cols)

        out = await sem_extract_async(
            docs=multimodal_data,
            model=lotus.settings.lm,
            output_cols=output_cols,
            extract_quotes=extract_quotes,
            postprocessor=postprocessor,
            safe_mode=safe_mode,
            progress_bar_desc=progress_bar_desc,
            return_explanations=return_explanations,
            strategy=strategy,
            batch_size=batch_size,
            use_batch_processing=use_batch_processing,
            max_concurrent_batches=max_concurrent_batches,
        )

        new_df = self._obj.copy()
        indices = new_df.index.to_list()
        for i, output_dict in enumerate(out.outputs):
            if i >= len(indices):
                break
            for key, value in output_dict.items():
                if key not in new_df.columns:
                    new_df[key] = None
                new_df.loc[indices[i], key] = value

        if return_raw_outputs:
            new_df["raw_output"] = out.raw_outputs

        if return_explanations:
            new_df["explanation"] = out.explanations

        return new_df
