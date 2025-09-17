import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pandas as pd

import lotus.models
from lotus.cache import operator_cache
from lotus.templates import task_instructions
from lotus.types import LMOutput, SemanticAggOutput


def sem_agg(
    docs: list[str],
    model: lotus.models.LM,
    user_instruction: str,
    partition_ids: list[int],
    safe_mode: bool = False,
    progress_bar_desc: str = "Aggregating",
    use_async: bool = False,
    max_concurrent_batches: int = 4,
    max_thread_workers: int = 8,
) -> SemanticAggOutput:
    """
    Aggregates multiple documents into a single answer using a language model.

    This function implements a hierarchical aggregation approach where documents are
    processed in batches and progressively combined until a single coherent answer
    is produced. The aggregation uses different templates for leaf-level documents
    and intermediate summaries. Can use async processing for better performance.

    Args:
        docs (list[str]): The list of documents to aggregate. Each document should
            be a string containing the text content to be aggregated.
        model (lotus.models.LM): The language model instance to use for aggregation.
            Must be properly configured with appropriate API keys and settings.
        user_instruction (str): The natural language instruction that guides the
            aggregation process. This instruction tells the model how to combine
            the information from multiple documents.
        partition_ids (list[int]): The partition IDs for the documents. Documents
            with the same partition ID will be aggregated together. This allows
            for grouping-related documents for more coherent aggregation.
        safe_mode (bool, optional): Whether to enable safe mode. Currently not
            implemented. Defaults to False.
        progress_bar_desc (str, optional): Description for the progress bar.
            Defaults to "Aggregating".
        use_async (bool, optional): Whether to use async processing for better
            performance. Defaults to False.
        max_concurrent_batches (int, optional): Maximum number of concurrent batches
            to process at each tree level when using async. Defaults to 4.
        max_thread_workers (int, optional): Maximum number of threads for CPU-intensive
            operations when using async. Defaults to 8.

    Returns:
        SemanticAggOutput: An object containing the aggregated outputs as a list
            of strings. Typically contains a single aggregated answer.

    Raises:
        ValueError: If the model is not properly configured or if there are
            issues with the input parameters.

    Example:
        >>> docs = ["Document 1 content", "Document 2 content"]
        >>> model = LM(model="gpt-4o")
        >>> result = sem_agg(docs, model, "Summarize the key points", [0, 0])
        >>> print(result.outputs[0])
        
        # Using async processing for better performance
        >>> result = sem_agg(docs, model, "Summarize the key points", [0, 0], use_async=True)
        >>> print(result.outputs[0])
    """
    # Create templates and formatters using shared functions
    leaf_template, node_template = _create_aggregation_templates(user_instruction)
    _, _, doc_formatter = _create_doc_formatters()

    if safe_mode:
        # TODO: implement safe mode
        lotus.logger.warning("Safe mode is not implemented yet")

    # Choose between sync and async processing
    if use_async:
        return asyncio.run(sem_agg_async(
            docs, model, user_instruction, partition_ids, 
            safe_mode, progress_bar_desc, max_concurrent_batches, max_thread_workers
        ))

    tree_level = 0
    summaries: list[str] = []
    new_partition_ids: list[int] = []
    
    while len(docs) != len(set(partition_ids)) or summaries == []:
        # Build batches using shared function
        batch, new_partition_ids = _build_batches(
            docs, partition_ids, model, leaf_template, node_template, 
            doc_formatter, tree_level, model.count_tokens
        )

        lm_output: LMOutput = model(batch, progress_bar_desc=progress_bar_desc)

        summaries = lm_output.outputs
        partition_ids = new_partition_ids
        new_partition_ids = []

        docs = summaries
        lotus.logger.debug(f"Model outputs from tree level {tree_level}: {summaries}")
        tree_level += 1
        if safe_mode:
            model.print_total_usage()

    return SemanticAggOutput(outputs=summaries)


def _create_aggregation_templates(user_instruction: str) -> tuple[str, str]:
    """
    Create the leaf and node instruction templates for aggregation.

    Args:
        user_instruction (str): The user's instruction for aggregation.

    Returns:
        tuple[str, str]: A tuple containing (leaf_template, node_template).
    """
    leaf_instr_template = (
        "Your job is to provide an answer to the user's instruction given the context below from multiple documents.\n"
        "Remember that your job is to answer the user's instruction by combining all relevant information from all provided documents, into a single coherent answer.\n"
        "Do NOT copy the format of the sources! Instead output your answer in a coherent, well-structured manner that best answers the user instruction.\n"
        "You have limited space to provide your answer, so be concise and to the point.\n\n---\n\n"
        "Follow the following format.\n\nContext: relevant facts from multiple documents\n\n"
        "Instruction: the instruction provided by the user\n\nAnswer: Write your answer\n\n---\n\n"
        "Context: {{docs_str}}\n\n"
        f"Instruction:  {user_instruction}\n\nAnswer:\n"
    )

    node_instr_template = (
        "Your job is to provide an answer to the user's instruction given the context below from multiple sources.\n"
        "Note that each source may be formatted differently and contain information about several different documents.\n"
        "Remember that your job is to answer the user's instruction by combining all relevant information from all provided sources, into a single coherent answer.\n"
        "The sources may provide opposing viewpoints or complementary information.\n"
        "Be sure to include information from ALL relevant sources in your answer.\n"
        "Do NOT copy the format of the sources, instead output your answer in a coherent, well-structured manner that best answers the user instruction.\n"
        "You have limited space to provide your answer, so be concise and to the point.\n"
        "You may need to draw connections between sources to provide a complete answer.\n\n---\n\n"
        "Follow the following format.\n\nContext: relevant facts from multiple sources\n\n"
        "Instruction: the instruction provided by the user\n\nAnswer: Write your answer\n\n---\n\n"
        "Context: {{docs_str}}\n\n"
        f"Instruction:  {user_instruction}\n\nAnswer:\n"
    )
    
    return leaf_instr_template, node_instr_template


def _create_doc_formatters() -> tuple[callable, callable, callable]:
    """
    Create the document formatting functions.

    Returns:
        tuple[callable, callable, callable]: A tuple containing (leaf_formatter, node_formatter, doc_formatter).
    """
    def leaf_doc_formatter(doc: str, ctr: int) -> str:
        """
        Format a leaf-level document for inclusion in the prompt.

        Args:
            doc (str): The document content to format.
            ctr (int): The document counter for numbering.

        Returns:
            str: The formatted document string with counter prefix.
        """
        return f"\n\tDocument {ctr}: {doc}"

    def node_doc_formatter(doc: str, ctr: int) -> str:
        """
        Format an intermediate summary document for inclusion in the prompt.

        Args:
            doc (str): The summary content to format.
            ctr (int): The summary counter for numbering.

        Returns:
            str: The formatted summary string with counter prefix.
        """
        return f"\n\tSource {ctr}: {doc}"

    def doc_formatter(tree_level: int, doc: str, ctr: int) -> str:
        """
        Format documents based on their position in the aggregation tree.

        Args:
            tree_level (int): The current level in the aggregation tree.
                0 indicates leaf documents, >0 indicates intermediate summaries.
            doc (str): The document or summary content to format.
            ctr (int): The counter for numbering.

        Returns:
            str: The formatted document string.
        """
        return leaf_doc_formatter(doc, ctr) if tree_level == 0 else node_doc_formatter(doc, ctr)
    
    return leaf_doc_formatter, node_doc_formatter, doc_formatter


def _build_batches(
    docs: list[str],
    partition_ids: list[int],
    model: lotus.models.LM,
    leaf_template: str,
    node_template: str,
    doc_formatter: callable,
    tree_level: int,
    count_tokens_func: callable,
) -> tuple[list[list[dict[str, str]]], list[int]]:
    """
    Build batches of prompts for processing (synchronous version).

    Args:
        docs (list[str]): The documents to process.
        partition_ids (list[int]): The partition IDs for the documents.
        model (lotus.models.LM): The language model instance.
        leaf_template (str): The leaf-level instruction template.
        node_template (str): The node-level instruction template.
        doc_formatter (callable): The document formatting function.
        tree_level (int): The current tree level.
        count_tokens_func (callable): The token counting function.

    Returns:
        tuple[list[list[dict[str, str]]], list[int]]: A tuple containing (batches, new_partition_ids).
    """
    if not docs or not partition_ids:
        return [], []
    
    cur_partition_id = partition_ids[0]
    do_fold = len(partition_ids) == len(set(partition_ids))
    context_str = ""
    batch = []
    
    if tree_level == 0:
        template = leaf_template
    else:
        template = node_template
        
    template_tokens = count_tokens_func(template)
    context_tokens = 0
    doc_ctr = 1  # num docs in current prompt
    new_partition_ids = []

    for idx in range(len(docs)):
        partition_id = partition_ids[idx]
        formatted_doc = doc_formatter(tree_level, docs[idx], doc_ctr)
        new_tokens = count_tokens_func(formatted_doc)

        # Check if we need to create a new batch due to partition boundary
        if partition_id != cur_partition_id and not do_fold:
            # Close the current prompt for the previous partition
            if context_str:  # Only create batch if there's content
                prompt = template.replace("{{docs_str}}", context_str)
                lotus.logger.debug(f"Prompt added to batch for partition {cur_partition_id}: {prompt}")
                batch.append([{"role": "user", "content": prompt}])
                new_partition_ids.append(cur_partition_id)
            
            # Start new partition
            cur_partition_id = partition_id
            doc_ctr = 1
            context_str = ""
            context_tokens = 0
        
        # Check if we need to create a new batch due to token limit
        elif new_tokens + context_tokens + template_tokens > model.max_ctx_len - model.max_tokens:
            # Close the current prompt
            prompt = template.replace("{{docs_str}}", context_str)
            lotus.logger.debug(f"Prompt added to batch (token limit): {prompt}")
            batch.append([{"role": "user", "content": prompt}])
            new_partition_ids.append(cur_partition_id)
            doc_ctr = 1
            context_str = ""
            context_tokens = 0

        context_str += formatted_doc
        context_tokens += new_tokens
        doc_ctr += 1

    # Add the last prompt if there's any content
    if context_str:
        prompt = template.replace("{{docs_str}}", context_str)
        lotus.logger.debug(f"Final prompt added to batch: {prompt}")
        batch.append([{"role": "user", "content": prompt}])
        new_partition_ids.append(cur_partition_id)

    return batch, new_partition_ids


async def _build_batches_async(
    docs: list[str],
    partition_ids: list[int],
    model: lotus.models.LM,
    leaf_template: str,
    node_template: str,
    doc_formatter: callable,
    tree_level: int,
    count_tokens_func: callable,
) -> tuple[list[list[dict[str, str]]], list[int]]:
    """
    Build batches of prompts for processing.

    Args:
        docs (list[str]): The documents to process.
        partition_ids (list[int]): The partition IDs for the documents.
        model (lotus.models.LM): The language model instance.
        leaf_template (str): The leaf-level instruction template.
        node_template (str): The node-level instruction template.
        doc_formatter (callable): The document formatting function.
        tree_level (int): The current tree level.
        count_tokens_func (callable): The token counting function.

    Returns:
        tuple[list[list[dict[str, str]]], list[int]]: A tuple containing (batches, new_partition_ids).
    """
    if not docs or not partition_ids:
        return [], []
    
    cur_partition_id = partition_ids[0]
    do_fold = len(partition_ids) == len(set(partition_ids))
    context_str = ""
    batch = []
    
    if tree_level == 0:
        template = leaf_template
    else:
        template = node_template
        
    template_tokens = await count_tokens_func(template)
    context_tokens = 0
    doc_ctr = 1  # num docs in current prompt
    new_partition_ids = []

    for idx in range(len(docs)):
        partition_id = partition_ids[idx]
        formatted_doc = doc_formatter(tree_level, docs[idx], doc_ctr)
        new_tokens = await count_tokens_func(formatted_doc)

        # Check if we need to create a new batch due to partition boundary
        if partition_id != cur_partition_id and not do_fold:
            # Close the current prompt for the previous partition
            if context_str:  # Only create batch if there's content
                prompt = template.replace("{{docs_str}}", context_str)
                lotus.logger.debug(f"Prompt added to batch for partition {cur_partition_id}: {prompt}")
                batch.append([{"role": "user", "content": prompt}])
                new_partition_ids.append(cur_partition_id)
            
            # Start new partition
            cur_partition_id = partition_id
            doc_ctr = 1
            context_str = ""
            context_tokens = 0
        
        # Check if we need to create a new batch due to token limit
        elif new_tokens + context_tokens + template_tokens > model.max_ctx_len - model.max_tokens:
            # Close the current prompt
            prompt = template.replace("{{docs_str}}", context_str)
            lotus.logger.debug(f"Prompt added to batch (token limit): {prompt}")
            batch.append([{"role": "user", "content": prompt}])
            new_partition_ids.append(cur_partition_id)
            doc_ctr = 1
            context_str = ""
            context_tokens = 0

        context_str += formatted_doc
        context_tokens += new_tokens
        doc_ctr += 1

    # Add the last prompt if there's any content
    if context_str:
        prompt = template.replace("{{docs_str}}", context_str)
        lotus.logger.debug(f"Final prompt added to batch: {prompt}")
        batch.append([{"role": "user", "content": prompt}])
        new_partition_ids.append(cur_partition_id)

    return batch, new_partition_ids


async def sem_agg_async(
    docs: list[str],
    model: lotus.models.LM,
    user_instruction: str,
    partition_ids: list[int],
    safe_mode: bool = False,
    progress_bar_desc: str = "Aggregating",
    max_concurrent_batches: int = 4,
    max_thread_workers: int = 8,
) -> SemanticAggOutput:
    """
    Async version of semantic aggregation with optimized concurrent processing.

    This function implements a hierarchical aggregation approach where documents are
    processed in batches and progressively combined until a single coherent answer
    is produced. The async version uses asyncio and thread pools for concurrent
    processing of batches and CPU-intensive operations.

    Args:
        docs (list[str]): The list of documents to aggregate. Each document should
            be a string containing the text content to be aggregated.
        model (lotus.models.LM): The language model instance to use for aggregation.
            Must be properly configured with appropriate API keys and settings.
        user_instruction (str): The natural language instruction that guides the
            aggregation process. This instruction tells the model how to combine
            the information from multiple documents.
        partition_ids (list[int]): The partition IDs for the documents. Documents
            with the same partition ID will be aggregated together. This allows
            for grouping-related documents for more coherent aggregation.
        safe_mode (bool, optional): Whether to enable safe mode. Currently not
            implemented. Defaults to False.
        progress_bar_desc (str, optional): Description for the progress bar.
            Defaults to "Aggregating".
        max_concurrent_batches (int, optional): Maximum number of concurrent batches
            to process at each tree level. Defaults to 4.
        max_thread_workers (int, optional): Maximum number of threads for CPU-intensive
            operations. Defaults to 8.

    Returns:
        SemanticAggOutput: An object containing the aggregated outputs as a list
            of strings. Typically contains a single aggregated answer.

    Raises:
        ValueError: If the model is not properly configured or if there are
            issues with the input parameters.

    Example:
        >>> import asyncio
        >>> docs = ["Document 1 content", "Document 2 content"]
        >>> model = LM(model="gpt-4o")
        >>> result = await sem_agg_async(docs, model, "Summarize the key points", [0, 0])
        >>> print(result.outputs[0])
    """
    # Create templates and formatters using shared functions
    leaf_template, node_template = _create_aggregation_templates(user_instruction)
    _, _, doc_formatter = _create_doc_formatters()

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
                    lambda: model(batch, progress_bar_desc=progress_bar_desc)
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

    if safe_mode:
        # TODO: implement safe mode
        lotus.logger.warning("Safe mode is not implemented yet")

    tree_level = 0
    summaries: list[str] = []
    new_partition_ids: list[int] = []
    
    # Create semaphore to limit concurrent batches
    semaphore = asyncio.Semaphore(max_concurrent_batches)
    
    while len(docs) != len(set(partition_ids)) or summaries == []:
        # Build batches using async function
        batch, new_partition_ids = await _build_batches_async(
            docs, partition_ids, model, leaf_template, node_template, 
            doc_formatter, tree_level, count_tokens_async
        )

        # Process all batches concurrently with semaphore control
        if batch:
            # Split batch into smaller chunks for better concurrency control
            batch_chunks = [batch[i:i+1] for i in range(0, len(batch), 1)]
            
            # Process chunks concurrently
            tasks = [process_batch_async(chunk, semaphore) for chunk in batch_chunks]
            lm_outputs = await asyncio.gather(*tasks)
            
            # Combine outputs from all chunks
            all_outputs = []
            for lm_output in lm_outputs:
                all_outputs.extend(lm_output.outputs)
            
            lm_output = LMOutput(outputs=all_outputs)
        else:
            lm_output = LMOutput(outputs=[])

        summaries = lm_output.outputs
        partition_ids = new_partition_ids
        new_partition_ids = []

        docs = summaries
        lotus.logger.debug(f"Model outputs from tree level {tree_level}: {summaries}")
        tree_level += 1
        if safe_mode:
            model.print_total_usage()

    return SemanticAggOutput(outputs=summaries)


@pd.api.extensions.register_dataframe_accessor("sem_agg")
class SemAggDataframe:
    """
    Apply semantic aggregation over a DataFrame.

    This method performs semantic aggregation on the DataFrame content using
    a natural language instruction. It can process all columns or specific
    columns identified in the instruction, and supports grouped aggregation.
    Can use async processing for better performance with large datasets.

    Args:
        user_instruction (str): The natural language instruction that guides
            the aggregation process. Should describe what kind of aggregation
            or summary is desired.
        all_cols (bool, optional): Whether to use all columns in the DataFrame
            for aggregation. If False, only columns mentioned in the instruction
            will be used. Defaults to False.
        suffix (str, optional): The suffix for the output column name.
            Defaults to "_output".
        group_by (list[str] | None, optional): Column names to group by before
            aggregation. Each group will be aggregated separately. Defaults to None.
        safe_mode (bool, optional): Whether to enable safe mode for aggregation.
            Defaults to False.
        progress_bar_desc (str, optional): Description for the progress bar.
            Defaults to "Aggregating".
        use_async (bool, optional): Whether to use async processing for better
            performance. Defaults to False.
        max_concurrent_batches (int, optional): Maximum number of concurrent batches
            to process at each tree level when using async. Defaults to 4.
        max_thread_workers (int, optional): Maximum number of threads for CPU-intensive
            operations when using async. Defaults to 8.

    Returns:
        pd.DataFrame: A DataFrame containing the aggregated results. The output
            will have one row per group (if group_by is specified) or one row
            for the entire dataset.

    Raises:
        ValueError: If the language model is not configured, if specified
            columns don't exist in the DataFrame, or if there are other
            configuration issues.

    Example:
        >>> import pandas as pd
        >>> import lotus
        >>> from lotus.models import LM
        >>> lotus.settings.configure(lm=LM(model="gpt-4o-mini"))
        >>> df = pd.DataFrame({
        ...     'journal': ['Harry is happy and love cats', 'Harry is feeling nauseous', "Harry is doing homework"],
        ...     'date': ['Monday', 'Tuesday', "Tuesday"]
        ... })

        # Example 1: simple aggregation
        >>> df.sem_agg("Summarize the key points", all_cols=True)
        Aggregating: 100%|████████████████████████████████████████████████████████████████ 1/1 LM calls [00:01<00:00,  1.44s/it]
                                                    _output
        0  Harry experienced a range of emotions and acti...

        # Example 2: grouped aggregation with async processing
        >>> df.sem_agg("Summarize the key points", all_cols=True, group_by=["date"], use_async=True)
        Aggregating: 100%|████████████████████████████████████████████████████████████████ 1/1 LM calls [00:00<00:00,  1.42it/s]
        Aggregating: 100%|████████████████████████████████████████████████████████████████ 1/1 LM calls [00:00<00:00,  1.40it/s]
                                                    _output     date
        0  Harry is happy and has a fondness for cats, as...   Monday
        0  Harry is feeling nauseous and is also doing ho...  Tuesday

        # Example 3: aggregation with column reference and async processing
        >>> df.sem_agg("Summarize the entries from {journal}", use_async=True)
        Aggregating: 100%|████████████████████████████████████████████████████████████████ 1/1 LM calls [00:01<00:00,  1.05s/it]
                                                    _output
        0  Harry is currently experiencing a mix of emoti...
    """

    def __init__(self, pandas_obj: Any):
        """
        Initialize the semantic aggregation accessor.

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
            TypeError: If the object is not a pandas DataFrame.
        """
        pass

    @staticmethod
    def process_group(args):
        """
        Process a group of data for semantic aggregation.

        This static method is used for parallel processing of grouped data.
        It applies semantic aggregation to each group and adds the group
        identifier to the result.

        Args:
            args (tuple): A tuple containing (group_name, group, user_instruction,
                         all_cols, group_by, suffix, progress_bar_desc, use_async,
                         max_concurrent_batches, max_thread_workers).

        Returns:
            pd.DataFrame: The aggregated result for the group with group identifier.
        """
        (group_name, group, user_instruction, all_cols, group_by, suffix, 
         progress_bar_desc, use_async, max_concurrent_batches, max_thread_workers) = args
        result = group.sem_agg(
            user_instruction, 
            all_cols, 
            suffix, 
            None, 
            progress_bar_desc=progress_bar_desc,
            use_async=use_async,
            max_concurrent_batches=max_concurrent_batches,
            max_thread_workers=max_thread_workers
        )
        result[group_by] = group_name
        return result

    @operator_cache
    def __call__(
        self,
        user_instruction: str,
        all_cols: bool = False,
        suffix: str = "_output",
        group_by: list[str] | None = None,
        safe_mode: bool = False,
        progress_bar_desc: str = "Aggregating",
        use_async: bool = False,
        max_concurrent_batches: int = 4,
        max_thread_workers: int = 8,
    ) -> pd.DataFrame:
        if lotus.settings.lm is None:
            raise ValueError(
                "The language model must be an instance of LM. Please configure a valid language model using lotus.settings.configure()"
            )

        lotus.logger.debug(f"User instruction: {user_instruction}")
        if all_cols:
            col_li = list(self._obj.columns)
        else:
            col_li = lotus.nl_expression.parse_cols(user_instruction)
        lotus.logger.debug(f"Columns: {col_li}")

        # check that column exists
        for column in col_li:
            if column not in self._obj.columns:
                raise ValueError(f"column {column} not found in DataFrame. Given usr instruction: {user_instruction}")

        if group_by:
            grouped = self._obj.groupby(group_by)
            group_args = [
                (group_name, group, user_instruction, all_cols, group_by, suffix, 
                 progress_bar_desc, use_async, max_concurrent_batches, max_thread_workers)
                for group_name, group in grouped
            ]
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=lotus.settings.parallel_groupby_max_threads) as executor:
                return pd.concat(list(executor.map(SemAggDataframe.process_group, group_args)))

        # Sort df by partition_id if it exists
        if "_lotus_partition_id" in self._obj.columns:
            self._obj = self._obj.sort_values(by="_lotus_partition_id")
            partition_ids = self._obj["_lotus_partition_id"].tolist()
        else:
            partition_ids = [0] * len(self._obj)

        df_txt = task_instructions.df2text(self._obj, col_li)
        lotus.logger.debug(f"df_txt: {df_txt}")
        formatted_usr_instr = lotus.nl_expression.nle2str(user_instruction, col_li)
        lotus.logger.debug(f"formatted_usr_instr: {formatted_usr_instr}")

        answer = sem_agg(
            df_txt,
            lotus.settings.lm,
            formatted_usr_instr,
            partition_ids,
            safe_mode=safe_mode,
            progress_bar_desc=progress_bar_desc,
            use_async=use_async,
            max_concurrent_batches=max_concurrent_batches,
            max_thread_workers=max_thread_workers,
        )

        # package answer in a dataframe
        answer_df = pd.DataFrame(answer.outputs, columns=[suffix])
        return answer_df


