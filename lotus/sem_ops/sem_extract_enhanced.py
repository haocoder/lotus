"""
Enhanced sem_extract with chunking, batching, and async support.

This module provides an enhanced version of sem_extract that supports:
1. Intelligent document chunking for large documents
2. Hybrid batch processing for mixed document sizes
3. Async processing for better performance
4. Smart result aggregation for chunked documents
"""

import asyncio
from typing import Any, Callable, Optional, Tuple, List, Dict
from concurrent.futures import ThreadPoolExecutor
import json
import logging

# import pandas as pd  # Not used in this module

import lotus
from lotus.cache import operator_cache
from lotus.models import LM
from lotus.templates import task_instructions
from lotus.types import LMOutput, ReasoningStrategy, SemanticExtractOutput, SemanticExtractPostprocessOutput
from lotus.utils import show_safe_mode

from .postprocessors import extract_postprocess


class DocumentClassifier:
    """
    Intelligent document classifier that determines if documents need chunking.
    
    This class analyzes document size and model context limits to decide
    whether documents should be processed as-is or chunked first.
    """
    
    def __init__(self, model: LM, chunk_threshold: float = 0.8):
        """
        Initialize the document classifier.
        
        Args:
            model: The language model instance
            chunk_threshold: Threshold ratio for chunking decision (0.0-1.0)
        """
        self.model = model
        self.chunk_threshold = chunk_threshold
        # Calculate available context space
        self.available_tokens = model.max_ctx_len - model.max_tokens - 1000  # Reserve space
        self.chunk_threshold_tokens = int(self.available_tokens * chunk_threshold)
        
    def classify_documents(self, docs: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Classify documents into those that need chunking and those that don't.
        
        Args:
            docs: List of documents to classify
            
        Returns:
            Tuple of (docs_to_chunk, docs_to_process)
        """
        docs_to_chunk = []
        docs_to_process = []
        
        for doc in docs:
            if self._should_chunk_document(doc):
                docs_to_chunk.append(doc)
            else:
                docs_to_process.append(doc)
        
        lotus.logger.debug(f"Document classification: {len(docs_to_chunk)} need chunking, {len(docs_to_process)} can process directly")
        return docs_to_chunk, docs_to_process
    
    def _should_chunk_document(self, doc: Dict[str, Any]) -> bool:
        """
        Determine if a single document needs chunking.
        
        Args:
            doc: Document to analyze
            
        Returns:
            True if document should be chunked
        """
        text = doc.get('text', '')
        if not text:
            return False
        
        # Quick token estimation
        estimated_tokens = self._estimate_tokens(text)
        
        # If estimated tokens exceed threshold, need chunking
        return estimated_tokens > self.chunk_threshold_tokens
    
    def _estimate_tokens(self, text: str) -> int:
        """
        Quick token estimation for decision making.
        
        Args:
            text: Text to estimate
            
        Returns:
            Estimated token count
        """
        if not text:
            return 0
        
        # More accurate estimation: ~1.3 tokens per word for English
        # For very long texts, we need to be more conservative
        word_count = len(text.split())
        
        # Base estimation
        base_tokens = int(word_count * 1.3)
        
        # For very long texts, add some overhead
        if word_count > 1000:
            base_tokens = int(base_tokens * 1.1)
        
        return base_tokens


class DocumentChunker:
    """
    Document chunker that splits large documents into manageable pieces.
    
    Supports multiple chunking strategies and maintains document metadata.
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 50,
        strategy: str = "token"
    ):
        """
        Initialize the document chunker.
        
        Args:
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between consecutive chunks
            strategy: Chunking strategy ("token", "sentence", "paragraph")
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy
    
    def chunk_document(self, doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split a document into chunks.
        
        Args:
            doc: Document to chunk
            
        Returns:
            List of chunked documents
        """
        text = doc.get('text', '')
        if not text:
            return [doc]
        
        if self.strategy == "token":
            return self._token_based_chunking(text, doc)
        elif self.strategy == "sentence":
            return self._sentence_based_chunking(text, doc)
        elif self.strategy == "paragraph":
            return self._paragraph_based_chunking(text, doc)
        else:
            raise ValueError(f"Unsupported chunking strategy: {self.strategy}")
    
    def _token_based_chunking(self, text: str, original_doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Token-based chunking using TokenTextSplitter."""
        try:
            from lotus.utils import TokenTextSplitter
            
            splitter = TokenTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            
            chunks = splitter.split_text(text)
            chunked_docs = []
            
            for i, chunk_text in enumerate(chunks):
                chunk_doc = original_doc.copy()
                chunk_doc['text'] = chunk_text
                chunk_doc['chunk_id'] = f"{original_doc.get('doc_id', 'doc')}_{i}"
                chunk_doc['chunk_index'] = i
                chunk_doc['total_chunks'] = len(chunks)
                chunk_doc['is_chunk'] = True
                chunked_docs.append(chunk_doc)
            
            return chunked_docs
            
        except ImportError:
            # Fallback to simple text splitting
            return self._simple_text_chunking(text, original_doc)
    
    def _sentence_based_chunking(self, text: str, original_doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Sentence-based chunking."""
        import re
        
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sentence_length = len(sentence.split())
            
            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Start new chunk
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        chunked_docs = []
        for i, chunk_text in enumerate(chunks):
            chunk_doc = original_doc.copy()
            chunk_doc['text'] = chunk_text
            chunk_doc['chunk_id'] = f"{original_doc.get('doc_id', 'doc')}_{i}"
            chunk_doc['chunk_index'] = i
            chunk_doc['total_chunks'] = len(chunks)
            chunk_doc['is_chunk'] = True
            chunked_docs.append(chunk_doc)
        
        return chunked_docs
    
    def _paragraph_based_chunking(self, text: str, original_doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Paragraph-based chunking."""
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = []
        current_length = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            paragraph_length = len(paragraph.split())
            
            if current_length + paragraph_length > self.chunk_size and current_chunk:
                # Start new chunk
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = [paragraph]
                current_length = paragraph_length
            else:
                current_chunk.append(paragraph)
                current_length += paragraph_length
        
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        chunked_docs = []
        for i, chunk_text in enumerate(chunks):
            chunk_doc = original_doc.copy()
            chunk_doc['text'] = chunk_text
            chunk_doc['chunk_id'] = f"{original_doc.get('doc_id', 'doc')}_{i}"
            chunk_doc['chunk_index'] = i
            chunk_doc['total_chunks'] = len(chunks)
            chunk_doc['is_chunk'] = True
            chunked_docs.append(chunk_doc)
        
        return chunked_docs
    
    def _simple_text_chunking(self, text: str, original_doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Simple text chunking fallback."""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = ' '.join(chunk_words)
            chunks.append(chunk_text)
        
        chunked_docs = []
        for i, chunk_text in enumerate(chunks):
            chunk_doc = original_doc.copy()
            chunk_doc['text'] = chunk_text
            chunk_doc['chunk_id'] = f"{original_doc.get('doc_id', 'doc')}_{i}"
            chunk_doc['chunk_index'] = i
            chunk_doc['total_chunks'] = len(chunks)
            chunk_doc['is_chunk'] = True
            chunked_docs.append(chunk_doc)
        
        return chunked_docs


class ChunkResultAggregator:
    """
    Aggregates results from chunked documents back to original documents.
    
    Supports multiple aggregation strategies for different use cases.
    """
    
    def __init__(self, strategy: str = "merge"):
        """
        Initialize the aggregator.
        
        Args:
            strategy: Aggregation strategy ("merge", "vote", "weighted")
        """
        self.strategy = strategy
    
    def aggregate_results(
        self, 
        chunk_results: List[Dict[str, Any]], 
        original_doc: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Aggregate chunk results into a single result.
        
        Args:
            chunk_results: List of results from chunks
            original_doc: Original document
            
        Returns:
            Aggregated result
        """
        if not chunk_results:
            return {}
        
        if self.strategy == "merge":
            return self._merge_strategy(chunk_results)
        elif self.strategy == "vote":
            return self._vote_strategy(chunk_results)
        elif self.strategy == "weighted":
            return self._weighted_strategy(chunk_results, original_doc)
        else:
            raise ValueError(f"Unsupported aggregation strategy: {self.strategy}")
    
    def _merge_strategy(self, chunk_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge strategy: combine all values."""
        merged = {}
        
        for result in chunk_results:
            for key, value in result.items():
                if key not in merged:
                    merged[key] = []
                
                if isinstance(value, list):
                    merged[key].extend(value)
                else:
                    merged[key].append(value)
        
        # Convert single-item lists to values
        for key, value in merged.items():
            if isinstance(value, list) and len(value) == 1:
                merged[key] = value[0]
        
        return merged
    
    def _vote_strategy(self, chunk_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Vote strategy: select most common values."""
        from collections import Counter
        
        voted = {}
        
        # Get all unique keys
        all_keys = set()
        for result in chunk_results:
            all_keys.update(result.keys())
        
        for key in all_keys:
            values = [result.get(key) for result in chunk_results if key in result]
            if values:
                # Simple voting: most common value
                counter = Counter(values)
                voted[key] = counter.most_common(1)[0][0]
        
        return voted
    
    def _weighted_strategy(self, chunk_results: List[Dict[str, Any]], original_doc: Dict[str, Any]) -> Dict[str, Any]:
        """Weighted strategy: weight results by chunk position."""
        weighted = {}
        
        # Get all unique keys
        all_keys = set()
        for result in chunk_results:
            all_keys.update(result.keys())
        
        for key in all_keys:
            weighted_values = []
            
            for i, result in enumerate(chunk_results):
                if key in result:
                    # Weight: middle chunks get higher weight
                    total_chunks = len(chunk_results)
                    weight = 1.0
                    
                    if total_chunks > 1:
                        # Higher weight for middle chunks
                        middle_weight = 2.0
                        edge_weight = 1.0
                        
                        if i == 0 or i == total_chunks - 1:
                            weight = edge_weight
                        else:
                            weight = middle_weight
                    
                    weighted_values.append((result[key], weight))
            
            if weighted_values:
                # Select most weighted value
                value_weights = {}
                for value, weight in weighted_values:
                    if value not in value_weights:
                        value_weights[value] = 0
                    value_weights[value] += weight
                
                weighted[key] = max(value_weights.items(), key=lambda x: x[1])[0]
        
        return weighted


class HybridBatchProcessor:
    """
    Hybrid batch processor that handles both regular and chunked documents.
    
    Combines the efficiency of batch processing with the flexibility of chunking.
    """
    
    def __init__(
        self,
        model: LM,
        batch_size: int = 10,
        chunk_size: int = 1000,
        chunk_overlap: int = 50,
        chunking_strategy: str = "token",
        aggregation_strategy: str = "merge"
    ):
        """
        Initialize the hybrid batch processor.
        
        Args:
            model: Language model instance
            batch_size: Batch size for processing
            chunk_size: Size of chunks
            chunk_overlap: Overlap between chunks
            chunking_strategy: Strategy for chunking
            aggregation_strategy: Strategy for aggregating results
        """
        self.model = model
        self.batch_size = batch_size
        self.chunker = DocumentChunker(chunk_size, chunk_overlap, chunking_strategy)
        self.aggregator = ChunkResultAggregator(aggregation_strategy)
    
    def process_mixed_documents(
        self,
        docs_to_chunk: List[Dict[str, Any]],
        docs_to_process: List[Dict[str, Any]],
        output_cols: Dict[str, str | None],
        extract_quotes: bool = False,
        postprocessor: Callable = extract_postprocess,
        safe_mode: bool = False,
        progress_bar_desc: str = "Extracting",
        return_explanations: bool = False,
        strategy: ReasoningStrategy | None = None,
    ) -> SemanticExtractOutput:
        """
        Process mixed document collection with chunking and batching.
        
        Args:
            docs_to_chunk: Documents that need chunking
            docs_to_process: Documents that can be processed directly
            output_cols: Output column definitions
            extract_quotes: Whether to extract quotes
            postprocessor: Post-processing function
            safe_mode: Whether to enable safe mode
            progress_bar_desc: Progress bar description
            return_explanations: Whether to return explanations
            strategy: Reasoning strategy
            
        Returns:
            SemanticExtractOutput with processed results
        """
        # 1. Chunk documents that need chunking
        chunked_docs = []
        chunk_mapping = {}
        
        for doc in docs_to_chunk:
            chunks = self.chunker.chunk_document(doc)
            chunked_docs.extend(chunks)
            
            # Record mapping for later aggregation
            doc_id = doc.get('doc_id', id(doc))
            chunk_mapping[doc_id] = {
                'original_doc': doc,
                'chunks': chunks,
                'chunk_count': len(chunks)
            }
        
        # 2. Combine all documents for batch processing
        all_docs = docs_to_process + chunked_docs
        
        if not all_docs:
            return SemanticExtractOutput(outputs=[], raw_outputs=[], explanations=[])
        
        # 3. Process with batch processing
        if len(all_docs) > 1:
            batch_result = self._process_with_batching(
                all_docs, output_cols, extract_quotes, postprocessor,
                safe_mode, progress_bar_desc, return_explanations, strategy
            )
        else:
            batch_result = self._process_individual(
                all_docs, output_cols, extract_quotes, postprocessor,
                safe_mode, progress_bar_desc, return_explanations, strategy
            )
        
        # 4. Separate and aggregate results
        return self._separate_and_aggregate_results(
            batch_result, docs_to_process, chunk_mapping
        )
    
    def _process_with_batching(
        self,
        docs: List[Dict[str, Any]],
        output_cols: Dict[str, str | None],
        extract_quotes: bool,
        postprocessor: Callable,
        safe_mode: bool,
        progress_bar_desc: str,
        return_explanations: bool,
        strategy: ReasoningStrategy | None,
    ) -> SemanticExtractOutput:
        """Process documents using batch processing."""
        from lotus.templates.batch_extract_formatter import batch_extract_formatter
        from lotus.sem_ops.postprocessors import batch_extract_parser
        
        try:
            # Use batch formatter
            batched_inputs = batch_extract_formatter(
                self.model, docs, output_cols, extract_quotes, strategy, self.batch_size
            )
            
            lotus.logger.debug(f"Hybrid batch inputs count: {len(batched_inputs)}")
            
            if safe_mode:
                estimated_total_calls = len(batched_inputs)
                estimated_total_cost = sum(self.model.count_tokens(input) for input in batched_inputs)
                show_safe_mode(estimated_total_cost, estimated_total_calls)
            
            # Call model with batch inputs
            lm_output: LMOutput = self.model(
                batched_inputs,
                response_format={"type": "json_object"},
                progress_bar_desc=progress_bar_desc
            )
            
            # Parse batch responses
            outputs, raw_outputs, explanations = batch_extract_parser(
                lm_output.outputs, self.model, len(docs)
            )
            
            if safe_mode:
                self.model.print_total_usage()
            
            return SemanticExtractOutput(
                raw_outputs=raw_outputs,
                outputs=outputs,
                explanations=explanations,
            )
            
        except Exception as e:
            lotus.logger.warning(f"Hybrid batch processing failed: {e}, falling back to individual processing")
            return self._process_individual(
                docs, output_cols, extract_quotes, postprocessor,
                safe_mode, progress_bar_desc, return_explanations, strategy
            )
    
    def _process_individual(
        self,
        docs: List[Dict[str, Any]],
        output_cols: Dict[str, str | None],
        extract_quotes: bool,
        postprocessor: Callable,
        safe_mode: bool,
        progress_bar_desc: str,
        return_explanations: bool,
        strategy: ReasoningStrategy | None,
    ) -> SemanticExtractOutput:
        """Process documents individually."""
        # Prepare model inputs
        inputs = []
        for doc in docs:
            prompt = task_instructions.extract_formatter(self.model, doc, output_cols, extract_quotes, strategy)
            inputs.append(prompt)
        
        # Check if safe_mode is enabled
        if safe_mode:
            estimated_cost = sum(self.model.count_tokens(input) for input in inputs)
            estimated_LM_calls = len(docs)
            show_safe_mode(estimated_cost, estimated_LM_calls)
        
        # Call model
        lm_output: LMOutput = self.model(
            inputs, 
            response_format={"type": "json_object"}, 
            progress_bar_desc=progress_bar_desc
        )
        
        # Post process results
        postprocess_output = postprocessor(lm_output.outputs, self.model, return_explanations)
        
        if safe_mode:
            self.model.print_total_usage()
        
        return SemanticExtractOutput(
            raw_outputs=postprocess_output.raw_outputs,
            outputs=postprocess_output.outputs,
            explanations=postprocess_output.explanations,
        )
    
    def _separate_and_aggregate_results(
        self,
        batch_result: SemanticExtractOutput,
        docs_to_process: List[Dict[str, Any]],
        chunk_mapping: Dict[str, Dict[str, Any]]
    ) -> SemanticExtractOutput:
        """Separate results and aggregate chunked results."""
        # Separate results
        original_results = []
        chunk_results = {}
        
        result_index = 0
        
        # Process original document results
        for doc in docs_to_process:
            if result_index < len(batch_result.outputs):
                original_results.append(batch_result.outputs[result_index])
                result_index += 1
        
        # Process chunked document results
        for doc_id, mapping_info in chunk_mapping.items():
            chunk_count = mapping_info['chunk_count']
            doc_chunk_results = []
            
            for _ in range(chunk_count):
                if result_index < len(batch_result.outputs):
                    doc_chunk_results.append(batch_result.outputs[result_index])
                    result_index += 1
            
            if doc_chunk_results:
                chunk_results[doc_id] = doc_chunk_results
        
        # Aggregate chunked results
        aggregated_results = []
        for doc_id, mapping_info in chunk_mapping.items():
            if doc_id in chunk_results:
                aggregated = self.aggregator.aggregate_results(
                    chunk_results[doc_id],
                    mapping_info['original_doc']
                )
                aggregated_results.append(aggregated)
            else:
                # Fallback: empty result
                aggregated_results.append({})
        
        # Combine final results
        final_outputs = original_results + aggregated_results
        final_raw_outputs = batch_result.raw_outputs
        final_explanations = batch_result.explanations
        
        return SemanticExtractOutput(
            outputs=final_outputs,
            raw_outputs=final_raw_outputs,
            explanations=final_explanations,
        )
    
    async def process_mixed_documents_async(
        self,
        docs_to_chunk: List[Dict[str, Any]],
        docs_to_process: List[Dict[str, Any]],
        output_cols: Dict[str, str | None],
        extract_quotes: bool = False,
        postprocessor: Callable = extract_postprocess,
        safe_mode: bool = False,
        progress_bar_desc: str = "Extracting",
        return_explanations: bool = False,
        strategy: ReasoningStrategy | None = None,
        max_concurrent_batches: int = 3,
    ) -> SemanticExtractOutput:
        """
        Async version of process_mixed_documents.
        
        Args:
            docs_to_chunk: Documents that need chunking
            docs_to_process: Documents that can be processed directly
            output_cols: Output column definitions
            extract_quotes: Whether to extract quotes
            postprocessor: Post-processing function
            safe_mode: Whether to enable safe mode
            progress_bar_desc: Progress bar description
            return_explanations: Whether to return explanations
            strategy: Reasoning strategy
            max_concurrent_batches: Maximum concurrent batches
            
        Returns:
            SemanticExtractOutput with processed results
        """
        # 1. Chunk documents that need chunking
        chunked_docs = []
        chunk_mapping = {}
        
        for doc in docs_to_chunk:
            chunks = self.chunker.chunk_document(doc)
            chunked_docs.extend(chunks)
            
            # Record mapping for later aggregation
            doc_id = doc.get('doc_id', id(doc))
            chunk_mapping[doc_id] = {
                'original_doc': doc,
                'chunks': chunks,
                'chunk_count': len(chunks)
            }
        
        # 2. Combine all documents for batch processing
        all_docs = docs_to_process + chunked_docs
        
        if not all_docs:
            return SemanticExtractOutput(outputs=[], raw_outputs=[], explanations=[])
        
        # 3. Process with async batch processing
        if len(all_docs) > 1:
            batch_result = await self._process_with_async_batching(
                all_docs, output_cols, extract_quotes, postprocessor,
                safe_mode, progress_bar_desc, return_explanations, strategy,
                max_concurrent_batches
            )
        else:
            batch_result = await self._process_individual_async(
                all_docs, output_cols, extract_quotes, postprocessor,
                safe_mode, progress_bar_desc, return_explanations, strategy
            )
        
        # 4. Separate and aggregate results
        return self._separate_and_aggregate_results(
            batch_result, docs_to_process, chunk_mapping
        )
    
    async def _process_with_async_batching(
        self,
        docs: List[Dict[str, Any]],
        output_cols: Dict[str, str | None],
        extract_quotes: bool,
        postprocessor: Callable,
        safe_mode: bool,
        progress_bar_desc: str,
        return_explanations: bool,
        strategy: ReasoningStrategy | None,
        max_concurrent_batches: int,
    ) -> SemanticExtractOutput:
        """Process documents using async batch processing."""
        from lotus.templates.batch_extract_formatter import batch_extract_formatter
        from lotus.sem_ops.postprocessors import batch_extract_parser
        
        try:
            # Use batch formatter
            batched_inputs = batch_extract_formatter(
                self.model, docs, output_cols, extract_quotes, strategy, self.batch_size
            )
            
            lotus.logger.debug(f"Async hybrid batch inputs count: {len(batched_inputs)}")
            
            if safe_mode:
                estimated_total_calls = len(batched_inputs)
                estimated_total_cost = sum(self.model.count_tokens(input) for input in batched_inputs)
                show_safe_mode(estimated_total_cost, estimated_total_calls)
            
            # Process batches concurrently with semaphore
            semaphore = asyncio.Semaphore(max_concurrent_batches)
            
            async def process_batch(batch_input: List[Dict[str, str]]) -> LMOutput:
                """Process a single batch asynchronously."""
                async with semaphore:
                    if hasattr(self.model, 'async_call'):
                        return await self.model.async_call(
                            batch_input,
                            response_format={"type": "json_object"},
                            progress_bar_desc=progress_bar_desc
                        )
                    else:
                        # Fallback to synchronous call
                        return self.model(
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
                    lotus.logger.warning(f"Async batch {i+1} failed: {result}")
                    failed_batches.append(i)
                else:
                    successful_outputs.extend(result.outputs)
            
            # If some batches failed, fall back to individual processing for failed documents
            if failed_batches:
                lotus.logger.warning(f"Failed async batches: {failed_batches}, falling back to individual processing")
                # Calculate which documents were in failed batches
                failed_docs = []
                for batch_idx in failed_batches:
                    start_idx = batch_idx * self.batch_size
                    end_idx = min(start_idx + self.batch_size, len(docs))
                    failed_docs.extend(docs[start_idx:end_idx])
                
                # Process failed documents individually
                if failed_docs:
                    individual_result = await self._process_individual_async(
                        failed_docs, output_cols, extract_quotes, postprocessor,
                        safe_mode, progress_bar_desc, return_explanations, strategy
                    )
                    successful_outputs.extend(individual_result.raw_outputs)
            
            # Parse batch responses
            outputs, raw_outputs, explanations = batch_extract_parser(
                successful_outputs, self.model, len(docs)
            )
            
            if safe_mode:
                self.model.print_total_usage()
            
            return SemanticExtractOutput(
                raw_outputs=raw_outputs,
                outputs=outputs,
                explanations=explanations,
            )
            
        except Exception as e:
            lotus.logger.warning(f"Async hybrid batch processing failed: {e}, falling back to individual processing")
            return await self._process_individual_async(
                docs, output_cols, extract_quotes, postprocessor,
                safe_mode, progress_bar_desc, return_explanations, strategy
            )
    
    async def _process_individual_async(
        self,
        docs: List[Dict[str, Any]],
        output_cols: Dict[str, str | None],
        extract_quotes: bool,
        postprocessor: Callable,
        safe_mode: bool,
        progress_bar_desc: str,
        return_explanations: bool,
        strategy: ReasoningStrategy | None,
    ) -> SemanticExtractOutput:
        """Process documents individually asynchronously."""
        # Prepare model inputs
        inputs = []
        for doc in docs:
            prompt = task_instructions.extract_formatter(self.model, doc, output_cols, extract_quotes, strategy)
            inputs.append(prompt)
        
        # Check if safe_mode is enabled
        if safe_mode:
            estimated_cost = sum(self.model.count_tokens(input) for input in inputs)
            estimated_LM_calls = len(docs)
            show_safe_mode(estimated_cost, estimated_LM_calls)
        
        # Call model asynchronously
        if hasattr(self.model, 'async_call'):
            lm_output: LMOutput = await self.model.async_call(
                inputs,
                response_format={"type": "json_object"},
                progress_bar_desc=progress_bar_desc
            )
        else:
            # Fallback to synchronous call
            lm_output: LMOutput = self.model(
                inputs,
                response_format={"type": "json_object"},
                progress_bar_desc=progress_bar_desc
            )
        
        # Post process results
        postprocess_output = postprocessor(lm_output.outputs, self.model, return_explanations)
        
        if safe_mode:
            self.model.print_total_usage()
        
        return SemanticExtractOutput(
            raw_outputs=postprocess_output.raw_outputs,
            outputs=postprocess_output.outputs,
            explanations=postprocess_output.explanations,
        )


def sem_extract_enhanced(
    docs: List[Dict[str, Any]],
    model: LM,
    output_cols: Dict[str, str | None],
    extract_quotes: bool = False,
    postprocessor: Callable = extract_postprocess,
    safe_mode: bool = False,
    progress_bar_desc: str = "Extracting",
    return_explanations: bool = False,
    strategy: ReasoningStrategy | None = None,
    # Batch processing parameters
    batch_size: int = 10,
    use_batch_processing: bool = True,
    # Chunking parameters
    enable_chunking: bool = False,
    chunk_size: int = 1000,
    chunk_overlap: int = 50,
    chunking_strategy: str = "token",
    aggregation_strategy: str = "merge",
    chunk_threshold: float = 0.8,
) -> SemanticExtractOutput:
    """
    Enhanced sem_extract with chunking, batching, and async support.
    
    This function implements a three-tier processing strategy:
    1. Document classification: Determine which documents need chunking
    2. Hybrid processing: Process regular and chunked documents together
    3. Result aggregation: Combine chunked results back to original documents
    
    Args:
        docs: List of documents to extract from
        model: Language model instance
        output_cols: Mapping of output column names to descriptions
        extract_quotes: Whether to extract supporting quotes
        postprocessor: Post-processing function
        safe_mode: Whether to enable safe mode
        progress_bar_desc: Description for progress bar
        return_explanations: Whether to return explanations
        strategy: Reasoning strategy to use
        batch_size: Number of documents per batch
        use_batch_processing: Whether to use batch processing
        enable_chunking: Whether to enable document chunking
        chunk_size: Size of document chunks
        chunk_overlap: Overlap between chunks
        chunking_strategy: Strategy for chunking ("token", "sentence", "paragraph")
        aggregation_strategy: Strategy for aggregating results ("merge", "vote", "weighted")
        chunk_threshold: Threshold for chunking decision (0.0-1.0)
        
    Returns:
        SemanticExtractOutput with extracted results
        
    Example:
        >>> docs = [
        ...     {"text": "Short document"},
        ...     {"text": "Very long document that exceeds context limits..."}
        ... ]
        >>> model = LM(model="gpt-4o")
        >>> output_cols = {"sentiment": "positive/negative/neutral"}
        >>> result = sem_extract_enhanced(
        ...     docs, model, output_cols,
        ...     enable_chunking=True,
        ...     use_batch_processing=True
        ... )
    """
    if not docs:
        return SemanticExtractOutput(outputs=[], raw_outputs=[], explanations=[])
    
    # Strategy 1: If chunking is disabled, use existing logic
    if not enable_chunking:
        if use_batch_processing and len(docs) > 1:
            from .sem_extract import _sem_extract_batch
            return _sem_extract_batch(
                docs, model, output_cols, extract_quotes, postprocessor,
                safe_mode, progress_bar_desc, return_explanations, strategy, batch_size
            )
        else:
            from .sem_extract import _sem_extract_individual
            return _sem_extract_individual(
                docs, model, output_cols, extract_quotes, postprocessor,
                safe_mode, progress_bar_desc, return_explanations, strategy
            )
    
    # Strategy 2: Enable chunking - use hybrid processing
    try:
        # Classify documents
        classifier = DocumentClassifier(model, chunk_threshold)
        docs_to_chunk, docs_to_process = classifier.classify_documents(docs)
        
        # Create hybrid processor
        processor = HybridBatchProcessor(
            model, batch_size, chunk_size, chunk_overlap,
            chunking_strategy, aggregation_strategy
        )
        
        # Process mixed documents
        return processor.process_mixed_documents(
            docs_to_chunk, docs_to_process, output_cols,
            extract_quotes, postprocessor, safe_mode,
            progress_bar_desc, return_explanations, strategy
        )
        
    except Exception as e:
        lotus.logger.warning(f"Enhanced processing failed: {e}, falling back to standard processing")
        # Fallback to standard processing
        if use_batch_processing and len(docs) > 1:
            from .sem_extract import _sem_extract_batch
            return _sem_extract_batch(
                docs, model, output_cols, extract_quotes, postprocessor,
                safe_mode, progress_bar_desc, return_explanations, strategy, batch_size
            )
        else:
            from .sem_extract import _sem_extract_individual
            return _sem_extract_individual(
                docs, model, output_cols, extract_quotes, postprocessor,
                safe_mode, progress_bar_desc, return_explanations, strategy
            )


async def sem_extract_enhanced_async(
    docs: List[Dict[str, Any]],
    model: LM,
    output_cols: Dict[str, str | None],
    extract_quotes: bool = False,
    postprocessor: Callable = extract_postprocess,
    safe_mode: bool = False,
    progress_bar_desc: str = "Extracting",
    return_explanations: bool = False,
    strategy: ReasoningStrategy | None = None,
    # Batch processing parameters
    batch_size: int = 10,
    use_batch_processing: bool = True,
    max_concurrent_batches: int = 3,
    # Chunking parameters
    enable_chunking: bool = False,
    chunk_size: int = 1000,
    chunk_overlap: int = 50,
    chunking_strategy: str = "token",
    aggregation_strategy: str = "merge",
    chunk_threshold: float = 0.8,
) -> SemanticExtractOutput:
    """
    Async version of enhanced sem_extract with chunking and batching.
    
    Provides the same functionality as sem_extract_enhanced but with async/await
    support for concurrent processing and better I/O utilization.
    
    Args:
        docs: List of documents to extract from
        model: Language model instance
        output_cols: Mapping of output column names to descriptions
        extract_quotes: Whether to extract supporting quotes
        postprocessor: Post-processing function
        safe_mode: Whether to enable safe mode
        progress_bar_desc: Description for progress bar
        return_explanations: Whether to return explanations
        strategy: Reasoning strategy to use
        batch_size: Number of documents per batch
        use_batch_processing: Whether to use batch processing
        max_concurrent_batches: Maximum concurrent batches
        enable_chunking: Whether to enable document chunking
        chunk_size: Size of document chunks
        chunk_overlap: Overlap between chunks
        chunking_strategy: Strategy for chunking
        aggregation_strategy: Strategy for aggregating results
        chunk_threshold: Threshold for chunking decision
        
    Returns:
        SemanticExtractOutput with extracted results
    """
    if not docs:
        return SemanticExtractOutput(outputs=[], raw_outputs=[], explanations=[])
    
    # Strategy 1: If chunking is disabled, use existing async logic
    if not enable_chunking:
        from .sem_extract import sem_extract_async
        return await sem_extract_async(
            docs, model, output_cols, extract_quotes, postprocessor,
            safe_mode, progress_bar_desc, return_explanations, strategy,
            batch_size, use_batch_processing, max_concurrent_batches
        )
    
    # Strategy 2: Enable chunking - use async hybrid processing
    try:
        # Classify documents
        classifier = DocumentClassifier(model, chunk_threshold)
        docs_to_chunk, docs_to_process = classifier.classify_documents(docs)
        
        # Create hybrid processor
        processor = HybridBatchProcessor(
            model, batch_size, chunk_size, chunk_overlap,
            chunking_strategy, aggregation_strategy
        )
        
        # Process mixed documents asynchronously
        return await processor.process_mixed_documents_async(
            docs_to_chunk, docs_to_process, output_cols,
            extract_quotes, postprocessor, safe_mode,
            progress_bar_desc, return_explanations, strategy,
            max_concurrent_batches
        )
        
    except Exception as e:
        lotus.logger.warning(f"Enhanced async processing failed: {e}, falling back to standard async processing")
        # Fallback to standard async processing
        from .sem_extract import sem_extract_async
        return await sem_extract_async(
            docs, model, output_cols, extract_quotes, postprocessor,
            safe_mode, progress_bar_desc, return_explanations, strategy,
            batch_size, use_batch_processing, max_concurrent_batches
        )
