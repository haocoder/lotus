#!/usr/bin/env python3
"""
Test cases for batch processing functionality in sem_filter.

This module contains comprehensive tests for the batch processing features
of the sem_filter operator, including shared prompt optimization and
batch response parsing.
"""

import pytest
from typing import Any, Dict, List
from unittest.mock import Mock, patch

import lotus
from lotus.models import LM
from lotus.sem_ops.sem_filter import sem_filter, _sem_filter_batch, _sem_filter_individual
from lotus.templates.task_instructions import batch_filter_formatter
from lotus.sem_ops.postprocessors import batch_filter_parser
from lotus.types import ReasoningStrategy, SemanticFilterOutput


class TestBatchFilterFormatter:
    """Test cases for batch filter formatter."""
    
    def test_batch_filter_formatter_basic(self) -> None:
        """Test basic batch filter formatter functionality."""
        model = Mock(spec=LM)
        model.is_deepseek.return_value = False
        
        docs = [
            {"text": "This is a positive review"},
            {"text": "This is a negative review"},
            {"text": "This is a neutral review"},
        ]
        user_instruction = "Is this a positive sentiment?"
        
        result = batch_filter_formatter(
            model=model,
            docs=docs,
            user_instruction=user_instruction,
            batch_size=3
        )
        
        assert len(result) == 1  # One batch for 3 documents
        assert len(result[0]) >= 2  # System prompt + user message
        
        # Check system prompt contains batch instructions
        system_prompt = result[0][0]["content"]
        assert "3 documents" in system_prompt
        assert "JSON format" in system_prompt
        assert "document_id" in system_prompt
        
        # Check user message contains all documents
        user_message = result[0][1]["content"]
        assert "Document 1:" in str(user_message)
        assert "Document 2:" in str(user_message)
        assert "Document 3:" in str(user_message)
        assert "Claim: Is this a positive sentiment?" in str(user_message)
    
    def test_batch_filter_formatter_with_examples(self) -> None:
        """Test batch filter formatter with few-shot examples."""
        model = Mock(spec=LM)
        model.is_deepseek.return_value = False
        
        docs = [
            {"text": "This is a positive review"},
            {"text": "This is a negative review"},
        ]
        user_instruction = "Is this a positive sentiment?"
        
        examples_multimodal_data = [
            {"text": "Great product!"},
            {"text": "Terrible quality"},
        ]
        examples_answer = [True, False]
        
        result = batch_filter_formatter(
            model=model,
            docs=docs,
            user_instruction=user_instruction,
            examples_multimodal_data=examples_multimodal_data,
            examples_answer=examples_answer,
            batch_size=2
        )
        
        assert len(result) == 1
        # Should have system prompt + examples + user message
        assert len(result[0]) >= 5  # system + 2 examples (user+assistant each) + user message
    
    def test_batch_filter_formatter_multiple_batches(self) -> None:
        """Test batch filter formatter with multiple batches."""
        model = Mock(spec=LM)
        model.is_deepseek.return_value = False
        
        docs = [
            {"text": f"Document {i}"} for i in range(1, 6)
        ]
        user_instruction = "Is this about technology?"
        
        result = batch_filter_formatter(
            model=model,
            docs=docs,
            user_instruction=user_instruction,
            batch_size=2
        )
        
        # Should create 3 batches: [doc1, doc2], [doc3, doc4], [doc5]
        assert len(result) == 3
        
        # Check first batch
        assert "2 documents" in result[0][0]["content"]
        assert "Document 1:" in str(result[0][1]["content"])
        assert "Document 2:" in str(result[0][1]["content"])
        
        # Check second batch
        assert "2 documents" in result[1][0]["content"]
        assert "Document 3:" in str(result[1][1]["content"])
        assert "Document 4:" in str(result[1][1]["content"])
        
        # Check third batch (smaller)
        assert "1 document" in result[2][0]["content"]
        assert "Document 5:" in str(result[2][1]["content"])
    
    def test_batch_filter_formatter_with_cot(self) -> None:
        """Test batch filter formatter with chain-of-thought reasoning."""
        model = Mock(spec=LM)
        model.is_deepseek.return_value = False
        
        docs = [{"text": "This is a positive review"}]
        user_instruction = "Is this a positive sentiment?"
        
        result = batch_filter_formatter(
            model=model,
            docs=docs,
            user_instruction=user_instruction,
            strategy=ReasoningStrategy.COT,
            reasoning_instructions="Think step by step",
            batch_size=1
        )
        
        system_prompt = result[0][0]["content"]
        assert "Think step by step" in system_prompt
        assert "reasoning" in system_prompt


class TestBatchFilterParser:
    """Test cases for batch filter parser."""
    
    def test_batch_filter_parser_json_format(self) -> None:
        """Test batch filter parser with proper JSON format."""
        model = Mock(spec=LM)
        
        batch_outputs = [
            '''{
                "results": [
                    {"document_id": 1, "answer": true, "reasoning": "Positive words present"},
                    {"document_id": 2, "answer": false, "reasoning": "Negative words present"}
                ]
            }'''
        ]
        
        outputs, raw_outputs, explanations = batch_filter_parser(
            batch_outputs, model, default=True, expected_doc_count=2
        )
        
        assert outputs == [True, False]
        assert len(raw_outputs) == 1
        assert explanations == ["Positive words present", "Negative words present"]
    
    def test_batch_filter_parser_malformed_json(self) -> None:
        """Test batch filter parser with malformed JSON."""
        model = Mock(spec=LM)
        
        batch_outputs = [
            "True\nFalse\nTrue"  # Fallback format
        ]
        
        outputs, raw_outputs, explanations = batch_filter_parser(
            batch_outputs, model, default=True, expected_doc_count=3
        )
        
        assert outputs == [True, False, True]
        assert len(raw_outputs) == 1
        assert len(explanations) == 3
    
    def test_batch_filter_parser_missing_results(self) -> None:
        """Test batch filter parser with missing results."""
        model = Mock(spec=LM)
        
        batch_outputs = [
            '''{
                "results": [
                    {"document_id": 1, "answer": true}
                ]
            }'''
        ]
        
        outputs, raw_outputs, explanations = batch_filter_parser(
            batch_outputs, model, default=False, expected_doc_count=2
        )
        
        assert outputs == [True, False]  # Second result padded with default
        assert explanations == ["", ""]  # Empty explanations
    
    def test_batch_filter_parser_multiple_batches(self) -> None:
        """Test batch filter parser with multiple batch outputs."""
        model = Mock(spec=LM)
        
        batch_outputs = [
            '''{
                "results": [
                    {"document_id": 1, "answer": true, "reasoning": "First batch"}
                ]
            }''',
            '''{
                "results": [
                    {"document_id": 2, "answer": false, "reasoning": "Second batch"}
                ]
            }'''
        ]
        
        outputs, raw_outputs, explanations = batch_filter_parser(
            batch_outputs, model, default=True, expected_doc_count=2
        )
        
        assert outputs == [True, False]
        assert len(raw_outputs) == 2
        assert explanations == ["First batch", "Second batch"]


class TestSemFilterBatch:
    """Test cases for sem_filter batch processing."""
    
    @patch('lotus.sem_ops.sem_filter.lotus.templates.task_instructions.batch_filter_formatter')
    @patch('lotus.sem_ops.sem_filter.batch_filter_parser')
    @patch('lotus.models.LM')
    def test_sem_filter_batch_success(
        self, 
        mock_model_class, 
        mock_parser, 
        mock_formatter
    ) -> None:
        """Test successful batch processing."""
        # Setup mocks
        mock_model = Mock(spec=LM)
        mock_model.count_tokens.return_value = 100
        
        mock_formatter.return_value = [["batch_input"]]
        mock_parser.return_value = ([True, False], ["raw1"], ["explanation1", "explanation2"])
        
        # Mock model call
        mock_lm_output = Mock()
        mock_lm_output.outputs = ["batch_response"]
        mock_lm_output.logprobs = None
        mock_model.return_value = mock_lm_output
        
        docs = [
            {"text": "Positive review"},
            {"text": "Negative review"},
        ]
        
        result = _sem_filter_batch(
            docs=docs,
            model=mock_model,
            user_instruction="Is this positive?",
            batch_size=2
        )
        
        # Verify calls
        mock_formatter.assert_called_once()
        mock_parser.assert_called_once_with(
            ["batch_response"], mock_model, True, 2
        )
        
        # Verify result
        assert isinstance(result, SemanticFilterOutput)
        assert result.outputs == [True, False]
        assert result.raw_outputs == ["raw1"]
        assert result.explanations == ["explanation1", "explanation2"]
    
    @patch('lotus.sem_ops.sem_filter._sem_filter_individual')
    @patch('lotus.sem_ops.sem_filter.lotus.templates.task_instructions.batch_filter_formatter')
    @patch('lotus.models.LM')
    def test_sem_filter_batch_fallback(
        self, 
        mock_model_class, 
        mock_formatter, 
        mock_individual
    ) -> None:
        """Test batch processing fallback to individual processing."""
        # Setup mocks
        mock_model = Mock(spec=LM)
        mock_formatter.side_effect = Exception("Batch processing failed")
        
        mock_individual_result = SemanticFilterOutput(
            raw_outputs=["raw1", "raw2"],
            outputs=[True, False],
            explanations=["exp1", "exp2"],
            logprobs=None
        )
        mock_individual.return_value = mock_individual_result
        
        docs = [
            {"text": "Positive review"},
            {"text": "Negative review"},
        ]
        
        result = _sem_filter_batch(
            docs=docs,
            model=mock_model,
            user_instruction="Is this positive?",
            batch_size=2
        )
        
        # Verify fallback was called
        mock_individual.assert_called_once()
        assert result == mock_individual_result


class TestSemFilterIntegration:
    """Integration tests for sem_filter with batch processing."""
    
    @patch('lotus.sem_ops.sem_filter._sem_filter_batch')
    @patch('lotus.models.LM')
    def test_sem_filter_uses_batch_processing(
        self, 
        mock_model_class, 
        mock_batch
    ) -> None:
        """Test that sem_filter uses batch processing when enabled."""
        mock_model = Mock(spec=LM)
        mock_batch_result = SemanticFilterOutput(
            raw_outputs=["raw1", "raw2"],
            outputs=[True, False],
            explanations=["exp1", "exp2"],
            logprobs=None
        )
        mock_batch.return_value = mock_batch_result
        
        docs = [
            {"text": "Positive review"},
            {"text": "Negative review"},
        ]
        
        result = sem_filter(
            docs=docs,
            model=mock_model,
            user_instruction="Is this positive?",
            use_batch_processing=True,
            batch_size=2
        )
        
        # Verify batch processing was called
        mock_batch.assert_called_once()
        assert result == mock_batch_result
    
    @patch('lotus.sem_ops.sem_filter._sem_filter_individual')
    @patch('lotus.models.LM')
    def test_sem_filter_uses_individual_processing(
        self, 
        mock_model_class, 
        mock_individual
    ) -> None:
        """Test that sem_filter uses individual processing when batch is disabled."""
        mock_model = Mock(spec=LM)
        mock_individual_result = SemanticFilterOutput(
            raw_outputs=["raw1", "raw2"],
            outputs=[True, False],
            explanations=["exp1", "exp2"],
            logprobs=None
        )
        mock_individual.return_value = mock_individual_result
        
        docs = [
            {"text": "Positive review"},
            {"text": "Negative review"},
        ]
        
        result = sem_filter(
            docs=docs,
            model=mock_model,
            user_instruction="Is this positive?",
            use_batch_processing=False
        )
        
        # Verify individual processing was called
        mock_individual.assert_called_once()
        assert result == mock_individual_result
    
    @patch('lotus.sem_ops.sem_filter._sem_filter_individual')
    @patch('lotus.models.LM')
    def test_sem_filter_single_document_uses_individual(
        self, 
        mock_model_class, 
        mock_individual
    ) -> None:
        """Test that sem_filter uses individual processing for single document."""
        mock_model = Mock(spec=LM)
        mock_individual_result = SemanticFilterOutput(
            raw_outputs=["raw1"],
            outputs=[True],
            explanations=["exp1"],
            logprobs=None
        )
        mock_individual.return_value = mock_individual_result
        
        docs = [{"text": "Positive review"}]
        
        result = sem_filter(
            docs=docs,
            model=mock_model,
            user_instruction="Is this positive?",
            use_batch_processing=True  # Even with batch enabled
        )
        
        # Verify individual processing was called for single document
        mock_individual.assert_called_once()
        assert result == mock_individual_result


class TestBatchProcessingEdgeCases:
    """Test edge cases for batch processing."""
    
    def test_empty_docs_list(self) -> None:
        """Test batch processing with empty documents list."""
        model = Mock(spec=LM)
        
        result = batch_filter_formatter(
            model=model,
            docs=[],
            user_instruction="Test instruction",
            batch_size=10
        )
        
        assert result == []
    
    def test_batch_size_larger_than_docs(self) -> None:
        """Test batch processing when batch size is larger than number of docs."""
        model = Mock(spec=LM)
        model.is_deepseek.return_value = False
        
        docs = [{"text": "Single document"}]
        
        result = batch_filter_formatter(
            model=model,
            docs=docs,
            user_instruction="Test instruction",
            batch_size=10
        )
        
        assert len(result) == 1
        # Should adjust system prompt for actual batch size
        assert "1 document" in result[0][0]["content"]
    
    def test_parser_with_no_expected_count(self) -> None:
        """Test batch parser without expected document count."""
        model = Mock(spec=LM)
        
        batch_outputs = [
            '''{
                "results": [
                    {"document_id": 1, "answer": true, "reasoning": "Test"}
                ]
            }'''
        ]
        
        outputs, raw_outputs, explanations = batch_filter_parser(
            batch_outputs, model, default=True, expected_doc_count=None
        )
        
        assert outputs == [True]
        assert explanations == ["Test"]
        assert len(raw_outputs) == 1


if __name__ == "__main__":
    pytest.main([__file__])
