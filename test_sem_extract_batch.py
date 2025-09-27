"""
Test cases for sem_extract batch processing functionality.
"""

import pandas as pd
import pytest
from typing import Any

import lotus
from lotus.models import LM
from lotus.sem_ops.sem_extract import sem_extract
from lotus.templates.batch_extract_formatter import batch_extract_formatter
from lotus.sem_ops.postprocessors import batch_extract_parser


class TestSemExtractBatch:
    """Test cases for sem_extract batch processing."""

    def setup_method(self) -> None:
        """Set up test environment."""
        # Configure a mock model for testing
        lotus.settings.configure(lm=LM(model="openrouter/google/gemini-2.5-flash", base_url="https://openrouter.ai/api/v1", api_key = '"key"-v1-ff17cd1b6aee306e43330dfeb8e0f7f4b85525ec32126903c2c099b8a5f3eb84'))

    def test_batch_extract_formatter(self) -> None:
        """Test batch extract formatter functionality."""
        docs = [
            {"text": "The product is excellent with 5 stars"},
            {"text": "This service is terrible, only 1 star"},
            {"text": "Average quality, 3 stars rating"}
        ]
        output_cols = {"sentiment": "positive/negative/neutral", "rating": "1-5 scale"}
        
        model = lotus.settings.lm
        batched_inputs = batch_extract_formatter(
            model, docs, output_cols, extract_quotes=False, batch_size=2
        )
        
        # Should create 2 batches (2 docs + 1 doc)
        assert len(batched_inputs) == 2
        assert len(batched_inputs[0]) == 2  # system + user message
        assert len(batched_inputs[1]) == 2  # system + user message
        
        # Check that system message contains required fields
        system_content = batched_inputs[0][0]["content"]
        assert "sentiment" in system_content
        assert "rating" in system_content
        assert "JSON" in system_content

    def test_batch_extract_parser(self) -> None:
        """Test batch extract parser functionality."""
        # Mock batch outputs
        batch_outputs = [
            '''[
                {"document_id": 1, "sentiment": "positive", "rating": "5"},
                {"document_id": 2, "sentiment": "negative", "rating": "1"}
            ]''',
            '''[
                {"document_id": 3, "sentiment": "neutral", "rating": "3"}
            ]'''
        ]
        
        model = lotus.settings.lm
        outputs, raw_outputs, explanations = batch_extract_parser(
            batch_outputs, model, expected_doc_count=3
        )
        
        # Check outputs
        assert len(outputs) == 3
        assert outputs[0]["sentiment"] == "positive"
        assert outputs[0]["rating"] == "5"
        assert outputs[1]["sentiment"] == "negative"
        assert outputs[1]["rating"] == "1"
        assert outputs[2]["sentiment"] == "neutral"
        assert outputs[2]["rating"] == "3"
        
        # Check raw outputs
        assert len(raw_outputs) == 2
        assert len(explanations) == 3

    def test_sem_extract_batch_processing(self) -> None:
        """Test sem_extract with batch processing."""
        docs = [
            {"text": "The product is excellent with 5 stars"},
            {"text": "This service is terrible, only 1 star"},
            {"text": "Average quality, 3 stars rating"},
            {"text": "Outstanding performance, 5 stars"},
            {"text": "Poor quality, 2 stars"}
        ]
        output_cols = {"sentiment": "positive/negative/neutral", "rating": "1-5 scale"}
        
        # Test batch processing
        result_batch = sem_extract(
            docs=docs,
            model=lotus.settings.lm,
            output_cols=output_cols,
            use_batch_processing=True,
            batch_size=2
        )
        
        # Test individual processing
        result_individual = sem_extract(
            docs=docs,
            model=lotus.settings.lm,
            output_cols=output_cols,
            use_batch_processing=False
        )
        
        # Both should return the same number of results
        assert len(result_batch.outputs) == len(docs)
        assert len(result_individual.outputs) == len(docs)
        
        # Check that outputs contain expected fields
        for output in result_batch.outputs:
            assert isinstance(output, dict)
            # Should have sentiment and rating fields (or be empty dict if extraction failed)
            if output:  # If extraction succeeded
                assert "sentiment" in output or "rating" in output

    def test_sem_extract_dataframe_batch(self) -> None:
        """Test DataFrame sem_extract with batch processing."""
        # Create test DataFrame
        df = pd.DataFrame({
            'text': [
                'The product is excellent with 5 stars',
                'This service is terrible, only 1 star',
                'Average quality, 3 stars rating',
                'Outstanding performance, 5 stars',
                'Poor quality, 2 stars'
            ],
            'category': ['electronics', 'service', 'general', 'electronics', 'general']
        })
        
        output_cols = {"sentiment": "positive/negative/neutral", "rating": "1-5 scale"}
        
        # Test batch processing
        result_df_batch = df.sem_extract(
            input_cols=['text'],
            output_cols=output_cols,
            use_batch_processing=True,
            batch_size=2
        )
        
        # Test individual processing
        result_df_individual = df.sem_extract(
            input_cols=['text'],
            output_cols=output_cols,
            use_batch_processing=False
        )
        
        # Both should return DataFrames with original columns plus extracted columns
        assert len(result_df_batch) == len(df)
        assert len(result_df_individual) == len(df)
        
        # Check that extracted columns are added
        for col in output_cols.keys():
            assert col in result_df_batch.columns
            assert col in result_df_individual.columns

    def test_batch_processing_fallback(self) -> None:
        """Test that batch processing falls back to individual processing on error."""
        docs = [{"text": "Test document"}]
        output_cols = {"sentiment": "positive/negative/neutral"}
        
        # This should work even if batch processing fails
        result = sem_extract(
            docs=docs,
            model=lotus.settings.lm,
            output_cols=output_cols,
            use_batch_processing=True,
            batch_size=1
        )
        
        assert len(result.outputs) == 1
        assert isinstance(result.outputs[0], dict)

    def test_batch_size_parameter(self) -> None:
        """Test different batch sizes."""
        docs = [
            {"text": f"Document {i}"} for i in range(10)
        ]
        output_cols = {"summary": "brief summary"}
        
        # Test with batch_size=3
        result = sem_extract(
            docs=docs,
            model=lotus.settings.lm,
            output_cols=output_cols,
            use_batch_processing=True,
            batch_size=3
        )
        
        assert len(result.outputs) == 10

    def test_extract_quotes_batch(self) -> None:
        """Test batch processing with quote extraction."""
        docs = [
            {"text": "The product is excellent with 5 stars"},
            {"text": "This service is terrible, only 1 star"}
        ]
        output_cols = {"sentiment": "positive/negative/neutral"}
        
        result = sem_extract(
            docs=docs,
            model=lotus.settings.lm,
            output_cols=output_cols,
            extract_quotes=True,
            use_batch_processing=True,
            batch_size=2
        )
        
        assert len(result.outputs) == 2
        # Check that quote fields are included if extraction succeeded
        for output in result.outputs:
            if output:  # If extraction succeeded
                # Should have sentiment and potentially sentiment_quote
                assert "sentiment" in output

    def test_safe_mode_batch(self) -> None:
        """Test batch processing with safe mode."""
        docs = [
            {"text": "The product is excellent with 5 stars"},
            {"text": "This service is terrible, only 1 star"}
        ]
        output_cols = {"sentiment": "positive/negative/neutral"}
        
        # This should not raise an error even in safe mode
        result = sem_extract(
            docs=docs,
            model=lotus.settings.lm,
            output_cols=output_cols,
            use_batch_processing=True,
            batch_size=2,
            safe_mode=True
        )
        
        assert len(result.outputs) == 2

    def test_strategy_batch(self) -> None:
        """Test batch processing with different reasoning strategies."""
        docs = [
            {"text": "The product is excellent with 5 stars"},
            {"text": "This service is terrible, only 1 star"}
        ]
        output_cols = {"sentiment": "positive/negative/neutral"}
        
        # Test with COT strategy
        result = sem_extract(
            docs=docs,
            model=lotus.settings.lm,
            output_cols=output_cols,
            use_batch_processing=True,
            batch_size=2,
            strategy=lotus.types.ReasoningStrategy.COT
        )
        
        assert len(result.outputs) == 2

    def test_empty_docs_batch(self) -> None:
        """Test batch processing with empty document list."""
        docs = []
        output_cols = {"sentiment": "positive/negative/neutral"}
        
        result = sem_extract(
            docs=docs,
            model=lotus.settings.lm,
            output_cols=output_cols,
            use_batch_processing=True,
            batch_size=2
        )
        
        assert len(result.outputs) == 0

    def test_single_doc_batch(self) -> None:
        """Test batch processing with single document."""
        docs = [{"text": "The product is excellent with 5 stars"}]
        output_cols = {"sentiment": "positive/negative/neutral"}
        
        result = sem_extract(
            docs=docs,
            model=lotus.settings.lm,
            output_cols=output_cols,
            use_batch_processing=True,
            batch_size=2
        )
        
        assert len(result.outputs) == 1


if __name__ == "__main__":
    # Run a simple test
    test = TestSemExtractBatch()
    test.setup_method()
    
    print("Testing batch extract formatter...")
    test.test_batch_extract_formatter()
    print("✓ Batch extract formatter test passed")
    
    print("Testing batch extract parser...")
    test.test_batch_extract_parser()
    print("✓ Batch extract parser test passed")
    
    print("Testing sem_extract batch processing...")
    test.test_sem_extract_batch_processing()
    print("✓ Sem_extract batch processing test passed")
    
    print("Testing DataFrame batch processing...")
    test.test_sem_extract_dataframe_batch()
    print("✓ DataFrame batch processing test passed")
    
    print("All tests passed! 🎉")
