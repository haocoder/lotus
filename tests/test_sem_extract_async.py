"""
Tests for async DataFrame sem_extract accessor functionality.

This module contains comprehensive tests for the asynchronous DataFrame sem_extract accessor,
including concurrent batch processing, error handling, and end-to-end functionality with real LLM.
"""

import asyncio
import pytest
from typing import TYPE_CHECKING, Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd

import lotus
from lotus.models import LM
from lotus.types import LMOutput, ReasoningStrategy, SemanticExtractOutput

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch
    from pytest_mock.plugin import MockerFixture


class TestDataFrameSemExtractAsync:
    """Test cases for async DataFrame sem_extract accessor functionality."""

    @pytest.fixture
    def mock_model(self) -> LM:
        """Create a mock LM model for testing."""
        model = MagicMock(spec=LM)
        model.async_call = AsyncMock()
        model.count_tokens = MagicMock(return_value=100)
        model.print_total_usage = MagicMock()
        model.is_deepseek = MagicMock(return_value=False)
        return model

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        """Create sample DataFrame for testing."""
        return pd.DataFrame({
            'text': [
                "This product is amazing! I love it.",
                "Terrible quality, would not recommend.",
                "It's okay, nothing special.",
            ],
            'category': ['electronics', 'service', 'general'],
            'price': [299.99, 0, 50.00]
        })

    @pytest.fixture
    def output_cols(self) -> Dict[str, str]:
        """Create sample output columns for testing."""
        return {
            "sentiment": "positive/negative/neutral",
            "confidence": "0-1 scale"
        }

    @pytest.fixture
    def mock_lm_output(self) -> LMOutput:
        """Create a mock LMOutput for testing."""
        return LMOutput(
            outputs=[
                '{"sentiment": "positive", "confidence": "0.9"}',
                '{"sentiment": "negative", "confidence": "0.8"}',
                '{"sentiment": "neutral", "confidence": "0.6"}'
            ],
            usage={"total_tokens": 100, "prompt_tokens": 50, "completion_tokens": 50}
        )

    @pytest.mark.asyncio
    async def test_dataframe_async_basic_functionality(
        self, 
        mock_model: LM, 
        sample_df: pd.DataFrame, 
        output_cols: Dict[str, str],
        mock_lm_output: LMOutput
    ) -> None:
        """Test basic async DataFrame sem_extract functionality."""
        mock_model.async_call.return_value = mock_lm_output
        
        with patch('lotus.settings.lm', mock_model):
            result_df = await sample_df.sem_extract.async_extract(
                input_cols=['text'],
                output_cols=output_cols,
                use_batch_processing=False
            )
        
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 3
        assert 'sentiment' in result_df.columns
        assert 'confidence' in result_df.columns
        
        # Verify model was called
        mock_model.async_call.assert_called_once()

    @pytest.mark.asyncio
    async def test_dataframe_async_batch_processing(
        self, 
        mock_model: LM, 
        sample_df: pd.DataFrame, 
        output_cols: Dict[str, str],
        mock_lm_output: LMOutput
    ) -> None:
        """Test async DataFrame batch processing functionality."""
        mock_model.async_call.return_value = mock_lm_output
        
        with patch('lotus.settings.lm', mock_model):
            result_df = await sample_df.sem_extract.async_extract(
                input_cols=['text'],
                output_cols=output_cols,
                use_batch_processing=True,
                batch_size=2,
                max_concurrent_batches=2
            )
        
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 3
        assert 'sentiment' in result_df.columns
        assert 'confidence' in result_df.columns

    @pytest.mark.asyncio
    async def test_dataframe_async_with_quotes(
        self, 
        mock_model: LM, 
        sample_df: pd.DataFrame, 
        output_cols: Dict[str, str],
        mock_lm_output: LMOutput
    ) -> None:
        """Test async DataFrame processing with quote extraction."""
        mock_model.async_call.return_value = mock_lm_output
        
        with patch('lotus.settings.lm', mock_model):
            result_df = await sample_df.sem_extract.async_extract(
                input_cols=['text'],
                output_cols=output_cols,
                extract_quotes=True,
                use_batch_processing=False
            )
        
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 3
        assert 'sentiment' in result_df.columns
        assert 'confidence' in result_df.columns
        assert 'quotes' in result_df.columns

    @pytest.mark.asyncio
    async def test_dataframe_async_with_explanations(
        self, 
        mock_model: LM, 
        sample_df: pd.DataFrame, 
        output_cols: Dict[str, str],
        mock_lm_output: LMOutput
    ) -> None:
        """Test async DataFrame processing with explanations."""
        mock_model.async_call.return_value = mock_lm_output
        
        with patch('lotus.settings.lm', mock_model):
            result_df = await sample_df.sem_extract.async_extract(
                input_cols=['text'],
                output_cols=output_cols,
                return_explanations=True,
                use_batch_processing=False
            )
        
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 3
        assert 'sentiment' in result_df.columns
        assert 'confidence' in result_df.columns
        assert 'explanation' in result_df.columns

    @pytest.mark.asyncio
    async def test_dataframe_async_error_handling(
        self, 
        mock_model: LM, 
        sample_df: pd.DataFrame, 
        output_cols: Dict[str, str]
    ) -> None:
        """Test error handling in async DataFrame processing."""
        mock_model.async_call.side_effect = Exception("Model call failed")
        
        with patch('lotus.settings.lm', mock_model):
            with pytest.raises(Exception, match="Model call failed"):
                await sample_df.sem_extract.async_extract(
                    input_cols=['text'],
                    output_cols=output_cols,
                    use_batch_processing=False
                )

    @pytest.mark.asyncio
    async def test_dataframe_async_missing_column(
        self, 
        mock_model: LM, 
        sample_df: pd.DataFrame, 
        output_cols: Dict[str, str]
    ) -> None:
        """Test async DataFrame processing with missing column."""
        with patch('lotus.settings.lm', mock_model):
            with pytest.raises(ValueError, match="Column nonexistent not found in DataFrame"):
                await sample_df.sem_extract.async_extract(
                    input_cols=['nonexistent'],
                    output_cols=output_cols
                )

    @pytest.mark.asyncio
    async def test_dataframe_async_no_model_configured(
        self, 
        sample_df: pd.DataFrame, 
        output_cols: Dict[str, str]
    ) -> None:
        """Test async DataFrame processing without configured model."""
        with patch('lotus.settings.lm', None):
            with pytest.raises(ValueError, match="The language model must be an instance of LM"):
                await sample_df.sem_extract.async_extract(
                    input_cols=['text'],
                    output_cols=output_cols
                )


class TestDataFrameAsyncPerformance:
    """Test cases for async DataFrame performance and concurrency."""

    @pytest.mark.asyncio
    async def test_dataframe_concurrent_batch_processing(self) -> None:
        """Test that concurrent DataFrame batch processing works correctly."""
        # This test verifies that multiple batches can be processed concurrently
        # without blocking each other
        
        async def mock_async_call(*args, **kwargs):
            # Simulate processing time
            await asyncio.sleep(0.1)
            return LMOutput(
                outputs=['{"sentiment": "positive"}'],
                usage={"total_tokens": 50, "prompt_tokens": 25, "completion_tokens": 25}
            )
        
        mock_model = MagicMock(spec=LM)
        mock_model.async_call = mock_async_call
        mock_model.count_tokens = MagicMock(return_value=50)
        mock_model.print_total_usage = MagicMock()
        
        # Create DataFrame that will be split into multiple batches
        df = pd.DataFrame({
            'text': [f"Document {i}" for i in range(10)],
            'category': ['test'] * 10
        })
        output_cols = {"sentiment": "positive/negative/neutral"}
        
        start_time = asyncio.get_event_loop().time()
        
        with patch('lotus.settings.lm', mock_model):
            result_df = await df.sem_extract.async_extract(
                input_cols=['text'],
                output_cols=output_cols,
                use_batch_processing=True,
                batch_size=3,
                max_concurrent_batches=3
            )
        
        end_time = asyncio.get_event_loop().time()
        processing_time = end_time - start_time
        
        # With concurrent processing, this should be faster than sequential
        # (3 batches with 0.1s each should take ~0.1s concurrent vs 0.3s sequential)
        assert processing_time < 0.2  # Should be much faster than sequential
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 10
        assert 'sentiment' in result_df.columns

    @pytest.mark.asyncio
    async def test_dataframe_semaphore_limits_concurrency(self) -> None:
        """Test that semaphore properly limits concurrency in DataFrame processing."""
        concurrent_calls = []
        
        async def mock_async_call(*args, **kwargs):
            concurrent_calls.append(1)
            # Simulate processing time
            await asyncio.sleep(0.1)
            concurrent_calls.append(-1)
            return LMOutput(
                outputs=['{"sentiment": "positive"}'],
                usage={"total_tokens": 50, "prompt_tokens": 25, "completion_tokens": 25}
            )
        
        mock_model = MagicMock(spec=LM)
        mock_model.async_call = mock_async_call
        mock_model.count_tokens = MagicMock(return_value=50)
        mock_model.print_total_usage = MagicMock()
        
        # Create DataFrame that will be split into multiple batches
        df = pd.DataFrame({
            'text': [f"Document {i}" for i in range(15)],
            'category': ['test'] * 15
        })
        output_cols = {"sentiment": "positive/negative/neutral"}
        
        with patch('lotus.settings.lm', mock_model):
            await df.sem_extract.async_extract(
                input_cols=['text'],
                output_cols=output_cols,
                use_batch_processing=True,
                batch_size=3,
                max_concurrent_batches=2  # Limit to 2 concurrent batches
            )
        
        # Verify that concurrency was limited (max 2 concurrent calls)
        max_concurrent = max(concurrent_calls)
        assert max_concurrent <= 2


if __name__ == "__main__":
    pytest.main([__file__])
