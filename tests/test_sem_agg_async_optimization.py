"""
Simplified tests for sem_agg async optimization functionality.

This module tests the core sem_agg functionality with both sync and async modes,
covering grouped and non-grouped scenarios with realistic DataFrame datasets.
"""

import asyncio
import os
from typing import TYPE_CHECKING

import pandas as pd
import pytest

import lotus
from lotus.models import LM
from lotus.sem_ops.sem_agg import sem_agg, sem_agg_async

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch
    from pytest_mock.plugin import MockerFixture


# Skip tests if no API key is available (including hardcoded fallback)
pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENROUTER_API_KEY") and not os.getenv("LOTUS_TEST_API_KEY"),
    reason="No API key available for real LM testing. Set LOTUS_TEST_API_KEY environment variable to run tests."
)


class TestSemAggEndToEnd:
    """End-to-end tests for sem_agg functionality with async optimization."""

    @pytest.fixture
    def real_lm(self) -> LM:
        """
        Create a real LM instance for testing.

        Returns:
            LM: A real LM instance using gpt-4o-mini.
        """
        # Use environment variable for API key with fallback
        api_key = (
            os.getenv("OPENROUTER_API_KEY") or 
            os.getenv("LOTUS_TEST_API_KEY") or
        )
        
        return LM(
            model="openrouter/google/gemini-2.5-flash",
            max_batch_size=4,
            temperature=0.0,
            max_tokens=256,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )

    @pytest.fixture
    def sample_dataframe(self) -> pd.DataFrame:
        """
        Create a realistic sample DataFrame for testing.

        Returns:
            pd.DataFrame: A sample DataFrame with text data and categories.
        """
        return pd.DataFrame({
            'content': [
                "Machine learning is a subset of artificial intelligence that focuses on algorithms and statistical models.",
                "Deep learning uses neural networks with multiple layers to process data and make predictions.",
                "Natural language processing enables computers to understand and generate human language.",
                "Computer vision allows machines to interpret and analyze visual information from images.",
                "Data science combines statistics, programming, and domain expertise to extract insights from data.",
                "Artificial intelligence aims to create systems that can perform tasks typically requiring human intelligence.",
            ],
            'category': ['ML', 'ML', 'NLP', 'CV', 'DS', 'AI'],
            'priority': ['High', 'Medium', 'High', 'Medium', 'Low', 'High'],
            'author': ['Alice', 'Bob', 'Alice', 'Charlie', 'Bob', 'Alice']
        })

    @pytest.fixture
    def large_dataframe(self) -> pd.DataFrame:
        """
        Create a larger DataFrame for testing performance scenarios.

        Returns:
            pd.DataFrame: A larger DataFrame with more data points.
        """
        categories = ['Technology', 'Science', 'Business', 'Health', 'Education']
        authors = ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve']
        priorities = ['High', 'Medium', 'Low']
        
        data = []
        for i in range(20):  # 20 documents
            data.append({
                'content': f"This is document {i+1} about {categories[i % len(categories)].lower()}. "
                          f"It contains important information about various topics and concepts. "
                          f"The content is designed to test the async aggregation functionality.",
                'category': categories[i % len(categories)],
                'author': authors[i % len(authors)],
                'priority': priorities[i % len(priorities)],
                'year': 2020 + (i % 4)
            })
        
        return pd.DataFrame(data)

    def test_sem_agg_sync_mode(self, real_lm: LM) -> None:
        """
        Test sem_agg in synchronous mode with basic functionality.

        Args:
            real_lm: Real LM instance.
        """
        docs = [
            "Machine learning is transforming industries.",
            "Deep learning enables advanced pattern recognition.",
            "AI systems are becoming more sophisticated."
        ]
        partition_ids = [0, 0, 1]
        
        result = sem_agg(
            docs=docs,
            model=real_lm,
            user_instruction="Summarize the key points about AI",
            partition_ids=partition_ids,
            use_async=False,
        )
        
        assert len(result.outputs) == 2  # Two partitions
        assert all(isinstance(output, str) for output in result.outputs)
        assert len(result.outputs[0]) > 0  # Non-empty output

    def test_sem_agg_async_mode(self, real_lm: LM) -> None:
        """
        Test sem_agg in asynchronous mode.

        Args:
            real_lm: Real LM instance.
        """
        docs = [
            "Machine learning is transforming industries.",
            "Deep learning enables advanced pattern recognition.",
            "AI systems are becoming more sophisticated."
        ]
        partition_ids = [0, 0, 1]
        
        result = sem_agg(
            docs=docs,
            model=real_lm,
            user_instruction="Summarize the key points about AI",
            partition_ids=partition_ids,
            use_async=True,
            max_concurrent_batches=2,
            max_thread_workers=4,
        )
        
        assert len(result.outputs) == 2  # Two partitions
        assert all(isinstance(output, str) for output in result.outputs)
        assert len(result.outputs[0]) > 0  # Non-empty output

    @pytest.mark.asyncio
    async def test_sem_agg_async_direct(self, real_lm: LM) -> None:
        """
        Test the direct async sem_agg_async function.

        Args:
            real_lm: Real LM instance.
        """
        docs = [
            "Machine learning is transforming industries.",
            "Deep learning enables advanced pattern recognition.",
            "AI systems are becoming more sophisticated."
        ]
        partition_ids = [0, 0, 1]
        
        result = await sem_agg_async(
            docs=docs,
            model=real_lm,
            user_instruction="Summarize the key points about AI",
            partition_ids=partition_ids,
            max_concurrent_batches=2,
            max_thread_workers=4,
        )
        
        assert len(result.outputs) == 2  # Two partitions
        assert all(isinstance(output, str) for output in result.outputs)
        assert len(result.outputs[0]) > 0  # Non-empty output

    def test_dataframe_sync_no_grouping(self, real_lm: LM, sample_dataframe: pd.DataFrame) -> None:
        """
        Test DataFrame accessor in sync mode without grouping.

        Args:
            real_lm: Real LM instance.
            sample_dataframe: Sample DataFrame for testing.
        """
        # Configure lotus settings
        lotus.settings.configure(lm=real_lm)
        
        result = sample_dataframe.sem_agg(
            user_instruction="Summarize the key themes in AI and technology",
            all_cols=True,
            use_async=False,
        )
        
        assert isinstance(result, pd.DataFrame)
        assert '_output' in result.columns
        assert len(result) == 1
        assert isinstance(result['_output'].iloc[0], str)
        assert len(result['_output'].iloc[0]) > 0

    def test_dataframe_async_no_grouping(self, real_lm: LM, sample_dataframe: pd.DataFrame) -> None:
        """
        Test DataFrame accessor in async mode without grouping.

        Args:
            mock_lm: Mock LM instance.
            sample_dataframe: Sample DataFrame for testing.
        """
        # Configure lotus settings
        lotus.settings.configure(lm=real_lm)
        
        result = sample_dataframe.sem_agg(
            user_instruction="Summarize the key themes in AI and technology",
            all_cols=True,
            use_async=True,
            max_concurrent_batches=2,
            max_thread_workers=4,
        )
        
        assert isinstance(result, pd.DataFrame)
        assert '_output' in result.columns
        assert len(result) == 1
        assert isinstance(result['_output'].iloc[0], str)
        assert len(result['_output'].iloc[0]) > 0

    def test_dataframe_sync_with_grouping(self, real_lm: LM, sample_dataframe: pd.DataFrame) -> None:
        """
        Test DataFrame accessor in sync mode with grouping.

        Args:
            mock_lm: Mock LM instance.
            sample_dataframe: Sample DataFrame for testing.
        """
        # Configure lotus settings
        lotus.settings.configure(lm=real_lm)
        
        result = sample_dataframe.sem_agg(
            user_instruction="Summarize the content for each category",
            all_cols=True,
            group_by=['category'],
            use_async=False,
        )
        
        assert isinstance(result, pd.DataFrame)
        assert '_output' in result.columns
        assert 'category' in result.columns
        assert len(result) == len(sample_dataframe['category'].unique())  # One row per category
        assert all(isinstance(output, str) and len(output) > 0 for output in result['_output'])

    def test_dataframe_async_with_grouping(self, real_lm: LM, sample_dataframe: pd.DataFrame) -> None:
        """
        Test DataFrame accessor in async mode with grouping.

        Args:
            mock_lm: Mock LM instance.
            sample_dataframe: Sample DataFrame for testing.
        """
        # Configure lotus settings
        lotus.settings.configure(lm=real_lm)
        
        result = sample_dataframe.sem_agg(
            user_instruction="Summarize the content for each category",
            all_cols=True,
            group_by=['category'],
            use_async=True,
            max_concurrent_batches=2,
            max_thread_workers=4,
        )
        
        assert isinstance(result, pd.DataFrame)
        assert '_output' in result.columns
        assert 'category' in result.columns
        assert len(result) == len(sample_dataframe['category'].unique())  # One row per category
        assert all(isinstance(output, str) and len(output) > 0 for output in result['_output'])

    def test_dataframe_multi_column_grouping(self, real_lm: LM, sample_dataframe: pd.DataFrame) -> None:
        """
        Test DataFrame accessor with multi-column grouping.

        Args:
            mock_lm: Mock LM instance.
            sample_dataframe: Sample DataFrame for testing.
        """
        # Configure lotus settings
        lotus.settings.configure(lm=real_lm)
        
        result = sample_dataframe.sem_agg(
            user_instruction="Summarize the content for each category and priority combination",
            all_cols=True,
            group_by=['category', 'priority'],
            use_async=True,
            max_concurrent_batches=2,
            max_thread_workers=4,
        )
        
        assert isinstance(result, pd.DataFrame)
        assert '_output' in result.columns
        assert 'category' in result.columns
        assert 'priority' in result.columns
        assert all(isinstance(output, str) and len(output) > 0 for output in result['_output'])

    def test_dataframe_large_dataset_async(self, real_lm: LM, large_dataframe: pd.DataFrame) -> None:
        """
        Test DataFrame accessor with larger dataset in async mode.

        Args:
            mock_lm: Mock LM instance.
            large_dataframe: Large DataFrame for testing.
        """
        # Configure lotus settings
        lotus.settings.configure(lm=real_lm)
        
        result = large_dataframe.sem_agg(
            user_instruction="Summarize the key themes across all documents",
            all_cols=True,
            group_by=['category'],
            use_async=True,
            max_concurrent_batches=4,
            max_thread_workers=8,
        )
        
        assert isinstance(result, pd.DataFrame)
        assert '_output' in result.columns
        assert 'category' in result.columns
        assert len(result) == len(large_dataframe['category'].unique())
        assert all(isinstance(output, str) and len(output) > 0 for output in result['_output'])

    def test_backward_compatibility(self, real_lm: LM) -> None:
        """
        Test that the original API still works without async parameters.

        Args:
            real_lm: Real LM instance.
        """
        docs = [
            "Machine learning is transforming industries.",
            "Deep learning enables advanced pattern recognition.",
        ]
        partition_ids = [0, 0]
        
        # Test original API without new parameters
        result = sem_agg(
            docs=docs,
            model=real_lm,
            user_instruction="Summarize the key points",
            partition_ids=partition_ids,
        )
        
        assert len(result.outputs) == 1
        assert isinstance(result.outputs[0], str)
        assert len(result.outputs[0]) > 0

    def test_column_specific_aggregation(self, real_lm: LM, sample_dataframe: pd.DataFrame) -> None:
        """
        Test aggregation with specific columns only.

        Args:
            mock_lm: Mock LM instance.
            sample_dataframe: Sample DataFrame for testing.
        """
        # Configure lotus settings
        lotus.settings.configure(lm=real_lm)
        
        result = sample_dataframe.sem_agg(
            user_instruction="Summarize the content from {content}",
            all_cols=False,  # Only use columns mentioned in instruction
            use_async=True,
        )
        
        assert isinstance(result, pd.DataFrame)
        assert '_output' in result.columns
        assert len(result) == 1
        assert isinstance(result['_output'].iloc[0], str)
        assert len(result['_output'].iloc[0]) > 0
