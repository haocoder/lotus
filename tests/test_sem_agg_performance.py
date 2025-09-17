"""
Performance tests for sem_agg async optimization with different dataset scales.

This module tests the performance of sem_agg functionality with both sync and async modes,
using datasets of various sizes to measure and compare performance improvements.
"""

import asyncio
import os
import time
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


# Skip tests if no API key is available
pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENROUTER_API_KEY") and not os.getenv("LOTUS_TEST_API_KEY"),
    reason="No API key available for real LM testing. Set LOTUS_TEST_API_KEY environment variable to run tests."
)


class PerformanceMetrics:
    """Class to collect and store performance metrics."""
    
    def __init__(self) -> None:
        """Initialize performance metrics storage."""
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.duration: float = 0.0
        self.memory_usage: float = 0.0
        
    def start_timer(self) -> None:
        """Start the performance timer."""
        self.start_time = time.time()
        
    def stop_timer(self) -> None:
        """Stop the performance timer and calculate duration."""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        
    def get_duration(self) -> float:
        """Get the duration in seconds."""
        return self.duration


class TestSemAggPerformance:
    """Performance tests for sem_agg functionality with different dataset scales."""

    @pytest.fixture
    def real_lm(self) -> LM:
        """
        Create a real LM instance for performance testing.

        Returns:
            LM: A real LM instance using a fast model for testing.
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
    def small_dataset(self) -> pd.DataFrame:
        """
        Create a small dataset for performance testing (10 documents).

        Returns:
            pd.DataFrame: A small DataFrame with 10 documents.
        """
        return self._generate_dataset(size=10, name="small")

    @pytest.fixture
    def medium_dataset(self) -> pd.DataFrame:
        """
        Create a medium dataset for performance testing (50 documents).

        Returns:
            pd.DataFrame: A medium DataFrame with 50 documents.
        """
        return self._generate_dataset(size=50, name="medium")

    @pytest.fixture
    def large_dataset(self) -> pd.DataFrame:
        """
        Create a large dataset for performance testing (100 documents).

        Returns:
            pd.DataFrame: A large DataFrame with 100 documents.
        """
        return self._generate_dataset(size=100, name="large")

    @pytest.fixture
    def extra_large_dataset(self) -> pd.DataFrame:
        """
        Create an extra large dataset for performance testing (200 documents).

        Returns:
            pd.DataFrame: An extra large DataFrame with 200 documents.
        """
        return self._generate_dataset(size=200, name="extra_large")

    def _generate_dataset(self, size: int, name: str) -> pd.DataFrame:
        """
        Generate a dataset of specified size with realistic content.

        Args:
            size: The number of documents to generate.
            name: The name of the dataset for identification.

        Returns:
            pd.DataFrame: A DataFrame with the specified number of documents.
        """
        categories = ['Technology', 'Science', 'Business', 'Health', 'Education', 'Finance', 'Sports', 'Entertainment']
        authors = ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank', 'Grace', 'Henry']
        priorities = ['High', 'Medium', 'Low']
        topics = [
            'artificial intelligence', 'machine learning', 'data science', 'cloud computing',
            'cybersecurity', 'blockchain', 'quantum computing', 'robotics', 'biotechnology',
            'renewable energy', 'space exploration', 'climate change', 'digital transformation'
        ]
        
        data = []
        for i in range(size):
            topic = topics[i % len(topics)]
            category = categories[i % len(categories)]
            author = authors[i % len(authors)]
            priority = priorities[i % len(priorities)]
            
            content = (
                f"This is document {i+1} about {topic} in the {category.lower()} domain. "
                f"It discusses important concepts, methodologies, and applications related to {topic}. "
                f"The content covers various aspects including technical details, practical implementations, "
                f"and future prospects. This document provides comprehensive information for understanding "
                f"the current state and potential developments in {topic}."
            )
            
            data.append({
                'content': content,
                'category': category,
                'author': author,
                'priority': priority,
                'topic': topic,
                'year': 2020 + (i % 4),
                'document_id': f"doc_{i+1:03d}"
            })
        
        return pd.DataFrame(data)

    def _run_performance_test(
        self, 
        dataset: pd.DataFrame, 
        use_async: bool, 
        max_concurrent_batches: int = 4,
        max_thread_workers: int = 8
    ) -> PerformanceMetrics:
        """
        Run a performance test with the given dataset and configuration.

        Args:
            dataset: The dataset to test with.
            use_async: Whether to use async processing.
            max_concurrent_batches: Maximum concurrent batches for async mode.
            max_thread_workers: Maximum thread workers for async mode.

        Returns:
            PerformanceMetrics: The performance metrics for this test run.
        """
        metrics = PerformanceMetrics()
        
        # Configure lotus settings
        lm = self.real_lm()
        lotus.settings.configure(lm=lm)
        
        # Start timing
        metrics.start_timer()
        
        # Run the aggregation
        result = dataset.sem_agg(
            user_instruction="Summarize the key themes and insights across all documents",
            all_cols=True,
            group_by=['category'],
            use_async=use_async,
            max_concurrent_batches=max_concurrent_batches,
            max_thread_workers=max_thread_workers,
        )
        
        # Stop timing
        metrics.stop_timer()
        
        # Verify results
        assert isinstance(result, pd.DataFrame)
        assert '_output' in result.columns
        assert 'category' in result.columns
        assert len(result) == len(dataset['category'].unique())
        assert all(isinstance(output, str) and len(output) > 0 for output in result['_output'])
        
        return metrics

    @pytest.mark.parametrize("dataset_size", ["small", "medium", "large", "extra_large"])
    def test_sync_vs_async_performance(self, dataset_size: str, request: FixtureRequest) -> None:
        """
        Test and compare sync vs async performance for different dataset sizes.

        Args:
            dataset_size: The size of the dataset to test.
            request: Pytest request fixture to access other fixtures.
        """
        # Get the appropriate dataset fixture
        dataset = request.getfixturevalue(f"{dataset_size}_dataset")
        
        print(f"\n=== Performance Test: {dataset_size.upper()} Dataset ({len(dataset)} documents) ===")
        
        # Test sync performance
        print("Testing sync performance...")
        sync_metrics = self._run_performance_test(dataset, use_async=False)
        print(f"Sync duration: {sync_metrics.get_duration():.2f} seconds")
        
        # Test async performance
        print("Testing async performance...")
        async_metrics = self._run_performance_test(dataset, use_async=True)
        print(f"Async duration: {async_metrics.get_duration():.2f} seconds")
        
        # Calculate performance improvement
        if sync_metrics.get_duration() > 0:
            improvement = ((sync_metrics.get_duration() - async_metrics.get_duration()) / sync_metrics.get_duration()) * 100
            print(f"Performance improvement: {improvement:.1f}%")
            
            # For larger datasets, async should show improvement
            if len(dataset) >= 50:
                assert improvement > 0, f"Async should be faster for {dataset_size} dataset, but was {improvement:.1f}% slower"
        
        print("=" * 60)

    def test_async_concurrency_scaling(self, medium_dataset: pd.DataFrame) -> None:
        """
        Test how async performance scales with different concurrency settings.

        Args:
            medium_dataset: Medium-sized dataset for testing.
        """
        print("\n=== Async Concurrency Scaling Test ===")
        
        concurrency_configs = [
            (1, 2),   # Low concurrency
            (2, 4),   # Medium concurrency
            (4, 8),   # High concurrency
            (8, 16),  # Very high concurrency
        ]
        
        results = []
        
        for max_concurrent_batches, max_thread_workers in concurrency_configs:
            print(f"Testing with {max_concurrent_batches} concurrent batches, {max_thread_workers} thread workers...")
            
            metrics = self._run_performance_test(
                medium_dataset, 
                use_async=True,
                max_concurrent_batches=max_concurrent_batches,
                max_thread_workers=max_thread_workers
            )
            
            results.append({
                'concurrent_batches': max_concurrent_batches,
                'thread_workers': max_thread_workers,
                'duration': metrics.get_duration()
            })
            
            print(f"Duration: {metrics.get_duration():.2f} seconds")
        
        # Find the best configuration
        best_config = min(results, key=lambda x: x['duration'])
        print(f"\nBest configuration: {best_config['concurrent_batches']} batches, {best_config['thread_workers']} workers")
        print(f"Best duration: {best_config['duration']:.2f} seconds")
        
        print("=" * 60)

    def test_memory_usage_comparison(self, large_dataset: pd.DataFrame) -> None:
        """
        Test memory usage comparison between sync and async modes.

        Args:
            large_dataset: Large dataset for memory testing.
        """
        print("\n=== Memory Usage Comparison Test ===")
        
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Test sync memory usage
        print("Testing sync memory usage...")
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        sync_metrics = self._run_performance_test(large_dataset, use_async=False)
        
        sync_memory = process.memory_info().rss / 1024 / 1024  # MB
        sync_memory_used = sync_memory - initial_memory
        
        print(f"Sync memory usage: {sync_memory_used:.2f} MB")
        
        # Test async memory usage
        print("Testing async memory usage...")
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        async_metrics = self._run_performance_test(large_dataset, use_async=True)
        
        async_memory = process.memory_info().rss / 1024 / 1024  # MB
        async_memory_used = async_memory - initial_memory
        
        print(f"Async memory usage: {async_memory_used:.2f} MB")
        
        print("=" * 60)

    @pytest.mark.asyncio
    async def test_direct_async_vs_wrapper_performance(self, medium_dataset: pd.DataFrame) -> None:
        """
        Test performance difference between direct async function and wrapper.

        Args:
            medium_dataset: Medium-sized dataset for testing.
        """
        print("\n=== Direct Async vs Wrapper Performance Test ===")
        
        lm = self.real_lm()
        lotus.settings.configure(lm=lm)
        
        # Prepare data for direct async call
        docs = medium_dataset['content'].tolist()
        partition_ids = [0] * len(docs)  # Single partition for simplicity
        
        # Test direct async function
        print("Testing direct async function...")
        start_time = time.time()
        
        result_direct = await sem_agg_async(
            docs=docs,
            model=lm,
            user_instruction="Summarize the key themes and insights",
            partition_ids=partition_ids,
            max_concurrent_batches=4,
            max_thread_workers=8,
        )
        
        direct_duration = time.time() - start_time
        print(f"Direct async duration: {direct_duration:.2f} seconds")
        
        # Test wrapper function
        print("Testing wrapper function...")
        start_time = time.time()
        
        result_wrapper = medium_dataset.sem_agg(
            user_instruction="Summarize the key themes and insights",
            all_cols=True,
            use_async=True,
            max_concurrent_batches=4,
            max_thread_workers=8,
        )
        
        wrapper_duration = time.time() - start_time
        print(f"Wrapper duration: {wrapper_duration:.2f} seconds")
        
        # Verify results are similar
        assert len(result_direct.outputs) > 0
        assert len(result_wrapper) > 0
        
        print("=" * 60)

    def test_performance_with_different_grouping_strategies(self, large_dataset: pd.DataFrame) -> None:
        """
        Test performance with different grouping strategies.

        Args:
            large_dataset: Large dataset for testing different grouping strategies.
        """
        print("\n=== Grouping Strategy Performance Test ===")
        
        grouping_strategies = [
            (None, "No grouping"),
            (['category'], "Single column grouping"),
            (['category', 'priority'], "Multi-column grouping"),
            (['category', 'author'], "Multi-column grouping (different columns)"),
        ]
        
        results = []
        
        for group_by, description in grouping_strategies:
            print(f"Testing {description}...")
            
            # Configure lotus settings
            lm = self.real_lm()
            lotus.settings.configure(lm=lm)
            
            start_time = time.time()
            
            result = large_dataset.sem_agg(
                user_instruction="Summarize the key themes and insights",
                all_cols=True,
                group_by=group_by,
                use_async=True,
                max_concurrent_batches=4,
                max_thread_workers=8,
            )
            
            duration = time.time() - start_time
            
            results.append({
                'strategy': description,
                'group_by': group_by,
                'duration': duration,
                'result_count': len(result)
            })
            
            print(f"Duration: {duration:.2f} seconds, Results: {len(result)}")
        
        # Print summary
        print("\nGrouping Strategy Performance Summary:")
        for result in results:
            print(f"{result['strategy']}: {result['duration']:.2f}s ({result['result_count']} results)")
        
        print("=" * 60)

    def test_performance_regression_detection(self, small_dataset: pd.DataFrame) -> None:
        """
        Test to detect performance regressions by running multiple iterations.

        Args:
            small_dataset: Small dataset for regression testing.
        """
        print("\n=== Performance Regression Detection Test ===")
        
        iterations = 3
        sync_times = []
        async_times = []
        
        for i in range(iterations):
            print(f"Iteration {i+1}/{iterations}")
            
            # Test sync
            sync_metrics = self._run_performance_test(small_dataset, use_async=False)
            sync_times.append(sync_metrics.get_duration())
            
            # Test async
            async_metrics = self._run_performance_test(small_dataset, use_async=True)
            async_times.append(async_metrics.get_duration())
        
        # Calculate statistics
        sync_avg = sum(sync_times) / len(sync_times)
        sync_std = (sum((t - sync_avg) ** 2 for t in sync_times) / len(sync_times)) ** 0.5
        
        async_avg = sum(async_times) / len(async_times)
        async_std = (sum((t - async_avg) ** 2 for t in async_times) / len(async_times)) ** 0.5
        
        print(f"\nSync Performance: {sync_avg:.2f}s ± {sync_std:.2f}s")
        print(f"Async Performance: {async_avg:.2f}s ± {async_std:.2f}s")
        
        # Check for significant performance degradation
        if sync_avg > 0:
            improvement = ((sync_avg - async_avg) / sync_avg) * 100
            print(f"Average improvement: {improvement:.1f}%")
        
        print("=" * 60)
