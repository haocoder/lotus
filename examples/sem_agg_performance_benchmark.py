#!/usr/bin/env python3
"""
Performance benchmark script for sem_agg async optimization.

This script generates datasets of different scales and compares the performance
of synchronous vs asynchronous sem_agg operations. It provides detailed metrics
and generates performance reports.

Usage:
    python examples/sem_agg_performance_benchmark.py [--output-dir OUTPUT_DIR] [--iterations ITERATIONS]
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

import lotus
from lotus.models import LM
from lotus.sem_ops.sem_agg import sem_agg, sem_agg_async


class PerformanceBenchmark:
    """Performance benchmark class for sem_agg operations."""
    
    def __init__(self, output_dir: str = "benchmark_results", iterations: int = 3) -> None:
        """
        Initialize the performance benchmark.
        
        Args:
            output_dir: Directory to save benchmark results.
            iterations: Number of iterations to run for each test.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.iterations = iterations
        self.results: List[Dict[str, Any]] = []
        
        # Initialize real-time logging files
        self.realtime_csv_path = self.output_dir / "realtime_results.csv"
        self.realtime_json_path = self.output_dir / "realtime_results.json"
        self.progress_log_path = self.output_dir / "progress_log.txt"
        self._initialize_realtime_files()
        
        # Progress tracking
        self.completed_tests = 0
        self.total_tests = 0
        self.start_time = None
        
        # Initialize LM
        self.lm = self._create_lm()
        lotus.settings.configure(lm=self.lm)
    
    def _initialize_realtime_files(self) -> None:
        """
        Initialize real-time logging files with headers.
        """
        # Initialize CSV file with headers
        csv_headers = [
            'test_name', 'dataset_size', 'use_async', 'max_concurrent_batches', 
            'max_thread_workers', 'group_by', 'duration', 'memory_used_mb', 
            'success', 'result_count', 'timestamp', 'error'
        ]
        
        with open(self.realtime_csv_path, 'w', encoding='utf-8') as f:
            f.write(','.join(csv_headers) + '\n')
        
        # Initialize JSON file with empty array
        with open(self.realtime_json_path, 'w', encoding='utf-8') as f:
            json.dump([], f, indent=2)
        
        # Initialize progress log file
        with open(self.progress_log_path, 'w', encoding='utf-8') as f:
            f.write(f"Performance Benchmark Progress Log\n")
            f.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n\n")
        
        print(f"Real-time logging initialized:")
        print(f"  CSV: {self.realtime_csv_path}")
        print(f"  JSON: {self.realtime_json_path}")
        print(f"  Progress: {self.progress_log_path}")
    
    def _write_realtime_result(self, result: Dict[str, Any]) -> None:
        """
        Write a single test result to real-time files.
        
        Args:
            result: Test result dictionary to write.
        """
        # Write to CSV file (append mode)
        with open(self.realtime_csv_path, 'a', encoding='utf-8') as f:
            # Convert result to CSV row
            row_data = [
                str(result.get('test_name', '')),
                str(result.get('dataset_size', '')),
                str(result.get('use_async', '')),
                str(result.get('max_concurrent_batches', '')),
                str(result.get('max_thread_workers', '')),
                str(result.get('group_by', '')),
                str(result.get('duration', '')),
                str(result.get('memory_used_mb', '')),
                str(result.get('success', '')),
                str(result.get('result_count', '')),
                str(result.get('timestamp', '')),
                str(result.get('error', ''))
            ]
            f.write(','.join(row_data) + '\n')
        
        # Write to JSON file (read, append, write)
        try:
            with open(self.realtime_json_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            existing_data = []
        
        existing_data.append(result)
        
        with open(self.realtime_json_path, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, indent=2, default=str)
        
        # Update progress log
        self._update_progress_log(result)
    
    def _update_progress_log(self, result: Dict[str, Any]) -> None:
        """
        Update the progress log with test completion information.
        
        Args:
            result: Test result dictionary.
        """
        self.completed_tests += 1
        
        # Calculate progress percentage
        progress_percent = (self.completed_tests / self.total_tests * 100) if self.total_tests > 0 else 0
        
        # Calculate elapsed time
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        
        # Estimate remaining time
        if self.completed_tests > 0:
            avg_time_per_test = elapsed_time / self.completed_tests
            remaining_tests = self.total_tests - self.completed_tests
            estimated_remaining = avg_time_per_test * remaining_tests
        else:
            estimated_remaining = 0
        
        # Format log entry
        log_entry = (
            f"[{datetime.now().strftime('%H:%M:%S')}] "
            f"Test {self.completed_tests}/{self.total_tests} ({progress_percent:.1f}%) - "
            f"{result['test_name']} - "
            f"{'✓' if result['success'] else '✗'} "
            f"{result['duration']:.2f}s"
        )
        
        if not result['success'] and 'error' in result:
            log_entry += f" - ERROR: {result['error'][:100]}"
        
        log_entry += f" - ETA: {estimated_remaining/60:.1f}min\n"
        
        # Write to progress log file
        with open(self.progress_log_path, 'a', encoding='utf-8') as f:
            f.write(log_entry)
        
        # Also print to console
        print(f"  Progress: {self.completed_tests}/{self.total_tests} ({progress_percent:.1f}%) - ETA: {estimated_remaining/60:.1f}min")
        
        # Generate real-time summary every 5 tests
        if self.completed_tests % 5 == 0:
            self._generate_realtime_summary()
    
    def _generate_realtime_summary(self) -> None:
        """
        Generate a real-time performance summary from current results.
        """
        try:
            # Read current results from JSON file
            with open(self.realtime_json_path, 'r', encoding='utf-8') as f:
                current_results = json.load(f)
            
            if not current_results:
                return
            
            # Convert to DataFrame for analysis
            df = pd.DataFrame(current_results)
            
            # Calculate summary statistics
            successful_tests = df[df['success'] == True]
            failed_tests = df[df['success'] == False]
            
            summary = {
                'timestamp': datetime.now().isoformat(),
                'total_tests': len(current_results),
                'successful_tests': len(successful_tests),
                'failed_tests': len(failed_tests),
                'success_rate': len(successful_tests) / len(current_results) * 100 if current_results else 0,
                'avg_duration': successful_tests['duration'].mean() if len(successful_tests) > 0 else 0,
                'min_duration': successful_tests['duration'].min() if len(successful_tests) > 0 else 0,
                'max_duration': successful_tests['duration'].max() if len(successful_tests) > 0 else 0,
                'avg_memory_mb': successful_tests['memory_used_mb'].mean() if len(successful_tests) > 0 else 0
            }
            
            # Performance comparison by mode
            sync_tests = successful_tests[successful_tests['use_async'] == False]
            async_tests = successful_tests[successful_tests['use_async'] == True]
            
            if len(sync_tests) > 0 and len(async_tests) > 0:
                sync_avg = sync_tests['duration'].mean()
                async_avg = async_tests['duration'].mean()
                improvement = ((sync_avg - async_avg) / sync_avg) * 100 if sync_avg > 0 else 0
                
                summary.update({
                    'sync_avg_duration': sync_avg,
                    'async_avg_duration': async_avg,
                    'performance_improvement': improvement
                })
            
            # Save summary to file
            summary_path = self.output_dir / "realtime_summary.json"
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, default=str)
            
            # Log summary to progress file
            with open(self.progress_log_path, 'a', encoding='utf-8') as f:
                f.write(f"\n--- Real-time Summary (Test {self.completed_tests}) ---\n")
                f.write(f"Success Rate: {summary['success_rate']:.1f}% ({summary['successful_tests']}/{summary['total_tests']})\n")
                f.write(f"Average Duration: {summary['avg_duration']:.2f}s\n")
                if 'performance_improvement' in summary:
                    f.write(f"Async vs Sync Improvement: {summary['performance_improvement']:.1f}%\n")
                f.write(f"Average Memory Usage: {summary['avg_memory_mb']:.2f} MB\n\n")
            
        except Exception as e:
            # Don't let summary generation errors break the benchmark
            with open(self.progress_log_path, 'a', encoding='utf-8') as f:
                f.write(f"Error generating real-time summary: {str(e)}\n")
    
    def _create_lm(self) -> LM:
        """
        Create a real LM instance for benchmarking.
        
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
    
    def generate_dataset(self, size: int, name: str) -> pd.DataFrame:
        """
        Generate a dataset of specified size with realistic content.
        
        Args:
            size: The number of documents to generate.
            name: The name of the dataset for identification.
            
        Returns:
            pd.DataFrame: A DataFrame with the specified number of documents.
        """
        categories = [
            'Technology', 'Science', 'Business', 'Health', 'Education', 
            'Finance', 'Sports', 'Entertainment', 'Politics', 'Environment'
        ]
        authors = [
            'Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank', 
            'Grace', 'Henry', 'Ivy', 'Jack'
        ]
        priorities = ['High', 'Medium', 'Low']
        topics = [
            'artificial intelligence', 'machine learning', 'data science', 'cloud computing',
            'cybersecurity', 'blockchain', 'quantum computing', 'robotics', 'biotechnology',
            'renewable energy', 'space exploration', 'climate change', 'digital transformation',
            'neural networks', 'deep learning', 'natural language processing', 'computer vision',
            'edge computing', 'internet of things', 'augmented reality', 'virtual reality'
        ]
        
        data = []
        for i in range(size):
            topic = topics[i % len(topics)]
            category = categories[i % len(categories)]
            author = authors[i % len(authors)]
            priority = priorities[i % len(priorities)]
            
            # Generate more detailed content for larger datasets
            if size > 100:
                content = (
                    f"This is document {i+1} about {topic} in the {category.lower()} domain. "
                    f"It provides comprehensive coverage of {topic} concepts, methodologies, "
                    f"and practical applications. The document discusses current trends, "
                    f"challenges, and future prospects in {topic}. It includes detailed "
                    f"technical information, case studies, and real-world examples that "
                    f"demonstrate the practical implementation of {topic} principles. "
                    f"The content is structured to provide both theoretical understanding "
                    f"and practical insights for professionals working in this field."
                )
            else:
                content = (
                    f"This is document {i+1} about {topic} in the {category.lower()} domain. "
                    f"It discusses important concepts, methodologies, and applications "
                    f"related to {topic}. The content covers various aspects including "
                    f"technical details, practical implementations, and future prospects. "
                    f"This document provides comprehensive information for understanding "
                    f"the current state and potential developments in {topic}."
                )
            
            data.append({
                'content': content,
                'category': category,
                'author': author,
                'priority': priority,
                'topic': topic,
                'year': 2020 + (i % 4),
                'document_id': f"doc_{i+1:04d}",
                'word_count': len(content.split()),
                'char_count': len(content)
            })
        
        return pd.DataFrame(data)
    
    def run_benchmark_test(
        self, 
        dataset: pd.DataFrame, 
        test_name: str,
        use_async: bool, 
        max_concurrent_batches: int = 4,
        max_thread_workers: int = 8,
        group_by: List[str] = None
    ) -> Dict[str, Any]:
        """
        Run a single benchmark test.
        
        Args:
            dataset: The dataset to test with.
            test_name: Name of the test for identification.
            use_async: Whether to use async processing.
            max_concurrent_batches: Maximum concurrent batches for async mode.
            max_thread_workers: Maximum thread workers for async mode.
            group_by: Columns to group by for aggregation.
            
        Returns:
            Dict containing test results and metrics.
        """
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            # Run the aggregation
            result = dataset.sem_agg(
                user_instruction="Summarize the key themes, insights, and important information across all documents",
                all_cols=True,
                group_by=group_by,
                use_async=use_async,
                max_concurrent_batches=max_concurrent_batches,
                max_thread_workers=max_thread_workers,
            )
            
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            duration = end_time - start_time
            memory_used = end_memory - start_memory
            
            # Verify results
            success = (
                isinstance(result, pd.DataFrame) and
                '_output' in result.columns and
                len(result) > 0 and
                all(isinstance(output, str) and len(output) > 0 for output in result['_output'])
            )
            
            result_dict = {
                'test_name': test_name,
                'dataset_size': len(dataset),
                'use_async': use_async,
                'max_concurrent_batches': max_concurrent_batches,
                'max_thread_workers': max_thread_workers,
                'group_by': group_by,
                'duration': duration,
                'memory_used_mb': memory_used,
                'success': success,
                'result_count': len(result),
                'timestamp': datetime.now().isoformat()
            }
            
            # Write result to real-time files immediately
            self._write_realtime_result(result_dict)
            
            return result_dict
            
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            
            error_result = {
                'test_name': test_name,
                'dataset_size': len(dataset),
                'use_async': use_async,
                'max_concurrent_batches': max_concurrent_batches,
                'max_thread_workers': max_thread_workers,
                'group_by': group_by,
                'duration': duration,
                'memory_used_mb': 0,
                'success': False,
                'error': str(e),
                'result_count': 0,
                'timestamp': datetime.now().isoformat()
            }
            
            # Write error result to real-time files immediately
            self._write_realtime_result(error_result)
            
            return error_result
    
    def _get_memory_usage(self) -> float:
        """
        Get current memory usage in MB.
        
        Returns:
            float: Memory usage in MB.
        """
        try:
            import psutil
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
    
    def run_comprehensive_benchmark(self) -> None:
        """Run comprehensive benchmark tests across different dataset sizes and configurations."""
        print("Starting comprehensive sem_agg performance benchmark...")
        print(f"Results will be saved to: {self.output_dir}")
        print("=" * 80)
        
        # Define test configurations
        # dataset_sizes = [10, 25, 50, 100, 200, 500]
        dataset_sizes = [50, 100]
        concurrency_configs = [
            # (1, 2),   # Low concurrency
            # (2, 4),   # Medium concurrency  
            (4, 8),   # High concurrency
            (8, 16),  # Very high concurrency
        ]
        grouping_strategies = [
            # (None, "no_grouping"),
            # (['category'], "single_grouping"),
            (['category', 'priority'], "multi_grouping"),
        ]
        
        self.total_tests = len(dataset_sizes) * len(concurrency_configs) * len(grouping_strategies) * 2 * self.iterations
        self.start_time = time.time()
        current_test = 0
        
        # Log benchmark start
        with open(self.progress_log_path, 'a', encoding='utf-8') as f:
            f.write(f"Benchmark Configuration:\n")
            f.write(f"  Dataset sizes: {dataset_sizes}\n")
            f.write(f"  Concurrency configs: {concurrency_configs}\n")
            f.write(f"  Grouping strategies: {grouping_strategies}\n")
            f.write(f"  Iterations: {self.iterations}\n")
            f.write(f"  Total tests: {self.total_tests}\n\n")
        
        for size in dataset_sizes:
            print(f"\nGenerating dataset of size {size}...")
            dataset = self.generate_dataset(size, f"dataset_{size}")
            
            for group_by, group_name in grouping_strategies:
                print(f"\nTesting grouping strategy: {group_name}")
                
                for max_concurrent_batches, max_thread_workers in concurrency_configs:
                    print(f"  Concurrency: {max_concurrent_batches} batches, {max_thread_workers} workers")
                    
                    # Test sync mode
                    for iteration in range(self.iterations):
                        current_test += 1
                        test_name = f"sync_{size}_{group_name}_{max_concurrent_batches}_{max_thread_workers}_iter{iteration}"
                        
                        print(f"    [{current_test}/{self.total_tests}] Sync test...", end=" ")
                        
                        result = self.run_benchmark_test(
                            dataset, test_name, use_async=False,
                            max_concurrent_batches=max_concurrent_batches,
                            max_thread_workers=max_thread_workers,
                            group_by=group_by
                        )
                        
                        self.results.append(result)
                        print(f"{result['duration']:.2f}s")
                    
                    # Test async mode
                    for iteration in range(self.iterations):
                        current_test += 1
                        test_name = f"async_{size}_{group_name}_{max_concurrent_batches}_{max_thread_workers}_iter{iteration}"
                        
                        print(f"    [{current_test}/{self.total_tests}] Async test...", end=" ")
                        
                        result = self.run_benchmark_test(
                            dataset, test_name, use_async=True,
                            max_concurrent_batches=max_concurrent_batches,
                            max_thread_workers=max_thread_workers,
                            group_by=group_by
                        )
                        
                        self.results.append(result)
                        print(f"{result['duration']:.2f}s")
        
        print("\n" + "=" * 80)
        print("Benchmark completed! Generating reports...")
        
        # Generate final summary
        self._generate_final_summary()
        
        # Generate reports
        self.generate_performance_report()
        self.save_results()
        
        print(f"Results saved to: {self.output_dir}")
        print(f"Real-time results available at:")
        print(f"  - {self.realtime_csv_path}")
        print(f"  - {self.realtime_json_path}")
        print(f"  - {self.progress_log_path}")
    
    def _generate_final_summary(self) -> None:
        """
        Generate a final summary of the benchmark results.
        """
        try:
            # Read all results from JSON file
            with open(self.realtime_json_path, 'r', encoding='utf-8') as f:
                all_results = json.load(f)
            
            if not all_results:
                return
            
            # Convert to DataFrame for analysis
            df = pd.DataFrame(all_results)
            
            # Calculate final statistics
            successful_tests = df[df['success'] == True]
            failed_tests = df[df['success'] == False]
            
            total_time = time.time() - self.start_time if self.start_time else 0
            
            final_summary = {
                'benchmark_completed_at': datetime.now().isoformat(),
                'total_tests': len(all_results),
                'successful_tests': len(successful_tests),
                'failed_tests': len(failed_tests),
                'success_rate': len(successful_tests) / len(all_results) * 100 if all_results else 0,
                'total_benchmark_time_minutes': total_time / 60,
                'avg_test_duration': successful_tests['duration'].mean() if len(successful_tests) > 0 else 0,
                'min_test_duration': successful_tests['duration'].min() if len(successful_tests) > 0 else 0,
                'max_test_duration': successful_tests['duration'].max() if len(successful_tests) > 0 else 0,
                'avg_memory_usage_mb': successful_tests['memory_used_mb'].mean() if len(successful_tests) > 0 else 0
            }
            
            # Performance comparison by mode
            sync_tests = successful_tests[successful_tests['use_async'] == False]
            async_tests = successful_tests[successful_tests['use_async'] == True]
            
            if len(sync_tests) > 0 and len(async_tests) > 0:
                sync_avg = sync_tests['duration'].mean()
                async_avg = async_tests['duration'].mean()
                improvement = ((sync_avg - async_avg) / sync_avg) * 100 if sync_avg > 0 else 0
                
                final_summary.update({
                    'sync_avg_duration': sync_avg,
                    'async_avg_duration': async_avg,
                    'overall_performance_improvement': improvement,
                    'sync_tests_count': len(sync_tests),
                    'async_tests_count': len(async_tests)
                })
            
            # Performance by dataset size
            dataset_performance = {}
            for size in df['dataset_size'].unique():
                size_data = df[df['dataset_size'] == size]
                size_successful = size_data[size_data['success'] == True]
                if len(size_successful) > 0:
                    dataset_performance[size] = {
                        'avg_duration': size_successful['duration'].mean(),
                        'test_count': len(size_successful),
                        'success_rate': len(size_successful) / len(size_data) * 100
                    }
            
            final_summary['dataset_performance'] = dataset_performance
            
            # Save final summary
            final_summary_path = self.output_dir / "final_summary.json"
            with open(final_summary_path, 'w', encoding='utf-8') as f:
                json.dump(final_summary, f, indent=2, default=str)
            
            # Log final summary to progress file
            with open(self.progress_log_path, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*50}\n")
                f.write(f"FINAL BENCHMARK SUMMARY\n")
                f.write(f"{'='*50}\n")
                f.write(f"Completed at: {final_summary['benchmark_completed_at']}\n")
                f.write(f"Total tests: {final_summary['total_tests']}\n")
                f.write(f"Successful: {final_summary['successful_tests']} ({final_summary['success_rate']:.1f}%)\n")
                f.write(f"Failed: {final_summary['failed_tests']}\n")
                f.write(f"Total benchmark time: {final_summary['total_benchmark_time_minutes']:.1f} minutes\n")
                f.write(f"Average test duration: {final_summary['avg_test_duration']:.2f}s\n")
                
                if 'overall_performance_improvement' in final_summary:
                    f.write(f"Overall async vs sync improvement: {final_summary['overall_performance_improvement']:.1f}%\n")
                    f.write(f"Sync tests: {final_summary['sync_tests_count']} (avg: {final_summary['sync_avg_duration']:.2f}s)\n")
                    f.write(f"Async tests: {final_summary['async_tests_count']} (avg: {final_summary['async_avg_duration']:.2f}s)\n")
                
                f.write(f"Average memory usage: {final_summary['avg_memory_usage_mb']:.2f} MB\n")
                f.write(f"\nDataset Performance:\n")
                for size, perf in dataset_performance.items():
                    f.write(f"  {size} docs: {perf['avg_duration']:.2f}s avg, {perf['success_rate']:.1f}% success\n")
            
            print(f"Final summary saved to: {final_summary_path}")
            
        except Exception as e:
            with open(self.progress_log_path, 'a', encoding='utf-8') as f:
                f.write(f"Error generating final summary: {str(e)}\n")
    
    def generate_performance_report(self) -> None:
        """Generate a comprehensive performance report."""
        if not self.results:
            print("No results to generate report from.")
            return
        
        # Convert results to DataFrame for analysis
        df = pd.DataFrame(self.results)
        
        # Calculate summary statistics
        summary_stats = []
        
        for size in df['dataset_size'].unique():
            for group_by in df['group_by'].unique():
                for use_async in [False, True]:
                    subset = df[
                        (df['dataset_size'] == size) & 
                        (df['group_by'] == group_by) & 
                        (df['use_async'] == use_async) &
                        (df['success'] == True)
                    ]
                    
                    if len(subset) > 0:
                        summary_stats.append({
                            'dataset_size': size,
                            'grouping': str(group_by) if group_by else 'none',
                            'mode': 'async' if use_async else 'sync',
                            'avg_duration': subset['duration'].mean(),
                            'std_duration': subset['duration'].std(),
                            'min_duration': subset['duration'].min(),
                            'max_duration': subset['duration'].max(),
                            'avg_memory_mb': subset['memory_used_mb'].mean(),
                            'test_count': len(subset)
                        })
        
        summary_df = pd.DataFrame(summary_stats)
        
        # Generate report
        report_path = self.output_dir / "performance_report.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# sem_agg Performance Benchmark Report\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Total tests run: {len(self.results)}\n")
            f.write(f"Successful tests: {len(df[df['success'] == True])}\n")
            f.write(f"Failed tests: {len(df[df['success'] == False])}\n\n")
            
            # Performance comparison by dataset size
            f.write("## Performance by Dataset Size\n\n")
            for size in sorted(df['dataset_size'].unique()):
                f.write(f"### Dataset Size: {size} documents\n\n")
                
                size_data = df[df['dataset_size'] == size]
                sync_data = size_data[(size_data['use_async'] == False) & (size_data['success'] == True)]
                async_data = size_data[(size_data['use_async'] == True) & (size_data['success'] == True)]
                
                if len(sync_data) > 0 and len(async_data) > 0:
                    sync_avg = sync_data['duration'].mean()
                    async_avg = async_data['duration'].mean()
                    improvement = ((sync_avg - async_avg) / sync_avg) * 100 if sync_avg > 0 else 0
                    
                    f.write(f"- **Sync average**: {sync_avg:.2f}s\n")
                    f.write(f"- **Async average**: {async_avg:.2f}s\n")
                    f.write(f"- **Performance improvement**: {improvement:.1f}%\n\n")
                
                # Best async configuration for this size
                if len(async_data) > 0:
                    best_async = async_data.loc[async_data['duration'].idxmin()]
                    f.write(f"- **Best async config**: {best_async['max_concurrent_batches']} batches, "
                           f"{best_async['max_thread_workers']} workers ({best_async['duration']:.2f}s)\n\n")
            
            # Detailed results table
            f.write("## Detailed Results\n\n")
            f.write(summary_df.to_markdown(index=False))
            f.write("\n\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            
            # Find best configurations
            best_configs = {}
            for size in df['dataset_size'].unique():
                size_async = df[
                    (df['dataset_size'] == size) & 
                    (df['use_async'] == True) & 
                    (df['success'] == True)
                ]
                if len(size_async) > 0:
                    best = size_async.loc[size_async['duration'].idxmin()]
                    best_configs[size] = {
                        'batches': best['max_concurrent_batches'],
                        'workers': best['max_thread_workers'],
                        'duration': best['duration']
                    }
            
            f.write("### Optimal Async Configurations by Dataset Size:\n\n")
            for size, config in best_configs.items():
                f.write(f"- **{size} documents**: {config['batches']} concurrent batches, "
                       f"{config['workers']} thread workers ({config['duration']:.2f}s)\n")
            
            f.write("\n### General Recommendations:\n\n")
            f.write("- Use async mode for datasets with 50+ documents\n")
            f.write("- Start with 4 concurrent batches and 8 thread workers\n")
            f.write("- Increase concurrency for larger datasets (200+ documents)\n")
            f.write("- Monitor memory usage with very high concurrency settings\n")
        
        print(f"Performance report generated: {report_path}")
    
    def save_results(self) -> None:
        """Save detailed results to JSON and CSV files."""
        # Save as JSON
        json_path = self.output_dir / "benchmark_results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Save as CSV
        csv_path = self.output_dir / "benchmark_results.csv"
        df = pd.DataFrame(self.results)
        df.to_csv(csv_path, index=False)
        
        print(f"Detailed results saved to: {json_path} and {csv_path}")


def main() -> None:
    """Main function to run the performance benchmark."""
    parser = argparse.ArgumentParser(description="Run sem_agg performance benchmark")
    parser.add_argument(
        "--output-dir", 
        default="benchmark_results",
        help="Directory to save benchmark results (default: benchmark_results)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of iterations to run for each test (default: 3)"
    )
    
    args = parser.parse_args()
    
    # Check for API key
    if not any([
        os.getenv("OPENROUTER_API_KEY"),
        os.getenv("LOTUS_TEST_API_KEY"),
        os.getenv("OPENAI_API_KEY"),
        os.getenv("ANTHROPIC_API_KEY")
    ]):
        print("Error: No API key found. Please set one of the following environment variables:")
        print("- OPENROUTER_API_KEY")
        print("- LOTUS_TEST_API_KEY") 
        print("- OPENAI_API_KEY")
        print("- ANTHROPIC_API_KEY")
        sys.exit(1)
    
    # Run benchmark
    benchmark = PerformanceBenchmark(
        output_dir=args.output_dir,
        iterations=args.iterations
    )
    
    try:
        benchmark.run_comprehensive_benchmark()
        print("\nBenchmark completed successfully!")
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nBenchmark failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
