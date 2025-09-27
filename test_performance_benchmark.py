"""
Performance benchmark for enhanced sem_extract functionality.

This benchmark compares the performance of different processing strategies:
1. Standard processing (no chunking, no batching)
2. Batch processing only
3. Chunking only
4. Enhanced processing (chunking + batching)
5. Async enhanced processing
"""

import asyncio
import time
import statistics
import pytest
from typing import List, Dict, Any
import pandas as pd

import lotus
from lotus.models import LM
from lotus.sem_ops.sem_extract import sem_extract, sem_extract_async
from lotus.sem_ops.sem_extract_enhanced import sem_extract_enhanced, sem_extract_enhanced_async


class PerformanceBenchmark:
    """Performance benchmark for sem_extract strategies."""
    
    def __init__(self, model_name: str = "gpt-4o-mini"):
        """Initialize benchmark with model configuration."""
        self.model = LM(model="openrouter/google/gemini-2.5-flash", base_url="https://openrouter.ai/api/v1", api_key = '"key"-v1-ff17cd1b6aee306e43330dfeb8e0f7f4b85525ec32126903c2c099b8a5f3eb84')
        self.results = {}
    
    def create_test_documents(self, num_short: int = 10, num_long: int = 5) -> List[Dict[str, Any]]:
        """Create test documents of different sizes."""
        docs = []
        
        # Short documents
        for i in range(num_short):
            docs.append({
                "text": f"This is a short document {i}. It contains basic information.",
                "doc_id": f"short_{i}"
            })
        
        # Long documents
        for i in range(num_long):
            docs.append({
                "text": f"This is a very long document {i}. " * 200,  # ~2000 words
                "doc_id": f"long_{i}"
            })
        
        return docs
    
    def benchmark_standard_processing(self, docs: List[Dict[str, Any]], output_cols: Dict[str, str]) -> Dict[str, Any]:
        """Benchmark standard processing (no chunking, no batching)."""
        print("Benchmarking standard processing...")
        
        start_time = time.time()
        try:
            result = sem_extract(
                docs=docs,
                model=self.model,
                output_cols=output_cols,
                use_batch_processing=False
            )
            end_time = time.time()
            
            return {
                "strategy": "standard",
                "duration": end_time - start_time,
                "success": True,
                "outputs_count": len(result.outputs),
                "error": None
            }
        except Exception as e:
            end_time = time.time()
            return {
                "strategy": "standard",
                "duration": end_time - start_time,
                "success": False,
                "outputs_count": 0,
                "error": str(e)
            }
    
    def benchmark_batch_processing(self, docs: List[Dict[str, Any]], output_cols: Dict[str, str]) -> Dict[str, Any]:
        """Benchmark batch processing only."""
        print("Benchmarking batch processing...")
        
        start_time = time.time()
        try:
            result = sem_extract(
                docs=docs,
                model=self.model,
                output_cols=output_cols,
                use_batch_processing=True,
                batch_size=5
            )
            end_time = time.time()
            
            return {
                "strategy": "batch",
                "duration": end_time - start_time,
                "success": True,
                "outputs_count": len(result.outputs),
                "error": None
            }
        except Exception as e:
            end_time = time.time()
            return {
                "strategy": "batch",
                "duration": end_time - start_time,
                "success": False,
                "outputs_count": 0,
                "error": str(e)
            }
    
    def benchmark_chunking_only(self, docs: List[Dict[str, Any]], output_cols: Dict[str, str]) -> Dict[str, Any]:
        """Benchmark chunking only (no batching)."""
        print("Benchmarking chunking only...")
        
        start_time = time.time()
        try:
            result = sem_extract_enhanced(
                docs=docs,
                model=self.model,
                output_cols=output_cols,
                enable_chunking=True,
                use_batch_processing=False,
                chunk_size=1000,
                chunk_overlap=100
            )
            end_time = time.time()
            
            return {
                "strategy": "chunking_only",
                "duration": end_time - start_time,
                "success": True,
                "outputs_count": len(result.outputs),
                "error": None
            }
        except Exception as e:
            end_time = time.time()
            return {
                "strategy": "chunking_only",
                "duration": end_time - start_time,
                "success": False,
                "outputs_count": 0,
                "error": str(e)
            }
    
    def benchmark_enhanced_processing(self, docs: List[Dict[str, Any]], output_cols: Dict[str, str]) -> Dict[str, Any]:
        """Benchmark enhanced processing (chunking + batching)."""
        print("Benchmarking enhanced processing...")
        
        start_time = time.time()
        try:
            result = sem_extract_enhanced(
                docs=docs,
                model=self.model,
                output_cols=output_cols,
                enable_chunking=True,
                use_batch_processing=True,
                batch_size=5,
                chunk_size=1000,
                chunk_overlap=100
            )
            end_time = time.time()
            
            return {
                "strategy": "enhanced",
                "duration": end_time - start_time,
                "success": True,
                "outputs_count": len(result.outputs),
                "error": None
            }
        except Exception as e:
            end_time = time.time()
            return {
                "strategy": "enhanced",
                "duration": end_time - start_time,
                "success": False,
                "outputs_count": 0,
                "error": str(e)
            }
    
    async def benchmark_async_enhanced(self, docs: List[Dict[str, Any]], output_cols: Dict[str, str]) -> Dict[str, Any]:
        """Benchmark async enhanced processing."""
        print("Benchmarking async enhanced processing...")
        
        start_time = time.time()
        try:
            result = await sem_extract_enhanced_async(
                docs=docs,
                model=self.model,
                output_cols=output_cols,
                enable_chunking=True,
                use_batch_processing=True,
                batch_size=5,
                max_concurrent_batches=3,
                chunk_size=1000,
                chunk_overlap=100
            )
            end_time = time.time()
            
            return {
                "strategy": "async_enhanced",
                "duration": end_time - start_time,
                "success": True,
                "outputs_count": len(result.outputs),
                "error": None
            }
        except Exception as e:
            end_time = time.time()
            return {
                "strategy": "async_enhanced",
                "duration": end_time - start_time,
                "success": False,
                "outputs_count": 0,
                "error": str(e)
            }
    
    async def run_benchmark(self, num_short: int = 10, num_long: int = 5, iterations: int = 3) -> Dict[str, Any]:
        """Run comprehensive benchmark."""
        print(f"Running benchmark with {num_short} short docs, {num_long} long docs, {iterations} iterations")
        
        output_cols = {
            "sentiment": "positive/negative/neutral",
            "confidence": "0-1 scale",
            "topic": "main topic"
        }
        
        all_results = []
        
        for iteration in range(iterations):
            print(f"\n--- Iteration {iteration + 1} ---")
            
            # Create test documents
            docs = self.create_test_documents(num_short, num_long)
            
            # Run all benchmarks
            iteration_results = {
                "iteration": iteration + 1,
                "doc_count": len(docs),
                "strategies": {}
            }
            
            # Standard processing
            result = self.benchmark_standard_processing(docs, output_cols)
            iteration_results["strategies"]["standard"] = result
            
            # Batch processing
            result = self.benchmark_batch_processing(docs, output_cols)
            iteration_results["strategies"]["batch"] = result
            
            # Chunking only
            result = self.benchmark_chunking_only(docs, output_cols)
            iteration_results["strategies"]["chunking_only"] = result
            
            # Enhanced processing
            result = self.benchmark_enhanced_processing(docs, output_cols)
            iteration_results["strategies"]["enhanced"] = result
            
            # Async enhanced processing
            result = await self.benchmark_async_enhanced(docs, output_cols)
            iteration_results["strategies"]["async_enhanced"] = result
            
            all_results.append(iteration_results)
        
        return self.analyze_results(all_results)
    
    def analyze_results(self, all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze benchmark results."""
        print("\n=== Benchmark Analysis ===")
        
        # Extract results by strategy
        strategy_results = {}
        for strategy in ["standard", "batch", "chunking_only", "enhanced", "async_enhanced"]:
            durations = []
            success_count = 0
            
            for iteration in all_results:
                if strategy in iteration["strategies"]:
                    result = iteration["strategies"][strategy]
                    if result["success"]:
                        durations.append(result["duration"])
                        success_count += 1
            
            if durations:
                strategy_results[strategy] = {
                    "avg_duration": statistics.mean(durations),
                    "min_duration": min(durations),
                    "max_duration": max(durations),
                    "success_rate": success_count / len(all_results),
                    "success_count": success_count,
                    "total_iterations": len(all_results)
                }
        
        # Calculate performance improvements
        if "standard" in strategy_results and "enhanced" in strategy_results:
            standard_avg = strategy_results["standard"]["avg_duration"]
            enhanced_avg = strategy_results["enhanced"]["avg_duration"]
            improvement = (standard_avg - enhanced_avg) / standard_avg * 100
            strategy_results["performance_improvement"] = improvement
        
        return {
            "summary": strategy_results,
            "detailed_results": all_results
        }
    
    def print_results(self, results: Dict[str, Any]):
        """Print benchmark results in a formatted way."""
        print("\n" + "=" * 60)
        print("PERFORMANCE BENCHMARK RESULTS")
        print("=" * 60)
        
        summary = results["summary"]
        
        print("\nStrategy Performance:")
        print("-" * 40)
        
        for strategy, data in summary.items():
            if strategy == "performance_improvement":
                continue
                
            print(f"\n{strategy.upper()}:")
            print(f"  Average Duration: {data['avg_duration']:.2f}s")
            print(f"  Min Duration: {data['min_duration']:.2f}s")
            print(f"  Max Duration: {data['max_duration']:.2f}s")
            print(f"  Success Rate: {data['success_rate']:.1%}")
            print(f"  Successful Runs: {data['success_count']}/{data['total_iterations']}")
        
        if "performance_improvement" in summary:
            improvement = summary["performance_improvement"]
            print(f"\nPerformance Improvement:")
            print(f"  Enhanced vs Standard: {improvement:.1f}%")
        
        print("\n" + "=" * 60)


async def run_performance_benchmark():
    """Run the complete performance benchmark."""
    print("Enhanced sem_extract Performance Benchmark")
    print("=" * 50)
    
    # Initialize benchmark
    benchmark = PerformanceBenchmark("gpt-4o-mini")
    
    # Run benchmark with different document sizes
    test_configs = [
        {"num_short": 5, "num_long": 2, "iterations": 2},
        {"num_short": 10, "num_long": 5, "iterations": 2},
        {"num_short": 20, "num_long": 10, "iterations": 1}
    ]
    
    for i, config in enumerate(test_configs):
        print(f"\n--- Test Configuration {i + 1} ---")
        print(f"Short docs: {config['num_short']}, Long docs: {config['num_long']}, Iterations: {config['iterations']}")
        
        try:
            results = await benchmark.run_benchmark(**config)
            benchmark.print_results(results)
        except Exception as e:
            print(f"Benchmark failed: {e}")
    
    print("\nBenchmark completed!")


def run_mock_benchmark():
    """Run a mock benchmark without actual API calls."""
    print("Mock Performance Benchmark (No API Calls)")
    print("=" * 50)
    
    # Simulate benchmark results
    mock_results = {
        "summary": {
            "standard": {
                "avg_duration": 10.5,
                "min_duration": 9.8,
                "max_duration": 11.2,
                "success_rate": 0.8,
                "success_count": 4,
                "total_iterations": 5
            },
            "batch": {
                "avg_duration": 6.2,
                "min_duration": 5.9,
                "max_duration": 6.5,
                "success_rate": 0.9,
                "success_count": 4,
                "total_iterations": 5
            },
            "chunking_only": {
                "avg_duration": 8.1,
                "min_duration": 7.8,
                "max_duration": 8.4,
                "success_rate": 0.85,
                "success_count": 4,
                "total_iterations": 5
            },
            "enhanced": {
                "avg_duration": 4.3,
                "min_duration": 4.0,
                "max_duration": 4.6,
                "success_rate": 0.95,
                "success_count": 4,
                "total_iterations": 5
            },
            "async_enhanced": {
                "avg_duration": 2.8,
                "min_duration": 2.5,
                "max_duration": 3.1,
                "success_rate": 0.9,
                "success_count": 4,
                "total_iterations": 5
            },
            "performance_improvement": 59.0
        }
    }
    
    # Print mock results
    benchmark = PerformanceBenchmark()
    benchmark.print_results(mock_results)
    
    print("\nMock benchmark completed!")
    print("Note: These are simulated results for demonstration purposes.")


# Add pytest test functions
def test_benchmark_initialization():
    """Test benchmark initialization."""
    benchmark = PerformanceBenchmark("gpt-4o-mini")
    assert benchmark.model is not None
    assert benchmark.results == {}


def test_create_test_documents():
    """Test document creation."""
    benchmark = PerformanceBenchmark("gpt-4o-mini")
    docs = benchmark.create_test_documents(num_short=3, num_long=2)
    
    assert len(docs) == 5
    assert all("text" in doc for doc in docs)
    assert all("doc_id" in doc for doc in docs)
    
    # Check short documents
    short_docs = [doc for doc in docs if "short" in doc["doc_id"]]
    assert len(short_docs) == 3
    
    # Check long documents
    long_docs = [doc for doc in docs if "long" in doc["doc_id"]]
    assert len(long_docs) == 2


def test_mock_benchmark():
    """Test mock benchmark functionality."""
    print("Testing mock benchmark...")
    run_mock_benchmark()
    print("Mock benchmark test completed successfully")


@pytest.mark.asyncio
async def test_benchmark_analysis():
    """Test benchmark analysis functionality."""
    benchmark = PerformanceBenchmark("gpt-4o-mini")
    
    # Create mock results
    mock_results = [
        {
            "iteration": 1,
            "doc_count": 5,
            "strategies": {
                "standard": {"duration": 10.0, "success": True},
                "batch": {"duration": 6.0, "success": True},
                "enhanced": {"duration": 4.0, "success": True}
            }
        }
    ]
    
    analysis = benchmark.analyze_results(mock_results)
    
    assert "summary" in analysis
    assert "detailed_results" in analysis
    assert "standard" in analysis["summary"]
    assert "batch" in analysis["summary"]
    assert "enhanced" in analysis["summary"]


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "mock":
        # Run mock benchmark
        run_mock_benchmark()
    else:
        # Run real benchmark (requires API key)
        print("Running real benchmark (requires API key)...")
        print("Use 'python test_performance_benchmark.py mock' for mock results")
        asyncio.run(run_performance_benchmark())
