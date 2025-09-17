#!/usr/bin/env python3
"""
Simple script to run sem_agg performance tests with real-time monitoring.

This script provides an easy way to run performance tests and monitor them
in real-time. It combines the benchmark and monitoring functionality.

Usage:
    python examples/run_performance_test.py [--size SIZE] [--iterations ITERATIONS] [--monitor]
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_api_key() -> bool:
    """
    Check if an API key is available.
    
    Returns:
        bool: True if API key is available.
    """
    api_keys = [
        "OPENROUTER_API_KEY",
        "LOTUS_TEST_API_KEY", 
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY"
    ]
    
    return any(os.getenv(key) for key in api_keys)


def run_quick_test(size: int, iterations: int) -> None:
    """
    Run a quick performance test.
    
    Args:
        size: Dataset size to test.
        iterations: Number of iterations to run.
    """
    print(f"Running quick performance test with {size} documents, {iterations} iterations...")
    
    cmd = [
        sys.executable, 
        "examples/quick_performance_test.py",
        "--size", str(size),
        "--iterations", str(iterations)
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("✅ Quick test completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Quick test failed with error: {e}")
        sys.exit(1)


def run_comprehensive_benchmark(output_dir: str, iterations: int, monitor: bool) -> None:
    """
    Run comprehensive benchmark with optional monitoring.
    
    Args:
        output_dir: Output directory for results.
        iterations: Number of iterations to run.
        monitor: Whether to start monitoring in parallel.
    """
    print(f"Running comprehensive benchmark...")
    print(f"Results will be saved to: {output_dir}")
    
    # Start benchmark in background if monitoring is enabled
    if monitor:
        print("Starting benchmark in background with monitoring...")
        
        cmd = [
            sys.executable,
            "examples/sem_agg_performance_benchmark.py",
            "--output-dir", output_dir,
            "--iterations", str(iterations)
        ]
        
        # Start benchmark process
        benchmark_process = subprocess.Popen(cmd)
        
        # Wait a moment for benchmark to start
        time.sleep(3)
        
        # Start monitoring
        print("Starting real-time monitoring...")
        monitor_cmd = [
            sys.executable,
            "examples/monitor_benchmark.py",
            "--benchmark-dir", output_dir,
            "--refresh-interval", "5"
        ]
        
        try:
            # Run monitor (this will block until benchmark completes)
            monitor_result = subprocess.run(monitor_cmd, check=True)
            
            # Wait for benchmark to complete
            benchmark_process.wait()
            
            print("✅ Comprehensive benchmark completed successfully!")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Monitoring failed: {e}")
            benchmark_process.terminate()
            sys.exit(1)
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user. Stopping benchmark...")
            benchmark_process.terminate()
            sys.exit(1)
    
    else:
        # Run benchmark without monitoring
        cmd = [
            sys.executable,
            "examples/sem_agg_performance_benchmark.py",
            "--output-dir", output_dir,
            "--iterations", str(iterations)
        ]
        
        try:
            result = subprocess.run(cmd, check=True)
            print("✅ Comprehensive benchmark completed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Benchmark failed: {e}")
            sys.exit(1)


def main() -> None:
    """Main function to run performance tests."""
    parser = argparse.ArgumentParser(description="Run sem_agg performance tests")
    parser.add_argument(
        "--mode",
        choices=["quick", "comprehensive"],
        default="quick",
        help="Test mode: quick or comprehensive (default: quick)"
    )
    parser.add_argument(
        "--size",
        type=int,
        default=50,
        help="Dataset size for quick test (default: 50)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of iterations to run (default: 3)"
    )
    parser.add_argument(
        "--output-dir",
        default="benchmark_results",
        help="Output directory for comprehensive benchmark (default: benchmark_results)"
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        help="Enable real-time monitoring for comprehensive benchmark"
    )
    
    args = parser.parse_args()
    
    # Check for API key
    if not check_api_key():
        print("❌ Error: No API key found. Please set one of the following environment variables:")
        print("   - OPENROUTER_API_KEY")
        print("   - LOTUS_TEST_API_KEY") 
        print("   - OPENAI_API_KEY")
        print("   - ANTHROPIC_API_KEY")
        sys.exit(1)
    
    print("🚀 sem_agg Performance Test Runner")
    print("=" * 50)
    
    if args.mode == "quick":
        run_quick_test(args.size, args.iterations)
    else:
        run_comprehensive_benchmark(args.output_dir, args.iterations, args.monitor)
    
    print("\n📊 Performance test completed!")
    print("Check the output files for detailed results and recommendations.")


if __name__ == "__main__":
    main()
