#!/usr/bin/env python3
"""
Real-time benchmark monitoring script.

This script monitors the progress of a running sem_agg performance benchmark
and displays real-time statistics and progress updates.

Usage:
    python examples/monitor_benchmark.py [--benchmark-dir BENCHMARK_DIR] [--refresh-interval SECONDS]
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))


class BenchmarkMonitor:
    """Real-time benchmark monitoring class."""
    
    def __init__(self, benchmark_dir: str = "benchmark_results", refresh_interval: int = 10) -> None:
        """
        Initialize the benchmark monitor.
        
        Args:
            benchmark_dir: Directory containing benchmark results.
            refresh_interval: Refresh interval in seconds.
        """
        self.benchmark_dir = Path(benchmark_dir)
        self.refresh_interval = refresh_interval
        self.realtime_json_path = self.benchmark_dir / "realtime_results.json"
        self.progress_log_path = self.benchmark_dir / "progress_log.txt"
        self.final_summary_path = self.benchmark_dir / "final_summary.json"
        
    def clear_screen(self) -> None:
        """Clear the terminal screen."""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def display_header(self) -> None:
        """Display the monitoring header."""
        print("=" * 80)
        print("sem_agg Performance Benchmark Monitor")
        print("=" * 80)
        print(f"Monitoring directory: {self.benchmark_dir}")
        print(f"Refresh interval: {self.refresh_interval} seconds")
        print(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
    
    def load_realtime_results(self) -> list:
        """
        Load real-time results from JSON file.
        
        Returns:
            list: List of test results.
        """
        try:
            if not self.realtime_json_path.exists():
                return []
            
            with open(self.realtime_json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []
    
    def load_final_summary(self) -> dict:
        """
        Load final summary if available.
        
        Returns:
            dict: Final summary data or empty dict.
        """
        try:
            if not self.final_summary_path.exists():
                return {}
            
            with open(self.final_summary_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}
    
    def display_progress_summary(self, results: list) -> None:
        """
        Display progress summary.
        
        Args:
            results: List of test results.
        """
        if not results:
            print("No test results available yet...")
            return
        
        df = pd.DataFrame(results)
        successful_tests = df[df['success'] == True]
        failed_tests = df[df['success'] == False]
        
        print(f"📊 PROGRESS SUMMARY")
        print(f"   Total tests: {len(results)}")
        print(f"   Successful: {len(successful_tests)} ({len(successful_tests)/len(results)*100:.1f}%)")
        print(f"   Failed: {len(failed_tests)} ({len(failed_tests)/len(results)*100:.1f}%)")
        
        if len(successful_tests) > 0:
            print(f"   Average duration: {successful_tests['duration'].mean():.2f}s")
            print(f"   Min duration: {successful_tests['duration'].min():.2f}s")
            print(f"   Max duration: {successful_tests['duration'].max():.2f}s")
            print(f"   Average memory: {successful_tests['memory_used_mb'].mean():.2f} MB")
    
    def display_performance_comparison(self, results: list) -> None:
        """
        Display performance comparison between sync and async modes.
        
        Args:
            results: List of test results.
        """
        if not results:
            return
        
        df = pd.DataFrame(results)
        successful_tests = df[df['success'] == True]
        
        if len(successful_tests) == 0:
            return
        
        sync_tests = successful_tests[successful_tests['use_async'] == False]
        async_tests = successful_tests[successful_tests['use_async'] == True]
        
        if len(sync_tests) > 0 and len(async_tests) > 0:
            sync_avg = sync_tests['duration'].mean()
            async_avg = async_tests['duration'].mean()
            improvement = ((sync_avg - async_avg) / sync_avg) * 100 if sync_avg > 0 else 0
            
            print(f"\n⚡ PERFORMANCE COMPARISON")
            print(f"   Sync mode: {sync_avg:.2f}s avg ({len(sync_tests)} tests)")
            print(f"   Async mode: {async_avg:.2f}s avg ({len(async_tests)} tests)")
            print(f"   Improvement: {improvement:.1f}% {'🚀' if improvement > 0 else '⚠️'}")
    
    def display_dataset_performance(self, results: list) -> None:
        """
        Display performance by dataset size.
        
        Args:
            results: List of test results.
        """
        if not results:
            return
        
        df = pd.DataFrame(results)
        successful_tests = df[df['success'] == True]
        
        if len(successful_tests) == 0:
            return
        
        print(f"\n📈 PERFORMANCE BY DATASET SIZE")
        for size in sorted(successful_tests['dataset_size'].unique()):
            size_data = successful_tests[successful_tests['dataset_size'] == size]
            avg_duration = size_data['duration'].mean()
            test_count = len(size_data)
            
            print(f"   {size:3d} documents: {avg_duration:.2f}s avg ({test_count} tests)")
    
    def display_recent_tests(self, results: list, count: int = 5) -> None:
        """
        Display recent test results.
        
        Args:
            results: List of test results.
            count: Number of recent tests to display.
        """
        if not results:
            return
        
        print(f"\n🕒 RECENT TESTS (last {count})")
        recent_tests = results[-count:]
        
        for test in recent_tests:
            status = "✅" if test['success'] else "❌"
            mode = "async" if test['use_async'] else "sync"
            duration = test['duration']
            size = test['dataset_size']
            
            print(f"   {status} {test['test_name']} - {mode} - {size} docs - {duration:.2f}s")
    
    def display_final_summary(self, summary: dict) -> None:
        """
        Display final benchmark summary.
        
        Args:
            summary: Final summary data.
        """
        if not summary:
            return
        
        print(f"\n🎯 FINAL BENCHMARK SUMMARY")
        print(f"   Completed at: {summary.get('benchmark_completed_at', 'N/A')}")
        print(f"   Total tests: {summary.get('total_tests', 0)}")
        print(f"   Success rate: {summary.get('success_rate', 0):.1f}%")
        print(f"   Total time: {summary.get('total_benchmark_time_minutes', 0):.1f} minutes")
        
        if 'overall_performance_improvement' in summary:
            improvement = summary['overall_performance_improvement']
            print(f"   Overall improvement: {improvement:.1f}% {'🚀' if improvement > 0 else '⚠️'}")
    
    def is_benchmark_complete(self) -> bool:
        """
        Check if benchmark is complete.
        
        Returns:
            bool: True if benchmark is complete.
        """
        return self.final_summary_path.exists()
    
    def monitor(self) -> None:
        """Start monitoring the benchmark."""
        print("Starting benchmark monitor...")
        print("Press Ctrl+C to stop monitoring")
        
        try:
            while True:
                self.clear_screen()
                self.display_header()
                
                # Check if benchmark is complete
                if self.is_benchmark_complete():
                    final_summary = self.load_final_summary()
                    self.display_final_summary(final_summary)
                    print(f"\n✅ Benchmark completed!")
                    break
                
                # Load and display current results
                results = self.load_realtime_results()
                
                if results:
                    self.display_progress_summary(results)
                    self.display_performance_comparison(results)
                    self.display_dataset_performance(results)
                    self.display_recent_tests(results)
                else:
                    print("⏳ Waiting for benchmark to start...")
                
                print(f"\n🔄 Refreshing in {self.refresh_interval} seconds...")
                time.sleep(self.refresh_interval)
                
        except KeyboardInterrupt:
            print(f"\n\n👋 Monitoring stopped by user.")
        except Exception as e:
            print(f"\n❌ Error during monitoring: {e}")


def main() -> None:
    """Main function to run the benchmark monitor."""
    parser = argparse.ArgumentParser(description="Monitor sem_agg performance benchmark")
    parser.add_argument(
        "--benchmark-dir",
        default="benchmark_results",
        help="Directory containing benchmark results (default: benchmark_results)"
    )
    parser.add_argument(
        "--refresh-interval",
        type=int,
        default=10,
        help="Refresh interval in seconds (default: 10)"
    )
    
    args = parser.parse_args()
    
    # Check if benchmark directory exists
    if not Path(args.benchmark_dir).exists():
        print(f"Error: Benchmark directory '{args.benchmark_dir}' does not exist.")
        print("Please run the benchmark first or specify the correct directory.")
        sys.exit(1)
    
    # Start monitoring
    monitor = BenchmarkMonitor(
        benchmark_dir=args.benchmark_dir,
        refresh_interval=args.refresh_interval
    )
    
    monitor.monitor()


if __name__ == "__main__":
    main()
