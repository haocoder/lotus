"""GPU configuration and performance monitoring for Lotus."""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import json
import threading
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class GPUPerformanceMetrics:
    """Performance metrics for GPU operations."""
    
    operation_name: str
    start_time: float
    end_time: float
    duration: float
    gpu_memory_before: Optional[float] = None
    gpu_memory_after: Optional[float] = None
    gpu_memory_peak: Optional[float] = None
    data_size: Optional[int] = None
    throughput: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None
    gpu_id: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary format."""
        return {
            'operation_name': self.operation_name,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.duration,
            'gpu_memory_before': self.gpu_memory_before,
            'gpu_memory_after': self.gpu_memory_after,
            'gpu_memory_peak': self.gpu_memory_peak,
            'data_size': self.data_size,
            'throughput': self.throughput,
            'success': self.success,
            'error_message': self.error_message,
            'gpu_id': self.gpu_id
        }


@dataclass
class GPUConfig:
    """Configuration for GPU acceleration in Lotus."""
    
    # GPU preferences
    prefer_gpu: bool = True
    fallback_to_cpu: bool = True
    gpu_device_ids: List[int] = field(default_factory=lambda: [0])
    
    # Memory management
    gpu_memory_fraction: float = 0.8
    allow_growth: bool = True
    
    # Performance settings
    batch_size_gpu: Optional[int] = None
    batch_size_cpu: Optional[int] = None
    
    # Monitoring
    enable_performance_monitoring: bool = True
    log_gpu_usage: bool = True
    metrics_file: Optional[str] = None
    
    # Vector store settings
    use_gpu_vector_store: bool = True
    gpu_index_factory: str = "IVF1024,Flat"
    gpu_metric: str = "METRIC_INNER_PRODUCT"
    
    # Clustering settings
    use_gpu_clustering: bool = True
    gpu_clustering_batch_size: Optional[int] = None
    
    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.gpu_memory_fraction <= 0 or self.gpu_memory_fraction > 1:
            raise ValueError("gpu_memory_fraction must be between 0 and 1")
        
        if not self.gpu_device_ids:
            self.gpu_device_ids = [0]


class GPUPerformanceMonitor:
    """Monitor GPU performance for Lotus operations."""
    
    def __init__(self, config: GPUConfig) -> None:
        """Initialize performance monitor."""
        self.config = config
        self.metrics: List[GPUPerformanceMetrics] = []
        self._lock = threading.Lock()
        self._gpu_available = False
        self._check_gpu_availability()
    
    def _check_gpu_availability(self) -> None:
        """Check if GPU monitoring is available."""
        try:
            import pynvml  # type: ignore # Optional dependency for GPU monitoring
            pynvml.nvmlInit()
            self._gpu_available = True
            logger.info("GPU monitoring enabled")
        except ImportError:
            logger.warning("pynvml not available, GPU memory monitoring disabled")
        except Exception as e:
            logger.warning(f"GPU monitoring setup failed: {e}")
    
    def _get_gpu_memory_usage(self, gpu_id: int = 0) -> Optional[float]:
        """Get current GPU memory usage in MB."""
        if not self._gpu_available:
            return None
        
        try:
            import pynvml  # type: ignore # Optional dependency for GPU monitoring
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return mem_info.used / 1024 / 1024  # Convert bytes to MB
        except Exception as e:
            logger.warning(f"Failed to get GPU memory usage: {e}")
            return None
    
    def start_operation(self, operation_name: str, gpu_id: int = 0, data_size: Optional[int] = None) -> str:
        """Start monitoring an operation."""
        if not self.config.enable_performance_monitoring:
            return ""
        
        operation_id = f"{operation_name}_{int(time.time() * 1000000)}"
        
        with self._lock:
            # Create initial metrics entry
            metrics = GPUPerformanceMetrics(
                operation_name=operation_name,
                start_time=time.time(),
                end_time=0.0,
                duration=0.0,
                gpu_memory_before=self._get_gpu_memory_usage(gpu_id),
                data_size=data_size,
                gpu_id=gpu_id
            )
            
            # Store with operation_id as key (we'll use index for now)
            self.metrics.append(metrics)
        
        return operation_id
    
    def end_operation(self, operation_id: str, success: bool = True, error_message: Optional[str] = None) -> None:
        """End monitoring an operation."""
        if not self.config.enable_performance_monitoring or not operation_id:
            return
        
        end_time = time.time()
        
        with self._lock:
            # Find the most recent matching operation (simple implementation)
            for i in reversed(range(len(self.metrics))):
                metrics = self.metrics[i]
                if metrics.end_time == 0.0:  # Not yet completed
                    metrics.end_time = end_time
                    metrics.duration = end_time - metrics.start_time
                    metrics.success = success
                    metrics.error_message = error_message
                    
                    if metrics.gpu_id is not None:
                        metrics.gpu_memory_after = self._get_gpu_memory_usage(metrics.gpu_id)
                    
                    # Calculate throughput if data_size is available
                    if metrics.data_size and metrics.duration > 0:
                        metrics.throughput = metrics.data_size / metrics.duration
                    
                    if self.config.log_gpu_usage:
                        self._log_operation(metrics)
                    
                    break
    
    def _log_operation(self, metrics: GPUPerformanceMetrics) -> None:
        """Log operation metrics."""
        status = "SUCCESS" if metrics.success else "FAILED"
        logger.info(
            f"GPU Operation {status}: {metrics.operation_name} "
            f"Duration: {metrics.duration:.3f}s"
            + (f" Throughput: {metrics.throughput:.1f} items/s" if metrics.throughput else "")
            + (f" GPU Memory: {metrics.gpu_memory_before:.1f}MB -> {metrics.gpu_memory_after:.1f}MB" 
               if metrics.gpu_memory_before and metrics.gpu_memory_after else "")
        )
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of performance metrics."""
        with self._lock:
            if not self.metrics:
                return {}
            
            completed_metrics = [m for m in self.metrics if m.end_time > 0]
            
            if not completed_metrics:
                return {}
            
            operations = {}
            for metrics in completed_metrics:
                op_name = metrics.operation_name
                if op_name not in operations:
                    operations[op_name] = {
                        'count': 0,
                        'total_duration': 0.0,
                        'avg_duration': 0.0,
                        'success_rate': 0.0,
                        'avg_throughput': 0.0
                    }
                
                op_stats = operations[op_name]
                op_stats['count'] += 1
                op_stats['total_duration'] += metrics.duration
                
                if metrics.throughput:
                    op_stats['avg_throughput'] = (
                        (op_stats['avg_throughput'] * (op_stats['count'] - 1) + metrics.throughput) 
                        / op_stats['count']
                    )
            
            # Calculate averages and success rates
            for op_name, stats in operations.items():
                stats['avg_duration'] = stats['total_duration'] / stats['count']
                successful = sum(1 for m in completed_metrics 
                               if m.operation_name == op_name and m.success)
                stats['success_rate'] = successful / stats['count']
            
            return {
                'total_operations': len(completed_metrics),
                'operations': operations,
                'monitoring_duration': max(m.end_time for m in completed_metrics) - min(m.start_time for m in completed_metrics)
            }
    
    def save_metrics(self, filename: Optional[str] = None) -> None:
        """Save metrics to file."""
        if filename is None:
            filename = self.config.metrics_file
        
        if not filename:
            return
        
        with self._lock:
            metrics_data = {
                'config': {
                    'prefer_gpu': self.config.prefer_gpu,
                    'gpu_device_ids': self.config.gpu_device_ids,
                    'gpu_memory_fraction': self.config.gpu_memory_fraction
                },
                'summary': self.get_metrics_summary(),
                'detailed_metrics': [m.to_dict() for m in self.metrics]
            }
        
        try:
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            with open(filename, 'w') as f:
                json.dump(metrics_data, f, indent=2)
            logger.info(f"GPU performance metrics saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")
    
    def clear_metrics(self) -> None:
        """Clear all collected metrics."""
        with self._lock:
            self.metrics.clear()


# Global configuration and monitor instances
_global_config: Optional[GPUConfig] = None
_global_monitor: Optional[GPUPerformanceMonitor] = None


def get_gpu_config() -> GPUConfig:
    """Get global GPU configuration."""
    global _global_config
    if _global_config is None:
        _global_config = GPUConfig()
    return _global_config


def set_gpu_config(config: GPUConfig) -> None:
    """Set global GPU configuration."""
    global _global_config, _global_monitor
    _global_config = config
    
    # Reinitialize monitor with new config
    if config.enable_performance_monitoring:
        _global_monitor = GPUPerformanceMonitor(config)
    else:
        _global_monitor = None


def get_gpu_monitor() -> Optional[GPUPerformanceMonitor]:
    """Get global GPU performance monitor."""
    global _global_monitor
    if _global_monitor is None:
        config = get_gpu_config()
        if config.enable_performance_monitoring:
            _global_monitor = GPUPerformanceMonitor(config)
    return _global_monitor


class gpu_operation:
    """Context manager for monitoring GPU operations."""
    
    def __init__(self, operation_name: str, gpu_id: int = 0, data_size: Optional[int] = None):
        """Initialize GPU operation monitor."""
        self.operation_name = operation_name
        self.gpu_id = gpu_id
        self.data_size = data_size
        self.operation_id = ""
        self.monitor = get_gpu_monitor()
    
    def __enter__(self) -> 'gpu_operation':
        """Start monitoring the operation."""
        if self.monitor:
            self.operation_id = self.monitor.start_operation(
                self.operation_name, self.gpu_id, self.data_size
            )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """End monitoring the operation."""
        if self.monitor and self.operation_id:
            success = exc_type is None
            error_message = str(exc_val) if exc_val else None
            self.monitor.end_operation(self.operation_id, success, error_message)


def configure_gpu(
    prefer_gpu: bool = True,
    fallback_to_cpu: bool = True,
    gpu_device_ids: Optional[List[int]] = None,
    gpu_memory_fraction: float = 0.8,
    enable_monitoring: bool = True,
    metrics_file: Optional[str] = None
) -> None:
    """
    Configure GPU settings for Lotus.
    
    Args:
        prefer_gpu: Whether to prefer GPU acceleration when available
        fallback_to_cpu: Whether to fallback to CPU if GPU fails
        gpu_device_ids: List of GPU device IDs to use
        gpu_memory_fraction: Fraction of GPU memory to use
        enable_monitoring: Whether to enable performance monitoring
        metrics_file: File to save performance metrics
    """
    config = GPUConfig(
        prefer_gpu=prefer_gpu,
        fallback_to_cpu=fallback_to_cpu,
        gpu_device_ids=gpu_device_ids or [0],
        gpu_memory_fraction=gpu_memory_fraction,
        enable_performance_monitoring=enable_monitoring,
        metrics_file=metrics_file
    )
    
    set_gpu_config(config)
    logger.info(f"GPU configuration updated: prefer_gpu={prefer_gpu}, devices={config.gpu_device_ids}")
