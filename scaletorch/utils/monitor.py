import json
import os
import time
from datetime import datetime
from typing import Any, Dict, Optional

import psutil
import torch

# Try to import pynvml for advanced GPU metrics (optional)
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False


class PerformanceMonitor:
    """
    Performance monitoring utility for tracking training metrics.
    """

    def __init__(self, config: Dict[str, Any], log_dir: Optional[str] = None):
        """
        Initialize performance monitor.

        Args:
            config: Training configuration
            log_dir: Directory to save performance logs
        """
        self.config = config
        self.log_dir = log_dir
        self.start_time = None
        self.iteration_start_time = None
        self.iteration_tokens_processed = 0
        self.stats = []

        # Create log directory if specified
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

    def start(self):
        """Start monitoring performance."""
        self.start_time = time.time()
        print(
            f"Performance monitoring started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

    def start_iteration(self, tokens_processed: int = 0):
        """Start tracking an iteration.

        Args:
            tokens_processed: Number of tokens processed in this iteration
        """
        self.iteration_start_time = time.time()
        self.iteration_tokens_processed = tokens_processed

    def end_iteration(self,
                      tokens_processed: Optional[int] = None
                      ) -> Dict[str, Any]:
        """End tracking an iteration and record statistics.

        Args:
            tokens_processed: Number of tokens processed in this iteration

        Returns:
            Dictionary containing iteration statistics
        """
        if self.iteration_start_time is None:
            raise RuntimeError(
                'start_iteration() must be called before end_iteration()')

        if tokens_processed is not None:
            self.iteration_tokens_processed = tokens_processed

        iteration_time = time.time() - self.iteration_start_time
        total_time = time.time(
        ) - self.start_time if self.start_time else iteration_time

        # Calculate throughput
        tokens_per_second = self.iteration_tokens_processed / iteration_time if iteration_time > 0 else 0

        # Collect GPU statistics with more detailed metrics
        gpu_stats = {}
        if torch.cuda.is_available():
            gpu_stats['gpu_utilization'] = torch.cuda.utilization()
            gpu_stats['gpu_memory_allocated'] = torch.cuda.memory_allocated(
            ) / (1024**2)  # MB
            gpu_stats['gpu_memory_reserved'] = torch.cuda.memory_reserved() / (
                1024**2)  # MB
            gpu_stats['gpu_memory_free'] = (
                torch.cuda.get_device_properties(0).total_memory -
                torch.cuda.memory_reserved()) / (1024**2)  # MB

            # Get detailed memory stats
            memory_stats = torch.cuda.memory_stats()
            gpu_stats['gpu_memory_fragmentation'] = memory_stats.get(
                'fragmentation.peak', 0)
            gpu_stats['gpu_memory_active'] = memory_stats.get(
                'active_bytes.all.current', 0) / (1024**2)  # MB
            gpu_stats['gpu_memory_inactive'] = memory_stats.get(
                'inactive_split_bytes.all.current', 0) / (1024**2)  # MB

            # Get GPU temperature and power if available (requires pynvml)
            if PYNVML_AVAILABLE:
                try:
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    gpu_stats[
                        'gpu_temperature'] = pynvml.nvmlDeviceGetTemperature(
                            handle, pynvml.NVML_TEMPERATURE_GPU)
                    gpu_stats[
                        'gpu_power_usage'] = pynvml.nvmlDeviceGetPowerUsage(
                            handle) / 1000.0  # Convert mW to W
                except Exception:
                    # Error accessing GPU metrics, skip these
                    pass

        # Collect CPU statistics
        cpu_stats = {
            'cpu_utilization': psutil.cpu_percent(),
            'cpu_memory_used': psutil.virtual_memory().used / (1024**2),  # MB
            'cpu_memory_available':
            psutil.virtual_memory().available / (1024**2),  # MB
        }

        # Collect memory statistics
        memory_stats = {}
        if torch.cuda.is_available():
            memory_stats['peak_memory'] = torch.cuda.max_memory_allocated() / (
                1024**2)  # MB
            torch.cuda.reset_peak_memory_stats()

        # Create iteration statistics
        iteration_stats = {
            'iteration':
            len(self.stats) + 1,
            'tokens_processed':
            self.iteration_tokens_processed,
            'iteration_time':
            iteration_time,
            'total_time':
            total_time,
            'tokens_per_second':
            tokens_per_second,
            'iterations_per_second':
            1.0 / iteration_time if iteration_time > 0 else 0,
            'gpu':
            gpu_stats,
            'cpu':
            cpu_stats,
            'memory':
            memory_stats,
            'timestamp':
            datetime.now().isoformat(),
        }

        self.stats.append(iteration_stats)
        return iteration_stats

    def get_average_stats(self) -> Dict[str, Any]:
        """Get average performance statistics.

        Returns:
            Dictionary containing average statistics
        """
        if not self.stats:
            return {}

        avg_stats = {
            'average_iteration_time':
            sum(s['iteration_time'] for s in self.stats) / len(self.stats),
            'average_tokens_per_second':
            sum(s['tokens_per_second'] for s in self.stats) / len(self.stats),
            'average_iterations_per_second':
            sum(s['iterations_per_second']
                for s in self.stats) / len(self.stats),
            'total_tokens_processed':
            sum(s['tokens_processed'] for s in self.stats),
            'total_time':
            self.stats[-1]['total_time'],
            'total_iterations':
            len(self.stats),
        }

        # Average GPU stats if available
        if torch.cuda.is_available() and 'gpu' in self.stats[0]:
            avg_stats['gpu'] = {
                'avg_gpu_utilization':
                sum(s['gpu']['gpu_utilization']
                    for s in self.stats) / len(self.stats),
                'avg_gpu_memory_allocated':
                sum(s['gpu']['gpu_memory_allocated']
                    for s in self.stats) / len(self.stats),
            }

        return avg_stats

    def _log_stats(self, stats: Dict[str, Any]) -> None:
        """Log performance statistics to stdout."""
        log_str = (f"Iteration {stats['iteration']}: "
                   f"Time={stats['iteration_time']:.4f}s, "
                   f"Tokens={stats['tokens_processed']}, "
                   f"Throughput={stats['tokens_per_second']:.2f} tokens/s")

        # Add GPU stats if available
        if 'gpu' in stats and stats['gpu']:
            log_str += (
                f", GPU Util={stats['gpu'].get('gpu_utilization', 0):.1f}%, "
                f"GPU Mem={stats['gpu'].get('gpu_memory_allocated', 0):.1f}MB")

        print(log_str)

    def log_iteration_stats(self, stats: Dict[str, Any]) -> None:
        """Log iteration statistics.

        Args:
            stats: Iteration statistics dictionary
        """
        self._log_stats(stats)

    def save_stats(self, filename: Optional[str] = None):
        """Save collected statistics to a JSON file.

        Args:
            filename: Custom filename for the statistics file
        """
        if not self.stats:
            print('No statistics to save.')
            return

        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'performance_stats_{timestamp}.json'

        # Determine full path
        if self.log_dir:
            filepath = os.path.join(self.log_dir, filename)
        else:
            filepath = filename

        # Create output dictionary with config and stats
        output = {
            'config': self.config,
            'start_time': self.start_time,
            'end_time': time.time(),
            'total_time':
            time.time() - self.start_time if self.start_time else 0,
            'stats': self.stats,
            'average_stats': self.get_average_stats()
        }

        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)

        print(f'Performance statistics saved to {filepath}')
        return filepath

    def print_summary(self):
        """Print a summary of performance statistics."""
        if not self.stats:
            print('No performance statistics available.')
            return

        avg_stats = self.get_average_stats()

        print('\n=== PERFORMANCE SUMMARY ===')
        print(f"Total Iterations: {avg_stats['total_iterations']}")
        print(f"Total Tokens Processed: {avg_stats['total_tokens_processed']}")
        print(f"Total Time: {avg_stats['total_time']:.2f} seconds")
        print(
            f"Average Iteration Time: {avg_stats['average_iteration_time']:.4f} seconds"
        )
        print(
            f"Average Throughput: {avg_stats['average_tokens_per_second']:.2f} tokens/second"
        )
        print(
            f"Average Iterations per Second: {avg_stats['average_iterations_per_second']:.2f} it/s"
        )

        # Print GPU summary if available
        if 'gpu' in avg_stats:
            print(
                f"Average GPU Utilization: {avg_stats['gpu']['avg_gpu_utilization']:.1f}%"
            )
            print(
                f"Average GPU Memory Allocated: {avg_stats['gpu']['avg_gpu_memory_allocated']:.1f} MB"
            )

        print('==========================')
