#!/usr/bin/env python3
"""
Benchmark script for comparing different training configurations.

This script runs multiple benchmark tests with different configurations
and generates a comparison report.
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from typing import Any, Dict, List

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Base configuration path
BASE_CONFIG = 'template/base_config.json'

# Load base configuration
with open(BASE_CONFIG, 'r') as f:
    BASE_CONFIG_DATA = json.load(f)

# Benchmark configurations
BENCHMARK_CONFIGS = [{
    'name': 'baseline',
    'config': {
        'training': {
            'batch_size': 2,
            'micro_batch_size': 1,
            'gradient_accumulation_steps': 2,
            'max_steps': 100
        },
        'model': {
            'hidden_size': 2048,
            'num_layers': 12,
            'num_attention_heads': 16,
            'intermediate_size': 8192,
            'max_sequence_length': 1024
        },
        'parallel': {
            'tensor_parallel_size': 1,
            'pipeline_parallel_size': 1,
            'context_parallel_size': 1
        },
        'optimization': {
            'gradient_checkpointing': False,
            'bfloat16': False
        },
        'logging': {
            'log_interval': 10,
            'use_wandb': False
        }
    }
}, {
    'name': 'gradient_checkpointing',
    'config': {
        'training': {
            'batch_size': 2,
            'micro_batch_size': 1,
            'gradient_accumulation_steps': 2,
            'max_steps': 100
        },
        'model': {
            'hidden_size': 2048,
            'num_layers': 12,
            'num_attention_heads': 16,
            'intermediate_size': 8192,
            'max_sequence_length': 1024
        },
        'parallel': {
            'tensor_parallel_size': 1,
            'pipeline_parallel_size': 1,
            'context_parallel_size': 1
        },
        'optimization': {
            'gradient_checkpointing': True,
            'bfloat16': False
        },
        'logging': {
            'log_interval': 10,
            'use_wandb': False
        }
    }
}, {
    'name': 'bfloat16',
    'config': {
        'training': {
            'batch_size': 2,
            'micro_batch_size': 1,
            'gradient_accumulation_steps': 2,
            'max_steps': 100
        },
        'model': {
            'hidden_size': 2048,
            'num_layers': 12,
            'num_attention_heads': 16,
            'intermediate_size': 8192,
            'max_sequence_length': 1024
        },
        'parallel': {
            'tensor_parallel_size': 1,
            'pipeline_parallel_size': 1,
            'context_parallel_size': 1
        },
        'optimization': {
            'gradient_checkpointing': False,
            'bfloat16': True
        },
        'logging': {
            'log_interval': 10,
            'use_wandb': False
        }
    }
}, {
    'name': 'full_optimization',
    'config': {
        'training': {
            'batch_size': 2,
            'micro_batch_size': 1,
            'gradient_accumulation_steps': 2,
            'max_steps': 100
        },
        'model': {
            'hidden_size': 2048,
            'num_layers': 12,
            'num_attention_heads': 16,
            'intermediate_size': 8192,
            'max_sequence_length': 1024
        },
        'parallel': {
            'tensor_parallel_size': 1,
            'pipeline_parallel_size': 1,
            'context_parallel_size': 1
        },
        'optimization': {
            'gradient_checkpointing': True,
            'bfloat16': True
        },
        'logging': {
            'log_interval': 10,
            'use_wandb': False
        }
    }
}]


def run_benchmark(config_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run a benchmark with the given configuration.

    Args:
        config_name: Name of the configuration
        config: Configuration dictionary

    Returns:
        Dictionary with benchmark results
    """
    print(f"\n{'='*60}")
    print(f'Running benchmark: {config_name}')
    print(f"{'='*60}")

    # Create a temporary config file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    config_file = f'temp_config_{config_name}_{timestamp}.json'
    log_file = f'benchmark_{config_name}_{timestamp}.log'

    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

    try:
        # Merge base configuration with benchmark specific configuration
        merged_config = BASE_CONFIG_DATA.copy()

        # Update with benchmark specific configuration
        for section, section_config in config.items():
            if section in merged_config:
                merged_config[section].update(section_config)
            else:
                merged_config[section] = section_config

        # Write merged configuration to temporary file
        with open(config_file, 'w') as f:
            json.dump(merged_config, f, indent=2)

        # Run the training script
        command = ['python', 'train.py', '--config', config_file]

        start_time = time.time()

        with open(log_file, 'w') as f:
            result = subprocess.run(
                command,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=3600  # 1 hour timeout
            )

        end_time = time.time()

        # Check if the run was successful
        if result.returncode != 0:
            print(
                f'Benchmark {config_name} failed with return code {result.returncode}'
            )
            return {
                'name': config_name,
                'success': False,
                'error': f'Return code {result.returncode}',
                'runtime': end_time - start_time
            }

        # Parse performance logs
        performance_logs = []
        for filename in os.listdir('.'):
            if filename.startswith(f'performance_logs_0_{timestamp}'
                                   ) and filename.endswith('.json'):
                with open(filename, 'r') as f:
                    performance_logs = json.load(f)
                os.remove(filename)
                break

        # Calculate average performance metrics
        if performance_logs:
            avg_tokens_per_sec = sum(
                log['tokens_per_sec']
                for log in performance_logs) / len(performance_logs)
            avg_iter_time = sum(
                log['iter_time']
                for log in performance_logs) / len(performance_logs)
            avg_loss = sum(log.get('loss', 0)
                           for log in performance_logs) / len(performance_logs)

            # Get peak memory usage
            peak_memory = max(
                log.get('peak_cuda_memory_mb', 0) for log in performance_logs)

        else:
            avg_tokens_per_sec = 0
            avg_iter_time = 0
            avg_loss = 0
            peak_memory = 0

        return {
            'name': config_name,
            'success': True,
            'runtime': end_time - start_time,
            'avg_tokens_per_sec': avg_tokens_per_sec,
            'avg_iter_time': avg_iter_time,
            'avg_loss': avg_loss,
            'peak_memory_mb': peak_memory,
            'config': config,
            'timestamp': timestamp
        }

    except subprocess.TimeoutExpired:
        print(f'Benchmark {config_name} timed out')
        return {
            'name': config_name,
            'success': False,
            'error': 'Timeout',
            'runtime': 3600
        }
    except Exception as e:
        print(f'Benchmark {config_name} failed with exception: {e}')
        return {
            'name': config_name,
            'success': False,
            'error': str(e),
            'runtime': end_time - start_time if 'end_time' in locals() else 0
        }
    finally:
        # Clean up
        if os.path.exists(config_file):
            os.remove(config_file)
        if os.path.exists(log_file):
            # Keep log files for debugging
            print(f'Log file saved as {log_file}')


def generate_report(results: List[Dict[str, Any]]) -> str:
    """
    Generate a benchmark report.

    Args:
        results: List of benchmark results

    Returns:
        Formatted report string
    """
    report = []
    report.append('\n' + '=' * 80)
    report.append('SCALETORCH BENCHMARK REPORT')
    report.append('=' * 80)
    report.append(f'Generated on: {datetime.now().isoformat()}')
    report.append(f'Number of configurations: {len(results)}')
    report.append('\n' + '=' * 80)

    # Table header
    report.append('\n' + 'CONFIGURATION RESULTS')
    report.append('-' * 80)
    report.append('{:<20} {:<10} {:<15} {:<15} {:<15} {:<15}'.format(
        'Configuration', 'Status', 'Avg Tokens/s', 'Avg Iter Time', 'Avg Loss',
        'Peak Memory (MB)'))
    report.append('-' * 80)

    # Table rows
    for result in results:
        status = 'SUCCESS' if result['success'] else 'FAILED'
        report.append(
            '{:<20} {:<10} {:<15.2f} {:<15.4f} {:<15.6f} {:<15.0f}'.format(
                result['name'], status, result.get('avg_tokens_per_sec', 0),
                result.get('avg_iter_time', 0), result.get('avg_loss', 0),
                result.get('peak_memory_mb', 0)))

    # Performance comparison
    report.append('\n' + '=' * 80)
    report.append('PERFORMANCE COMPARISON')
    report.append('-' * 80)

    # Find baseline
    baseline = None
    for result in results:
        if result['name'] == 'baseline' and result['success']:
            baseline = result
            break

    if baseline:
        for result in results:
            if result['success'] and result['name'] != 'baseline':
                speedup = result['avg_tokens_per_sec'] / baseline[
                    'avg_tokens_per_sec'] if baseline[
                        'avg_tokens_per_sec'] > 0 else 0
                memory_ratio = result['peak_memory_mb'] / baseline[
                    'peak_memory_mb'] if baseline['peak_memory_mb'] > 0 else 0
                report.append(f"{result['name']}:")
                report.append(f'  Speedup vs baseline: {speedup:.2f}x')
                report.append(
                    f'  Memory usage vs baseline: {memory_ratio:.2f}x')
                report.append(
                    f'  Peak memory reduction: {100 - (memory_ratio * 100):.1f}%'
                )
                report.append('')

    report.append('=' * 80)
    return '\n'.join(report)


def main():
    """Main benchmark function."""
    print('Starting ScaleTorch benchmark...')
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Run all benchmarks
    results = []
    for config in BENCHMARK_CONFIGS:
        result = run_benchmark(config['name'], config['config'])
        results.append(result)

    # Generate report
    report = generate_report(results)
    print(report)

    # Save report
    report_file = f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_file, 'w') as f:
        f.write(report)

    # Save raw results
    results_file = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f'\nReport saved as {report_file}')
    print(f'Raw results saved as {results_file}')

    print('\nBenchmark completed!')


if __name__ == '__main__':
    main()
