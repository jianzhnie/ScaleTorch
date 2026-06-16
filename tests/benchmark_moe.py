#!/usr/bin/env python3
"""
Benchmark script for the MoE model implementation.
Measures performance under various configurations.
"""

import time
import torch
from typing import Dict, List

from scaletorch.models.moe_model import GPT, GPTConfig


def benchmark_configuration(config_kwargs: Dict, batch_size: int = 2, seq_len: int = 32, 
                          num_iterations: int = 10, warmup_iterations: int = 3) -> Dict:
    """
    Benchmark a specific configuration.
    
    Args:
        config_kwargs: Configuration parameters for GPTConfig
        batch_size: Batch size for benchmarking
        seq_len: Sequence length for benchmarking
        num_iterations: Number of iterations to measure
        warmup_iterations: Number of warmup iterations
    
    Returns:
        Dictionary with benchmark results
    """
    # Create model
    config = GPTConfig(**config_kwargs)
    model = GPT(config)
    model.eval()  # Set to evaluation mode for consistent timing
    
    # Create dummy input
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    idx = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(device)
    
    # Warmup iterations
    for _ in range(warmup_iterations):
        with torch.no_grad():
            _, _ = model(idx)
    
    # Benchmark iterations
    times = []
    for _ in range(num_iterations):
        start_time = time.time()
        with torch.no_grad():
            _, _ = model(idx)
        end_time = time.time()
        times.append(end_time - start_time)
    
    # Calculate statistics
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    # Calculate throughput
    tokens_per_batch = batch_size * seq_len
    avg_throughput = tokens_per_batch / avg_time
    
    return {
        'avg_time': avg_time,
        'min_time': min_time,
        'max_time': max_time,
        'avg_throughput': avg_throughput,
        'total_params': model.get_num_params()
    }


def run_benchmarks() -> None:
    """Run a series of benchmarks comparing different configurations."""
    base_config = {
        'n_embd': 256,
        'n_layer': 4,
        'n_head': 4,
        'block_size': 128,
        'vocab_size': 512,
    }
    
    configs = [
        {
            'name': 'Standard GPT',
            'config': {**base_config, 'use_moe': False}
        },
        {
            'name': 'MoE (2 experts)',
            'config': {**base_config, 'use_moe': True, 'n_experts': 2}
        },
        {
            'name': 'MoE (4 experts)',
            'config': {**base_config, 'use_moe': True, 'n_experts': 4}
        },
        {
            'name': 'MoE (8 experts)',
            'config': {**base_config, 'use_moe': True, 'n_experts': 8}
        },
        {
            'name': 'MoE (4 experts, top-1)',
            'config': {**base_config, 'use_moe': True, 'n_experts': 4, 'top_k': 1}
        },
        {
            'name': 'MoE (4 experts, top-4)',
            'config': {**base_config, 'use_moe': True, 'n_experts': 4, 'top_k': 4}
        }
    ]
    
    print("Running MoE Model Benchmarks")
    print("=" * 50)
    print(f"{'Configuration':<25} {'Avg Time (s)':<15} {'Throughput (tok/s)':<20} {'Params (M)':<15}")
    print("-" * 75)
    
    results = []
    for config_info in configs:
        name = config_info['name']
        config_kwargs = config_info['config']
        
        try:
            result = benchmark_configuration(config_kwargs)
            results.append((name, result))
            
            print(f"{name:<25} {result['avg_time']:<15.4f} {result['avg_throughput']:<20.2f} "
                  f"{result['total_params']/1e6:<15.2f}")
        except Exception as e:
            print(f"{name:<25} Failed with error: {str(e)}")
    
    # Summary
    print("\nSummary:")
    if results:
        # Find fastest configuration
        fastest = min(results, key=lambda x: x[1]['avg_time'])
        print(f"Fastest configuration: {fastest[0]} ({fastest[1]['avg_time']:.4f}s)")
        
        # Find most memory efficient (lowest parameter count)
        smallest = min(results, key=lambda x: x[1]['total_params'])
        print(f"Smallest model: {smallest[0]} ({smallest[1]['total_params']/1e6:.2f}M parameters)")


if __name__ == "__main__":
    run_benchmarks()