#!/bin/bash

# Manual test script for create_config.py
# This script runs several test cases to verify the functionality of the configuration creation utility

set -e  # Exit on any error

echo "=== Manual Test Script for ScaleTorch Configuration Creation ==="
echo

# Test 1: Basic configuration creation
echo "Test 1: Creating basic configuration..."
python -m scaletorch.utils.create_config \
    --data_parallel_size 1 \
    --tensor_parallel_size 1 \
    --pipeline_parallel_size 1 \
    --context_parallel_size 1 \
    --pipeline_parallel_engine 1f1b \
    --model_name_or_path hf-internal-testing/tiny-random-gpt2 \
    --grad_acc_steps 1 \
    --micro_batch_size 1 \
    --sequence_length 128 \
    --experiment_name basic_test \
    --template_dir template \
    --output_dir ./work_dir/hf_models/

echo "✓ Basic configuration created successfully"
echo

# Verify that the config files were created
echo "Verifying created configurations:"
for config in work_dir/hf_models/*/config.json; do
    if [ -f "$config" ]; then
        echo "✓ $config exists"
        # Show a snippet of the config file
        echo "  Sample content:"
        head -n 10 "$config" | sed 's/^/    /'
        echo
    fi
done

echo
echo "Test outputs are stored in ./test_outputs/"
echo "You can inspect individual configuration files in their respective experiment directories."
