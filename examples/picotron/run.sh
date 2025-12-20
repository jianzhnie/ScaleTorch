#!/bin/bash

torchrun  --nnodes=1 --nproc_per_node=8 \
    examples/picotron/train.py \
    --model_name_or_path /home/jianzhnie/llmtuner/hfhub/models/facebook/opt-125m
