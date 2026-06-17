#!/bin/bash
# Single GPU
python examples/imagenet/dist_train.py -a resnet18 --dummy

# Single node, multiple GPUs (mp.spawn):
# python examples/imagenet/dist_train.py -a resnet18 --dummy \
#     --dist-url 'tcp://127.0.0.1:8080' \
#     --dist-backend 'nccl' \
#     --multiprocessing-distributed \
#     --batch-size 32 \
#     --world-size 1 \
#     --rank 0

# Single node, multiple GPUs (torchrun):
# torchrun examples/imagenet/dist_train.py -a resnet18 --dummy \
#     --dist-url 'tcp://127.0.0.1:29501' \
#     --dist-backend 'nccl' \
#     --batch-size 128
