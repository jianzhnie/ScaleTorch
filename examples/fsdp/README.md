# MNIST Example

## Single GPU

```bash
python examples/mnist/basic_mnist.py
```

## Multi-processing Distributed Data Parallel Training

You should always use the NCCL backend for multi-processing distributed training since it currently provides the best distributed training performance.

### Single node, multiple GPUs:

```bash
python examples/mnist/multigpu_mnist.py
```

## Torchrun

```bash
torchrun  --nnodes=1 --nproc_per_node=2 --rdzv_id=100 --rdzv_backend=c10d examples/mnist/torchrun_mnist.py
```
