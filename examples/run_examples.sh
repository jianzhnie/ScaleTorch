# example1
python examples/basicddp/demo_ddp.py   .py
torchrun  --nnodes=1 --nproc_per_node=2  examples/basicddp/elastic_ddp.py

# example2
torchrun  --nnodes=1 --nproc_per_node=2 mingpt/main.py

# example3
python examples/mnist/multigpu_mnist.py
torchrun  --nnodes=1 --nproc_per_node=2  examples/mnist/torchrun_mnist.py
