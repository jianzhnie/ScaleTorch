# distributed-pytorch

Code for the DDP tutorial series at https://pytorch.org/tutorials/beginner/ddp_series_intro.html

Each code file extends upon the previous one. The series starts with a non-distributed script that runs on a single GPU and incrementally updates to end with multinode training on a Slurm cluster.

## Files

- [single_gpu.py](single_gpu.py): Non-distributed training script

- [multigpu.py](multigpu.py): DDP on a single node

- [multigpu_torchrun.py](multigpu_torchrun.py): DDP on a single node using Torchrun

- [multinode.py](multinode.py): DDP on multiple nodes using Torchrun
