"""Parallelism strategies for distributed training.

Modules:
    - data_parallel: BasicDataParallel and DataParallelBucket (bucketed all-reduce)
    - tensor_parallel: ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding
    - pipeline_parallel: PipelineParallel with AFAB and 1F1B scheduling
    - context_parallel: Ring attention with causal masking
    - sequence_parallel: All-gather / reduce-scatter for sequence parallelism
    - expert_parallel: Expert parallelism communication primitives
    - process_group: ProcessGroupManager for 5D parallelism
"""

from scaletorch.parallel.data_parallel.data_parallel import (
    BasicDataParallel,
    DataParallelBase,
    DataParallelBucket,
)
from scaletorch.parallel.pipeline_parallel.pipeline_parallel import (
    PipelineParallel,
    train_step_pipeline_1f1b,
    train_step_pipeline_afab,
)
from scaletorch.parallel.process_group import (
    ProcessGroupManager,
    get_process_group_manager,
    process_group_manager,
    setup_process_group_manager,
)
from scaletorch.parallel.tensor_parallel.tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
    apply_tensor_parallel,
)

__all__ = [
    "BasicDataParallel",
    "ColumnParallelLinear",
    "DataParallelBase",
    "DataParallelBucket",
    "PipelineParallel",
    "ProcessGroupManager",
    "RowParallelLinear",
    "VocabParallelEmbedding",
    "apply_tensor_parallel",
    "get_process_group_manager",
    "process_group_manager",
    "setup_process_group_manager",
    "train_step_pipeline_1f1b",
    "train_step_pipeline_afab",
]
