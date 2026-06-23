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
