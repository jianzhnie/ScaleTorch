"""Model implementations with parallelism support.

Models:
    - Llama: Transformer decoder with flash attention, TP, CP, and RoPE
    - Qwen3: Qwen3 transformer with QK norms and explicit head_dim
    - Qwen3MoE: Qwen3 Mixture-of-Experts with Expert Parallelism
    - GPT (MoE): GPT-style transformer with MoE support
    - LeNet: Simple CNN for educational purposes
"""
