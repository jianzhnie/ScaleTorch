"""Profile Qwen3 actual MFU and identify optimization opportunities."""

import gc
import os
import time

os.environ.setdefault("FLASH_ATTEN", "0")
os.environ.setdefault("DTYPE", "bfloat16")

import torch
from torch.amp import autocast
from transformers import AutoConfig

from scaletorch.models.model_qwen3 import Qwen3
from scaletorch.utils.checkpoint import (
    init_model_with_dematerialized_weights,
    init_model_with_materialized_weights,
)

MODEL_PATH = "/workspace/models/qwen3"
BS = 8
SEQ = 2048
STEPS = 15
WARMUP = 3

torch.npu.set_device(0)
device = "npu:0"

cfg = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
H = cfg.hidden_size
L = cfg.num_hidden_layers
V = cfg.vocab_size
I = cfg.intermediate_size
n_heads = cfg.num_attention_heads
n_kv = cfg.num_key_value_heads
hd = getattr(cfg, "head_dim", H // n_heads)

print(f"Model: hidden={H}, layers={L}, heads={n_heads}, kv={n_kv}, head_dim={hd}")
print(f"       inter={I}, vocab={V}, tie={getattr(cfg, 'tie_word_embeddings', False)}")

# Parameter count
N_params = (
    V * H  # embedding
    + L * H * n_heads * hd  # q_proj
    + L * H * n_kv * hd  # k_proj
    + L * H * n_kv * hd  # v_proj
    + L * n_heads * hd * H  # o_proj
    + L * H * I  # gate_proj
    + L * H * I  # up_proj
    + L * I * H  # down_proj
    + L * hd  # q_norm
    + L * hd  # k_norm
    + L * H * 2  # layernorm (input + post_attn)
    + H  # final_norm
)
print(f"Params (computed): {N_params / 1e6:.1f}M")

# FLOPs per token (forward pass)
# Linear layers: 2 * M * N per token
linear_flops = (
    2
    * (
        H * n_heads * hd  # q_proj
        + H * n_kv * hd  # k_proj
        + H * n_kv * hd  # v_proj
        + n_heads * hd * H  # o_proj
        + H * I  # gate_proj
        + H * I  # up_proj
        + I * H  # down_proj
    )
    * L
)
# Attention: 2 * n_heads * hd * SEQ per token (QK^T + AV)
attn_flops = 2 * 2 * n_heads * hd * SEQ * L
# Embedding + LM head
embed_flops = 2 * 2 * V * H
fwd_flops = linear_flops + attn_flops + embed_flops
# Training: fwd + bwd ≈ 3x fwd
train_flops = 3 * fwd_flops

print("\nFLOPs/token breakdown:")
print(f"  Linear layers: {linear_flops / 1e9:.2f} GFLOPs")
print(f"  Attention:     {attn_flops / 1e9:.2f} GFLOPs")
print(f"  Embed+LMhead:  {embed_flops / 1e9:.2f} GFLOPs")
print(f"  Forward total: {fwd_flops / 1e9:.2f} GFLOPs")
print(f"  Train (3x):    {train_flops / 1e9:.2f} GFLOPs")

# Device info
dev_name = torch.npu.get_device_name(0)
peak_tflops = 320.0 if "910B" in dev_name or "910b" in dev_name else 256.0
print(f"\nDevice: {dev_name}")
print(f"Peak bf16: {peak_tflops} TFLOPS")

# Build model
with init_model_with_dematerialized_weights():
    model = Qwen3(config=cfg)
model = init_model_with_materialized_weights(model, cfg, save_dir=MODEL_PATH + "/")
model = model.to(torch.bfloat16).to(device)
model.train()
real_params = sum(p.numel() for p in model.parameters())
print(f"Actual params: {real_params / 1e6:.1f}M")

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=True)


def train_step(gc_enabled=True):
    x = torch.randint(0, cfg.vocab_size, (BS, SEQ), device=device)
    with autocast(device_type="npu", dtype=torch.bfloat16):
        out = model(x, gradient_checkpointing=gc_enabled)
        loss = torch.nn.functional.cross_entropy(
            out[:, :-1].reshape(-1, out.size(-1)), x[:, 1:].reshape(-1)
        )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return loss.item()


def benchmark(label, gc_enabled, steps=STEPS):
    for _ in range(WARMUP):
        train_step(gc_enabled)
    torch.npu.synchronize()
    torch.npu.reset_peak_memory_stats()
    gc.collect()
    torch.npu.empty_cache()

    start = time.time()
    for _ in range(steps):
        train_step(gc_enabled)
    torch.npu.synchronize()
    elapsed = time.time() - start

    tps = BS * SEQ * steps / elapsed
    achieved = tps * train_flops / 1e12
    mfu = achieved / peak_tflops * 100
    mem = torch.npu.max_memory_allocated() / 1e9
    print(f"\n[{label}]")
    print(f"  Step time: {elapsed / steps * 1000:.1f}ms")
    print(f"  Tokens/s:  {tps:.0f}")
    print(f"  TFLOPS:    {achieved:.2f}")
    print(f"  MFU:       {mfu:.1f}%")
    print(f"  Peak mem:  {mem:.2f} GB")
    return mfu, tps, mem


print("\n" + "=" * 50)
benchmark("BS=8 GC+fused (current)", gc_enabled=True)
