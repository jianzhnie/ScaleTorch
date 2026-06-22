"""Benchmark script for ScaleTorch on NPU - measures speed and memory."""

import gc
import os
import time

import torch
from torch.amp import autocast
from transformers import AutoConfig

os.environ.setdefault("FLASH_ATTEN", "0")
os.environ.setdefault("DTYPE", "bfloat16")

from scaletorch.models.llama import Llama
from scaletorch.models.model_qwen3 import Qwen3
from scaletorch.utils.checkpoint import (
    init_model_with_dematerialized_weights,
    init_model_with_materialized_weights,
)


def _create_model(config):
    """Auto-select model class based on config."""
    model_type = getattr(config, "model_type", "llama")
    if model_type in ("qwen3",):
        return Qwen3(config=config)
    return Llama(config=config)


def benchmark(
    model_path="/workspace/models/qwen3",
    batch_size=4,
    seq_len=2048,
    steps=10,
    warmup=3,
    gradient_checkpointing=False,
    use_fused_adam=False,
    device_id=0,
):
    device = f"npu:{device_id}"
    torch.npu.set_device(device_id)
    torch.npu.empty_cache()
    gc.collect()

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    with init_model_with_dematerialized_weights():
        model = _create_model(config)
    model = init_model_with_materialized_weights(
        model, config, save_dir=model_path + "/"
    )
    model = model.to(torch.bfloat16).to(device)
    model.train()

    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model params: {num_params:.1f}M")

    opt_kwargs = {}
    if use_fused_adam:
        opt_kwargs["fused"] = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, **opt_kwargs)

    def train_step():
        x = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
        with autocast(device_type="npu", dtype=torch.bfloat16):
            out = model(x, gradient_checkpointing=gradient_checkpointing)
            loss = torch.nn.functional.cross_entropy(
                out[:, :-1].reshape(-1, out.size(-1)),
                x[:, 1:].reshape(-1),
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return loss.item()

    # Warmup
    for _ in range(warmup):
        train_step()

    torch.npu.synchronize()
    torch.npu.reset_peak_memory_stats()
    gc.collect()
    torch.npu.empty_cache()

    # Benchmark
    start = time.time()
    losses = []
    for _i in range(steps):
        loss_val = train_step()
        losses.append(loss_val)

    torch.npu.synchronize()
    elapsed = time.time() - start

    peak_mem = torch.npu.max_memory_allocated() / 1e9
    reserved_mem = torch.npu.memory_reserved() / 1e9
    tokens_per_sec = batch_size * seq_len * steps / elapsed
    step_time_ms = elapsed / steps * 1000

    print(
        f"Config: BS={batch_size}, SEQ={seq_len}, GC={gradient_checkpointing}, FUSED={use_fused_adam}"
    )
    print(f"Steps: {steps}, Total time: {elapsed:.2f}s")
    print(f"Step time: {step_time_ms:.1f}ms")
    print(f"Tokens/s: {tokens_per_sec:.0f}")
    print(f"Peak Memory: {peak_mem:.2f} GB")
    print(f"Reserved Memory: {reserved_mem:.2f} GB")
    print(f"Avg Loss: {sum(losses) / len(losses):.4f}")
    print("---")
    return step_time_ms, peak_mem, tokens_per_sec


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/workspace/models/qwen3")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--fused_adam", action="store_true")
    parser.add_argument("--device_id", type=int, default=0)
    args = parser.parse_args()

    benchmark(
        model_path=args.model_path,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        steps=args.steps,
        gradient_checkpointing=args.gradient_checkpointing,
        use_fused_adam=args.fused_adam,
        device_id=args.device_id,
    )
