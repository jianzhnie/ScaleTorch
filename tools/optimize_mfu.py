"""Test multiple optimization strategies for Qwen3 MFU."""

import gc
import os
import time

os.environ["FLASH_ATTEN"] = "0"
os.environ["DTYPE"] = "bfloat16"

import torch
from torch.amp import autocast
from transformers import AutoConfig

from scaletorch.models.model_qwen3 import Qwen3
from scaletorch.utils.checkpoint import (
    init_model_with_dematerialized_weights,
    init_model_with_materialized_weights,
)

MODEL = os.environ.get("MODEL_PATH", "/workspace/models/qwen3")
SEQ = 2048
WARMUP, STEPS = 3, 15

torch.npu.set_device(0)
cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)

L = cfg.num_hidden_layers
n_h = cfg.num_attention_heads
hd = getattr(cfg, "head_dim", cfg.hidden_size // n_h)
N = sum(p.numel() for p in Qwen3(cfg).parameters()) if False else 596_000_000
train_flops_per_tok = 6 * N + 12 * L * n_h * hd * SEQ
dev_name = torch.npu.get_device_name(0)
peak = 320e12 if "910B" in dev_name else 256e12
print(f"Device: {dev_name}, Peak: {peak / 1e12:.0f} TFLOPS")


def build_model():
    with init_model_with_dematerialized_weights():
        m = Qwen3(config=cfg)
    m = init_model_with_materialized_weights(m, cfg, save_dir=MODEL + "/")
    return m.to(torch.bfloat16).to("npu:0")


def run(label, bs, gc_on, compile_model=False):
    model = build_model()
    model.train()
    if compile_model:
        try:
            model = torch.compile(model, backend="inductor")
            print("  torch.compile enabled")
        except Exception as e:
            print(f"  torch.compile failed: {e}")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=True)

    def step():
        x = torch.randint(0, cfg.vocab_size, (bs, SEQ), device="npu:0")
        with autocast(device_type="npu", dtype=torch.bfloat16):
            out = model(x, gradient_checkpointing=gc_on)
            loss = torch.nn.functional.cross_entropy(
                out[:, :-1].reshape(-1, out.size(-1)), x[:, 1:].reshape(-1)
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)

    for _ in range(WARMUP):
        step()
    torch.npu.synchronize()
    torch.npu.reset_peak_memory_stats()

    start = time.time()
    for _ in range(STEPS):
        step()
    torch.npu.synchronize()
    elapsed = time.time() - start

    tps = bs * SEQ * STEPS / elapsed
    achieved_flops = tps * train_flops_per_tok
    mfu = achieved_flops / peak * 100
    mem = torch.npu.max_memory_allocated() / 1e9
    tflops = achieved_flops / 1e12

    print(f"[{label}] BS={bs} GC={gc_on}")
    print(
        f"  {elapsed / STEPS * 1000:.0f}ms/step | {tps:.0f} tok/s | {tflops:.1f} TFLOPS | MFU={mfu:.1f}% | Mem={mem:.1f}GB"
    )
    del model, opt
    gc.collect()
    torch.npu.empty_cache()
    return mfu


results = []
results.append(("A: BS=4 baseline", run("A", 4, False)))
results.append(("B: BS=4 GC", run("B", 4, True)))
results.append(("C: BS=8 GC", run("C", 8, True)))
results.append(("D: BS=5 no-GC", run("D", 5, False)))

print("\n" + "=" * 50)
print("Summary:")
for name, mfu in results:
    print(f"  {name}: MFU={mfu:.1f}%")
