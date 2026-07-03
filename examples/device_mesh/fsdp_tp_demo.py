"""2-D Parallelism demo: FSDP + Tensor Parallel on Ascend NPU / CUDA.

Combines Fully Sharded Data Parallel (FSDP) with Tensor Parallel (TP)
on a minimal Llama-style transformer via a 2-D DeviceMesh:
  * ``dp`` – FSDP across data-parallel replicas (cross-host)
  * ``tp`` – TP within each host (intra-host, e.g. 8 GPUs/NPUs)

Example for 8 devices:
  dp_size = 4, tp_size = 2 → 4 dp replicas × 2 tp ranks

Reference:
  https://pytorch.org/tutorials/intermediate/TP_tutorial.html
"""

import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed._tensor import Replicate, Shard
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)

# -- detect backend ---------------------------------------------------------
try:
    import torch_npu  # noqa: F401
except ImportError:
    device_mod, backend, device_type = torch.cuda, "nccl", "cuda"
else:
    if torch.npu.is_available():
        device_mod, backend, device_type = torch.npu, "hccl", "npu"
    else:
        sys.exit(
            "[fsdp_tp] torch_npu found but NPU is not available. "
            "Is the CANN toolkit sourced?\n"
        )

# -- bootstrap --------------------------------------------------------------
rank = int(os.environ["RANK"])
local_rank = int(os.environ.get("LOCAL_RANK", os.environ["RANK"]))
world_size = int(os.environ["WORLD_SIZE"])

dist.init_process_group(backend)
device_mod.set_device(local_rank)

num_devices = device_mod.device_count()
print(
    f"[rank={rank}] local_rank={local_rank} world_size={world_size} "
    f"device={device_type} device_count={num_devices}",
    flush=True,
)

# -- 2-D mesh: dp × tp -----------------------------------------------------
tp_size = 2
if world_size % tp_size != 0:
    dist.destroy_process_group()
    sys.exit(
        f"[fsdp_tp] world_size={world_size} not divisible by tp_size={tp_size}.\n"
    )

dp_size = world_size // tp_size

mesh_2d = init_device_mesh(
    device_type,
    mesh_shape=(dp_size, tp_size),
    mesh_dim_names=("dp", "tp"),
)
tp_mesh = mesh_2d["tp"]
dp_mesh = mesh_2d["dp"]

dp_rank = dp_mesh.get_local_rank()

print(
    f"[rank={rank}] FSDP+TP mesh: {dp_size}×{tp_size} "
    f"(dp={dp_mesh} tp={tp_mesh})",
    flush=True,
)

# -- model (minimal Llama-style, self-contained) ---------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class Attention(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        head_dim = dim // n_heads
        self.wq = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wv = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, dim, bias=False)
        self.n_heads = n_heads
        self.head_dim = head_dim

    def forward(self, x):
        B, S, D = x.shape
        q = self.wq(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(B, S, D)
        return self.wo(attn)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, hidden_dim: int):
        super().__init__()
        self.attention_norm = RMSNorm(dim)
        self.attention = Attention(dim, n_heads)
        self.ffn_norm = RMSNorm(dim)
        self.feed_forward = FeedForward(dim, hidden_dim)

    def forward(self, x):
        x = x + self.attention(self.attention_norm(x))
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class ToyTransformer(nn.Module):
    def __init__(self, dim: int, n_layers: int, n_heads: int, vocab_size: int):
        super().__init__()
        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList(
            [TransformerBlock(dim, n_heads, dim * 4) for _ in range(n_layers)]
        )
        self.norm = RMSNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, x):
        x = self.tok_embeddings(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.output(x)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, std=0.02)


model = ToyTransformer(
    dim=256, n_layers=2, n_heads=16, vocab_size=32000,
).to(device_mod.current_device())
model.init_weights()

# -- TP parallelization plan ------------------------------------------------
# Embedding: replicate input, shard output along dim 1.
model = parallelize_module(
    model,
    tp_mesh,
    {
        "tok_embeddings": RowwiseParallel(
            input_layouts=Replicate(),
            output_layouts=Shard(1),
        ),
        "norm": SequenceParallel(),
        "output": ColwiseParallel(
            input_layouts=Shard(1),
            output_layouts=Replicate(),
        ),
    },
)

for transformer_block in model.layers:
    parallelize_module(
        module=transformer_block,
        device_mesh=tp_mesh,
        parallelize_plan={
            "attention_norm": SequenceParallel(),
            "attention": PrepareModuleInput(
                input_layouts=(Shard(1),),
                desired_input_layouts=(Replicate(),),
            ),
            "attention.wq": ColwiseParallel(use_local_output=False),
            "attention.wk": ColwiseParallel(use_local_output=False),
            "attention.wv": ColwiseParallel(use_local_output=False),
            "attention.wo": RowwiseParallel(output_layouts=Shard(1)),
            "ffn_norm": SequenceParallel(),
            "feed_forward": PrepareModuleInput(
                input_layouts=(Shard(1),),
                desired_input_layouts=(Replicate(),),
            ),
            "feed_forward.w1": ColwiseParallel(),
            "feed_forward.w2": RowwiseParallel(output_layouts=Shard(1)),
            "feed_forward.w3": ColwiseParallel(),
        },
    )

# -- FSDP over dp dimension ------------------------------------------------
sharded_model = fully_shard(model, mesh=dp_mesh)

# -- smoke test -------------------------------------------------------------
lr = 3e-3
optimizer = torch.optim.AdamW(sharded_model.parameters(), lr=lr, foreach=True)

num_iters = 10
batch_size = 2
for i in range(num_iters):
    # Seed with dp_rank so all tp ranks in the same replica get identical input.
    torch.manual_seed(i + dp_rank)
    inp = torch.randint(32000, (batch_size, 256), device=device_mod.current_device())
    output = sharded_model(inp)
    output.sum().backward()
    optimizer.step()
    optimizer.zero_grad()

print(
    f"[rank={rank}] FSDP+TP smoke test passed — "
    f"{num_iters} iterations on {dp_size}×{tp_size} mesh ✓",
    flush=True,
)

dist.destroy_process_group()
print(f"[rank={rank}] Done.", flush=True)
