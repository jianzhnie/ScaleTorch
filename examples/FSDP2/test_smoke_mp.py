"""CPU multi-process smoke test for FSDP2 + TP Llama2.

Simulates 4 processes on CPU with a 2×2 mesh (dp=2, tp=2) to verify
forward/backward/step works correctly end-to-end.
"""

import os
import sys
sys.path.insert(0, '/home/jianzhnie/llmtuner/llm/ScaleTorch/examples/FSDP2')

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from torch.distributed._tensor import Replicate, Shard
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)

# FSDP2: PyTorch 2.4-2.6 uses _composable.fsdp; 2.7+ uses torch.distributed.fsdp
try:
    from torch.distributed._composable.fsdp import (
        FSDPModule,
        MixedPrecisionPolicy,
        fully_shard,
    )
except ImportError:
    from torch.distributed.fsdp import (
        FSDPModule,
        MixedPrecisionPolicy,
        fully_shard,
    )

from llama2 import LlamaConfig, LlamaForPretraining


def _build_tp_plan():
    layer_plan = {
        "input_layernorm": SequenceParallel(),
        # NOTE: self_attn PrepareModuleInput omitted — LlamaDecoderLayer
        # handles redistribution from Shard(1) → Replicate manually so the
        # optional attn_mask argument does not break the TP hook.
        "self_attn.q_proj": ColwiseParallel(use_local_output=False),
        "self_attn.k_proj": ColwiseParallel(use_local_output=False),
        "self_attn.v_proj": ColwiseParallel(use_local_output=False),
        "self_attn.o_proj": RowwiseParallel(output_layouts=Shard(1)),
        "post_attention_layernorm": SequenceParallel(),
        "mlp": PrepareModuleInput(
            input_layouts=(Shard(1),),
            desired_input_layouts=(Replicate(),),
        ),
        "mlp.gate_proj": ColwiseParallel(),
        "mlp.up_proj": ColwiseParallel(),
        "mlp.down_proj": RowwiseParallel(output_layouts=Shard(1)),
    }
    return layer_plan


def _apply_tp(model, tp_mesh):
    layer_plan = _build_tp_plan()
    for layer in model.base_model.layers:
        parallelize_module(module=layer, device_mesh=tp_mesh, parallelize_plan=layer_plan)
    model = parallelize_module(
        model,
        tp_mesh,
        {
            "base_model.embed_tokens": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(1),
            ),
            "base_model.norm": SequenceParallel(),
            "lm_head": ColwiseParallel(
                input_layouts=Shard(1),
                output_layouts=Replicate(),
            ),
        },
    )
    return model


def run_worker(rank, world_size):
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29505"

    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    tp_size = 2
    dp_size = world_size // tp_size

    mesh_2d = init_device_mesh("cpu", mesh_shape=(dp_size, tp_size), mesh_dim_names=("dp", "tp"))
    tp_mesh = mesh_2d["tp"]
    dp_mesh = mesh_2d["dp"]
    dp_rank = dp_mesh.get_local_rank()

    config = LlamaConfig.from_preset("debug")
    config.num_hidden_layers = 2
    config.max_position_embeddings = 32
    config.vocab_size = 128

    # Create on CPU, init weights, then apply TP and FSDP2
    model = LlamaForPretraining(config)
    model.reset_parameters()

    model = _apply_tp(model, tp_mesh)

    fsdp_kwargs = {"mesh": dp_mesh, "reshard_after_forward": True}
    for layer in model.base_model.layers:
        fully_shard(layer, **fsdp_kwargs)
    fully_shard(model.base_model, **fsdp_kwargs)
    fully_shard(model, **fsdp_kwargs)

    assert isinstance(model, FSDPModule), f"Expected FSDPModule, got {type(model)}"

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Training steps
    losses = []
    for step in range(10):
        torch.manual_seed(42 + dp_rank + step * 100)
        bs, seq_len = 2, 16
        input_ids = torch.randint(0, config.vocab_size, (bs, seq_len))
        target_ids = torch.randint(0, config.vocab_size, (bs, seq_len))

        logits = model(input_ids)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if rank == 0 and step % 2 == 0:
            print(f"[rank={rank}] step={step} loss={loss.item():.4f}", flush=True)

    assert not any(torch.isnan(torch.tensor(l)) for l in losses), f"NaN in losses: {losses}"
    assert not any(torch.isinf(torch.tensor(l)) for l in losses), f"Inf in losses: {losses}"

    print(f"[rank={rank}] All steps passed — losses: {[f'{l:.4f}' for l in losses]}", flush=True)

    dist.destroy_process_group()


def main():
    world_size = 4
    mp.spawn(run_worker, args=(world_size,), nprocs=world_size, join=True)
    print("\n[smoke-mp] All CPU multi-process tests passed!")


if __name__ == "__main__":
    main()
