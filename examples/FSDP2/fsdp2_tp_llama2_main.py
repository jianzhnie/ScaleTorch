"""FSDP2 + Tensor Parallel Llama 2 pre-training — Ascend NPU / CUDA.

Combines Fully Sharded Data Parallel (FSDP) with Tensor Parallel (TP)
on Llama 2 via a 2-D DeviceMesh:
  * ``dp`` – FSDP across data-parallel replicas (cross-host)
  * ``tp`` – TP within each host (intra-host, e.g. 8 GPUs/NPUs)

Reference:
  https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch.distributed._tensor import Replicate, Shard
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
    loss_parallel,
)
from torch.utils.data.distributed import DistributedSampler

from checkpoint import load_checkpoint_dcp, save_checkpoint_dcp
from data import PretrainingDataset, create_causal_mask, create_padding_mask
from llama2 import LlamaConfig, LlamaForPretraining

# -- backend auto-detection (NPU → CUDA fallback) ---------------------------
try:
    import torch_npu  # noqa: F401
except ImportError:
    device_mod, backend, device_type = torch.cuda, "nccl", "cuda"
else:
    if torch.npu.is_available():
        device_mod, backend, device_type = torch.npu, "hccl", "npu"
    else:
        sys.exit(
            "[fsdp2_tp_llama2] torch_npu found but NPU is not available. "
            "Is the CANN toolkit sourced?\n"
        )

# FSDP2: PyTorch 2.4-2.6 uses _composable.fsdp; 2.7+ uses torch.distributed.fsdp
try:
    from torch.distributed._composable.fsdp import (
        CPUOffloadPolicy,
        FSDPModule,
        MixedPrecisionPolicy,
        fully_shard,
    )
except ImportError:
    from torch.distributed.fsdp import (
        CPUOffloadPolicy,
        FSDPModule,
        MixedPrecisionPolicy,
        fully_shard,
    )


def _verify_min_devices(min_devices: int = 2) -> bool:
    return device_mod.device_count() >= min_devices


class _SyntheticDataset(torch.utils.data.Dataset):
    """Fallback dataset that generates random token sequences."""

    def __init__(self, vocab_size: int, seq_length: int, num_samples: int = 1000):
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.num_samples = num_samples

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.randint(0, self.vocab_size, (self.seq_length,), dtype=torch.int64)
        y = torch.randint(0, self.vocab_size, (self.seq_length,), dtype=torch.int64)
        return x, y


def _build_tp_plan():
    """Build the Tensor Parallel parallelization plan for Llama2 model.

    Reference: https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html
    """
    # Plan for each decoder layer
    layer_plan = {
        "input_layernorm": SequenceParallel(),
        # NOTE: self_attn PrepareModuleInput is intentionally omitted.
        # The input to self_attn carries an optional attn_mask argument whose
        # arity varies (1 if no mask, 2 with mask), which PrepareModuleInput
        # cannot handle.  Instead, LlamaDecoderLayer.forward redistributes
        # hidden_states from Shard(1) → Replicate before calling self_attn.
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


def _apply_tp(model: nn.Module, tp_mesh) -> nn.Module:
    """Apply Tensor Parallel to the Llama2 model."""
    layer_plan = _build_tp_plan()

    # Apply TP to each decoder layer
    for layer in model.base_model.layers:
        parallelize_module(
            module=layer,
            device_mesh=tp_mesh,
            parallelize_plan=layer_plan,
        )

    # Apply TP to the base model (embeddings + final norm) and lm_head
    # Use use_local_output=False on lm_head so output stays as DTensor for loss_parallel
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


def _build_model(config: LlamaConfig, args, device, rank):
    """Create LlamaForPretraining, apply TP then FSDP2.

    Construction steps (matches fsdp_tp_demo.py pattern):
      1. Create on CPU, initialise weights (avoids meta-device DTensor NaN)
      2. Move to target device while the model is still a plain nn.Module
      3. Apply Tensor Parallel — shards parameters into DTensors on-device
      4. Apply FSDP2 over the dp dimension

    NOTE: Do NOT use meta device + to_empty for TP+FSDP2 models — DTensor
    parameters constructed from uninitialised memory cannot be properly
    reset by ``reset_parameters()``, causing NaN losses.
    """
    # 1. Create on CPU with standard weight initialisation.
    model = LlamaForPretraining(config)
    model.reset_parameters()

    # 2. Move the full (unsharded) model to the target device.
    #    For very large models that do not fit on a single device, keep the
    #    model on CPU through steps 2-3 and call model.to(device) *after*
    #    FSDP2 wrapping (see the "cpu_first" alternative below).
    model = model.to(device)

    # 3. Tensor Parallel — shard intra-host (within a node).
    model = _apply_tp(model, args.tp_mesh)

    # 4. FSDP2 — shard across data-parallel replicas (cross-host).
    fsdp_kwargs = {"mesh": args.dp_mesh, "reshard_after_forward": True}
    if args.mixed_precision:
        fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )
    if args.cpu_offload:
        fsdp_kwargs["offload_policy"] = CPUOffloadPolicy()

    # Shard each decoder layer individually, then the top-level modules.
    for layer in model.base_model.layers:
        fully_shard(layer, **fsdp_kwargs)
    fully_shard(model.base_model, **fsdp_kwargs)
    fully_shard(model, **fsdp_kwargs)

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(
            f"\n[fsdp2_tp_llama2] Model: {args.model_size} "
            f"({total_params/1e6:.1f}M params) "
            f"dp_size={args.dp_size} tp_size={args.tp_size}\n",
            flush=True,
        )
    return model


def _build_dataloader(args, config, rank, world_size):
    """Build a DataLoader with DistributedSampler.

    Uses a real HuggingFace dataset when available; falls back to synthetic
    random data when ``--use-synthetic-data`` is set or the real dataset
    cannot be loaded.
    """
    seq_length = args.seq_len

    if args.use_synthetic_data:
        dataset = _SyntheticDataset(config.vocab_size, seq_length)
        if rank == 0:
            print("[fsdp2_tp_llama2] Using synthetic random data.", flush=True)
    elif args.dataset_path:
        # Load a local Arrow dataset from disk (e.g., wikitext2).
        try:
            import datasets as hf_datasets
            import tokenizers as tok_mod

            tokenizer = tok_mod.Tokenizer.from_file(args.tokenizer_path)
            dataset = hf_datasets.load_from_disk(args.dataset_path)
            dataset = PretrainingDataset(dataset, tokenizer, seq_length)
            if rank == 0:
                print(
                    f"[fsdp2_tp_llama2] Loaded local dataset: {args.dataset_path} "
                    f"({len(dataset)} samples after tokenization)",
                    flush=True,
                )
        except Exception as e:
            if rank == 0:
                print(
                    f"[fsdp2_tp_llama2] Failed to load local dataset: {e}\n"
                    f"[fsdp2_tp_llama2] Falling back to synthetic data.",
                    flush=True,
                )
            dataset = _SyntheticDataset(config.vocab_size, seq_length)
    else:
        try:
            import datasets
            import tokenizers

            tokenizer = tokenizers.Tokenizer.from_file(args.tokenizer_path)
            dataset = datasets.load_dataset(
                args.dataset_name, args.dataset_split, split="train"
            )
            dataset = PretrainingDataset(dataset, tokenizer, seq_length)
            if rank == 0:
                print(
                    f"[fsdp2_tp_llama2] Loaded dataset: {args.dataset_name} "
                    f"(split={args.dataset_split})",
                    flush=True,
                )
        except Exception as e:
            if rank == 0:
                print(
                    f"[fsdp2_tp_llama2] Failed to load real dataset: {e}\n"
                    f"[fsdp2_tp_llama2] Falling back to synthetic data.",
                    flush=True,
                )
            dataset = _SyntheticDataset(config.vocab_size, seq_length)

    # Note: DistributedSampler uses the dp mesh for data parallelism.
    # All ranks in the same TP group should see the same data.
    sampler = DistributedSampler(
        dataset,
        num_replicas=args.dp_size,
        rank=args.dp_rank,
        shuffle=True,
        drop_last=True,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        pin_memory=True,
        shuffle=False,
        num_workers=2,
        prefetch_factor=2,
    )
    padding_token_id = getattr(
        getattr(dataset, "pad", None), "token_to_id", lambda x: -1
    )("[PAD]") if hasattr(dataset, "pad") else -1

    return dataloader, padding_token_id


def main(args):
    # -- guard: minimum device count -----------------------------------------
    if not _verify_min_devices(min_devices=2):
        print(
            f"[fsdp2_tp_llama2] Need at least 2 {device_type.upper()} devices, "
            f"found {device_mod.device_count()}. Exiting.",
            flush=True,
        )
        sys.exit(1)

    # -- bootstrap -----------------------------------------------------------
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ["RANK"]))
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group(backend)
    device_mod.set_device(local_rank)
    device = torch.device(f"{device_type}:{local_rank}")

    print(
        f"[rank={rank}] local_rank={local_rank} world_size={world_size} "
        f"device={device_type} device_count={device_mod.device_count()}",
        flush=True,
    )

    # -- 2-D mesh: dp × tp ---------------------------------------------------
    tp_size = args.tp_size
    if world_size % tp_size != 0:
        dist.destroy_process_group()
        sys.exit(
            f"[fsdp2_tp_llama2] world_size={world_size} not divisible by tp_size={tp_size}.\n"
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

    args.tp_mesh = tp_mesh
    args.dp_mesh = dp_mesh
    args.dp_size = dp_size
    args.tp_size = tp_size
    args.dp_rank = dp_rank

    print(
        f"[rank={rank}] FSDP+TP mesh: {dp_size}×{tp_size} "
        f"(dp={dp_mesh} tp={tp_mesh})",
        flush=True,
    )

    # -- model ---------------------------------------------------------------
    torch.manual_seed(0)
    config = LlamaConfig.from_preset(args.model_size)
    # Allow overrides from CLI.
    config.max_position_embeddings = args.seq_len
    if args.vocab_size is not None:
        config.vocab_size = args.vocab_size
    model = _build_model(config, args, device, rank)
    assert isinstance(model, FSDPModule), (
        f"Expected FSDPModule, got {type(model)}"
    )

    # -- explicit prefetching (optional) -------------------------------------
    if args.explicit_prefetching > 0:
        # After fully_shard, model.base_model.layers contains FSDPModule wrappers
        modules = list(model.base_model.layers)
        num_pf = args.explicit_prefetching
        for i, m in enumerate(modules):
            if hasattr(m, 'set_modules_to_forward_prefetch'):
                if i < len(modules) - 1:
                    m.set_modules_to_forward_prefetch(
                        modules[i + 1 : i + 1 + num_pf]
                    )
                if i > 0:
                    m.set_modules_to_backward_prefetch(
                        modules[max(0, i - num_pf) : i]
                    )
            elif rank == 0 and i == 0:
                print(
                    "[fsdp2_tp_llama2] Warning: explicit_prefetching enabled but "
                    "FSDPModule does not expose prefetch API. Skipping.",
                    flush=True,
                )

    # -- activation checkpointing (optional) ---------------------------------
    if args.activation_checkpointing:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            apply_activation_checkpointing,
            checkpoint_wrapper,
        )
        from llama2 import LlamaDecoderLayer

        def _is_decoder_layer(module):
            inner = module
            if hasattr(module, '_orig_module'):
                inner = module._orig_module
            return isinstance(inner, LlamaDecoderLayer)

        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=checkpoint_wrapper,
            check_fn=_is_decoder_layer,
        )

    # -- data pipeline -------------------------------------------------------
    dataloader, padding_token_id = _build_dataloader(args, config, rank, world_size)
    num_training_steps = len(dataloader) * args.epochs

    # -- optimizer & scheduler -----------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.99),
        eps=1e-8,
        weight_decay=0.1,
    )

    num_warmup_steps = min(args.warmup_steps, num_training_steps // 4)
    warmup_scheduler = lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=num_warmup_steps,
    )
    cosine_scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_training_steps - num_warmup_steps,
        eta_min=0,
    )
    scheduler = lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[num_warmup_steps],
    )

    # -- resume from checkpoint (if exists) ----------------------------------
    if os.path.exists(args.checkpoint_dir):
        if rank == 0:
            print(
                f"[fsdp2_tp_llama2] Resuming from checkpoint: {args.checkpoint_dir}",
                flush=True,
            )
        load_checkpoint_dcp(model, optimizer, scheduler, args.checkpoint_dir)

    model.train()

    # -- training loop -------------------------------------------------------
    global_step = 0
    for epoch in range(args.epochs):
        import tqdm

        if rank == 0:
            pbar = tqdm.tqdm(
                dataloader, desc=f"Epoch {epoch+1}/{args.epochs}"
            )
        else:
            pbar = dataloader  # tqdm only on rank 0

        for batch_id, batch in enumerate(pbar):
            # Periodic checkpointing
            if global_step > 0 and global_step % args.save_every == 0:
                save_checkpoint_dcp(
                    model, optimizer, scheduler, args.checkpoint_dir
                )

            # Explicit prefetching: unshard before sending data
            model.unshard()

            input_ids, target_ids = batch
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            # Attention mask: causal + padding
            attn_mask = create_causal_mask(input_ids)
            if padding_token_id >= 0:
                attn_mask = attn_mask + create_padding_mask(
                    input_ids, padding_token_id
                )

            # Forward / backward / step
            logits = model(input_ids, attn_mask)

            # Use loss_parallel for efficient cross-entropy with sharded vocab dim
            with loss_parallel():
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    target_ids.reshape(-1),
                    ignore_index=padding_token_id if padding_token_id >= 0 else -100,
                )

            optimizer.zero_grad(
                set_to_none=False if args.cpu_offload else True
            )
            loss.backward()
            if not args.cpu_offload:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if isinstance(pbar, tqdm.tqdm):
                pbar.set_postfix(loss=f"{loss.item():.4f}")
            elif rank == 0 and global_step % 100 == 0:
                print(
                    f"  [step {global_step}] loss={loss.item():.4f}", flush=True
                )

            global_step += 1

        if isinstance(pbar, tqdm.tqdm):
            pbar.close()

    # -- final checkpoint ----------------------------------------------------
    save_checkpoint_dcp(model, optimizer, scheduler, args.checkpoint_dir)

    dist.destroy_process_group()
    if rank == 0:
        print("[fsdp2_tp_llama2] Training complete ✓", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FSDP2 + TP Llama 2 pre-training (NPU/CUDA)"
    )
    # Model
    parser.add_argument(
        "--model-size", type=str, default="debug",
        choices=["debug", "1B", "7B"],
        help="Model size preset (default: debug)",
    )
    parser.add_argument(
        "--vocab-size", type=int, default=None,
        help="Override vocab size (default: use model preset's value)",
    )
    # Tensor Parallel
    parser.add_argument(
        "--tp-size", type=int, default=2,
        help="Tensor parallel size (default: 2). Must divide world_size.",
    )
    # Dataset
    parser.add_argument(
        "--use-synthetic-data", action="store_true", default=False,
        help="Use random synthetic data instead of real dataset",
    )
    parser.add_argument(
        "--dataset-path", type=str, default=None,
        help="Path to a local Arrow dataset on disk (e.g., wikitext2). "
             "Uses datasets.load_from_disk().",
    )
    parser.add_argument(
        "--tokenizer-path", type=str, default="bpe_50K.json",
        help="Path to tokenizer JSON file",
    )
    parser.add_argument(
        "--dataset-name", type=str, default="HuggingFaceFW/fineweb",
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--dataset-split", type=str, default="sample-10BT",
        help="HuggingFace dataset split",
    )
    # Training
    parser.add_argument(
        "--epochs", type=int, default=3,
        help="Number of training epochs (default: 3)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Peak learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Per-rank batch size (default: 4)",
    )
    parser.add_argument(
        "--seq-len", type=int, default=512,
        help="Sequence length (default: 512)",
    )
    parser.add_argument(
        "--warmup-steps", type=int, default=1000,
        help="LR warmup steps (default: 1000)",
    )
    parser.add_argument(
        "--save-every", type=int, default=1000,
        help="Save checkpoint every N global steps (default: 1000)",
    )
    # FSDP2 options
    parser.add_argument(
        "--mixed-precision", action="store_true", default=False,
        help="Enable bfloat16 mixed precision",
    )
    parser.add_argument(
        "--cpu-offload", action="store_true", default=False,
        help="Enable CPU offloading of parameters",
    )
    parser.add_argument(
        "--explicit-prefetching", type=int, default=0,
        help="Number of layers to prefetch (0 = disabled, default: 0)",
    )
    parser.add_argument(
        "--activation-checkpointing", action="store_true", default=False,
        help="Enable per-layer activation checkpointing",
    )
    # Checkpoint
    parser.add_argument(
        "--checkpoint-dir", type=str, default="checkpoint-dist",
        help="Directory for DCP checkpoints (default: checkpoint-dist)",
    )

    args = parser.parse_args()
    main(args)
