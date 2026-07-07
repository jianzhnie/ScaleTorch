"""FSDP2 Llama 2 pre-training — Ascend NPU / CUDA.

Full-featured FSDP2 training script for Llama 2 with:
  * Auto-detection of NPU (HCCL) vs CUDA (NCCL) backend
  * 1-D DeviceMesh for data-parallel sharding
  * Mixed precision (bfloat16 param, float32 reduce)
  * AdamW + warmup + cosine LR scheduler
  * DistributedSampler + DataLoader
  * Activation checkpointing (optional)
  * Explicit forward/backward prefetching (optional)
  * CPU offloading (optional)
  * DCP (Distributed Checkpoint) save/load

Usage:
  torchrun --nproc_per_node=8 fsdp2_llama2_main.py
  torchrun --nproc_per_node=8 fsdp2_llama2_main.py --model-size 7B
  torchrun --nproc_per_node=8 fsdp2_llama2_main.py --model-size 1B --mixed-precision
  torchrun --nproc_per_node=8 fsdp2_llama2_main.py --model-size debug \\
      --use-synthetic-data --epochs 1
"""

from __future__ import annotations

import argparse
import functools
import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch.distributed.device_mesh import init_device_mesh
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
            "[fsdp2_llama2] torch_npu found but NPU is not available. "
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
        # targets are just shifted (for loss computation)
        y = torch.randint(0, self.vocab_size, (self.seq_length,), dtype=torch.int64)
        return x, y


def _build_model(config: LlamaConfig, args, device, rank):
    """Create LlamaForPretraining on meta device, wrap with FSDP2, materialise."""
    with torch.device("meta"):
        model = LlamaForPretraining(config)

    # -- FSDP2 kwargs ---------------------------------------------------------
    fsdp_kwargs = {"mesh": args.mesh, "reshard_after_forward": True}
    if args.mixed_precision:
        fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )
    if args.cpu_offload:
        fsdp_kwargs["offload_policy"] = CPUOffloadPolicy()

    # Shard each decoder layer, then the root modules.
    for layer in model.base_model.layers:
        fully_shard(layer, **fsdp_kwargs)
    fully_shard(model.base_model, **fsdp_kwargs)
    fully_shard(model, **fsdp_kwargs)

    model.to_empty(device="cpu" if args.cpu_offload else device)
    model.reset_parameters()

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(
            f"\n[fsdp2_llama2] Model: {args.model_size} "
            f"({total_params/1e6:.1f}M params)\n",
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
            print("[fsdp2_llama2] Using synthetic random data.", flush=True)
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
                    f"[fsdp2_llama2] Loaded dataset: {args.dataset_name} "
                    f"(split={args.dataset_split})",
                    flush=True,
                )
        except Exception as e:
            if rank == 0:
                print(
                    f"[fsdp2_llama2] Failed to load real dataset: {e}\n"
                    f"[fsdp2_llama2] Falling back to synthetic data.",
                    flush=True,
                )
            dataset = _SyntheticDataset(config.vocab_size, seq_length)

    sampler = DistributedSampler(dataset, shuffle=True, drop_last=True)
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
            f"[fsdp2_llama2] Need at least 2 {device_type.upper()} devices, "
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

    # -- 1-D device mesh for FSDP2 data-parallel sharding --------------------
    mesh = init_device_mesh(
        device_type,
        mesh_shape=(world_size,),
        mesh_dim_names=("dp",),
    )
    args.mesh = mesh

    # -- model ---------------------------------------------------------------
    torch.manual_seed(0)
    config = LlamaConfig.from_preset(args.model_size)
    # Allow overrides from CLI.
    config.max_position_embeddings = args.seq_len
    model = _build_model(config, args, device, rank)
    assert isinstance(model, FSDPModule), (
        f"Expected FSDPModule, got {type(model)}"
    )

    # -- explicit prefetching (optional) -------------------------------------
    if args.explicit_prefetching > 0:
        modules = list(model.base_model.layers)
        num_pf = args.explicit_prefetching
        for i, m in enumerate(modules):
            if i < len(modules) - 1:
                m.set_modules_to_forward_prefetch(
                    modules[i + 1 : i + 1 + num_pf]
                )
            if i > 0:
                m.set_modules_to_backward_prefetch(
                    modules[max(0, i - num_pf) : i]
                )

    # -- activation checkpointing (optional) ---------------------------------
    if args.activation_checkpointing:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            apply_activation_checkpointing,
            checkpoint_wrapper,
        )
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

        from llama2 import LlamaDecoderLayer

        wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={LlamaDecoderLayer, nn.Embedding},
        )
        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=checkpoint_wrapper,
            auto_wrap_policy=wrap_policy,
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
                f"[fsdp2_llama2] Resuming from checkpoint: {args.checkpoint_dir}",
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
        print("[fsdp2_llama2] Training complete ✓", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FSDP2 Llama 2 pre-training (NPU/CUDA)"
    )
    # Model
    parser.add_argument(
        "--model-size", type=str, default="debug",
        choices=["debug", "1B", "7B"],
        help="Model size preset (default: debug)",
    )
    # Dataset
    parser.add_argument(
        "--use-synthetic-data", action="store_true", default=False,
        help="Use random synthetic data instead of real dataset",
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
