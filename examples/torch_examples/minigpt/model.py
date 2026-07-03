"""Full definition of a GPT Language Model.

Adapted from https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from scaletorch.utils.logger_utils import get_logger

logger = get_logger(__name__)


@dataclass
class GPTConfig:
    """Configuration class for GPT model architecture."""

    model_type: str = "gpt2"
    n_layer: int | None = 12
    n_head: int | None = 12
    n_embd: int | None = 768
    vocab_size: int = 50257
    block_size: int = 1024
    embd_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    attn_pdrop: float = 0.1

    def __post_init__(self):
        if self.model_type and all(
            val is None for val in [self.n_layer, self.n_head, self.n_embd]
        ):
            model_configs: dict[str, dict[str, int]] = {
                "openai-gpt": {
                    "n_layer": 12,
                    "n_head": 12,
                    "n_embd": 768,
                },
                "gpt2": {"n_layer": 12, "n_head": 12, "n_embd": 768},
                "gpt2-medium": {
                    "n_layer": 24,
                    "n_head": 16,
                    "n_embd": 1024,
                },
                "gpt2-large": {
                    "n_layer": 36,
                    "n_head": 20,
                    "n_embd": 1280,
                },
                "gpt2-xl": {
                    "n_layer": 48,
                    "n_head": 25,
                    "n_embd": 1600,
                },
                "gopher-44m": {"n_layer": 8, "n_head": 16, "n_embd": 512},
                "gpt-mini": {"n_layer": 6, "n_head": 6, "n_embd": 192},
                "gpt-micro": {"n_layer": 4, "n_head": 4, "n_embd": 128},
                "gpt-nano": {"n_layer": 3, "n_head": 3, "n_embd": 48},
            }
            config = model_configs.get(self.model_type, {})
            self.n_layer = config.get("n_layer", self.n_layer)
            self.n_head = config.get("n_head", self.n_head)
            self.n_embd = config.get("n_embd", self.n_embd)


@dataclass
class OptimizerConfig:
    """Configuration class for optimizer hyperparameters."""

    learning_rate: float = 3e-4
    weight_decay: float = 0.1


class MultiheadAttentionLayer(nn.Module):
    """A multi-head masked self-attention layer with projection."""

    def __init__(
        self, config: GPTConfig, device: str = "cpu", dtype: torch.dtype = torch.float32
    ) -> None:
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError(
                "Embedding dimension must be divisible by number of heads, "
                f"got n_embd={config.n_embd} and n_head={config.n_head}"
            )

        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.c_proj = nn.Linear(
            config.n_embd, config.n_embd, device=device, dtype=dtype
        )

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

        self.attn = nn.MultiheadAttention(
            embed_dim=config.n_embd,
            num_heads=config.n_head,
            dropout=config.attn_pdrop,
            batch_first=True,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, seq_size, _ = x.size()
        y = self.attn(x, x, x, attn_mask=self.mask[0, 0, :seq_size, :seq_size])[0]
        y = self.resid_drop(self.c_proj(y))
        return y


class Block(nn.Module):
    """A standard Transformer block with self-attention and feed-forward
    layers."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = MultiheadAttentionLayer(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class EmbeddingStem(nn.Module):
    """Embedding layer combining token and positional embeddings."""

    def __init__(
        self, config: GPTConfig, device: str = "cpu", dtype: torch.dtype = torch.float32
    ) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(
            config.vocab_size, config.n_embd, device=device, dtype=dtype
        )
        self.pos_emb = nn.Parameter(
            torch.zeros(1, config.block_size, config.n_embd, device=device, dtype=dtype)
        )
        self.drop = nn.Dropout(config.embd_pdrop)
        self.block_size = config.block_size

    def reset_parameters(self) -> None:
        self.tok_emb.reset_parameters()

    def forward(self, idx: torch.LongTensor) -> torch.Tensor:
        b, t = idx.size()
        if t > self.block_size:
            raise ValueError(
                f"Cannot forward sequence of length {t}, "
                f"block size is only {self.block_size}"
            )

        token_embeddings = self.tok_emb(idx)
        position_embeddings = self.pos_emb[:, :t, :]
        return self.drop(token_embeddings + position_embeddings)


class GPT(nn.Module):
    """GPT Language Model implementation."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.block_size = config.block_size

        config = self._set_model_config(config)

        self.emb_stem = EmbeddingStem(config)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                p.data.normal_(mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        n_params = sum(p.numel() for p in self.parameters())
        logger.info("Number of parameters: %.2fM", n_params / 1e6)

    def _set_model_config(self, config: GPTConfig) -> GPTConfig:
        return config

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self, idx: torch.LongTensor, targets: torch.LongTensor | None = None
    ) -> tuple[torch.Tensor, None] | tuple[torch.Tensor, torch.Tensor]:
        x = self.emb_stem(idx)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.LongTensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        do_sample: bool = False,
        top_k: int | None = None,
    ) -> torch.LongTensor:
        for _ in range(max_new_tokens):
            idx_cond = (
                idx if idx.size(1) <= self.block_size else idx[:, -self.block_size :]
            )

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("Inf")

            probs = F.softmax(logits, dim=-1)

            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)

            idx = torch.cat((idx, idx_next), dim=1)

        return idx


def create_optimizer(
    model: torch.nn.Module, opt_config: OptimizerConfig
) -> torch.optim.AdamW:
    """Create an AdamW optimizer with separate weight decay for different
    parameter types."""
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear,)
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)

    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = f"{mn}.{pn}" if mn else pn

            if pn.endswith("bias"):
                no_decay.add(fpn)
            elif (
                pn.endswith("weight") and isinstance(m, whitelist_weight_modules)
            ) or pn.endswith("in_proj_weight"):
                decay.add(fpn)
            elif (
                pn.endswith("weight") and isinstance(m, blacklist_weight_modules)
            ) or pn.endswith("pos_emb"):
                no_decay.add(fpn)

    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    if inter_params:
        raise ValueError(
            "parameters made it into both decay/no_decay sets: %s" % str(inter_params)
        )
    if param_dict.keys() - union_params:
        raise ValueError(
            "parameters not separated into either decay/no_decay set: %s"
            % str(param_dict.keys() - union_params)
        )

    optim_groups = [
        {
            "params": [param_dict[pn] for pn in sorted(list(decay))],
            "weight_decay": opt_config.weight_decay,
        },
        {
            "params": [param_dict[pn] for pn in sorted(list(no_decay))],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(
        optim_groups, lr=opt_config.learning_rate, betas=(0.9, 0.95)
    )
    return optimizer
