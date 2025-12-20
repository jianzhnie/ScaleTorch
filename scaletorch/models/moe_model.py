"""
Full definition of a GPT Language Model with Mixture of Experts, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py

OpenMoE:
Based upon ColossalAI OpenMoE: https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/moe/experts.py
"""

import inspect
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias,
                            1e-5)


class CausalSelfAttention(nn.Module):
    """Causal self-attention module with optional flash attention support."""

    def __init__(self, config: 'GPTConfig'):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd,
                                3 * config.n_embd,
                                bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional,
                             'scaled_dot_product_attention')
        if not self.flash:
            print(
                'WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0'
            )
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                'bias',
                torch.tril(torch.ones(config.block_size,
                                      config.block_size)).view(
                                          1, 1, config.block_size,
                                          config.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size(
        )  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head,
                   C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head,
                   C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head,
                   C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(
            B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """Standard MLP module for non-MoE layers."""

    def __init__(self, config: 'GPTConfig'):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd,
                              4 * config.n_embd,
                              bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd,
                                config.n_embd,
                                bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """Transformer block with self-attention and MLP."""

    def __init__(self, config: 'GPTConfig'):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class MLPExperts(nn.Module):
    """Group of MLP experts for MoE layers."""

    def __init__(
        self,
        d: int,
        n_exp: int = 8,
        bias: bool = False,
        dropout: float = 0.2,
    ):
        """
        Arguments:
        d: size of embedding dimension
        n_exp: the number of experts to create in the expert layer
        bias: whether or not to use bias in linear layers
        dropout: probability of dropout
        """
        super().__init__()
        self.bias = bias
        self.c_fc = nn.Parameter(torch.empty(n_exp, d, 4 * d))
        self.c_proj = nn.Parameter(torch.empty(n_exp, 4 * d, d))
        self.fc_bias = nn.Parameter(torch.empty(n_exp, 1, 4 *
                                                d)) if self.bias else None
        self.proj_bias = nn.Parameter(torch.empty(n_exp, 1,
                                                  d)) if self.bias else None
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        # Initialize parameters
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with normal distribution."""
        torch.nn.init.normal_(self.c_fc, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.c_proj, mean=0.0, std=0.02)
        if self.bias:
            torch.nn.init.zeros_(self.fc_bias)
            torch.nn.init.zeros_(self.proj_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process input through expert MLPs.

        Args:
            x: Input tensor of shape [n_exp, exp_capacity, d]

        Returns:
            Output tensor of shape [n_exp, exp_capacity, d]
        """
        x = torch.bmm(x, self.c_fc)
        if self.bias:
            x += self.fc_bias
        x = self.gelu(x)
        x = torch.bmm(x, self.c_proj)
        if self.bias:
            x += self.proj_bias
        x = self.dropout(x)
        return x


class Router(nn.Module):
    """Router module for selecting experts in MoE layers."""

    def __init__(
        self,
        d: int,
        n_exp: int = 8,
        top_k: int = 2,
        use_noisy_top_k: bool = True,
        capacity_factor: float = 1.25,
    ):
        """
        Arguments:
        d: size of embedding dimension
        n_exp: the number of experts to create in the expert layer
        top_k: the number of active experts for each token
        use_noisy_top_k: whether to add noise when computing expert output
        capacity_factor: used to compute expert capacity
        """
        super().__init__()

        self.d = d
        self.n_exp = n_exp
        self.top_k = top_k
        assert self.top_k >= 1 and self.top_k <= n_exp
        self.use_noisy_top_k = use_noisy_top_k
        self.capacity_factor = capacity_factor
        self.w_g = nn.Linear(d, n_exp, bias=False)
        self.w_noise = nn.Linear(d, n_exp,
                                 bias=False) if self.use_noisy_top_k else None

    def forward(
            self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route tokens to experts.

        Args:
            x: Input tensor of shape [B, C, d]

        Returns:
            Tuple of:
            - exp_weights: Expert weights of shape [B * C, n_exp, exp_capacity]
            - exp_mask: Expert mask of shape [B * C, n_exp, exp_capacity]
            - expert_batches: Expert batches of shape [n_exp, exp_capacity, d]
        """
        # get the total number of tokens in the batch
        B, C, _ = x.size()
        num_tokens = B * C

        # eq (4) in https://arxiv.org/abs/1701.06538
        logits = self.w_g(x)  # [B, C, d] -> [B, C, n_exp]
        if self.use_noisy_top_k:
            # (optionally) add noise into the router
            noise = F.softplus(self.w_noise(x))
            noise *= torch.randn_like(noise)
            logits += noise

        # top-K expert selection, compute probabilities over active experts
        top_k_logits, top_k_indices = logits.topk(self.top_k,
                                                  dim=-1)  # [B, C, K]
        router_probs = torch.full_like(logits, float('-inf'))  # [B, C, n_exp]
        router_probs.scatter_(-1, top_k_indices, top_k_logits)
        router_probs = F.softmax(router_probs, dim=-1)

        # compute the expert capacity
        exp_capacity = math.floor(self.top_k * self.capacity_factor *
                                  num_tokens / self.n_exp)
        exp_capacity += exp_capacity % 2  # make sure expert capacity is an even integer
        exp_capacity = int(exp_capacity)

        # make a multi-hot mask of chosen experts
        # values are 0 if expert not chosen, 1 if expert chosen
        exp_mask = F.one_hot(top_k_indices,
                             num_classes=self.n_exp)  # [B, C, K, n_exp]
        exp_mask = exp_mask.view(num_tokens, self.top_k,
                                 self.n_exp)  # [B * C, K, n_exp]
        exp_mask = exp_mask.permute(1, 0, 2)  # [K, B * C, n_exp]

        # compute index for each token in expert batch
        # NOTE: cumsum counts top-1 first, top-2 second, etc.
        # to prioritize top experts when dropping tokens
        exp_rank = exp_mask.reshape(self.top_k * num_tokens,
                                    self.n_exp)  # [K * B * C, n_exp]
        exp_rank = torch.cumsum(
            exp_rank,
            dim=0) - 1  # cumsum of expert selections [K * B * C, n_exp]
        exp_rank = exp_rank.reshape(self.top_k, num_tokens,
                                    self.n_exp)  # [K, B * C, n_exp]

        # mask entries beyond expert capacity and compute used capacity
        exp_mask *= torch.lt(exp_rank, exp_capacity)  # [K, B * C, n_exp]

        # matrix storing token position in batch of corresponding expert
        exp_rank = torch.sum(exp_mask * exp_rank, dim=-1)  # [K, B * C]

        # mask probabilities to only include selected experts
        router_probs = router_probs.view(
            num_tokens, self.n_exp)[None, :]  # [1, B * C, n_exp]
        exp_weights = exp_mask * router_probs  # [K, B * C, n_exp]

        # position of each token within the capacity of the selected expert
        exp_rank_sc = F.one_hot(
            exp_rank, num_classes=exp_capacity)  # [K, B * C, exp_capacity]

        # weight of selected expert for each token at position the capacity of that expert
        exp_weights = torch.sum(exp_weights.unsqueeze(3) *
                                exp_rank_sc.unsqueeze(2),
                                dim=0)  # [B * C, n_exp, exp_capacity]
        exp_mask = exp_weights.bool(
        )  # binary mask of selected experts for each token

        # reshape tokens into batches for each expert, return both weights and batches
        # [n_exp, exp_capacity, B * C] * [B * C, d] -> [n_exp, exp_capacity, d]
        x = x.view(num_tokens, self.d)
        expert_batches = exp_mask.permute(1, 2, 0).type_as(x) @ x
        return exp_weights, exp_mask, expert_batches


class MOELayer(nn.Module):
    """Mixture of Experts layer implementing expert routing and processing."""

    def __init__(
        self,
        d: int,
        n_exp: int = 8,
        top_k: int = 2,
        use_noisy_top_k: bool = True,
        capacity_factor: float = 1.25,
        bias: bool = False,
        dropout: float = 0.2,
    ):
        """
        Arguments:
        d: size of embedding dimension
        n_exp: the number of experts to create in the expert layer
        top_k: the number of active experts for each token
        use_noisy_top_k: whether to add noise when computing expert output
        capacity_factor: used to compute expert capacity
        bias: whether or not to use bias in linear layers
        dropout: probability of dropout
        """
        super().__init__()
        self.router = Router(  # (noisy) top k router
            d=d,
            n_exp=n_exp,
            top_k=top_k,
            use_noisy_top_k=use_noisy_top_k,
            capacity_factor=capacity_factor,
        )
        self.experts = MLPExperts(  # group of MLPs (experts)
            d=d,
            n_exp=n_exp,
            bias=bias,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MoE layer.

        Args:
            x: Input tensor of shape [B, C, d]

        Returns:
            Output tensor of shape [B, C, d]
        """
        B, C, d = x.size()  # track original shape of input
        num_tokens = B * C

        # pass each token through the router
        exp_weight, exp_mask, exp_batches = self.router(x)

        # compute expert output
        exp_out = self.experts(exp_batches)  # [n_exp, exp_capacity, d]

        # aggregate expert outputs based on router weights
        # eq (2) on page 4 of ST-MoE (https://arxiv.org/abs/2202.08906)
        exp_weight = exp_weight.view(num_tokens,
                                     -1)  # [B * C, n_exp * exp_capacity]
        exp_out = exp_out.view(-1, d)  # [n_exp * exp_capacity, d]
        output = exp_weight @ exp_out  # [B * C, d]

        # resize output before return
        return output.view(B, C, d)


class MoEBlock(nn.Module):
    """Transformer block with MoE layer instead of standard MLP."""

    def __init__(
        self,
        config: 'GPTConfig',
        n_exp: int = 8,
        top_k: int = 2,
        use_noisy_top_k: bool = True,
        capacity_factor: float = 1.25,
    ):
        """
        Arguments:
        config: GPT configuration
        n_exp: the number of experts to create in the expert layer
        top_k: the number of active experts for each token
        use_noisy_top_k: whether to add noise when computing expert output
        capacity_factor: used to compute expert capacity
        """
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MOELayer(
            d=config.n_embd,
            n_exp=n_exp,
            top_k=top_k,
            use_noisy_top_k=use_noisy_top_k,
            capacity_factor=capacity_factor,
            bias=config.bias,
            dropout=config.dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    """Configuration class for GPT model."""
    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster


class GPT(nn.Module):
    """GPT Language Model with optional MoE layers."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config)
                                 for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight  # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p,
                                      mean=0.0,
                                      std=0.02 / math.sqrt(2 * config.n_layer))

        # report number of parameters
        print('number of parameters: %.2fM' % (self.get_num_params() / 1e6, ))

    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module: nn.Module):
        """Initialize weights for different module types."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the model.

        Args:
            idx: Input tensor of token indices with shape (b, t)
            targets: Target tensor for training with shape (b, t)

        Returns:
            Tuple of logits and loss (loss is None during inference)
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f'Cannot forward sequence of length {t}, block size is only {self.config.block_size}'
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(
            idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(
            pos)  # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1),
                                   ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(
                x[:,
                  [-1], :])  # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def crop_block_size(self, block_size: int):
        """
        Model surgery to decrease the block size if necessary.
        E.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        but want to use a smaller block size for some smaller, simpler model.
        """
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(
            self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:, :, :block_size, :
                                                  block_size]

    @classmethod
    def from_pretrained(cls,
                        model_type: str,
                        override_args: Optional[dict] = None):
        """
        Load a pretrained GPT-2 model from Hugging Face.

        Args:
            model_type: Type of GPT-2 model ('gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl')
            override_args: Dictionary of arguments to override in the config

        Returns:
            GPT model with loaded weights
        """
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {}  # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print('loading weights from pretrained gpt: %s' % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium': dict(n_layer=24, n_head=16,
                                n_embd=1024),  # 350M params
            'gpt2-large': dict(n_layer=36, n_head=20,
                               n_embd=1280),  # 774M params
            'gpt2-xl': dict(n_layer=48, n_head=25,
                            n_embd=1600),  # 1558M params
        }[model_type]
        print('forcing vocab_size=50257, block_size=1024, bias=True')
        config_args[
            'vocab_size'] = 50257  # always 50257 for GPT model checkpoints
        config_args[
            'block_size'] = 1024  # always 1024 for GPT model checkpoints
        config_args['bias'] = True  # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')
                   ]  # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')
        ]  # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')
                      ]  # same, just the mask (buffer)
        transposed = [
            'attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight',
            'mlp.c_proj.weight'
        ]
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(
            sd_keys), f'mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}'
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay: float, learning_rate: float,
                             betas: Tuple[float, float], device_type: str):
        """
        Configure the AdamW optimizer with weight decay.

        Args:
            weight_decay: Weight decay coefficient
            learning_rate: Learning rate
            betas: AdamW beta parameters
            device_type: Type of device ('cuda' or 'cpu')

        Returns:
            Configured optimizer
        """
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [{
            'params': decay_params,
            'weight_decay': weight_decay
        }, {
            'params': nodecay_params,
            'weight_decay': 0.0
        }]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f'num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters'
        )
        print(
            f'num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters'
        )
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(
            torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups,
                                      lr=learning_rate,
                                      betas=betas,
                                      **extra_args)
        print(f'using fused AdamW: {use_fused}')

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter: int, dt: float) -> float:
        """
        Estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS.

        Args:
            fwdbwd_per_iter: Number of forward/backward passes per iteration
            dt: Time per iteration in seconds

        Returns:
            MFU value
        """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self,
                 idx: torch.Tensor,
                 max_new_tokens: int,
                 temperature: float = 1.0,
                 top_k: Optional[int] = None) -> torch.Tensor:
        """
        Generate new tokens from the model.

        Args:
            idx: Conditioning sequence of indices with shape (b, t)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter

        Returns:
            Generated sequence with shape (b, t+max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(
                1) <= self.config.block_size else idx[:,
                                                      -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
