"""Verify Qwen3 model init + weight loading for all available sizes."""

import os

import torch

os.environ["FLASH_ATTEN"] = "0"
os.environ["DTYPE"] = "bfloat16"

from transformers import AutoConfig

from scaletorch.models.model_qwen3 import Qwen3
from scaletorch.utils.checkpoint import (
    init_model_with_dematerialized_weights,
    init_model_with_materialized_weights,
)

MODELS = [
    ("/workspace/models/Qwen3-0.6B", "single sft, tie=True"),
    ("/workspace/models/Qwen3-1.7B", "sharded sft, tie=True"),
    ("/workspace/models/Qwen3-4B", "sharded sft, tie=True"),
    ("/workspace/models/Qwen3-8B", "sharded sft, tie=False"),
]

device = torch.device("npu:0")
torch.npu.set_device(0)

for model_path, desc in MODELS:
    if not os.path.exists(model_path):
        print(f"SKIP {os.path.basename(model_path)}: not mounted")
        continue

    print(f"\n{'=' * 60}")
    print(f"Testing {os.path.basename(model_path)} ({desc})")
    print(f"{'=' * 60}")

    try:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        print(
            f"  Config: hidden={config.hidden_size}, layers={config.num_hidden_layers}, "
            f"heads={config.num_attention_heads}, kv={config.num_key_value_heads}, "
            f"head_dim={getattr(config, 'head_dim', '?')}, "
            f"tie={getattr(config, 'tie_word_embeddings', '?')}"
        )

        with init_model_with_dematerialized_weights():
            model = Qwen3(config=config)

        model = init_model_with_materialized_weights(
            model, config, save_dir=model_path + "/"
        )

        model = model.to(torch.bfloat16).to(device)

        num_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  Params: {num_params:.1f}M")

        if getattr(config, "tie_word_embeddings", False):
            assert (
                model.final_proj.weight.data_ptr() == model.embedding.weight.data_ptr()
            ), "tie_word_embeddings: weights should share memory"
            print("  Tie check: PASS (shared memory)")

        x = torch.randint(0, config.vocab_size, (1, 32), device=device)
        with (
            torch.no_grad(),
            torch.amp.autocast(device_type="npu", dtype=torch.bfloat16),
        ):
            out = model(x)
        assert out.shape == (1, 32, config.vocab_size), f"Bad shape: {out.shape}"
        assert not torch.isnan(out).any(), "NaN in output"
        assert not torch.isinf(out).any(), "Inf in output"
        print(f"  Forward: PASS (shape={out.shape}, no NaN/Inf)")

        model.train()
        x = torch.randint(0, config.vocab_size, (2, 64), device=device)
        with torch.amp.autocast(device_type="npu", dtype=torch.bfloat16):
            out = model(x)
            loss = torch.nn.functional.cross_entropy(
                out[:, :-1].reshape(-1, out.size(-1)), x[:, 1:].reshape(-1)
            )
        loss.backward()
        grads_ok = sum(1 for p in model.parameters() if p.grad is not None)
        print(f"  Backward: PASS (loss={loss.item():.2f}, grads={grads_ok} params)")

        del model, out, loss
        torch.npu.empty_cache()
        print("  RESULT: OK")

    except Exception as e:
        print(f"  RESULT: FAILED - {e}")
        import traceback

        traceback.print_exc()

print(f"\n{'=' * 60}")
print("All verification complete")
