import torch

from scaletorch.models.moe_model import GPT, GPTConfig


def test_moe_model():
    config = GPTConfig(n_layer=2,
                       n_head=4,
                       n_embd=128,
                       use_moe=True,
                       n_experts=4,
                       top_k=2,
                       moe_layers=[1])
    model = GPT(config)
    print('Model created successfully')

    # Create dummy input
    idx = torch.randint(0, config.vocab_size, (2, 32))

    # Forward pass
    logits, loss = model(idx)
    print(f'Forward pass successful. Logits shape: {logits.shape}')

    # Check if MoE was used
    moe_layers = [
        m for m in model.modules() if m.__class__.__name__ == 'MOELayer'
    ]
    print(f'Number of MoE layers: {len(moe_layers)}')
    assert len(moe_layers) == 1


if __name__ == '__main__':
    test_moe_model()
