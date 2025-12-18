import argparse

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoConfig, AutoModel

from scaletorch.dist import cleanup_dist, init_dist
from scaletorch.utils import get_current_device, get_dist_info


def main():
    parser = argparse.ArgumentParser(
        description='Training script for transformers model')
    parser.add_argument('--model_name_or_path',
                        type=str,
                        default='openai/gpt2')
    parser.add_argument('--micro_batch_size', type=int, default=4)
    parser.add_argument('--seq_length', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    args = parser.parse_args()

    init_dist(launcher='pytorch', backend='nccl')

    rank, world_size, local_rank = get_dist_info()
    device = get_current_device()
    model_config = AutoConfig.from_pretrained(args.model_name_or_path)

    model = AutoModel.from_pretrained(args.model_name_or_path,
                                      config=model_config)

    model = model.to(device)
    dist.barrier()

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    dist.barrier()

    # Create dummy data
    input_ids = torch.randint(0,
                              model_config.vocab_size,
                              (args.micro_batch_size, args.seq_length),
                              device=device)
    target_ids = torch.randint(0,
                               model_config.vocab_size,
                               (args.micro_batch_size, args.seq_length),
                               device=device)

    # Training step
    optimizer.zero_grad()

    # Forward pass
    outputs = model(input_ids=input_ids)

    # Compute loss
    target_ids = target_ids.reshape(-1)
    outputs = outputs.view(-1, model_config.vocab_size)
    loss = F.cross_entropy(outputs, target_ids)

    # Backward pass
    loss.backward()

    # Optimizer step
    optimizer.step()

    print(f'Loss: {loss.item():.4f}')

    cleanup_dist()
