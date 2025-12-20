import argparse

import torch
import torch.distributed as dist
from torch.optim import AdamW
from transformers import AutoConfig, AutoModelForCausalLM

import scaletorch.parallel.pg_manager as pgm
from scaletorch.dist import cleanup_dist, init_dist
from scaletorch.parallel.pg_manager import setup_process_group_manager
from scaletorch.utils import get_current_device, get_dist_info


def main():
    parser = argparse.ArgumentParser(
        description='Training script for transformers model')
    parser.add_argument('--model_name_or_path',
                        type=str,
                        default='openai/gpt2')
    parser.add_argument('--micro_batch_size', type=int, default=8)
    parser.add_argument('--seq_length', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--tp_size',
                        type=int,
                        default=1,
                        help='Tensor Parallel size')
    parser.add_argument('--dp_size',
                        type=int,
                        default=8,
                        help='Data Parallel size')
    parser.add_argument('--cp_size',
                        type=int,
                        default=1,
                        help='Context Parallel size')
    parser.add_argument('--pp_size',
                        type=int,
                        default=1,
                        help='Pipeline Parallel size')
    parser.add_argument('--pp_engine',
                        type=str,
                        default='afab',
                        choices=['1f1b', 'afab'])
    args = parser.parse_args()

    init_dist(launcher='pytorch', backend='nccl')

    rank, world_size, local_rank = get_dist_info()
    device = get_current_device()

    setup_process_group_manager(dp_size=args.dp_size,
                                cp_size=args.cp_size,
                                pp_size=args.pp_size,
                                tp_size=args.tp_size)

    model_config = AutoConfig.from_pretrained(args.model_name_or_path)

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,
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
    outputs = model(input_ids=input_ids, labels=target_ids)

    # Compute loss
    loss = outputs.loss

    # Backward pass
    loss.backward()

    # Optimizer step
    optimizer.step()

    print(f'[rank {pgm.process_group_manager.global_rank}], Loss: {loss:.4f}')

    cleanup_dist()


if __name__ == '__main__':
    main()
