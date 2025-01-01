import argparse

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

from scaletorch.utils import (cleanup_distribute_environment,
                              setup_distributed_environment)
from scaletorch.utils.net_utils import LeNet


class FSDPTrainer:

    def __init__(self, model, args, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        self.args = args

        # Set device
        torch.cuda.set_device(rank)

        # Initialize model with FSDP
        self.model = FSDP(model.to(rank))

        # Setup optimizer and scheduler
        self.optimizer = torch.optim.Adadelta(self.model.parameters(),
                                              lr=args.lr)
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=args.gamma)

        # Setup data
        self._setup_data()

        # CUDA events for timing
        self.init_start_event = torch.cuda.Event(enable_timing=True)
        self.init_end_event = torch.cuda.Event(enable_timing=True)

    def _setup_data(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, ))
        ])

        # Datasets
        train_dataset = datasets.MNIST('./data',
                                       train=True,
                                       download=True,
                                       transform=transform)
        test_dataset = datasets.MNIST('./data',
                                      train=False,
                                      transform=transform)

        # Samplers
        self.train_sampler = DistributedSampler(train_dataset,
                                                rank=self.rank,
                                                num_replicas=self.world_size,
                                                shuffle=True)
        self.test_sampler = DistributedSampler(test_dataset,
                                               rank=self.rank,
                                               num_replicas=self.world_size)

        # DataLoaders
        train_kwargs = {
            'batch_size': self.args.batch_size,
            'sampler': self.train_sampler,
            'num_workers': 2,
            'pin_memory': True,
            'shuffle': False,
        }
        test_kwargs = {
            'batch_size': self.args.test_batch_size,
            'sampler': self.test_sampler,
            'num_workers': 2,
            'pin_memory': True,
            'shuffle': False,
        }

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, **train_kwargs)
        self.test_loader = torch.utils.data.DataLoader(test_dataset,
                                                       **test_kwargs)

    def train_epoch(self, epoch):
        self.model.train()
        ddp_loss = torch.zeros(2).to(self.rank)
        self.train_sampler.set_epoch(epoch)

        for data, target in self.train_loader:
            data, target = data.to(self.rank), target.to(self.rank)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target, reduction='sum')
            loss.backward()
            self.optimizer.step()
            ddp_loss[0] += loss.item()
            ddp_loss[1] += len(data)

        dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
        if self.rank == 0:
            print('Train Epoch: {} \tLoss: {:.6f}'.format(
                epoch, ddp_loss[0] / ddp_loss[1]))

    def test(self):
        self.model.eval()
        ddp_loss = torch.zeros(3).to(self.rank)

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.rank), target.to(self.rank)
                output = self.model(data)
                ddp_loss[0] += F.nll_loss(output, target,
                                          reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                ddp_loss[1] += pred.eq(target.view_as(pred)).sum().item()
                ddp_loss[2] += len(data)

        dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)

        if self.rank == 0:
            test_loss = ddp_loss[0] / ddp_loss[2]
            print(
                'Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.
                format(
                    test_loss,
                    int(ddp_loss[1]),
                    int(ddp_loss[2]),
                    100.0 * ddp_loss[1] / ddp_loss[2],
                ))

    def train(self):
        self.init_start_event.record()

        for epoch in range(1, self.args.epochs + 1):
            self.train_epoch(epoch)
            self.test()
            self.scheduler.step()

        self.init_end_event.record()

        if self.rank == 0:
            print(
                f'CUDA event elapsed time: {self.init_start_event.elapsed_time(self.init_end_event) / 1000}sec'
            )
            print(f'{self.model}')

    def save_model(self):
        if self.args.save_model:
            dist.barrier()
            states = self.model.state_dict()
            if self.rank == 0:
                torch.save(states, 'mnist_cnn.pt')


def fsdp_main(rank, world_size, args):
    setup_distributed_environment(rank, world_size)

    # 创建模型
    model = LeNet()

    # 初始化训练器
    trainer = FSDPTrainer(model, args, rank, world_size)

    # 开始训练
    trainer.train()

    # 保存模型
    trainer.save_model()

    cleanup_distribute_environment()


if __name__ == '__main__':
    # 保持原有的参数解析代码不变
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    # ... 参数设置代码 ...
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    WORLD_SIZE = torch.cuda.device_count()
    mp.spawn(fsdp_main, args=(WORLD_SIZE, args), nprocs=WORLD_SIZE, join=True)
