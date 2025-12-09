import torch
import scaletorch.dist as dist


if __name__ == "__main__":

    # 初始化分布式环境（如果处于分布式环境）

    dist.init_dist(launcher='pytorch', backend='nccl')

    # 创建本地数据
    data = torch.tensor([1.0, 2.0, 3.0])

    # 执行全局求和归约
    dist.all_reduce(data, op='sum')
    print(f"Rank {dist.get_rank()}: {data}")

    # 执行全局平均归约
    data = torch.tensor([1.0, 2.0, 3.0])
    dist.all_reduce(data, op='mean')
    print(f"Rank {dist.get_rank()}: {data}")