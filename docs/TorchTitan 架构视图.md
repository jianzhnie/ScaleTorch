#  TorchTitan 架构视图

## 项目概述

TorchTitan 是 Meta 开发的 PyTorch 原生大规模生成式 AI 模型训练平台，专为快速实验和大规模训练而设计。作为 PyTorch 原生扩展技术的最小化清洁实现，TorchTitan 为开发者提供了灵活的基础框架。

**核心设计原则**：

- 易于理解、使用和扩展
- 应用多维并行时对模型代码的最小更改
- 倾向于简洁、最小化的代码库，同时提供可复用/可替换的组件

## 1. 逻辑视图 (Logical View)

逻辑视图描述了系统的功能分解和模块关系，展示了 TorchTitan 的核心抽象和组件组织。

### 1.1 核心模块层次结构

```text
TorchTitan 逻辑架构
├── 训练协调层 (Training Orchestration)
│   ├── Trainer - 主训练循环和状态管理
│   └── ConfigManager - 配置管理和验证
├── 并行化层 (Parallelization)
│   ├── ParallelDims - 并行维度管理
│   ├── FSDP2 - 全分片数据并行
│   ├── Tensor Parallel - 张量并行
│   ├── Pipeline Parallel - 流水线并行
│   ├── Context Parallel - 上下文并行
│   └── Expert Parallel - 专家并行 (MoE)
├── 模型层 (Model)
│   ├── ModelProtocol - 模型接口协议
│   ├── Llama3/Llama4 - 模型实现
│   ├── DeepSeek V3 - MoE 模型
│   └── Attention - 注意力机制
├── 训练组件层 (Training Components)
│   ├── DataLoader - 数据加载
│   ├── Optimizer - 优化器容器
│   ├── LRScheduler - 学习率调度
│   ├── Loss - 损失函数
│   ├── Metrics - 指标处理
│   ├── Checkpoint - 检查点管理
│   └── Validator - 验证器
└── 基础设施层 (Infrastructure)
    ├── Logging - 日志系统
    ├── Profiling - 性能分析
    ├── FaultTolerance - 容错管理
    └── Utils - 工具函数
```

### 1.2 关键抽象和接口

### 1.2.1 TrainSpec 协议

TrainSpec 是 TorchTitan 的核心协议，定义了训练规范：

```text
@dataclass
class TrainSpec:
    model_cls: type[ModelProtocol]                    # 模型类
    model_args: Mapping[str, BaseModelArgs]          # 模型参数映射
    parallelize_fn: ParallelizeFunction               # 并行化函数
    pipelining_fn: PipeliningFunction                 # 流水线函数
    build_optimizers_fn: OptimizersBuilder            # 优化器构建器
    build_lr_schedulers_fn: LRSchedulersBuilder       # 学习率调度器
    build_dataloader_fn: DataLoaderBuilder            # 数据加载器
    build_tokenizer_fn: TokenizerBuilder              # 分词器
    build_loss_fn: LossFunctionBuilder                # 损失函数
    build_validator_fn: ValidatorBuilder              # 验证器
    build_metrics_processor_fn: MetricsProcessorBuilder # 指标处理器
    state_dict_adapter: type[BaseStateDictAdapter]    # 状态字典适配器
```

**设计优势**：

- 解耦训练逻辑和模型实现
- 支持多模型架构（Llama、DeepSeek、Qwen 等）
- 可扩展的构建器模式

### 1.2.2 ParallelDims 并行维度

ParallelDims 管理多维并行配置：

```text
@dataclass
class ParallelDims:
    dp_replicate: int  # 数据并行复制维度
    dp_shard: int      # 数据并行分片维度 (FSDP)
    cp: int            # 上下文并行维度
    tp: int            # 张量并行维度
    pp: int            # 流水线并行维度
    ep: int            # 专家并行维度 (MoE)
    etp: int           # 专家内张量并行维度
    world_size: int    # 总进程数
```

**约束条件**：

```text
world_size = dp_replicate × dp_shard × cp × tp × pp
```

**示例配置**：

- Llama 3 8B (8 GPU): `dp_shard=8, tp=1, pp=1`
- Llama 3 70B (64 GPU): `dp_shard=2, tp=4, pp=8`
- DeepSeek V3 MoE (128 GPU): `dp_shard=2, tp=4, pp=4, ep=4`

### 1.3 模块依赖关系

```text
┌─────────────────────────────────────────────────────────────┐
│                         train.py                             │
│                    (主训练循环 - 745 行)                        │
└───────────────────┬─────────────────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
┌───────▼───────┐      ┌───────▼────────┐
│  TrainSpec    │      │  ParallelDims  │
│  (协议层)      │      │  (并行管理)     │
│  88 行         │      │  260 行        │
└───────┬───────┘      └───────┬────────┘
        │                      │
        │                      │
┌───────▼───────────────────────▼─────────┐
│          distributed/                    │
│  - parallel_dims.py (260 行)             │
│  - tensor_parallel.py (30 行)            │
│  - pipeline_parallel.py (476 行)         │
│  - activation_checkpoint.py (391 行)     │
│  - expert_parallel.py (365 行)           │
└──────────────────┬───────────────────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
┌───────▼────────┐   ┌───────▼────────┐
│  components/   │   │   models/      │
│  - optimizer   │   │   - llama3     │
│  - checkpoint  │   │   - llama4     │
│  - dataloader  │   │   - deepseek   │
│  - metrics     │   │   - attention  │
└────────────────┘   └────────────────┘
```

## 2. 过程视图 (Process View)

过程视图描述了运行时的进程、线程结构和通信流程。

### 2.1 进程拓扑结构

### 2.1.1 设备网格 (DeviceMesh) 构建

TorchTitan 使用多维设备网格组织分布式进程：

```text
设备网格示例 (64 GPU 训练 Llama 3 70B):
配置: pp=8, dp_replicate=1, dp_shard=2, tp=4

┌─────────────────────────────────────────────────────────────┐
│  Pipeline Stage 0 (Node 0-1, 8 GPUs)                        │
│  ├─ DP Shard 0: TP Group [GPU 0-3]                          │
│  └─ DP Shard 1: TP Group [GPU 4-7]                          │
├─────────────────────────────────────────────────────────────┤
│  Pipeline Stage 1 (Node 2-3, 8 GPUs)                        │
│  ├─ DP Shard 0: TP Group [GPU 8-11]                         │
│  └─ DP Shard 1: TP Group [GPU 12-15]                        │
├─────────────────────────────────────────────────────────────┤
│  Pipeline Stage 2-7 (继续...)                                │
└─────────────────────────────────────────────────────────────┘

进程组层次:
World Group (64 processes)
├── PP Group (8 stages × 8 ranks/stage)
├── DP Group (2 shards × 32 ranks/shard)
└── TP Group (4 GPUs/group × 16 groups)
```

### 2.1.2 Rank 到 GPU 的映射

```text
# 每个进程的身份确定
rank = int(os.environ['RANK'])              # Global rank: 0-63
local_rank = int(os.environ['LOCAL_RANK'])  # Local rank: 0-7
world_size = int(os.environ['WORLD_SIZE'])  # Total: 64

# 通过 DeviceMesh 获取各维度的 rank
pp_rank = world_mesh["pp"].get_local_rank()     # 0-7
tp_rank = world_mesh["tp"].get_local_rank()     # 0-3
dp_rank = world_mesh["dp"].get_local_rank()     # 0-15

# 设备分配
device = torch.device(f"cuda:{local_rank}")
```

### 2.2 训练循环执行流程

### 2.2.1 初始化阶段详细流程

```text
┌────────────────────────────────────────────────────────────┐
│  阶段 1: 分布式环境初始化 (约 5-10 秒)                       │
├────────────────────────────────────────────────────────────┤
│  1. init_process_group(backend="nccl")                    │
│     ├─ 设置 MASTER_ADDR/MASTER_PORT                       │
│     ├─ 初始化 NCCL 通信后端                                │
│     └─ 建立进程间连接                                       │
│                                                            │
│  2. build_device_mesh()                                    │
│     ├─ 构建多维网格: [pp, dp_replicate, dp_shard, tp]     │
│     ├─ 创建子网格 (dp, dp_cp, ep 等)                       │
│     └─ 初始化所有进程组                                     │
└────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│  阶段 2: 模型构建 (约 10-30 秒)                             │
├────────────────────────────────────────────────────────────┤
│  1. Meta 设备初始化                                         │
│     with torch.device("meta"):                            │
│         model = LlamaModel(model_args)                    │
│     # 此时模型占用 0 字节内存                               │
│                                                            │
│  2. 应用模型转换器                                          │
│     ├─ Float8 量化转换 (可选)                              │
│     ├─ 自定义层替换 (可选)                                 │
│     └─ 模型结构修改 (可选)                                 │
│                                                            │
│  3. Pipeline Parallel 切分 (如果启用)                      │
│     ├─ 计算每阶段的层分配                                   │
│     ├─ 复制模型并删除不属于当前阶段的层                      │
│     └─ 创建 PipelineStage 对象                            │
│                                                            │
│  4. Tensor Parallel 应用                                   │
│     ├─ 列并行: nn.Linear → ColwiseParallel                │
│     ├─ 行并行: nn.Linear → RowwiseParallel                │
│     └─ Sequence Parallel (可选)                           │
│                                                            │
│  5. FSDP2 包装                                             │
│     ├─ 按层应用 fully_shard()                              │
│     ├─ 设置分片策略 (FULL_SHARD/HYBRID_SHARD)             │
│     └─ 配置混合精度策略                                     │
│                                                            │
│  6. 参数实体化到 GPU                                        │
│     ├─ DTensor 分布式初始化                                │
│     ├─ 参数分片到各 GPU                                    │
│     └─ 内存分配: ~60GB/GPU (70B 模型)                      │
└────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│  阶段 3: 优化器和数据加载器初始化 (约 5-15 秒)               │
├────────────────────────────────────────────────────────────┤
│  1. 构建优化器                                              │
│     ├─ AdamW(model.parameters(), lr=3e-4)                 │
│     ├─ 优化器状态分片 (FSDP)                               │
│     └─ 内存分配: ~35GB/GPU (优化器状态)                    │
│                                                            │
│  2. 构建学习率调度器                                        │
│     └─ WarmupStableDecayScheduler                         │
│                                                            │
│  3. 构建数据加载器                                          │
│     ├─ 设置 DP rank 和 world size                         │
│     ├─ 分配数据分片                                         │
│     └─ 启动数据预取                                         │
│                                                            │
│  4. 初始化检查点管理器                                       │
│     └─ 配置 DCP 保存/加载路径                              │
│                                                            │
│  5. 初始化指标处理器                                        │
│     └─ TensorBoard/WandB 连接                             │
└────────────────────────────────────────────────────────────┘
```

### 2.2.2 训练迭代详细时序

```text
单步训练详细分解 (Llama 3 70B, 64 GPU, 微批次=4):

时间 (ms)  操作                          通信                    内存
─────────────────────────────────────────────────────────────────
0          获取批次数据                   无                     +2GB
           └─ DataLoader.next()

20         微批次 0: 前向传播开始          
           ├─ Token Embedding
           ├─ Layer 0-9 (PP Stage 0)
           │  ├─ Attention                All-Gather (FSDP)     Peak +5GB
           │  │  └─ QKV Linear            All-Reduce (TP)
           │  └─ FFN                      All-Reduce (TP)
           └─ 激活传递到下一阶段            P2P Send (PP)

150        微批次 1: 前向传播开始
           (流水线并发)

280        微批次 2: 前向传播开始

410        微批次 3: 前向传播开始

540        所有微批次前向完成
           └─ PP 最后阶段计算损失

560        微批次 3: 反向传播开始
           ├─ Loss.backward()
           └─ 激活梯度传递                 P2P Recv (PP)

690        微批次 2: 反向传播
           └─ 梯度累积                     Reduce-Scatter (FSDP)

820        微批次 1: 反向传播

950        微批次 0: 反向传播完成
           └─ 所有梯度累积完成

970        梯度裁剪
           └─ clip_grad_norm_()

990        优化器步进
           ├─ optimizer.step()            All-Gather (FSDP)
           └─ 参数更新                     -3GB

1020       学习率调度
           └─ lr_scheduler.step()

1030       指标记录
           ├─ 计算吞吐量                   All-Reduce (损失)
           ├─ 记录 GPU 内存
           └─ 日志输出

1050       垃圾回收 (每 10 步)
           └─ gc.collect()

总时间: ~1050ms/步
吞吐量: ~500K tokens/秒 (64 GPU)
MFU: ~45%
```

### 2.3 通信原语和模式

### 2.3.1 FSDP2 通信模式

```text
# 前向传播: All-Gather 参数
for layer in model.layers:
    # 每层执行:
    params = all_gather(sharded_params, group=dp_shard_group)
    # 形状: [shard_size] -> [full_size]
    # 带宽: ~900 GB/s (NVLink) 或 ~200 GB/s (IB)
    
    output = layer.forward(input, params)
    
    # 立即释放全量参数
    del params  # 节省内存

# 反向传播: Reduce-Scatter 梯度
for layer in reversed(model.layers):
    grad_output = layer.backward()
    
    # 梯度规约和分片
    sharded_grad = reduce_scatter(grad_output, group=dp_shard_group)
    # 形状: [full_size] -> [shard_size]
```

### 2.3.2 Tensor Parallel 通信模式

```text
# 列并行 (Column Parallel)
class ColumnParallelLinear:
    def forward(self, input):
        # input: [batch, seq, hidden]  # 完整输入
        # weight: [hidden, hidden/tp] # 分片权重
        
        # 本地矩阵乘法 (无通信)
        output = F.linear(input, self.weight)
        # output: [batch, seq, hidden/tp]
        
        # All-Reduce 聚合输出 (如果需要)
        if self.gather_output:
            output = all_reduce(output, group=tp_group)
            # output: [batch, seq, hidden]
        
        return output

# 行并行 (Row Parallel)
class RowParallelLinear:
    def forward(self, input):
        # input: [batch, seq, hidden/tp]  # 分片输入
        # weight: [hidden/tp, hidden]     # 分片权重
        
        # All-Gather 收集输入
        if self.input_is_parallel:
            input = all_gather(input, group=tp_group)
            # input: [batch, seq, hidden]
        
        # 本地矩阵乘法
        output = F.linear(input, self.weight)
        
        return output
```

### 2.3.3 Pipeline Parallel 通信模式

```text
# Zero Bubble V Schedule 通信
class PipelineStage:
    def forward_step(self, microbatch_id):
        # 接收上一阶段的激活
        if not self.is_first:
            activation = p2p_recv(
                src=self.prev_stage_rank,
                tag=microbatch_id
            )
        
        # 本地前向计算
        output = self.model(activation)
        
        # 发送激活到下一阶段
        if not self.is_last:
            p2p_send(
                output,
                dst=self.next_stage_rank,
                tag=microbatch_id
            )
        
        return output
    
    def backward_step(self, microbatch_id):
        # 接收下一阶段的梯度
        if not self.is_last:
            grad_output = p2p_recv(
                src=self.next_stage_rank,
                tag=microbatch_id
            )
        
        # 本地反向计算
        grad_input = self.model.backward(grad_output)
        
        # 发送梯度到上一阶段
        if not self.is_first:
            p2p_send(
                grad_input,
                dst=self.prev_stage_rank,
                tag=microbatch_id
            )
```

### 2.3.4 Context Parallel Ring Attention

```text
# Ring Attention 通信模式
class RingAttention:
    def forward(self, query, key, value):
        # 序列分片到 CP 维度
        # query: [batch, seq/cp, heads, dim]
        
        output = torch.zeros_like(query)
        
        # Ring 迭代
        for step in range(cp_degree):
            # 计算当前块的注意力
            attn_out = scaled_dot_product_attention(
                query, key, value
            )
            output += attn_out
            
            # Ring 传递 KV
            send_rank = (cp_rank + 1) % cp_degree
            recv_rank = (cp_rank - 1) % cp_degree
            
            key, value = ring_exchange(
                send_tensor=(key, value),
                send_rank=send_rank,
                recv_rank=recv_rank,
                group=cp_group
            )
        
        return output
```

### 2.4 线程和异步执行模型

### 2.4.1 主训练进程线程结构

```text
主进程 (每个 GPU 一个进程)
├── 主训练线程 (Python Main Thread)
│   ├── 数据加载迭代
│   ├── 模型前向传播
│   ├── 损失计算
│   ├── 反向传播
│   ├── 优化器更新
│   └── 指标记录
│
├── CUDA 默认流 (Default Stream)
│   └── 主要计算 kernel
│
├── NCCL 通信流 (Communication Streams)
│   ├── 流 1: FSDP All-Gather
│   ├── 流 2: FSDP Reduce-Scatter  
│   ├── 流 3: TP All-Reduce
│   └── 流 4: PP P2P 通信
│
├── 异步检查点线程 (Checkpoint Thread)
│   ├── 后台状态字典序列化
│   ├── 异步文件写入
│   └── Safetensors 合并
│
├── Purge 线程 (Cleanup Thread)
│   └── 删除旧检查点
│
├── 垃圾回收线程 (GC Thread - 可选)
│   └── 定期执行 gc.collect()
│
└── 数据预取线程 (DataLoader Workers)
    ├── Worker 1: 数据读取
    ├── Worker 2: 数据读取
    ├── Worker 3: 数据读取
    └── Worker 4: 数据读取
```

### 2.4.2 异步 Tensor Parallel 执行

```text
# Async TP 实现
class AsyncTensorParallel:
    def __init__(self):
        self.comm_stream = torch.cuda.Stream()
        self.compute_stream = torch.cuda.current_stream()
    
    def all_reduce_async(self, tensor):
        # 在通信流上启动 All-Reduce
        with torch.cuda.stream(self.comm_stream):
            handle = dist.all_reduce(
                tensor,
                group=self.tp_group,
                async_op=True  # 非阻塞
            )
        
        # 主流继续计算 (重叠通信)
        return handle
    
    def forward(self, x):
        # 阶段 1: 启动通信
        comm_handle = self.all_reduce_async(x)
        
        # 阶段 2: 并行计算其他操作
        y = self.compute_other_ops()
        
        # 阶段 3: 等待通信完成
        comm_handle.wait()
        
        # 阶段 4: 使用通信结果
        z = x + y
        return z
```

### 2.5 内存管理和优化策略

### 2.5.1 GPU 内存分配时序

```text
Llama 3 70B 单 GPU 内存使用 (TP=4, FSDP=2):

初始化:
├─ 模型参数: 18GB  (70B / 4 TP / 2 FSDP)
├─ 优化器状态: 36GB (2 × 参数大小, Adam)
└─ 梯度缓冲: 18GB
   总计: 72GB

训练中峰值:
├─ 基线: 72GB
├─ 激活 (前向): +8GB  
├─ FSDP All-Gather 临时: +18GB (瞬时)
└─ 峰值: ~98GB → 优化后 80GB

优化技术:
1. 激活检查点: -6GB  (重计算激活)
2. CPU Offload: -36GB (优化器状态到 CPU)
3. Flash Attention: -2GB (减少注意力内存)
```

### 2.5.2 激活检查点策略

```text
# 选择性激活检查点
def apply_activation_checkpointing(
    model: nn.Module,
    checkpoint_mode: str
):
    if checkpoint_mode == "selective":
        # 只对 Attention 和 FFN 应用检查点
        for layer in model.layers:
            checkpoint_wrapper(
                layer.attention,
                checkpoint_impl="torch.utils.checkpoint"
            )
            checkpoint_wrapper(
                layer.feed_forward,
                checkpoint_impl="torch.utils.checkpoint"
            )
    
    elif checkpoint_mode == "full":
        # 全层检查点
        for layer in model.layers:
            checkpoint_wrapper(layer)
    
    # 内存节省: ~30-50%
    # 计算开销: +20-30% (重计算)
```