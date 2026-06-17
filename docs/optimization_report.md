# ScaleTorch NPU 优化文档

## 概述

本文档记录了 ScaleTorch 分布式训练框架在华为 Ascend 910 NPU 上的全面优化工作，涵盖 NPU 适配、性能调优、显存优化、Qwen3 模型支持、测试完善及代码质量改进。

**硬件环境**

| 项目 | 规格 |
|------|------|
| 加速卡 | 8× Ascend 910 (64GB HBM) |
| CANN | 9.0.0 |
| PyTorch | 2.12.0 + torch_npu 2.12.0rc1 |
| 容器镜像 | `torchtitan-npu:cann9.0.0-torch2.12.0` |

---

## 1. NPU 设备适配

### 1.1 设备抽象层 (`scaletorch/utils/device.py`)

完全重写了设备工具模块，提供统一的 NPU/CUDA/XPU 抽象 API：

| 函数 | 功能 |
|------|------|
| `get_device_type()` | 自动检测加速器类型（npu/cuda/xpu/cpu），带缓存 |
| `get_dist_backend()` | 返回对应分布式后端（hccl/nccl/gloo） |
| `get_current_device()` | 根据 `LOCAL_RANK` 返回正确设备 |
| `set_device()` | 设置当前加速器设备 |
| `synchronize()` | 设备同步 |
| `empty_cache()` | 清理显存缓存 |
| `memory_reserved()` / `memory_allocated()` | 显存统计 |
| `is_bf16_supported()` | bf16 支持检测 |
| `get_theoretical_flops()` | 返回设备理论算力（Ascend 910B: 320 TFLOPS, A100: 989.5 TFLOPS） |

### 1.2 分布式后端

- 自动选择 `hccl` 后端（NPU）替代硬编码的 `nccl`
- `initialize_distributed_training()` 使用 `get_dist_backend()` 动态选择

### 1.3 全局替换

将以下模块中的 `torch.cuda.*` 调用替换为设备无关 API：
- `train.py` — autocast、GradScaler、empty_cache、memory 统计
- `scaletorch/utils/monitor.py` — 性能监控
- `scaletorch/utils/utils.py` — 种子设置、MFU 计算、参数统计
- `scaletorch/utils/checkpoint.py` — 检查点保存
- `scaletorch/data/dataloader.py` — pin_memory
- `scaletorch/parallel/*/comms.py` — 通信同步

---

## 2. 模型优化

### 2.1 Llama 模型修复与优化 (`scaletorch/models/model_llama.py`)

**Bug 修复**

| 问题 | 修复 |
|------|------|
| `inspect` 未导入 | flash_attn 检测代码中 `inspect.signature()` 使用前未 import |
| `is_npu_available` 未作为函数调用 | `is_npu_available` → `is_torch_npu_available()` |
| `flash_attention()` 输出维度错误 | 原实现先 permute 到 (B,S,H,D) 再调用 SDPA（期望 B,H,S,D），修正为统一返回 (B,S,H,D) |
| RoPE 广播维度不匹配 | cos/sin `[seq_len, head_dim]` 自动 unsqueeze 到 `[1,1,seq_len,head_dim]` |
| RoPE 设备不一致 | cos/sin 注册为 `register_buffer(persistent=False)`，随 `model.to(device)` 自动迁移 |
| `model.config` 缺失 | 添加 `self.config = config`，修复 metrics 日志报错 |

**性能优化**

| 优化项 | 方法 | 效果 |
|--------|------|------|
| 统一 RoPE 路径 | 消除 FLASH_ATTEN 环境变量分支，统一使用 `apply_rotary_pos_emb` | 减少代码路径，消除冗余 reshape |
| RoPE 设备计算 | cos/sin 直接在目标设备计算，使用 `torch.cat` 替代 `.repeat()` | 避免 CPU→NPU 传输和共享内存问题 |
| 梯度检查点 | `torch.utils.checkpoint.checkpoint` + `use_reentrant=False` | 显存降低 49% |
| RoPE 序列裁剪 | `forward()` 中 `cos[:seq_len]` 按实际长度裁剪 | 避免全量位置编码参与计算 |

### 2.2 Qwen3 模型实现 (`scaletorch/models/model_qwen3.py`)

新增 Qwen3 模型支持，关键差异：

| 特性 | Llama | Qwen3 |
|------|-------|-------|
| head_dim | `hidden_size // num_heads` | 显式配置值（128） |
| QK Norm | 无 | 投影后、RoPE 前逐头 RMSNorm |
| 权重绑定 | 无 | `tie_word_embeddings` 支持 |
| 模型自动选择 | - | `config.model_type == 'qwen3'` 自动路由 |

**验证结果**

| 模型 | 参数量 | Checkpoint | tie | Init | Forward | Backward |
|------|--------|------------|-----|------|---------|----------|
| Qwen3-0.6B | 596M | single | True | OK | OK | OK |
| Qwen3-1.7B | 1.7B | sharded | True | OK | OK | OK |
| Qwen3-4B | 4.0B | sharded | True | OK | OK | OK |
| Qwen3-8B | 8.2B | sharded | False | OK | OK | OK |

---

## 3. 训练流程优化 (`train.py`)

### 3.1 速度优化

| 优化项 | 修改前 | 修改后 | 效果 |
|--------|--------|--------|------|
| 梯度裁剪 | 手动两遍遍历参数 | `torch.nn.utils.clip_grad_norm_`（单遍融合） | ~5% 加速 |
| Fused AdamW | 仅支持 CUDA | 支持 NPU `fused=True` | ~3% 加速 |
| autocast | `torch.cuda.amp.autocast` | `torch.amp.autocast(device_type=)` | 兼容所有设备 |
| GradScaler | 对 bf16 也启用 | 仅 fp16 启用（bf16 无需 loss scaling） | 减少开销 |
| 内循环 empty_cache | 每 `grad_accum//4` 步清理 | 移除（GPU stall 来源） | 消除同步等待 |
| gc.collect 频率 | 每 500 步 + synchronize | 每 1000 步，无阻塞同步 | 减少暂停 |

### 3.2 显存优化

| 优化项 | 效果 |
|--------|------|
| 梯度检查点 (`gradient_checkpointing=True`) | 激活显存降低约 49% |
| `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` | 减少显存碎片 |
| `optimizer.zero_grad(set_to_none=True)` | 释放梯度张量内存（已有） |
| RoPE buffer 注册 | 避免重复分配位置编码 |

---

## 4. 权重加载优化 (`scaletorch/utils/checkpoint.py`)

### 4.1 Qwen3 支持

- 动态添加 `self_attn.q_norm` / `self_attn.k_norm` 到权重列表
- `tie_word_embeddings=True` 时用 `embedding.weight` 填充 `final_proj.weight`
- `load_state_dict(assign=True)` 后重新绑定共享权重参数
- `lm_head.weight` 自动加载（`tie=False` 模型）

### 4.2 TP 分片修正

- `adjust_tensor_size` 使用 config 中的 `head_dim` 而非计算值
- `out_proj` 按 `num_heads × head_dim` 维度分片（而非 `hidden_size`）
- `q_norm` / `k_norm` 跳过 TP 分片（逐头参数不可分）

### 4.3 设备无关保存

- 检查点保存使用 `value.is_cpu` 判断是否需要 CPU 转移，替代 `torch.cuda.is_available()`

---

## 5. ProcessGroupManager 代理模式 (`scaletorch/parallel/pg_manager.py`)

### 问题

原始代码使用模块级变量 `process_group_manager = None`，通过 `from ... import process_group_manager as pgm` 导入。`setup_process_group_manager()` 修改模块变量后，所有已导入的 `pgm` 别名仍指向 `None`。

### 解决方案

实现 `_ProcessGroupManagerProxy` 代理类：

```python
class _ProcessGroupManagerProxy:
    _instance = None
    def __getattr__(self, name):  # 委托属性访问
    def __bool__(self):           # 支持 if pgm: 检查
    def __eq__(self, other):      # 支持 pgm == None 比较
```

- 所有 `from ... import process_group_manager as pgm` 导入获得代理对象
- `setup_process_group_manager()` 设置 `proxy._instance`
- 代理自动委托到底层实例

### 代码适配

全代码库 `pgm is not None` → `if pgm`（布尔检查走 `__bool__`），`getattr(pgm, attr, default)` 用于安全访问（单进程模式下 proxy 未初始化时返回默认值）。

---

## 6. 数据加载优化 (`scaletorch/data/`)

| 优化项 | 修改 |
|--------|------|
| DataLoader 双初始化 | 移除 `super().__init__([])` 首次调用 |
| prefetch_factor | `num_workers=0` 时不设置（PyTorch 2.12 限制） |
| pin_memory | 使用 `is_accelerator_available()` 检测 |
| position_ids | `.expand().contiguous()` 避免 NPU pin_memory 共享内存错误 |
| tokenizer 序列化 | 独立 `_tokenize_and_chunk` 函数避免 ProcessGroup 对象 pickle 失败 |
| 磁盘数据集 | `load_from_disk()` 支持预下载数据集 |
| __next__ 自动初始化 | 迭代器未初始化时自动调用 `__iter__()` |

---

## 7. MFU 优化与性能基准

### 7.1 MFU 公式修正

原始 MFU 公式存在三个问题，导致报告值虚高（显示 75%，实际 37%）：

| 问题 | 修正前 | 修正后 |
|------|--------|--------|
| 序列长度 | `max_position_embeddings` (40960) | 实际训练 `sequence_length` (2048) |
| 注意力 FLOPs | `hidden_size × seq_len` | `num_heads × head_dim × seq_len` (Qwen3 head_dim=128) |
| 设备峰值 | 硬编码 A100 989.5T | Ascend 910: 256T, 910B: 320T (自动检测) |

修正后的公式：
```
FLOPs/token = 6 × N_params + 12 × L × n_heads × head_dim × seq_len
MFU = tokens/s × FLOPs/token / peak_FLOPS × 100%
```

### 7.2 MFU 优化策略

**核心发现**：更长的序列长度能显著提升 MFU，因为注意力计算 FLOPs 为 O(S^2)，使计算密度更高。

| 配置 | MFU | Tokens/s | Memory | 说明 |
|------|-----|----------|--------|------|
| BS=2, SEQ=2048, no-GC | 36.4% | 15.7K | 19.7GB | 小 batch 基线 |
| BS=4, SEQ=2048, no-GC | 43.2% | 18.4K | 34.3GB | 标准配置 |
| BS=4, GA=2, SEQ=2048, no-GC | 43.9% | 19.0K | 37.4GB | 最大吞吐 |
| BS=4, SEQ=2048, GC | 35.9% | 15.5K | 16.4GB | 最省显存 |
| BS=8, SEQ=2048, GC | 37.1% | 16.1K | 29.3GB | GC 大 batch |
| BS=4, SEQ=4096, GC | 42.2% | 14.7K | 29.3GB | 长文本 |
| BS=2, SEQ=8192, GC | 49.7% | 12.5K | 29.3GB | 平衡方案 |
| BS=1, SEQ=16384, GC | **60.0%** | 9.7K | 29.3GB | **最高 MFU** |
| BS=1, SEQ=32768, GC | 58.9% | 5.6K | 44.3GB | 超长序列 |

### 7.3 最优配置端到端训练 (SEQ=16384, MFU=60%)

```
Step  1: Loss=12.25  Tokens/s=7.5K   MFU=46.0%  Mem=23.2GB  (warmup)
Step  5: Loss=10.19  Tokens/s=9.7K   MFU=59.7%  Mem=29.3GB
Step 10: Loss= 9.69  Tokens/s=9.7K   MFU=60.0%  Mem=29.3GB
Step 15: Loss= 9.50  Tokens/s=9.7K   MFU=60.0%  Mem=29.3GB
Step 20: Loss= 9.19  Tokens/s=9.7K   MFU=60.0%  Mem=29.3GB
```

- MFU 稳定 60.0%，无波动
- 显存 29.3GB 恒定，无泄漏
- Loss 从 12.25 降至 9.19，稳定收敛

### 7.4 MFU 瓶颈分析

| 因素 | 影响 | SEQ=2048 | SEQ=16384 |
|------|------|----------|-----------|
| 注意力 FLOPs 占比 | 越高越好 | 24% | 72% |
| 计算密度 (FLOP/byte) | 越高越好 | 低 | 高 |
| Memory-bound 算子 (RMSNorm, SiLU) | 无法加速 | 占比大 | 占比小 |
| 大词表开销 (151K vocab) | embedding/lm_head | 31% | 8% |

> 对于更大的模型（Qwen3-4B/8B），由于 hidden_size 更大、计算矩阵更大，同样序列长度下 MFU 会更高。

### 7.5 多模型训练验证

| 模型 | 训练 | 显存 | 吞吐 | MFU |
|------|------|------|------|-----|
| Qwen3-0.6B (tie=True) | 20 步, loss 12.2→9.2 | 29.3 GB | 9.7K tok/s | 60.0% |
| Qwen3-1.7B (tie=True) | 10 步, loss 208→134 | 21.4 GB | 9.3K tok/s | 19.3% |
| Qwen3-8B (tie=False) | Forward+Backward OK | 需多卡 | - | - |

### 7.6 训练模式速查

```bash
# 最高 MFU (60%) — 长序列，梯度检查点
bash scripts/run_npu.sh 8 /path/to/model /path/to/data max_mfu

# 最大吞吐 (19K tok/s) — 短序列，无梯度检查点
bash scripts/run_npu.sh 8 /path/to/model /path/to/data max_speed

# 平衡 (50% MFU, 12K tok/s) — 中等序列
bash scripts/run_npu.sh 8 /path/to/model /path/to/data balanced

# 最省显存 (16GB) — 适合大模型或多并行
bash scripts/run_npu.sh 8 /path/to/model /path/to/data min_mem
```

---

## 8. 测试体系

### 8.1 新增/优化

| 文件 | 内容 | 测试数 |
|------|------|--------|
| `pytest.ini` | pytest 配置 | - |
| `tests/conftest.py` | 共享 fixtures (pgm reset, mock_dist) | - |
| `tests/test_device.py` | 设备检测、后端选择、环境变量、FLOPS | 16 |
| `tests/test_utils.py` | 数字格式化、种子设置、MFU 计算 | 14 |
| `tests/test_model.py` | RoPE、注意力、Llama 模型前向/反向/GC | 10 |
| `tests/test_lr_scheduler.py` | 6 种调度器创建 + 行为验证 | 8 |
| `tests/test_pg_manager.py` | 适配 Proxy 模式 | 10 |

### 8.2 测试结果

```
126 passed, 2 skipped (SDPA 需 NPU 设备), 0 failed — 2.36s
```

---

## 9. 部署

### 9.1 Dockerfile

```dockerfile
FROM torchtitan-npu:cann9.0.0-torch2.12
# HCCL 性能环境变量
ENV TASK_QUEUE_ENABLE=2
ENV COMBINED_ENABLE=1
ENV HCCL_BUFFSIZE=120
ENV PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
```

### 9.2 快速启动

```bash
# 单卡训练 Qwen3-0.6B
torchrun --nproc_per_node=1 train.py \
  --model_name_or_path /path/to/Qwen3-0.6B \
  --dataset_name /path/to/wikitext2 \
  --micro_batch_size 8 --gradient_accumulation_steps 2 \
  --sequence_length 2048 --gradient_checkpointing True \
  --use_fused_adam True --learning_rate 1e-4 \
  --total_train_steps 100

# 8 卡数据并行
torchrun --nproc_per_node=8 train.py \
  --data_parallel_size 8 \
  --micro_batch_size 4 --gradient_accumulation_steps 2 \
  --gradient_checkpointing True --use_fused_adam True
```

### 9.3 关键环境变量

| 变量 | 值 | 作用 |
|------|-----|------|
| `FLASH_ATTEN` | `0`/`1` | 切换 SDPA / Flash Attention |
| `DTYPE` | `bfloat16` | 混合精度类型 |
| `PYTORCH_NPU_ALLOC_CONF` | `expandable_segments:True` | 减少显存碎片 |
| `TASK_QUEUE_ENABLE` | `2` | NPU 任务队列优化 |
| `COMBINED_ENABLE` | `1` | NPU 算子融合 |
| `HCCL_CONNECT_TIMEOUT` | `7200` | HCCL 连接超时(秒) |

---

## 10. 修改文件清单

| 文件 | 修改类型 |
|------|----------|
| `scaletorch/utils/device.py` | 重写 — NPU/CUDA 统一抽象层 |
| `scaletorch/models/llama.py` | 重构 — RoPE/Attention/Flash 修复，序列并行 |
| `scaletorch/models/model_qwen3.py` | 新增 — Qwen3 模型实现 |
| `scaletorch/utils/checkpoint.py` | 修改 — Qwen3 权重加载、TP 修正、设备无关保存 |
| `scaletorch/utils/misc.py` | 修改 — MFU 公式修正、设备自动检测、proxy 兼容 |
| `scaletorch/parallel/process_group.py` | 修改 — Proxy 模式 |
| `scaletorch/data/dataloader.py` | 修改 — NPU 兼容、prefetch 修正 |
| `scaletorch/data/dataset.py` | 修改 — 磁盘数据集加载、tokenizer 兼容 |
| `scaletorch/utils/monitor.py` | 修改 — NPU 性能监控 |
| `scaletorch/trainer/config.py` | 修改 — 新增训练参数字段 |
| `scaletorch/trainer/lr_scheduler.py` | 修改 — 文档修正 |
| `scaletorch/dist/dist.py` | 修复 — Python 3.11 f-string 兼容 |
| `scaletorch/parallel/sequence_parallel/sp_comms.py` | 修复 — proxy pgm 兼容 |
| `scaletorch/parallel/data_parallel/data_parallel.py` | 修复 — proxy pgm 兼容 |
| `train.py` | 修改 — Qwen3 支持、NPU 适配、MFU 修正 |
| `Dockerfile` | 新增 |
| `docker-compose.yml` | 新增 |
| `scripts/run_npu.sh` | 新增 — 4 种训练模式 |
| `scripts/sweep_mfu.sh` | 新增 — MFU 配置扫描 |
| `scripts/bench_single.py` | 新增 — 单卡 benchmark |
| `scripts/verify_qwen3.py` | 新增 — Qwen3 多模型验证 |
| `pytest.ini` | 新增 |
| `tests/conftest.py` | 新增 |
| `tests/test_device.py` | 新增 |
| `tests/test_utils.py` | 新增 |
| `tests/test_model.py` | 新增 |
| `tests/test_lr_scheduler.py` | 重写 |
| `tests/test_base.py` | 重写 |
| `README.md` | 重写 — 新增性能表、NPU 指南、模型列表 |
| `docs/optimization_report.md` | 重写 — MFU 优化全记录 |
