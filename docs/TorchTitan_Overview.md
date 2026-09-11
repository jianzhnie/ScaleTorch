# TorchTitan 代码库深度分析

> TorchTitan 是 Meta 开源的 **PyTorch 原生大规模生成式 AI 训练平台**，面向大语言模型与多模态模型的预训练 / 微调 / 后训练。核心理念是：**一切并行能力都来自 PyTorch 自身**（`torch.distributed`、DTensor、`fully_shard`、`torch.compile`），TorchTitan 只负责把它们组装成一个可读、可扩展的最小实现。
>
> 本文合并了"架构视图"（4+1 视图）与"代码库分析"（源码级拆解）两份文档，并以当前代码为准做了全面校正。**文中所有行数、类名、API 签名均取自当前 `main` 分支实际代码**。

---

## 目录

- [第一部分：总览](#第一部分总览)
  - [1. 项目定位与设计原则](#1-项目定位与设计原则)
  - [2. 目录结构](#2-目录结构)
  - [3. 端到端数据流与调用链](#3-端到端数据流与调用链)
- [第二部分：核心抽象](#第二部分核心抽象)
  - [4. Configurable 模式：Config → build()](#4-configurable-模式config--build)
  - [5. Module 协议：声明式分片的核心](#5-module-协议声明式分片的核心)
  - [6. 配置系统：CLI、config_registry 与 override](#6-配置系统cliconfig_registry-与-override)
- [第三部分：模型层](#第三部分模型层)
  - [7. 模型继承体系与共享组件](#7-模型继承体系与共享组件)
  - [8. 已支持的模型家族](#8-已支持的模型家族)
- [第四部分：分布式并行系统](#第四部分分布式并行系统)
  - [9. DeviceMesh：所有并行的地基](#9-devicemesh所有并行的地基)
  - [10. 三种 SPMD 后端](#10-三种-spmd-后端)
  - [11. ShardingConfig：声明式分片](#11-shardingconfig声明式分片)
  - [12. FSDP2 数据并行](#12-fsdp2-数据并行)
  - [13. Tensor Parallel](#13-tensor-parallel)
  - [14. Context Parallel](#14-context-parallel)
  - [15. Pipeline Parallel](#15-pipeline-parallel)
  - [16. Expert Parallel](#16-expert-parallel)
  - [17. 并行化应用顺序与原因](#17-并行化应用顺序与原因)
- [第五部分：训练组件与运行时](#第五部分训练组件与运行时)
  - [18. Trainer 初始化](#18-trainer-初始化)
  - [19. 训练循环逐步拆解](#19-训练循环逐步拆解)
  - [20. 数据管道](#20-数据管道)
  - [21. 损失函数](#21-损失函数)
  - [22. 优化器与学习率调度](#22-优化器与学习率调度)
  - [23. 检查点](#23-检查点)
  - [24. 激活重计算](#24-激活重计算)
  - [25. torch.compile 与 CUDA Graph](#25-torchcompile-与-cuda-graph)
  - [26. 分布式工具函数](#26-分布式工具函数)
  - [27. 可观测性](#27-可观测性)
- [第六部分：运行时视图](#第六部分运行时视图)
  - [28. 进程拓扑与线程模型](#28-进程拓扑与线程模型)
  - [29. 通信原语与内存管理](#29-通信原语与内存管理)
- [第七部分：工程实践](#第七部分工程实践)
  - [30. 实验性功能](#30-实验性功能)
  - [31. 测试、验证与数值正确性](#31-测试验证与数值正确性)
- [附录：常见误区与勘误](#附录常见误区与勘误)

---

## 第一部分：总览

### 1. 项目定位与设计原则

TorchTitan 的目标是成为"PyTorch 原生扩展技术的最小化清洁实现"：任何一项新的分布式训练技术，都应该能在这里被读一遍、改一行、跑起来。

**五条设计原则**（同时是代码审查的判据）：

| 原则 | 含义 | 在代码中的体现 |
|------|------|----------------|
| **PyTorch 原生** | 核心不依赖非 PyTorch 库 | 并行原语全部来自 `torch.distributed`；数据加载用 `grain` 但可替换 |
| **根因优先** | 不打补丁，先理解为什么 | 例如 `disable_fsdp_gradient_division()` 的理由写在注释里 |
| **复用优先** | 先查已有实现，不造轮子 | `models/common/` 承载全部共享组件；跨模型只保留一份 wrapper |
| **实验隔离** | `experiments/` 不污染核心 | 核心代码里没有 `if experiment_x:` 分支 |
| **保护战场代码** | 不改动已收敛的行为 | 非计算类改动必须产出 bit-wise 相同的 loss |

### 2. 目录结构

```
torchtitan/                        # 非实验核心
├── train.py                       # 入口: main() → ConfigManager → Trainer (72 行)
├── trainer.py                     # 核心 Trainer 类 (1156 行, 项目最重文件的 1/10)
│
├── config/                        # 配置系统
│   ├── configurable.py            # Configurable 基类: Config → build() (180 行)
│   ├── manager.py                 # ConfigManager: --module/--config 解析 (314 行)
│   ├── configs.py                 # TrainingConfig / ParallelismConfig / ... (407 行)
│   ├── override.py                # @override 组件替换机制 (520 行)
│   ├── function.py                # 配置工具函数 (40 行)
│   └── README.md
│
├── protocols/                     # 协议与抽象基类
│   ├── module.py                  # Module: nn.Module + Configurable + 声明式分片 (783 行)
│   ├── model.py                   # BaseModel: + preprocess_inputs / init_weights (128 行)
│   ├── model_spec.py              # ModelSpec: 模型 + 各类 builder callable (77 行)
│   ├── sharding.py                # ShardingConfig / LocalMapConfig / resolve_placements (159 行)
│   └── state_dict_adapter.py      # HF 权重互转协议 (136 行)
│
├── models/                        # 模型实现
│   ├── common/                    # 共享组件 (19 个 .py + MOE_SHARDING.md)
│   │   ├── decoder.py             # Decoder / TransformerBlock 基类
│   │   ├── decoder_sharding.py    # 通用解码器分片（colwise/rowwise/norm/GQA/FFN）
│   │   ├── attention.py           # GQAttention / FlexAttention / VarlenAttention / SDPA
│   │   ├── feed_forward.py        # FeedForward (SwiGLU)
│   │   ├── moe.py                 # GroupedExperts / RoutedExperts / TopKRouter / MoE
│   │   ├── moe_sharding.py        # MoE 分片 + MOE_SHARDING.md
│   │   ├── token_dispatcher.py    # 9 个 dispatch 类，见 §16.2
│   │   ├── embedding.py           # Embedding + weight tying
│   │   ├── rope.py / nn_modules.py / linear.py / param_init.py
│   │   ├── vision_encoder.py      # VLM 视觉编码器 (Qwen3.5 / Kimi-VL / Muse-Glimmer)
│   │   ├── multimodal.py / aux_loss.py / config_utils.py / dist_gemm.py
│   │   └── ...
│   ├── llama3/                    # 每个模型: model.py + config_registry.py
│   ├── deepseek_v3/ deepseek_v4/  #              + sharding.py + parallelize.py
│   ├── gpt_oss/ qwen3/ qwen3_5/   #              + state_dict_adapter.py
│   ├── qwen3_6/ qwen3_8/ kimi_k2_7/ kimi_k3/ muse_glimmer/
│   └── flux/                      # 图像生成（自带 trainer.py，独立管线）
│
├── distributed/                   # 分布式与并行
│   ├── parallel_dims.py           # ParallelDims + DeviceMesh 构建 (609 行)
│   ├── fsdp.py                    # FSDP2 / HSDP / MoE 双 mesh (434 行)
│   ├── tensor_parallel.py         # NoParallel 策略 (95 行)
│   ├── linear.py                  # AllGatherLinear / LinearReduceScatter 融合算子 (418 行)
│   ├── context_parallel/          # CP: api.py (prepare_context_parallel_input)
│   ├── pipeline_parallel.py       # PP 切分 + schedule 构建 (640 行)
│   ├── activation_checkpoint.py   # AC: selective / region / full / memory-budget (425 行)
│   ├── spmd_types.py              # spmd_types 后端胶水层 (526 行)
│   ├── compile.py                 # torch.compile + regional inductor (151 行)
│   ├── cudagraph.py               # CUDA Graph 捕获与重放 (434 行)
│   ├── utils.py                   # 梯度裁剪 / 确定性 / 初始化 / 通信原语 (740 行)
│   ├── flex_shard/                # FlexShard + DistMuon（可独立成包）
│   ├── deepep/                    # DeepEP / HybridEP
│   └── minimal_async_ep/          # 轻量异步 EP
│
├── components/                    # 训练组件（注意: 没有 components/__init__.py）
│   ├── checkpointer/              # BaseCheckpointManager + DCP / torch_checkpointing
│   ├── data/                      # GrainDataLoader + dataset/sources/packing/collators
│   ├── optimizer/                 # OptimizersContainer / LRSchedulersContainer
│   ├── quantization/              # float8 / mxfp8 / nvfp4
│   ├── loss.py                    # CrossEntropyLoss / ChunkedLossWrapper (866 行)
│   ├── metrics.py                 # MetricsProcessor + TensorBoard/WandB (590 行)
│   ├── tokenizer.py / validate.py / lora.py
│
├── hf_datasets/                   # HuggingFace 数据集加载
│   ├── text_datasets.py           # 文本预训练 / SFT
│   └── multimodal/                # 多模态数据集 + collator + image/video/text 工具
│
├── observability/
│   ├── structured_logger/         # 结构化日志 + 甘特图生成（贯穿全流程的 trace span）
│   └── sdc_replayer.py            # 静默数据损坏 (SDC) 重放检测
│
├── tools/                         # logging / profiler / utils / filesystem
├── overrides/                     # 内核级替换: fused_swiglu / fused_mla / helion_rope ...
├── ops/                           # 自定义算子 (scatter_add)
└── experiments/                   # 实验性功能，不影响核心
    ├── graph_trainer/             # FX 图级训练 + 自动并行（llama3/deepseek_v3/qwen3/muse_glimmer）
    ├── autoparallel/              # 自动并行探索
    ├── rl/                        # RL 后训练（alphabet_sort / dapo_math / search_r1 / verifiers）
    ├── torchft/                   # 容错训练
    └── transformers_modeling_backend/  # HF Transformers 后端
```

**规模估算**：核心（不含 `experiments/`）约 200 个 Python 文件、2 万行量级；其中 `trainer.py` 一个文件就占约 1100 行，是理解全局的必经之路。

### 3. 端到端数据流与调用链

先看一张全景图，后面每一节都会放大其中一个方框。

```
                        用户 CLI
   python -m torchtitan.train --module llama3 --config llama3_debugmodel
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────────┐
│ ConfigManager.parse_args()                        config/manager.py  │
│  1. 手工扫出 --module / --config，其余参数留给 tyro                    │
│  2. import torchtitan.models.llama3.config_registry                  │
│  3. 调用 llama3_debugmodel() → 得到一份完整的 Trainer.Config 默认值    │
│  4. tyro.cli(Trainer.Config, args=剩余参数, default=默认值)           │
│     └─ CLI > config_registry 默认值                                  │
│  5. _validate_config(): 检查 HF assets / tokenizer 路径               │
└──────────────────────────────┬───────────────────────────────────────┘
                               │ config
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│ Trainer.__init__(config)                          trainer.py:336     │
│                                                                      │
│  ① init_distributed()        → ParallelDims.from_config(...)          │
│  ② set_determinism()         → 种子分配 + 确定性算法                   │
│  ③ model_config.update_from_config(config)                           │
│        └─ 顺带把 sharding_config 挂到每个子模块的 Config 上            │
│  ④ apply_overrides(config.override, config)   ← @override 替换点      │
│  ⑤ with torch.device("meta"): model = model_config.build()           │
│        └─ meta 设备上建模型，内存占用 0                                │
│  ⑥ model = parallelize_fn(model, ...)  ← 或经 pipelining_fn 走 PP     │
│        └─ Module.parallelize → AC → compile → FSDP（顺序见 §17）      │
│  ⑦ model.to_empty(device); model.init_weights()                      │
│        └─ 真正分配显存并初始化参数                                     │
│  ⑧ optimizer / lr_scheduler / tokenizer / dataloader / checkpointer   │
└──────────────────────────────┬───────────────────────────────────────┘
                               │ trainer.train()
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│ train() → 主循环                                      trainer.py:1074 │
│                                                                      │
│  for step in range(...):                                             │
│    train_step():                                                     │
│      1. zero_grad()                                                  │
│      2. 取满 gradient_accumulation_steps × num_pp_microbatches 个 batch│
│      3. 统计 global_valid_tokens（DP 维 all-reduce，device 上不落 CPU）│
│      4. 每个 grad-accum group:                                        │
│           forward_backward_step()                                    │
│             ├─ preprocess_inputs()   ← CP 分片 / SPMD 标注 / mask 构建 │
│             ├─ PP ? pp_schedule.step()  : model(x)                   │
│             └─ loss_fn(pred, labels, global_valid_tokens) → backward │
│      5. clip_grad_norm_() → optimizer.step() → lr_scheduler.step()   │
│      6. 非有限 loss/grad → torch._assert_async 立即中止               │
│    metrics.log() / checkpointer.save() / validator.validate()        │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 第二部分：核心抽象

### 4. Configurable 模式：Config → build()

TorchTitan 的第一个核心模式是 **声明式构建**：每个组件都是「一个 Config 数据类 + 一个持有它的类」，二者通过 `__init_subclass__` 自动绑定。

```python
class RMSNorm(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        eps: float = 1e-5

    def __init__(self, config: Config):
        super().__init__()
        self.dim = config.dim
        self.weight = nn.Parameter(torch.ones(config.dim))

# 使用：先配置，再构建
cfg = RMSNorm.Config(dim=4096)
norm = cfg.build()            # 等价于 RMSNorm(config=replace(cfg))
```

关键机制在 `config/configurable.py`：

- **`__init_subclass__` 自动绑定**：外层类定义 `Config` 时，`config_cls._owner = cls` 自动生效，无需写 `build()`。
- **`build()` 两种模式**：
  - 无 kwargs：`self._owner(config=replace(self))` —— 纯配置构建。
  - 带 kwargs：`self._owner(config=..., **kwargs)` —— 用于 tokenizer / dataloader / optimizer 这类需要**运行时对象**（tokenizer 实例、dp_world_size 等）的组件。kwargs 与 config 字段重叠会直接报错，防止配置悄悄失效。
- **强制约束**：`__init_subclass__` 校验每个 `Config` 必须是 `@dataclass(kw_only=True, slots=True)`，且所有字段 keyword-only。这是硬性检查，不遵守直接 `TypeError`。
- **`to_dict()`**：递归序列化（含嵌套 Config、list、dict、callable 转 repr），用于结构化日志与 metrics 上报。

> **为什么用 `replace(self)` 而不是直接传 `self`？** 因为 `build()` 可能被多次调用（例如 `traverse` 找到同一 Config 后分别构建），浅拷贝避免构建方的就地修改污染原始配置树。

**`traverse()`：配置树的通用遍历**。`Configurable.Config.traverse(config_cls)` 深度优先遍历整个配置树，产出 `(fqn, config, parent, field_name)` 四元组。它是两套机制的公共基础设施：

1. `@override` 机制靠它定位要替换的节点（见 §6）。
2. `Trainer.Config.__post_init__` 的校验靠它扫描（例如「是否存在 FlexAttention」）。

`ModelSpec` 是普通 dataclass 而非 Configurable，所以它自己实现了 `traverse()`，把 `self.model` 暴露出去，让遍历能穿透到模型配置——这正是 `protocols/model_spec.py:60` 那段注释要说明的事。

### 5. Module 协议：声明式分片的核心

`Module = nn.Module + Configurable`（`protocols/module.py:44`），是所有模型组件的基类。它带来三件事：

```
nn.Module ──┐
            ├──► Module ──► BaseModel ──► Decoder ──► Llama3Model / Qwen3Model / ...
Configurable┘      │
                   ├─ init_states()          # 递归初始化参数与缓冲区
                   ├─ parallelize()          # 递归分片 + 输入输出重分布
                   ├─ _sharding_config       # 声明式分片规范（见 §11）
                   └─ _param_init            # 每参数初始化器字典
```

#### 5.1 init_states：参数初始化的统一入口

`init_states()` 分三步，**自动递归**：

1. 广度优先遍历所有 `Module` 子模块并递归调用 `init_states`。遇到非 `Module` 的普通包装器（`CheckpointWrapper`、`torch.compile` 包装）会**看进去**继续找 `Module` 后代——这一点很关键，因为 AC 和 compile 会在 `init_states` 之前或之后改变模块树形状。
2. 调用 `_init_self_parameters()`：优先级是
   `_param_init` 字典查询 → `reset_parameters()`（PyTorch 约定）→ 报错。
3. 调用 `_init_self_buffers(buffer_device=...)`：默认 no-op，RoPE / MoE 计数器等需要覆盖。

**缓冲区的一个隐蔽陷阱**：`_init_self_buffers` 常常会 `self.cache = self._precompute()` 直接重新赋值，把原本是 DTensor 的 buffer 变成普通 tensor。`init_states` 因此在调用前后做了 **save/restore**：先记下每个 DTensor buffer 的 `(device_mesh, placements)`，调用后对变成普通 tensor 的重新 `distribute_tensor` 回去。同样的 save/restore 逻辑也用在 `Module._apply()` 上，因为 `model.to_empty()` 会重建 buffer 数据、抹掉 SPMD 标注。见 `_preserve_buffer_spmd_types()`（`module.py:164`）。

#### 5.2 parallelize：递归分片的四步

`Module.parallelize(parallel_dims)` 对树中每个带 `_sharding_config` 的模块执行：

```
1. _distribute_states()      → 参数 / buffer 按 state_shardings 分发到 mesh
2. self.forward 被替换为:
     forward_with_redistribution(*args, **kwargs):
         args, kwargs = _redistribute_inputs(...)      # 输入对齐
         outputs = fn(*args, **kwargs)                 # fn = 可选的 local_map 包装
         return _redistribute_outputs(...)             # 输出还原
3. 入口处已用 _parallelized 标志位防止重复调用（二次调用直接 ValueError）
4. fully_shard 的 hooks 挂在 __call__ 上，包在这个 forward 外面
```

`_redistribute_inputs` 的逻辑（`module.py:595`）值得单独记住：

- 输入若已在 `in_src_shardings` 声明且是普通 tensor → 用 `DTensor.from_local` 包成 DTensor；
- 若已是 DTensor，**校验**其 placements 与 `in_src_shardings` 一致，不一致就报错（不静默接受）；
- 若声明了 `in_dst_shardings` 且当前 placement 不同 → `redistribute(placements=desired, async_op=True)`。

`async_op=True` 是所有 redistribute 的默认选择：把通信交给 NCCL 流异步推进，让后续可以重叠计算。

#### 5.3 ModuleList / ModuleDict / Sequential

`nn.ModuleList` 等容器不满足 `Module` 协议。TorchTitan 提供了三个等价子类（`module.py:769`）让它们同时继承 `nn` 容器与 `Module`，从而能被 `init_states` / `parallelize` 正常递归。

`BaseModel.verify_module_protocol()` 会在构建后立刻扫描整棵模块树，把不满足协议的模块连同 FQN 一起报出来——比等到 `parallelize` 阶段出现晦涩错误要友好得多。

### 6. 配置系统：CLI、config_registry 与 override

#### 6.1 ConfigManager：从命令行到配置对象

`config/manager.py` 的实现刻意避开 tyro 的直接介入，先手工解析出路由参数：

```python
python -m torchtitan.train --module llama3 --config llama3_debugmodel \
       --training.steps 100 --parallelism.tensor_parallel_degree 8
```

1. `_load_config()` 手工扫描 `--module` / `--config`（同时支持 `--module=x` 与 `--module x` 两种写法），把它们从 argv 中剥离。
2. `--module` 支持**短名**与**全限定路径**两种形式：
   - 短名：在 `torchtitan.models` → `torchtitan.experiments` → `torchtitan.experiments.rl.examples` 三个前缀下依次尝试 `import {prefix}.{name}.config_registry`。
   - 全限定：先试 `{path}.config_registry`，失败再直接 import `{path}`。
3. 从模块里取同名函数调用，得到默认配置实例。
4. `tyro.cli(config_cls, args=rest, default=loaded_config)` 把 CLI 参数合并进去，**CLI 优先级高于 config_registry 默认值**。
5. `_validate_config()` 做资产路径检查（含对旧版 `tokenizer.model` 路径的兼容告警）。

`ConfigManager` 还注册了两条 tyro 规则（`register_tyro_rules`）：

- `list[str]` 支持逗号分隔写法，如 `--compile.components model,loss`；
- `list[OverrideImport]` 支持 `target={"k": v}` 形式的 kwargs，用于 override 传参。

#### 6.2 config_registry：每个模型的门面

每个模型目录下的 `config_registry.py` 导出若干**零参或近零参函数**，每个函数返回一份完整的 `Trainer.Config`。命名惯例是 `{model}_{规模或用途}`：

```python
# torchtitan/models/llama3/config_registry.py
def llama3_debugmodel(seq_len=None) -> Trainer.Config:
    model_spec = model_registry("debugmodel", seq_len=seq_len)
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(global_vocab_size=...),
        ),
        model_spec=model_spec,
        optimizer=default_adamw(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=2, decay_ratio=0.8),
        training=TrainingConfig(num_tokens_per_microbatch_per_dp_rank=..., steps=10),
        dataloader=GrainDataLoader.Config(dataset=...),
        parallelism=ParallelismConfig(pipeline_parallel_schedule="Interleaved1F1B"),
        activation_checkpoint=SelectiveAC.Config(),
        ...
    )
```

`model_spec` 由模型目录下的 `model_registry()` 函数（定义在 `llama3/__init__.py`）按 flavor 名构建，返回 `ModelSpec`：

```python
@dataclass
class ModelSpec:
    name: str                                    # "llama3"
    flavor: str                                  # "8B" / "debugmodel"
    model: BaseModel.Config                      # 模型超参配置（含 sharding_config）
    max_context_length: int
    parallelize_fn: Callable                     # 本模型的并行化函数
    pipelining_fn: Callable | None               # PP 切分函数（不支持 PP 的模型为 None）
    post_optimizer_build_fn: Callable | None     # 优化器构建后的钩子（如 DistMuon 布局对齐）
    state_dict_adapter: type[BaseStateDictAdapter] | None
```

> **与旧文档的差异**：早年的 `TrainSpec` 把 optimizer / dataloader / loss / tokenizer / metrics 的 builder 也塞进协议里。当前版本只保留 **5 个**字段——模型**架构**相关的部分。优化器、数据、损失等一律回到 `Trainer.Config` 的顶级字段。这个收敛让「换模型」和「换训练组件」两件事彻底正交。

#### 6.3 override：不改核心代码替换组件

`config/override.py` 提供了一套**按 FQN 精准替换任意 Configurable** 的机制。override 作者写一个模块，用 `@override` 装饰器注册工厂函数，用户通过 `--override.imports` 激活：

```bash
--override.imports torchtitan.overrides.moe_token_dispatcher.hybridep_override
# 带 kwargs：
--override.imports 'my_pkg.triton_rope.triton_rope={"block_size": 256}'
```

三个设计要点：

1. **按 FQN glob 精准定位**：`@override(..., fqns=["layers.0.*"])` 可以只替换第 0 层的某个组件，做 A/B 对照（例如"只在这几层上用融合 MoE"）。
2. **冲突检测是 per-node 的**：两个 override 只有claim 同一个节点才算冲突，替换同一个 Config 类的不同实例不冲突。
3. **`exact=True`**：默认情况下替换也会匹配目标 Config 的**子类**；当替换只对具体契约有效时，加 `exact=True` 收紧匹配。

应用时机在 `Trainer.__init__` 的 ④ 步：`update_from_config` 之后、任何 `build()` 之前。因为 override 可能改任意字段，应用后会**重新跑一遍 `config.__post_init__()`** 做全量校验（`__post_init__` 只抛异常、不改状态，所以可安全重跑）。

---

## 第三部分：模型层

### 7. 模型继承体系与共享组件

#### 7.1 继承链

```
                    Module  (nn.Module + Configurable + 声明式分片)
                       │
                   BaseModel            # 增加 preprocess_inputs / init_weights / verify_module_protocol
                       │
                    Decoder             # 增加 tok_embeddings / layers / norm / lm_head 的通用结构
          ┌────────────┼────────────────────────┐
          │            │                        │
   Llama3Model   Qwen3Model   DeepSeekV3Model   ...   MuseGlimmerModel
```

`Decoder` 是**所有**自回归解码器模型的基类（`models/common/decoder.py:66`）。它约定了一套标准结构，并让 `parallelize_fn` / `apply_fsdp_to_decoder` / `apply_ac` 能够在不知道具体模型的情况下工作——这就是「模型无关」原则的落点。

#### 7.2 共享组件全景

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    Decoder（通用解码器骨架）                              │
│                                                                          │
│  tok_embeddings ──► layers[0..N-1] ──► norm ──► lm_head                  │
│   (Embedding)        (TransformerBlock)  (RMSNorm)  (Linear)             │
│                                                                          │
│  TransformerBlock（pre-norm 残差结构）:                                   │
│     x ──► attention_norm ──► Attention ──┐                               │
│           (RMSNorm)          │           ├──► + ──► h                    │
│                     x ───────┘           │                               │
│     h ──► ffn_norm ──► FeedForward / MoE ─┐                              │
│           (RMSNorm)      │                ├──► + ──► out                 │
│                     h ───┘                │                              │
│                                                                          │
│  GQAttention（分组查询注意力，QKV 投影可插拔）:                             │
│     x ──► qkv_linear ──► RoPE ──► inner_attention ──► wo                 │
│            ├─ QKVLinear（三路独立）                                        │
│            └─ FusedQKVLinear（单路融合，配合 Async-TP / dist_gemm）        │
│                          inner_attention ∈ {FlexAttention,               │
│                                             VarlenAttention,             │
│                                             ScaledDotProductAttention}   │
│                                                                          │
│  FeedForward（SwiGLU）:                                                   │
│     x ──► w1 ──► SiLU ──┐                                                │
│     x ──► w3 ───────────┼──► * ──► w2 ──► out                            │
│                                                                          │
│  MoE:                                                                    │
│     x ──► TokenChoiceTopKRouter ──► GroupedExperts(dispatch)             │
│            │                              │                              │
│            │                    ┌─────────┴──────────┐                   │
│            │                 Expert 0 ... Expert N-1 │                   │
│            │                    └─────────┬──────────┘                   │
│            └──► 加权合并 ──────────────────┘ ──► out                      │
│         配套: AuxLoss（负载均衡辅助损失），MicrobatchWiseLoadBalanceLoss    │
└──────────────────────────────────────────────────────────────────────────┘
```

**几个容易忽略的点**：

- **`GQAttention` 的 `rope=None` 表示 NoPE**（无位置编码）。这是为交错 RoPE / NoPE 的模型（如 iRoPE 类架构）按层配置预留的口子——位置信息只能通过 attention mask 传入。见 `attention.py:923` 的类文档。
- **`qk_norm` 是可选的**（Qwen3 系列使用）。字段类型 `RMSNorm.Config | None = None`，为 None 时不做 QK 归一化。
- **MoE 的专家权重挂在 `moe.routed_experts.inner_experts` 这个 grouped-GEMM 子模块上**。`apply_fsdp_to_decoder` 依赖这个路径来识别专家参数（`fsdp.py:299` 附近的注释与断言）——这是跨文件的隐式契约，改动 MoE 结构时必须同步。
- **`token_dispatcher.py` 里有 9 个 dispatch 类，构成一条继承链**（完整清单见 §16.2）。`Trainer._validate_cuda_graphs()` 会遍历配置树找到所有 `LocalTokenDispatcher.Config`，逐个检查 CUDA Graph 兼容性——**这是配置期校验，不是运行时的 isinstance 分派**。

#### 7.3 分片的声明：模型的 sharding.py

模型目录下的 `sharding.py` **只描述「这个模型长什么样」**，把通用分片规则委托给 `models/common/decoder_sharding.py`：

```python
# models/llama3/sharding.py
def set_llama3_sharding_config(config, *, enable_sp):
    set_decoder_sharding_config(config, enable_sp=enable_sp)   # tok_embeddings / norm / lm_head
    for layer_cfg in config.layers:
        norm = norm_config(enable_sp=enable_sp)
        layer_cfg.attention_norm.sharding_config = norm
        layer_cfg.ffn_norm.sharding_config = norm
        set_gqa_attention_sharding(layer_cfg.attention, enable_sp=enable_sp)
        set_gqa_inner_attention_local_map(layer_cfg.attention.inner_attention)
        set_dense_ffn_sharding(layer_cfg.feed_forward, attn_x_layout=..., enable_sp=enable_sp)
```

这些函数在 `update_from_config` 里被调用（见 `models/llama3/model.py` 的 `Llama3Model.Config.update_from_config`）。**注意分片规范是无条件填充的**：运行时真正生效哪些声明，由 `Module.parallelize()` 收到的 mesh 决定——mesh 上没有的轴，对应的声明会被跳过。所以同一个模型配置可以跑 FSDP-only，也可以跑 TP+FSDP，无需改配置。

### 8. 已支持的模型家族

`torchtitan/models/__init__.py` 的 `_supported_models` 是权威列表：

| 模型 | 目录 | 类型 | 主要配置函数（节选） | 特点 |
|------|------|------|---------------------|------|
| **Llama 3** | `llama3/` | Dense | `llama3_debugmodel`, `llama3_8b`, `llama3_70b`, `llama3_405b`, `sft_debugmodel` | 标准 Transformer 基准实现 |
| **DeepSeek V3** | `deepseek_v3/` | MoE | `deepseek_v3_debugmodel`, `deepseek_v3_16b`, `deepseek_v3_671b` | MLA + MoE + MTP；含 hybridep / minimal_async_ep 变体 |
| **DeepSeek V4** | `deepseek_v4/` | MoE | `deepseek_v4_debugmodel`, `deepseek_v4_flash`, `deepseek_v4_pro` | 含 `compressor.py` / `mhc.py` / `mtp.py` |
| **GPT-OSS** | `gpt_oss/` | MoE | `gpt_oss_debugmodel`, `gpt_oss_20b`, `gpt_oss_120b` | MoE 变体 |
| **Qwen 3** | `qwen3/` | Dense | `qwen3_0_6b` … `qwen3_32b`, `qwen3_30b_a3b` | QK-Norm |
| **Qwen 3.5** | `qwen3_5/` | 多模态 | `qwen35_0_8b` … `qwen35_397b_a17b` | + Vision Encoder + `gdn.py` |
| **Qwen 3.6** | `qwen3_6/` | 多模态 | `qwen36_27b`, `qwen36_35b_a3b` | 多模态 |
| **Qwen 3.8** | `qwen3_8/` | 多模态 | `qwen38_27b`, `qwen38_2_4t_a95b` | 多模态 |
| **Kimi K2.7** | `kimi_k2_7/` | MoE + VLM | `kimi_k2_5`, `moonlight_16b_a3b`, `kimi_vl_a3b` | **首个 FlexShard / DistMuon 集成**，含 `qk_clip.py` |
| **Kimi K3** | `kimi_k3/` | MoE + VLM | `kimi_k3_debugmodel` | 含 `kda.py` |
| **Muse Glimmer** | `muse_glimmer/` | Dense + VLM | `muse_glimmer_30b`, `muse_glimmer_30b_mm` | 多模态 |
| **Flux** | `flux/` | 图像生成 | `flux/` 自带 `trainer.py` | DiT 架构，**独立训练管线**，不经过 `Trainer` |

> **Flux 是唯一例外**：它有自己的 `trainer.py`、`validate.py`、`tokenizer.py`，因为输入不是 token 序列而是图像 latent。它也是唯一还会调用 `cp_shard()`（而非 `prepare_context_parallel_input()`）的模型，见 `context_parallel/__init__.py` 的说明。

---

## 第四部分：分布式并行系统

> 这一部分是全篇的重点。TorchTitan 的并行能力可概括为一句话：**把 world 组织成一个多维 DeviceMesh，然后让每个并行策略只关心自己那一个轴。**

### 9. DeviceMesh：所有并行的地基

#### 9.1 ParallelDims

`ParallelDims`（`distributed/parallel_dims.py`）是一个 dataclass，由 `ParallelismConfig` + `world_size` 构造：

```python
@dataclass
class ParallelDims:
    dp_replicate: int     # HSDP 复制维（DDP 语义）
    dp_shard: int         # FSDP 分片维
    cp: int               # Context Parallel
    tp: int               # Tensor Parallel
    pp: int               # Pipeline Parallel
    ep: int               # Expert Parallel
    world_size: int
    spmd_backend: Literal["partial_dtensor", "spmd_types"]
```

`_validate()` 强制两条约束：

```
world_size == dp_replicate × dp_shard × cp × tp × pp
ep 必须整除 dp_shard × cp × tp            # 稀疏区与稠密区覆盖同一批 rank
```

`dp_shard = -1` 是唯一的负值特例，表示「吃掉所有剩余的 rank」（在扣除 dp_replicate/cp/tp/pp 之后）。

#### 9.2 一次 build_mesh，切出全部子网格

`build_mesh()` 的思路是：先建一个一维 world mesh，然后反复 `_unflatten` 成不同形状的视图。同一批设备被组织成多个**语义不同**的网格：

| 网格名 | 形状 | 用途 |
|--------|------|------|
| `dataloading` | `[pp, batch, cp, tp]` | 数据加载：确定全局 batch、本 rank 读哪片数据 |
| `loss` | `[loss]` = batch×cp 展平 | loss 的 all-reduce 维度 |
| `dense`（spmd_types） | `[pp, dp_replicate, dp_shard, cp, tp]` | 传给 `fully_shard` 的存储网格 |
| `dense`（partial_dtensor） | `[pp, dp_replicate, fsdp, tp]` | `dp_shard` 与 `cp` 已折叠成 `fsdp` |
| `spmd_dense_for_fwdbwd`（spmd_types） | `[dp, cp, tp]` | **SPMD 类型系统**做前反向类型检查用的网格 |
| `sparse` | `[pp, dp_replicate, efsdp, ep]` | MoE 专家所在的稀疏网格 |
| `spmd_sparse_for_fwdbwd` | `[dp_replicate, efsdp, ep]` | 稀疏区的 SPMD 类型网格 |

注意 `batch = dp_replicate × dp_shard`，`fsdp = dp_shard × cp`，`efsdp = fsdp × tp // ep`。**EP 是从 FSDP 和 TP 那里"借"rank 的**，所以稠密区 `dp_shard × cp × tp` 与稀疏区 `efsdp × ep` 覆盖同一批 rank——这是 §16 的核心约束。

#### 9.3 Fake 后端：避免无谓的进程组

`_mesh_exist()` 决定某个轴是否需要真实的 NCCL 进程组。degree == 1 的轴（以及 `batch` 维）会被 `_unflatten(..., backend_override={name: "fake"})` 标记为 Fake 后端，从而**不创建进程组**。三个例外必须保活：

- `fsdp`：即使 degree == 1 也保留真实后端，因为 `fully_shard()` 需要它来安装 `MixedPrecisionPolicy`。
- `dp_shard`（spmd_types 下）：同上，`fully_shard` 要能从多维网格里辨别出 DP 子网格。
- `efsdp`：当 `ep > 1` 时保活，MoE 层需要 FSDP 帮忙做混合精度。

#### 9.4 寻址 API

`ParallelDims` 对外暴露四个查询入口，语义各不相同，**用错会导致难以定位的问题**：

| 方法 | 语义 | 典型用途 |
|------|------|---------|
| `get_optional_mesh(dims)` | 取子网格；**任一轴未启用就返回 None** | 大多数场景 |
| `get_mesh(dims)` | 同上，但未启用直接抛 `ValueError` | 确定该轴一定存在时 |
| `get_activated_mesh(axes)` | **过滤掉未启用的轴**，剩下的组成子网格 | 存储网格（`resolve_fsdp_mesh`） |
| `resolve_mesh(axes)` | 按后端决定哪些轴是「带内」的（partial_dtensor 只有 tp/ep，spmd_types 还有 dp/cp），其余丢弃 | 分片边界解析 |

属性 `dp_enabled` / `cp_enabled` / `pp_enabled` / `fsdp_enabled`（= dp_shard>1 或 cp>1）/ `ep_enabled` 是日常判断的主要入口。另外 `seq_len_divisor = tp × cp × 2`：SP 要求序列长度能被 TP 整除，CP 负载均衡要求能被 `2 × CP` 整除。

> **命名约定（TorchTitan 强制）**：`axis` 专指 DeviceMesh 的轴，`dim` 专指张量维度。调用 PyTorch 上游 API 时（`mesh_dim_names`）保留上游拼写，但进入本地变量后一律改名为 `mesh_axis_names`。

### 10. 三种 SPMD 后端

当前 TorchTitan 同时存在三条并行运行时路径，由 `parallelism.spmd_backend` 选择（`config/configs.py`）：

```python
spmd_backend: Literal["partial_dtensor", "spmd_types"] = "spmd_types"   # 默认已切到 spmd_types
```

| | `partial_dtensor` | `spmd_types` |
|---|---|---|
| **参数表示** | DTensor@`(tp,)`，其余是普通 tensor | 普通 tensor + **显式 SPMD 类型标注**（`spmd_types` 包） |
| **哪些轴是「带内」** | 只有 `tp` / `ep` | `dp` / `cp` / `tp` / `ep` |
| **DP 处理** | FSDP hooks（带外 out-of-band） | FSDP hooks + 存储网格（`dp_shard` 轴） |
| **CP 处理** | 手动 allgather 包装 forward | `spmd.redistribute` 声明式 |
| **网格形状** | `[pp, dp_replicate, fsdp, tp]`（已折叠） | `[pp, dp_replicate, dp_shard, cp, tp]`（不折叠，另有 `[dp, cp, tp]` 视图） |
| **类型检查** | 无（依赖 DTensor 运行时） | 编译期 + 运行时（`debug.spmd_typechecking`） |
| **状态** | **正在被弃用**（见下） | 当前默认与推荐 |

**为什么会有两套？** `partial_dtensor` 把 `dp_shard` 和 `cp` 提前折叠成 `fsdp` 一个轴，好处是 DTensor 的 placement 列表短、心智负担小；代价是「FSDP 的 DP 语义」和「CP 的序列语义」被混在一个轴上，模型代码无法区分。`spmd_types` 保留完整的轴信息，让每个并行维度的语义显式可查，代价是 mesh 更多、类型系统更复杂。

**`spmd_types` 的类型系统**（外部包 `spmd_types`，TorchTitan 在 `distributed/spmd_types.py` 做胶水）：

```
SpmdType = (local_type: {axis_name -> Shard(d)/Replicate/Partial/Varying}, partition_spec)

标注：   spmd.assert_type(tensor, {DP: S(0), CP: S(1), TP: R})
检查：   with spmd.typecheck(): 每个 op 验证输入类型、推断输出类型
重分布： spmd.redistribute(tensor, src=S(1), dst=R)   → 自动推导出 all-gather
局部计算：spmd.no_typecheck(in_types=..., out_types=...)(fn)
```

TorchTitan 侧的胶水函数包括 `spmd_distribute_tensor` / `spmd_redistribute_per_axis` / `spmd_validate_redistributions` / `annotate_input_spmd_types` / `annotate_replicated_parameters`，以及 mesh 上下文栈 `set_current_spmd_mesh()` / `current_spmd_mesh()`（TLS 栈，因为类型系统需要一个「当前网格」的概念来做轴名解析）。

**过渡期注意事项**（代码里明确标注的 TODO）：

- 开启 `debug.spmd_typechecking` 时**不支持** pipeline parallelism；
- 开启 `debug.spmd_typechecking` 且使用 `SelectiveAC` 时**不支持** FlexAttention；
- `spmd_types` 不支持**不均匀**的模型并行参数切分（`_validate_even_model_parallel_param_sharding`，`module.py:363`）。

### 11. ShardingConfig：声明式分片

这是理解 TorchTitan 当前架构的**关键数据结构**（`protocols/sharding.py:66`）。它把「某个模块的分片方式」从命令式代码变成一份可读的声明。

```python
@dataclass
class ShardingConfig:
    state_shardings:  dict[str, SpmdType]           # 参数 / buffer 的 placement
    in_src_shardings: dict[str, SpmdType] | None    # 输入的"当前" placement
    in_dst_shardings: dict[str, SpmdType] | None    # 输入的"期望" placement（不一致则 redistribute）
    out_src_shardings: SpmdType | tuple | None      # 输出的"当前" placement
    out_dst_shardings: SpmdType | None              # 输出的"期望" placement
    local_map: LocalMapConfig | None                # 若设置，forward 在 local tensor 上执行

@dataclass
class LocalMapConfig:
    in_grad_placements: tuple[SpmdType | None, ...] | None
```

**`src` / `dst` 成对出现的原因**：重分布被建模为 `(source, destination)`。对 DTensor 来说 source 通常隐含在 tensor 的 `placements` 里，但显式声明让契约统一——未来的「类型擦除」系统需要两侧都有信息。声明了 src 还能让运行时**校验**输入确实在预期的 placement 上（不一致就报错，而不是静默算错）。

#### 11.1 两个具体例子

**Colwise（列并行）**——输出按最后一维切开，无需通信：

```
colwise_config():  weight → Shard(0)@TP     # 按 out_features 切
                   out_src → Shard(-1)@TP

构建时:  config.w1.sharding_config = colwise_config()
执行时:  _distribute_states():  param [4096, 11008] → 每 rank 持有 [4096, 11008/tp]
        forward:  x[B,S,4096] (Replicate@TP) @ weight.T → [B,S,11008/tp] (Shard(-1)@TP)
```

**Rowwise + Sequence Parallel**——输出是 Partial，reduce-scatter 到序列维：

```
rowwise_config(output_sp=True):  weight → Shard(1)@TP   # 按 in_features 切
                                 out_src → Partial@TP
                                 out_dst → Shard(1)@TP  # SP 布局

执行时:  x[B,S,11008/tp] (Shard(-1)@TP) @ weight.T → [B,S,4096]，各 rank 持有 Partial
        _redistribute_outputs(): Partial → Shard(1)@TP（即 reduce-scatter）
        最终: [B,S/tp,4096]
```

#### 11.2 resolve_placements 的两个坑

`resolve_placements(layout, mesh)`（`sharding.py:107`）把 `SpmdType` 解析成 `tuple[Placement, ...]`，有两条容易踩的规则：

1. **必须为 mesh 的每个轴显式声明 placement**。缺一个就 `ValueError`，附带已声明/需要声明的轴列表。这是刻意的：静默取默认值会让「配置写漏了」变成「精度悄悄错了」。
2. **1-rank 轴上的 `Shard(d)` / `Partial` 会被归一化成 `Replicate()`**。三者在一个 rank 上操作完全等价，但 DTensor 的算子规则（placement 相等性检查、view/reshape strict 模式）把它们当作不同东西并会拒绝——归一化是绕开这个问题的标准做法。

#### 11.3 local_map：让部分计算在本地张量上进行

`local_map` 用于那些**必须在本地（非 DTensor）张量上执行**的算子——典型是 attention。配置里声明 `local_map`，`Module.parallelize` 就会用 `torch.distributed.tensor.experimental.local_map` 包装 forward：DTensor 进 → 转 local → 执行 → 转回 DTensor。

输入 placement 取自 `in_dst_shardings`（与 `_redistribute_inputs` 用的同一份，保证一致），输出 placement 取自 `out_src_shardings`。`LocalMapConfig` 只额外携带 `in_grad_placements`——因为反向的输入梯度放置没有别的地方可以声明（`sharding.py:37` 的注释解释了这点）。

> **`in_grad_placements` 必须显式给出**：这是 `DTensor.to_local` 类调用的固有陷阱——原始 placement 是 `Replicate` 或 `Partial` 时，反向梯度放置不是唯一确定的。TorchTitan 的规则是**必须声明**，不允许 "just in case" 的默认值。

#### 11.4 分片的实际归属

注意一个常见的误解：**列切/行切等 TP 策略不再定义在 `distributed/tensor_parallel.py` 里**。那个文件现在只有 `NoParallel`（95 行）。真正的分片声明在：

- `models/common/decoder_sharding.py` —— 通用解码器的 `colwise_config()` / `rowwise_config()` / `norm_config()` / `set_gqa_attention_sharding()` / `set_dense_ffn_sharding()`；
- `models/common/moe_sharding.py` —— MoE 相关；
- 各模型的 `sharding.py` —— 模型特有的部分。

`tensor_parallel.py` 剩下的 `NoParallel` 是一个特殊 `ParallelStyle`：**不分片**，但做两件事——(1) 把参数注册成 TP mesh 上的 `Replicate` DTensor，(2) 在模块边界插入 `Tensor ↔ DTensor` 转换 hook。**为什么必须存在？** 因为梯度裁剪和融合优化器都假设「所有参数在同一个 mesh 上」。MoE 的 router gate 就用它。

---

### 12. FSDP2 数据并行

`distributed/fsdp.py` 的 `apply_fsdp_to_decoder()` 是所有模型 `parallelize_fn` 的最后一步，统一处理 Dense 与 MoE。

#### 12.1 执行流程

```
apply_fsdp_to_decoder(model, dp_mesh, ...)
│
├─ 1. MixedPrecisionPolicy(param_dtype, reduce_dtype, cast_forward_inputs=False)
│
├─ 2. 非权值绑定路径:
│     ├─ fully_shard(model.tok_embeddings)
│     └─ fully_shard([model.norm, model.lm_head])  ← 合并成一个 FSDP 单元
│        理由: 最后几层会被立即预取，默认不 reshard
│
├─ 3. 逐层 fully_shard TransformerBlock（MoE 有三分支，见下）
│
├─ 4. fully_shard(model)  ← 顶层
│
├─ 5. 后处理:
│     ├─ enable_symm_mem（可选）: set_force_sum_reduction_for_comms + set_symm_mem_for_comm
│     └─ disable_fsdp_gradient_division(): set_gradient_divide_factor(1.0)
│
└─ 6. ep_degree > 1 时设置显式 prefetch（见 12.4）
```

#### 12.2 权值绑定（weight tying）的特殊处理

当 `model.enable_weight_tying=True` 时，`tok_embeddings` 与 `lm_head` 共享同一个参数。FSDP 必须把它们**打包成一个 FSDP 单元**，否则共享参数会被 all-gather 两次。注意此时 `reshard_after_forward` 强制取 `policy == "always"`。

#### 12.3 MoE 层的三种分片情形

```
┌─ Case A: ep_degree == 1 且不需要 Shard(1)
│    全部参数在同一个 mesh 上 → 普通 fully_shard
│
├─ Case B: ep_degree == 1 但 efsdp_ep_size > num_experts
│    用 shard_placement_fn: expert 参数 → Shard(1)，其余 → Shard(0)
│    目的: 按 hidden 维切分，避免在 expert 维上切出 padding
│
└─ Case C: ep_degree > 1（真正的 Expert Parallel）
     用 shard_placement_fn 返回 ShardPlacementResult:
       expert 参数 → (placement, edp_mesh_info)      ← 稀疏 mesh
       其余参数   → (Shard(0), dp_mesh_info)          ← 稠密 mesh
     其中 edp_mesh_info / dp_mesh_info 由 FSDP2 内部
     `_get_mesh_info(mesh, mesh_dims)` 构建，会在多维 SPMD
     网格里提取并**展平** DP 子网格。
```

判定 `expert_shard_placement` 的公式：

```python
efsdp_ep_size = edp_mesh["efsdp"].size() * ep_degree   # ep>1
              = fsdp_config["mesh"].size()             # ep==1
expert_shard_placement = Shard(1) if efsdp_ep_size > num_experts else Shard(0)
```

**直觉**：专家权重形状是 `[num_experts, hidden, inter]`。若「EFSDP × EP 的 rank 数」超过专家数，按第 0 维切会让部分 rank 分不到专家（pad 出空分片），此时改成按 hidden 维（第 1 维）切更均衡。

#### 12.4 EP 场景的显式预取

EP 的 dispatch 过程中有一次 **D2H 同步**（把 all-to-all 的元数据从 GPU 拷回 CPU），这会让 FSDP 的**隐式**预取失效（同步点打断了预取链）。因此在 `ep_degree > 1` 时，TorchTitan 手工设置整条链路的预取关系：

```
前向: tok_embeddings → layers[0] → layers[1] → ... → [norm, lm_head]
      set_modules_to_forward_prefetch(...)
反向: lm_head → layers[N-1] → ... → layers[0] → tok_embeddings
      set_modules_to_backward_prefetch(...)
```

#### 12.5 reshard_after_forward 策略

```python
def get_fsdp_reshard_after_forward_policy(policy: str, pp_enabled: bool) -> bool
```

| 策略 | 行为 | 适用 |
|------|------|------|
| `"always"` | 前向后立即 reshard | 内存受限 |
| `"never"` | 保留全量参数不 reshard | 延迟敏感 |
| `"default"` | **PP 开启时不 reshard，否则 reshard** | 自动 |

**默认策略的理由**：PP 模式下每个 microbatch 都要前向一次，每层都 reshard + all-gather 开销巨大且无法与计算重叠。

#### 12.6 disable_fsdp_gradient_division

FSDP 默认会把梯度除以 `world_size`。TorchTitan **关掉它**（`set_gradient_divide_factor(1.0)`），因为训练循环自己用 `global_valid_tokens` 做归一化（见 §21）。梯度既被 FSDP 除一次又被 token 数归一化一次，结果就错了——这是典型的「必须理解为什么」的改动。

#### 12.7 双 mesh 分片示意

```
                稠密参数                          专家参数
       (attention, norm, dense FFN)            (routed experts)
                  │                                  │
                  ▼                                  ▼
            ┌───────────┐                     ┌───────────┐
            │  dp_mesh  │                     │  edp_mesh │
            │ dp_rep,   │                     │ dp_rep,   │
            │ dp_shard, │                     │ efsdp,    │
            │ cp, tp    │                     │ ep        │
            └───────────┘                     └───────────┘
             Shard(0)                       Shard(0) 或 Shard(1)
```

---

### 13. Tensor Parallel

TP 在 TorchTitan 中有**两条实现路径**，加上一个辅助策略：

#### 13.1 声明式路径（当前主力）

分片规则写在 `ShardingConfig` 里（§11），由 `Module.parallelize()` 执行。`parallelize_llama` 的判断是：

```python
if parallelism.spmd_backend == "spmd_types" or parallel_dims.tp_enabled:
    model.parallelize(parallel_dims)
```

即：**spmd_types 后端下无条件调用**（因为 dp/cp 也走这套机制），partial_dtensor 后端下只在 TP 开启时调用。

#### 13.2 NoParallel（`tensor_parallel.py:18`）

一个「不做任何张量分片」的 `ParallelStyle`，用途见 §11.4。它做两件事：

```
输入 (plain Tensor)
    │  _prepare_input_fn
    ▼
DTensor.from_local(input, mesh, [Replicate()])   # 未声明 layout 时
    │  （若 input_layout != desired_input_layout 则 redistribute）
    ▼  module.forward()
    │  _prepare_output_fn
    ├─ use_local_output=True  → .to_local()
    └─ use_local_output=False → DTensor
```

#### 13.3 Async TP / dist_gemm

- **Async TP**（`compile.py:_maybe_enable_async_tp`）：设置 `torch._inductor.config._micro_pipeline_tp = True`，让 Inductor 把 TP 通信（all-reduce / reduce-scatter）与矩阵乘**重叠**。**前提是开启 `torch.compile` 且包含 `model` 组件**（`CompileConfig.__post_init__` 里有硬校验）。同时会调用 `enable_symm_mem_for_group(group_name)` 为进程组开启对称内存——这是一个临时的 workaround，等 PyTorch 自动注册后应移除（代码里有对应 TODO 与 issue 链接）。

- **`distributed/linear.py`**：提供 `AllGatherLinear` / `AllGatherLinearMulti` / `LinearReduceScatter` 三个自定义 autograd Function，把 TP 的集合通信**融合进 GEMM**（计算通信一体化）。由 `llama3_debugmodel_dist_gemm` 这类配置启用（`tp_gemm_backend="dist_gemm"`），并且**要求 spmd_types 后端**——因为融合算子收发的是 plain local tensor，这正是该后端的契约。

---

### 14. Context Parallel

CP 把**序列维度**切开，支持超长上下文训练。当前 **CP 要求 `spmd_backend="spmd_types"`**（`validate_cp_backend` 会硬校验），因为 CP 的分片是声明式的。

#### 14.1 输入分片：prepare_context_parallel_input

在 `BaseModel.preprocess_inputs()` 里被调用（`distributed/context_parallel/api.py:54`）：

```
input_dict {input, labels, positions, attention_masks, ...}
    │
    ├─ _cp_shard_dims(input_shardings)
    │     只挑出「CP 轴是 Shard(d)」的输入 → {name: seq_dim}
    │     （CP 轴是 Replicate/Partial 的输入，例如不分片的图像流，会被跳过）
    │
    ├─ cp_shard(cp_mesh, buffers, attention_masks, load_balancer_type, ...)
    │     └─ 在各自声明的 seq_dim 上分片
    │     └─ attention_masks (BlockMask) 单独按 Q 序列维分片
    │
    └─ 原地写回 input_dict
```

`input_shardings` 为 `None` 时退化为默认 `{"input": 0, "labels": 0, "positions": 0}`（标准解码器输入）。注意返回的是**同一个 dict 对象**（原地修改），不是新对象。

#### 14.2 负载均衡：HeadTail vs PTRR

因果注意力下，序列**前段 token 的计算量远小于后段**（每个 query 只能看到自己之前的 KV）。若按连续块均匀切分，各 rank 负载会严重不均。两种负载均衡：

| 策略 | 配置值 | 适用 | 原理 |
|------|--------|------|------|
| **HeadTail** | `"headtail"`（默认） | SDPA | 把序列的头部和尾部交错分配给同一 rank，使各 rank token 数相等 |
| **PTRR** | `"ptrr"` | FlexAttention | Per-Token-Round-Robin，基于 `BlockMask` 的精确负载均衡 |
| 关闭 | `None` | 均匀分片 | 无均衡 |

`context_parallel_ptrr_mask_key` 用于 `attention_masks` 是 `dict[str, BlockMask]` 时，指定用哪个 mask 构建 balancer。

#### 14.3 FlashAttention / FlexAttention 的 KV 处理

在 sdp / flex 后端下，Q 保持 CP 分片（每个 rank 只算自己那段 Q），K/V 需要拿到全局：

```
Q: Shard(seq)@CP   → 各 rank 只算自己那段的注意力输出
K/V: 需要完整序列   → 由底层 attention backend 处理（flex 走 block mask 的 `noop` 掩码逻辑，
                       DTensor 路径走 _context_parallel_dispatcher 的 Ring Attention）
```

---

### 15. Pipeline Parallel

`distributed/pipeline_parallel.py`（640 行）负责把模型按层切到各 PP rank，并构建调度器。

#### 15.1 两个入口

```python
pipeline_llm(model, parallel_dims, ...)                    # :65
pipeline_with_first_stage_modules(...)                     # :144  （首 stage 含额外模块，如 VLM）
```

#### 15.2 主流程

```
pipeline_llm(model, parallel_dims, training, parallelism, compile_config,
             ac_config, dump_folder, device, model_config, parallelize_fn, loss_fn)
│
├─ 1. _get_pipeline_metadata()              :194
│     ├─ schedule_class = get_schedule_class(parallelism.pipeline_parallel_schedule)
│     └─ 判定是否为 looped（多 virtual stage）schedule
│
├─ 2. _generate_llm_fqn_per_model_part()    :357   层分配算法（见 15.3）
│
├─ 3. _pipeline_module_split()              :572
│     ├─ _get_pp_rank_to_stage_indices_mapping()   :532
│     ├─ _split_module(whole_model, module_names)  :470
│     └─ 创建 PipelineStage 对象
│
├─ 4. 对每个 model_part 应用 parallelize_fn（TP + AC + compile + FSDP）
│
└─ 5. _build_pipeline_schedule()            :267
      ├─ CSV 模式: _PipelineScheduleRuntime + schedule._load_csv(path)
      ├─ looped:   schedule_class(stages, n_microbatches, loss_fn=_scalar_loss_fn, scale_grads=False)
      └─ 单 stage: schedule_class(stages[0], ...)
```

#### 15.3 层分配算法（加权均衡）

`_generate_llm_fqn_per_model_part(num_stages, num_layers, input_weight, output_weight)`：

```
输入: num_stages=4, num_layers=12, input_weight=1, output_weight=1
有效层数 = 12 + 1 (embedding) + 1 (lm_head) = 14
每 stage 基准 = 14 / 4 = 3，余 2

Stage 0: tok_embeddings + layers.0 + layers.1          （3 + 余1 − embedding 权重1）
Stage 1: layers.2 + layers.3 + layers.4 + layers.5     （3 + 余1）
Stage 2: layers.6 + layers.7 + layers.8                （3）
Stage 3: layers.9 + layers.10 + layers.11 + norm + lm_head （3 − lm_head 权重1）
```

**`input_weight` / `output_weight` 的直觉**：embedding 和 lm_head 的计算量不可忽略，把各等价为若干 transformer 层，就能让首尾 stage 少分几个 transformer 层，同时总计算量更均衡。对应配置字段是 `pipeline_parallel_first_stage_less_layers` / `pipeline_parallel_last_stage_less_layers` / `pipeline_parallel_layers_per_stage`，也可以用 `module_fqns_per_model_part` 完全手工指定。

#### 15.4 Stage 到 Rank 的映射

`_get_pp_rank_to_stage_indices_mapping()` 按 schedule 类型分成两种风格：

```
Loop 风格 (1F1B / Interleaved1F1B)      V 风格 (ZBV / DualPipeV)
  Rank 0: stage 0, 4                      Rank 0: stage 0, 7
  Rank 1: stage 1, 5                      Rank 1: stage 1, 6
  Rank 2: stage 2, 6                      Rank 2: stage 2, 5
  Rank 3: stage 3, 7                      Rank 3: stage 3, 4

  连续交错                                  V 形对称，减少 bubble
```

判断逻辑就是 `schedule_class in (ScheduleZBVZeroBubble, ScheduleDualPipeV)`。

#### 15.5 支持的 Schedule

| Schedule | 类型 | 特点 |
|----------|------|------|
| `GPipe` | Single | 全 forward 再全 backward，bubble 最大 |
| `1F1B` | Single | 1-forward-1-backward 交替，经典 |
| `Interleaved1F1B` | Multi (looped) | 多 virtual stage 交错，减少 bubble |
| `ScheduleZBVZeroBubble` | Multi-V | 零气泡 |
| `ScheduleDualPipeV` | Multi-V | 双管道 V 形 |
| `_PipelineScheduleRuntime` | CSV | 从 CSV 文件加载自定义调度 |

配置校验：`ParallelismConfig.__post_init__` 会调用 `get_schedule_class()` 提前验证 scheduler 名合法，非法名在**配置阶段**就报错。`PipelineScheduleSingle` 是抽象基类，直接指定会 `ValueError`。

#### 15.6 Stage 间的张量传输

PP stage 之间传的是 **plain tensor**，不是 DTensor——因为 `ProcessGroup` 不可序列化，DTensor 跨进程无法直接传递。接收端通过 `_build_get_mesh_callback()`（`:41`）在运行时**重建** DTensor：

```python
get_mesh(mesh_dim_names, mesh_layout) → parallel_dims.get_mesh(list(mesh_dim_names))
```

#### 15.7 loss 在 schedule 中的传递

`_build_pipeline_schedule` 把 loss_fn 转成 `_scalar_loss_fn`（丢弃 loss_fn 返回的第二个元素，只留标量），因为 PyTorch 的 pipelining schedule 期望一个裸标量。`scale_grads=False` 同样关键——梯度缩放由 TorchTitan 自己用 `global_valid_tokens` 处理，不能让 schedule 再缩一次。

---

### 16. Expert Parallel

EP 是 MoE 特有的并行维度，也是**唯一一个会「借用」其他维度 rank** 的策略。

#### 16.1 稠密区与稀疏区

```
            同一批 rank，两套网格视图
   ┌────────────────────────────────────────────┐
   │  稠密区: dp_shard × cp × tp                 │
   │  ↕ 覆盖完全相同的 rank                        │
   │  稀疏区: efsdp × ep                         │
   └────────────────────────────────────────────┘
   约束: efsdp = dp_shard × cp × tp / ep        （必须整除）
```

配置约束在主 `ParallelismConfig` 里说得很直白：**EP 从 FSDP 和 TP 借 rank**。`pp` 与 `dp_replicate` 是外层维度，不受该约束影响。

#### 16.2 dispatch 后端全景

`models/common/token_dispatcher.py` 里的实现构成一条**继承链**（不是并列的几个选项）：

```
LocalTokenDispatcher (Configurable)                  # EP=1：纯本地重排，无 all-to-all
      │
      └─ BaseEPTokenDispatcher (LocalTokenDispatcher, ABC)
            │
            └─ AllToAllTokenDispatcher                   # EP>1：标准 all-to-all dispatch/combine
                  │
                  ├─ TorchAOTokenDispatcher               # 量化 grouped GEMM 需要的 token padding
                  │     torchao permute_and_pad，把每个专家的 token 组
                  │     pad 到 pad_multiple 的倍数（FP8 需 16，MXFP8 需 32）
                  │     支持 EP=1（单卡调试数值）与 EP>1
                  │
                  └─ DeepEPTokenDispatcher                # DeepEP v2 统一 ElasticBuffer
                        compact（主机同步、可反向） vs expand（无主机同步、cudagraph-capturable、仅推理）

            └─ HybridEPTokenDispatcher                   # 高性能 HybridEP；支持 non_blocking_capacity_factor
            └─ MinimalAsyncEPTokenDispatcher             # 轻量异步 EP；自带 kernels.py；要求 EP>1 与 EP mesh
```

**dispatcher 的选择方式**：`RoutedExperts.Config.token_dispatcher` 是一个 `LocalTokenDispatcher.Config` 字段（`moe.py:141`），由模型的 MoE 配置指定；`RoutedExperts.parallelize` 通过 `wire_meshes()` 把 EP mesh 注入进去（`moe.py:189`）。

**CUDA Graph 兼容性——是配置期校验，不是运行时分派**。`Trainer.Config._validate_cuda_graphs()`（`trainer.py:178`）在 `ep > 1` 时遍历配置树，对每个 `LocalTokenDispatcher.Config` 检查：

```python
if isinstance(cfg, MinimalAsyncEPTokenDispatcher.Config):
    continue                                   # 支持
if isinstance(cfg, HybridEPTokenDispatcher.Config) and cfg.non_blocking_capacity_factor is not None:
    continue                                   # 支持
raise ValueError("CUDA graphs support only expert parallel token dispatcher "
                 "configurations without CPU synchronization. ...")
```

即：**只有 MinimalAsyncEP，或配了 `non_blocking_capacity_factor` 的 HybridEP 能通过校验**；其余（AllToAll / TorchAO / DeepEP / 未配置 capacity factor 的 HybridEP）在启用 CUDA Graph 时**直接报错**，而不是运行时降级。这个「配置期大声失败」的做法与 §25.2 的 regional inductor 校验是同一风格。

#### 16.3 EP 相关的分布式工具

- **梯度裁剪**：EP 参数与稠密参数在不同 mesh 上，`full_tensor()` 会在不同进程组规约，**不能混合计算**。`distributed/utils.py` 的 `clip_grad_norm_` 因此分两条路径，EP 路径见 §26.1。
- **激活重计算**：AC 的 `MUST_SAVE` 集合里包含 `all_to_all_single` / `deepep.dispatch` / `deepep.combine` / `hybridep.*`，避免重计算时重新通信。
- **FSDP 分片**：见 §12.3 Case C。
- **FlexShard / DistMuon**：`distributed/flex_shard/` 为专家参数规划存储布局到计算布局的重分布，Kimi K2.7 是首个集成方。

---

### 17. 并行化应用顺序与原因

`parallelize_fn` 里各步骤的顺序是**强约束**的，顺序错了会出现难以诊断的问题。以 `parallelize_llama` 为准：

```
parallelize_llama(model, *, parallel_dims, training, parallelism,
                  compile_config, ac_config, dump_folder, skip_dp=False)
│
├─ ① model.parallelize(parallel_dims)          ← TP / CP 声明式分片
│     条件: spmd_types 后端 或 tp_enabled
│     必须最先：参数要先进正确的 mesh，后续操作才有意义
│
├─ ② ac_config.build(dump_folder).apply(model)  ← 激活重计算
│     必须在 compile 之前：AC 需要包装原始 forward
│
├─ ③ apply_compile(model, compile_config, parallel_dims)
│     ├─ _maybe_enable_async_tp()   （Async TP 配置）
│     └─ 对每个 TransformerBlock: block.compile(backend, fullgraph=True)
│     必须在 AC 之后：compile 在 AC wrapper 外层
│     必须在 FSDP 之前：FSDP 的 hooks 不能被编译进去
│
└─ ④ apply_fsdp_to_decoder(model, dp_mesh, ...)  ← 数据并行
      必须最后：FSDP 的 pre/post forward hooks 管理 all-gather /
      reduce-scatter 通信，这些 hooks 必须落在 compile 范围之外
```

**为什么是这个顺序？** 一句话总结：**外层包内层**。

- AC 是 forward 层面的语义变换 → 要在 compile 之前，让编译器看到重计算的图；
- compile 是图层面的变换 → 要在 FSDP 之前，让编译器不用管 FSDP 的通信 hook；
- FSDP 是模块调用层面的 hook → 放最外层，靠 `__call__` 拦截。

模型自己还要多做一步：`model_config.update_from_config(config)` 在 build **之前**把 `sharding_config` 挂到配置树上，否则 `model.parallelize()` 面对的是空声明。

---

## 第五部分：训练组件与运行时

### 18. Trainer 初始化

`Trainer.__init__`（`trainer.py:336`）是**整个项目最重要的构造函数**，约 380 行。按顺序它做了：

| 步骤 | 代码位置 | 关键点 |
|------|---------|--------|
| 1. 设备与通信初始化 | `init_distributed()` `:721` | `init_distributed(comm_config, ...)` → `ParallelDims.from_config(config.parallelism, world_size)` |
| 2. 序列长度整除性校验 | `:352` | `num_tokens_per_pp_microbatch % seq_len_divisor`，其中 `seq_len_divisor = (tp if enable_sp else 1) × (2·cp if cp>1 else 1)` |
| 3. `set_spmd_backend()` | `:368` | 全局设定 SPMD 后端（过渡期机制，见 §10） |
| 4. `set_determinism()` | `:386` | 先于模型构建，保证初始化种子一致；见 §26.2 |
| 5. `model_config.update_from_config(config)` | `:396` | 设置模型超参，并挂载 `sharding_config` |
| 6. `apply_overrides()` + 重跑 `__post_init__` | `:407` | override 替换组件，之后全量重新校验 |
| 7. meta 设备建模型 | `:418` | `with torch.device("meta"): model = model_config.build()`，此时显存占用为 0 |
| 8. `verify_module_protocol()` | `:426` | 扫描整棵树，找出不满足 `Module` 协议的模块 |
| 9. 构建 metrics processor | `:429` | 需要 `config.to_dict()` 作为日志元数据 |
| 10. 计算参数量与 FLOPs | `:442` | `get_nparams_and_flops(model, max_context_length)` |
| 11. 模型并行化 | `:507` | PP 路径走 `pipelining_fn`，非 PP 走 `parallelize_fn`（顺序见 §17） |
| 12. `to_empty()` + `init_weights()` | `:520`–`:554` | **真正分配显存**的地方；PP 下逐 model_part，非 PP 下单个模型 |
| 13. `ChunkedLossWrapper.set_lm_head()` | `:559` | 把 lm_head 从模型"移交"给 loss，并置模型 `_skip_lm_head=True`；PP 下只有末 stage 做 |
| 14. 构建 optimizer / lr_scheduler | `:595` | 优化器必须在 FSDP **之后**建（要看到分片后的参数） |
| 15. `post_optimizer_build_fn` | `:596` | 例如 DistMuon 的 compute layout 对齐（见 §22.3） |
| 16. 构建 tokenizer / dataloader / checkpointer | `:631`–`:644` | dataloader 需要 `dp_world_size` / `dp_rank` |
| 17. `train_context` | `:658` | `get_spmd_context(...)`，SPMD mesh / 类型检查上下文 |
| 18. 选择 `fwd_bwd_fn` | `:666`–`:671` | PP → `_pp_forward_backward_body`，否则 `_forward_backward_body` |
| 19. CUDA Graph 包装 | `:673` | `wrap_with_cuda_graph(fwd_bwd_fn, ...)`，见 §25.3 |
| 20. 构建 validator | `:694` | 可选；注意 CUDA Graph + PP 下 validator 被禁止（§25.3） |

**几个必须记住的顺序约束**：

- **`set_determinism` 必须在建模之前**：种子决定了参数初始化，顺序错了就得不到可复现的初始权重。
- **优化器必须在 FSDP 之后构建**：优化器看到的是 `fully_shard` 包装后的参数，才能正确分片优化器状态。
- **`to_empty()` 必须在 `init_weights()` 之前**：meta 设备上的模型只有形状，`to_empty` 才真正分配存储。
- **CUDA Graph 包装必须在 `fwd_bwd_fn` 选定之后**：它包装的是完整的 forward+backward 调用，而不是模型本身。

### 19. 训练循环逐步拆解

#### 19.1 train_step（`trainer.py:878`）

```
train_step(data_iterator)
│
├─ 1. optimizers.zero_grad(set_to_none=disable_cuda_graphs)
│      （CUDA Graph 下不能用 set_to_none，否则图内张量地址失效）
│
├─ 2. 取满一个 optimizer step 的全部数据
│      for _ in range(gradient_accumulation_steps):
│          for _ in range(num_pp_microbatches):
│              input_dict = next(data_iterator)
│              local_valid_tokens += input_dict.pop("num_valid_tokens")
│      └─ 注意 pop：保证送进模型的 batch 只剩下 forward kwargs
│
├─ 3. 计算 global_valid_tokens
│      local_valid_tokens_tensor 在 device 上创建（避免 CPU 同步）
│      dp_enabled 时: dist_utils.dist_sum_tensor(…, dp_mesh)
│      AuxLoss.set_step_denominator(global_valid_tokens)
│
├─ 4. 逐个 grad-accum group:
│      ├─ tensor 搬到 device（non_blocking=True）
│      ├─ HSDP: 只有最后一个 group 才 set_requires_all_reduce(True)
│      │   （accum>1 且非 cuda graph 时才切换，因为图在第一个 group 捕获）
│      ├─ sdc_replayer 存在且是第一个 group → run_fwd_bwd（重放校验）
│      ├─ 否则直接 fwd_bwd()
│      ├─ loss_is_finite.logical_and_(isfinite(local_loss).all())
│      └─ should_log 时累积 accumulated_loss（第一个取 clone，后续 add_）
│
├─ 5. 优化器步:
│      grad_norm = clip_grad_norm_(all_params, max_norm, foreach=True,
│                                 pp_mesh=..., ep_enabled=...)
│      ├─ loss_is_finite 在 loss_mesh（DP×CP）上 all_reduce(MIN)
│      ├─ 再在 pp_mesh 上 all_reduce(MIN)
│      ├─ step_is_finite = loss_is_finite AND isfinite(grad_norm).all()
│      └─ torch._assert_async(step_is_finite, ...)  ← 不阻塞主机
│      checkpointer.maybe_wait_for_staging()
│      optimizers.step(); lr_schedulers.step()
│
└─ 6. 记录 metrics（仅 should_log 时）
       global_avg_loss = sum(accumulated_loss) / ...   ← 数学推导见下面注释
       global_max_loss = max(local_avg_loss)
```

**两个值得注意的设计**：

1. **`torch._assert_async` 而非 Python `assert`**：它在 device 上做检查，不触发主机同步（每个 step 一次同步代价很大）。但注意：断言失败时 CPU 侧可以捕获成 `RuntimeError`，而 CUDA 侧的断言失败会直接使进程失效——这是刻意的权衡。
2. **`loss_is_finite` 用 int32 而非 bool**：NCCL 的归约不支持 bool。MIN 归约意味着「只要有一个 rank 出了非有限值，全局就判定为失败」。

**global_avg_loss 的推导**（代码里有对应注释）：

```
accumulated_loss = local_loss_sum / global_valid_tokens
global_avg_loss  = sum_all_ranks(local_loss_sum) / global_valid_tokens
                 = sum_all_ranks(accumulated_loss)
local_avg_loss   = local_loss_sum / local_valid_tokens
                 = accumulated_loss × global_valid_tokens / local_valid_tokens
global_max_loss  = max_all_ranks(local_avg_loss)
```

#### 19.2 forward_backward_step（`trainer.py:760`）

```
forward_backward_step(input_dict, global_valid_tokens)
│
├─ PP 路径:
│    for each microbatch:
│       inputs_mb, labels_mb, extra_kwargs_mb = model_parts[0].preprocess_inputs(...)
│       arg_mbs  (仅首 stage) / kwarg_mbs / target_mbs (仅末 stage)
│    return fwd_bwd_fn(arg_mbs, kwarg_mbs, target_mbs, global_valid_tokens)
│
└─ 非 PP 路径:
     inputs, labels, extra_kwargs = model_parts[0].preprocess_inputs(...)
     return fwd_bwd_fn(inputs, labels, global_valid_tokens, extra_kwargs)
```

`preprocess_inputs` 是**模型侧的责任**（`BaseModel.preprocess_inputs`），它负责构建 attention mask、做 CP 分片、SPMD 类型标注、把 `input`/`labels` 从 batch 里拆出来。这是"并行细节不进训练循环"原则的体现。

#### 19.3 非 PP 的 fwd/bwd body

```python
def _forward_backward_body(inputs, labels, global_valid_tokens, extra_kwargs):
    with self.train_context():                      # SPMD mesh / loss_parallel / typecheck
        pred = self.model_parts[0](inputs, **extra_kwargs)
        loss, _ = self.loss_fn(pred, labels, global_valid_tokens, **loss_kwargs)
        del pred                                    # 尽早释放 logits
        with spmd.no_typecheck():
            # 只在反向屏蔽类型检查：让类型穿过 BWD 会和 AC 等
            # torch_function 内部机制产生不必要的冲突；FWD 已足够验证。
            loss.backward()
    return loss   # 返回的是 local SUM loss / global_valid_tokens
```

#### 19.4 PP 的 fwd/bwd body

```python
self.pp_schedule.step(arg_mbs=..., kwarg_mbs=..., target_mbs=...,
                      losses=losses, loss_kwargs={"global_valid_tokens": ...},
                      return_outputs=False)
# 只有末 stage 有 losses；detach 后 sum 成一个标量返回
# 非末 stage 返回一个哨兵常量 _pp_loss_sentinel_on_non_last_stage
```

### 20. 数据管道

#### 20.1 组件分层

```
  DataSource（数据来源）           DatasetConfig（怎么组）          Packing
  ┌────────────────────┐         ┌──────────────────────┐      ┌────────────────┐
  │ HuggingFaceRandom  │         │ SingleDatasetConfig  │      │ ConcatThenSplit│
  │  AccessSource      │────────▶│ DatasetMixConfig     │─────▶│ FirstFit       │
  │ HuggingFaceStream  │         │ DatasetConcatConfig  │      │ PackingConfig  │
  │  ingSource         │         │ WeightedDataset      │      └────────────────┘
  │ IndexedJsonlSource │         └──────────────────────┘
  └────────────────────┘
            │
            ▼
  ┌────────────────────┐     ┌─────────────────┐     ┌──────────────────────┐
  │  SampleProcessor   │────▶│    Collator     │────▶│    BaseDataLoader    │
  │ (ChatProcessor …)  │     │ (TextCollator)  │     │ GrainDataLoader      │
  └────────────────────┘     └─────────────────┘     └──────────────────────┘
```

- **`BaseDataLoader`**（`components/data/loader.py:39`）：`Stateful + ABC + Configurable`，是数据加载器的基类。当前唯一实现是 `GrainDataLoader`（基于 `grain`）。
- **`Stateful`**：数据加载器状态（已消费到哪儿）会进入 checkpoint，保证断点续训时数据位置一致。
- **DP 分片**：`config.dataloader.build(dp_world_size=..., dp_rank=..., tokenizer=..., max_context_length=..., num_tokens_per_batch=...)`，由数据加载器自身处理分片与 shuffle。

#### 20.2 batch_generator

```python
def batch_generator(self, data_iterable) -> Iterator[TrainerBatch]:
    data_iterator = iter(data_iterable)
    while True:
        try:
            input_dict = next(data_iterator)
        except StopIteration:
            # 梯度累积途中数据耗尽 → 整个 step 不执行
            raise DataloaderExhaustedError()
        self.metrics_processor.ntokens_since_last_log += input_dict["labels"].numel()
        self.metrics_processor.data_loading_times.append(time.perf_counter() - start)
        yield input_dict     # 注意：tensor 仍在 CPU 上
```

**「tensor 留在 CPU」是刻意的**：直到真正要用的时候才搬到 GPU，这样梯度累积期间多个 microbatch 的输入不会同时占显存。

#### 20.3 数据流全景

```
              HuggingFace Dataset / JSONL
                        │
                        ▼
              DataSource + SampleProcessor
                        │
                        ▼
              Packing (ConcatThenSplit / FirstFit)
                        │
                        ▼
         ┌──── GrainDataLoader (DP 分片, Stateful) ────┐
         │                                              │
         └────────► yields TrainerBatch (CPU tensor) ◄──┘
                        │
                        ▼
                 batch_generator()          # 计时 + token 统计
                        │  × gradient_accumulation_steps × num_pp_microbatches
                        ▼
             Tensor.to(device, non_blocking=True)
                        │
                        ▼
              preprocess_inputs()              # 模型侧：mask 构建 / CP 分片 / SPMD 标注
                        │
        ┌───────────────┴────────────────┐
        ▼                                ▼
  非 PP: model(x) → pred          PP: pp_schedule.step()
        │                                │
        └───────────┬────────────────────┘
                    ▼
        loss_fn(pred, labels, global_valid_tokens) → backward
                    │
                    ▼
        clip_grad_norm_ → optimizer.step() → lr_scheduler.step()
                    │
                    ▼
        Metrics / Checkpoint / Validator
```

### 21. 损失函数

`components/loss.py`（866 行，第三大文件）提供：

| 类 | 说明 |
|----|------|
| `cross_entropy_loss()` | 基础 CE，**sum reduction**（配合 `global_valid_tokens` 归一化） |
| `_LossParallelCrossEntropy` | 自定义 autograd Function，实现 TP 维度的 loss parallel |
| `CrossEntropyLoss` | 标准 CE Loss 组件 |
| `MSELoss` | 均方误差（Flux 等使用） |
| `GradAccumulator` | 跨梯度累积步的辅助 |
| `ChunkedLossWrapper` | **分块计算 CE，避免物化完整 logits** |
| `_DecoderOutputGradientBackProp` | 自定义反向，配合 chunked 前向 |

**ChunkedLossWrapper 的机制**（`loss.py:506`）：

1. `Trainer.__init__` 通过 `set_lm_head(lm_head)` 把模型的 lm_head **移交给 loss**，并设置模型的 `_skip_lm_head = True`。PP 场景下只有最后一个 stage 做这件事。
2. loss 内部按 chunk 计算 `hidden @ lm_head.weight.T` → 立即算 CE → 释放 logits。
3. 反向时 `_DecoderOutputGradientBackProp` 把梯度回传给 hidden。

**为什么重要？** 一个 `[B, S, 128256]` 的 logits 张量在 bf16 下就是 `B×S×0.25 MB` 量级；chunk 化后峰值内存降一个数量级，而数学结果完全一致。

**归一化契约**：所有 loss 都按 **sum** 归约，除以 `global_valid_tokens` 由训练循环的 `loss_fn(pred, labels, global_valid_tokens)` 完成。这就是为什么 FSDP 必须关掉自动梯度除法（§12.6）——否则梯度会被缩两次。

### 22. 优化器与学习率调度

#### 22.1 OptimizersContainer（`components/optimizer/optimizer.py`）

```
OptimizersContainer(Optimizer, Stateful, Configurable, Generic[T])
├─ 支持多 optimizer：通过 ParamGroupConfig 按 regex 匹配参数 FQN
│     └─ 每组独立 optimizer_name + optimizer_kwargs
│     └─ 用例: MoE 的 router gate 用独立超参；Kimi 的 DistMuon + AdamW 混合
├─ default_adamw(lr, **kwargs) 工厂函数
├─ register_moe_load_balancing_hook()   MoE 负载均衡钩子
└─ 实现 Stateful → 优化器状态进 checkpoint
```

`build()` 的入参是 `model_parts=[...]`（PP 下是多个模块），容器负责把它们的参数收集起来。构建时机在 **FSDP 之后**（§18 步骤 13）。

#### 22.2 LRSchedulersContainer（`components/optimizer/lr_scheduler.py`）

```
LRSchedulersContainer(Stateful, Configurable)
├─ decay_type: linear / cosine / cosine_sqrt / ...
├─ warmup_steps + decay_ratio + min_lr_factor
└─ 每个 optimizer 对应一个 scheduler
```

`build(optimizers=..., training_steps=...)`；`get_metrics()` 返回各组的当前学习率，供日志记录。

#### 22.3 FlexShard / DistMuon（`distributed/flex_shard/`）

这是较新的一层，解决的是一个具体问题：**优化器的计算布局可能与参数的持久存储布局不同**。

```
Storage layout (DTensor 持久存储)
        │  FlexShard 规划重分布
        ▼
Compute layout (优化器实际计算用的临时分片)
        │  与计算重叠
        ▼
      DistMuon / AdamW
```

公开 API 包括 `ComputeLayout`（用 `BlockShard` / `Owned` 描述临时切分）、`BucketConfig`（按 FQN 分桶、打包重分布以重叠通信与计算）、`build_dist_muon()`。Kimi K2.7 的 `config_registry.py` 是首个集成方。

代码里明确标注了包边界意图：**FlexShard 未来会被抽成独立仓库**，所以它必须保持自包含、只依赖 PyTorch，不能反向依赖 TorchTitan 的模型或训练基础设施。

### 23. 检查点

`components/checkpointer/`（从旧的单文件 `checkpoint.py` 拆出）：

```
base.py                    # BaseCheckpointManager (ABC) / ModelWrapper / 各状态键
├─ MODEL / OPTIMIZER / LR_SCHEDULER / DATALOADER / TRAIN_STATE  ← 状态字典的顶层键
├─ ModelWrapper            # 把 nn.Module 列表统一成 Stateful 接口
└─ purge_thread()          # 后台删除旧检查点

dcp.py                     # CheckpointManager — 基于 torch.distributed.checkpoint
├─ AsyncMode: DISABLED / ASYNC / ASYNC_WITH_PINNED_MEM
└─ _FilesystemCheckpointStorage

torch_checkpointing.py     # 另一套底层实现
```

**能力清单**：

- 周期性保存（`interval`）+ 最后一步强制保存；
- 三种异步模式：同步 / 后台线程 / 后台线程 + pinned memory staging；
- `maybe_wait_for_staging()` 在优化器步之前被调用，保证前一次异步保存不与其他工作冲突（`trainer.py` 的 `optim` span 里）；
- **Seed checkpoint 创建**：`train_config.checkpoint.create_seed_checkpoint=True` 时，`train.py` 会断言 `WORLD_SIZE == 1` 并直接保存 step 0，用于生成一个"干净的初始权重"；
- `state_dict_adapter`：把 TorchTitan 的 state dict 转成 HF SafeTensors 格式（或反向），实现与 HuggingFace 生态的互操作。

### 24. 激活重计算

`distributed/activation_checkpoint.py` 提供了**四种**策略，通过 tyro 子命令选择：

```python
ActivationCheckpointingConfig = (
    Annotated[SelectiveAC.Config,    tyro.conf.subcommand("selective")]
  | Annotated[RegionAC.Config,       tyro.conf.subcommand("region")]
  | Annotated[FullAC.Config,         tyro.conf.subcommand("full")]
  | Annotated[MemoryBudgetAC.Config, tyro.conf.subcommand("memory-budget")]
  | Annotated[None,                  tyro.conf.subcommand("none")]
)
```

| 策略 | CLI | 原理 | 适用 |
|------|-----|------|------|
| `SelectiveAC` | `--activation_checkpoint selective` | **逐 op 选择性**：保存昂贵的 op，重算便宜的 | 默认，通用 |
| `RegionAC` | `--activation_checkpoint region` | 只保留模型**显式声明**的 region，其余全部重算 | 需要精细控制，见 `docs/remat.md` |
| `FullAC` | `--activation_checkpoint full` | 整个 block 全部重算 | 内存最紧 |
| `MemoryBudgetAC` | `--activation_checkpoint memory-budget` | 给定内存预算，由编译器决定策略 | **要求开启 compile** |
| `None` | `--activation_checkpoint none` | 关闭 | — |

#### 24.1 SelectiveAC 的 save 集合

`_get_default_save_ops()` 构造 `MUST_SAVE` 集合，分两类：

**计算密集型 op**（重算太贵）：

```
_scaled_dot_product_cudnn_attention / _scaled_dot_product_attention_math
_scaled_dot_product_fused_attention_overrideable
flex_attention (torch._higher_order_ops.flex_attention)
linear / mm
max                       # 低精度训练的缩放因子，必须保存
topk                      # MoE 路由 —— 非确定性，必须保存以免重算后专家分配不一致
inductor_compiled_code    # torch.compile 的融合 kernel
torch_attn._varlen_attn
```

**通信 op**（重算 = 重新通信，代价极高）：

```
reduce_scatter_tensor          # FSDP
all_to_all_single              # EP
deepep.dispatch / deepep.combine
hybridep.dispatch / hybridep.combine
```

另外还会并入 PyTorch 默认的 `get_default_op_list().compute_intensive_ops`。op 列表支持「条件注册」写法（`(module_obj, "dotted.path")`），未安装时静默跳过——所以 deepep/hybridep 未装也不报错。

#### 24.2 自定义策略

```python
def wrapped_policy(ctx, func, *args, **kwargs) -> CheckpointPolicy:
    # 1. CUDA→CPU 拷贝必须保存（MoE 的 D2H 同步元数据）
    if func == aten._to_copy and "cuda" in str(args[0].device) and kwargs.get("device") == "cpu":
        return MUST_SAVE

    # 2. mm/linear: 每隔一个保存一个（奇偶交替）
    if func in mm_ops:
        weight_shape = normalize(args[1].shape)     # linear 的 weight 是 (out, in)，转成 (in, out) 对齐 mm
        if weight_shape in mm_recompute_shapes:
            return PREFER_RECOMPUTE                  # 配置里点名的形状强制重算
        meta[mm_count_key] += 1

    # 3. 在 save_ops 里的：保存（但 mm 类偶数次仍重算）
    # 4. 其余：PREFER_RECOMPUTE
```

**mm 奇偶交替的直觉**：SwiGLU FFN 里有 w1/w3 两个并行的 colwise linear，输入相同。只保存其中一个即可，另一个重算成本很低，省下的内存却很可观。

`SelectiveAC.Config.force_recompute_mm_shapes_by_fqns`（默认 `["moe.router.gate"]`）有一个**值得注意的语义**：它是按**形状**匹配的，不是按 FQN。指定 `"moe.router.gate"` 意味着「任何形状为 `(*, in) x (in, out)` 的 mm 都会被强制重算」——因为 router gate 的矩阵很小，重算比重存划算。

#### 24.3 一个隐蔽的 workaround

`_disable_dynamo_lru_cache()`（`:84`）关闭了 dynamo 的 LRU 缓存。原因是 SAC + PP + FlexAttention 三者的交互：第二个 PP microbatch 会触发动态形状的重新编译，此时存在两份合法图，dynamo 默认选最新的（动态形状版），而 SAC 缓存的是静态图的 inductor HOP 输出，缺少 symint 导致断言失败。关掉 LRU 缓存、按插入顺序选图可以绕开。**注释里直接给了上游 issue 链接**——这是「根因优先」原则的样板。

### 25. torch.compile 与 CUDA Graph

#### 25.1 apply_compile（`distributed/compile.py:41`）

```python
def apply_compile(model, *, compile_config, parallel_dims):
    _maybe_enable_async_tp(compile_config, tp_mesh)          # Async TP 配置
    torch._dynamo.config.capture_scalar_outputs = True        # MoE token-choice 的 data-dependent 形状
    torch._dynamo.config.skip_fwd_side_effects_in_bwd_under_checkpoint = True
    backend = _maybe_regional_inductor_backend(model, compile_config.backend)

    for layer_id, transformer_block in model.layers.named_children():
        transformer_block.compile(backend=backend, fullgraph=True)
```

**按 TransformerBlock 粒度编译**而非整个模型，理由有三：

1. 编译只发生一次，所有 block 共享同一份编译结果；
2. 图更小，Inductor 的优化更聚焦；
3. `fullgraph=True` 要求无 graph break，block 粒度更容易满足。

`skip_fwd_side_effects_in_bwd_under_checkpoint` 这条配置的注释说明了它与 AC 的交互：eager AC 会在反向重放 forward 的 Python 副作用，但 compile 的图无法轻易重放 Python 变更。打开这个 flag 表示**接受 eager 与 compile 的这点差异**——即 RoPE cache 更新这类副作用不会在反向重算时重复执行。

#### 25.2 Regional Inductor

当 backend 是 `aot_eager` 且模型用了 FlexAttention 时，会启用 `regional_inductor`：

```
          aot_eager 后端
               │
      ┌────────┼────────┐
   普通 op  FlexAttn  普通 op
   (eager) (需 inductor)
               │
      ┌────────▼─────────┐
      │ regional_inductor │  只对标注区域 lower 到 inductor
      └───────────────────┘
```

标注由 `maybe_regional_inductor()` 返回的上下文管理器完成（在 `FlexAttention.forward` 里用），内部是 `fx_traceback.annotate({"compile_with_inductor": {...}})`。

`_maybe_regional_inductor_backend` 有明确的防御：

- 模型**不用** FlexAttention → 原样返回 backend，不启用；
- backend 是 `inductor` → 原样返回（本来就能 lower flex）；
- backend 是**其他**非 `aot_eager` 也非 `inductor` → **直接报错**，因为 flex region 会分解成 eager aten op（没有 Triton kernel），不能静默降级。

这是一个很好的「不静默降级」范例：能透明处理就处理，不能就大声失败。

#### 25.3 CUDA Graph（`distributed/cudagraph.py`）

`TrainingConfig.disable_cuda_graphs` 控制是否启用（默认 `False`，即默认启用）。启用时 `Trainer.__init__` 用 `wrap_with_cuda_graph(fwd_bwd_fn, gradient_accumulation_steps, sdc_num_steps, sdc_num_replays)` 把整个 forward+backward 包起来（`trainer.py:673`）；`Trainer.close()` 里调用 `cudagraph_teardown()`。

**`Trainer.Config._validate_cuda_graphs()`（`trainer.py:178`）的五条硬约束**——全部是**配置期报错**，不是运行时降级：

| # | 约束 | 报错信息要点 |
|---|------|-------------|
| 1 | **PP + validator 不能同时开启** | validation 会重新初始化共享的 pipeline schedule，破坏图 |
| 2 | **PP 不能用 looped schedule** | `schedule_class` 是 `PipelineScheduleMulti` 子类就报错；单 stage schedule（GPipe / 1F1B）可以 |
| 3 | **VarlenAttention 必须固定形状** | 模型含 `VarlenAttention.Config` 时，`dataloader.max_num_documents` 必须设置（作为每本地 batch 文档数的上界） |
| 4 | **ep == 1 时直接通过** | 只有 EP 开启才需要检查 dispatcher |
| 5 | **EP 的 dispatcher 不能有主机同步** | 只有 `MinimalAsyncEPTokenDispatcher`，或配了 `non_blocking_capacity_factor` 的 `HybridEPTokenDispatcher` 能通过（详见 §16.2） |

除此之外，基础前提（写在 `TrainingConfig.disable_cuda_graphs` 的 docstring 里）是：**输入形状固定、捕获区域内没有 CPU↔GPU 同步**；且与 `torch.compile(mode="reduce-overhead")` **相互独立**——后者自己做图捕获。

**两处容易被忽略的配套改动**：

```python
# trainer.py:879 —— 图启用时必须 set_to_none=False
self.optimizers.zero_grad(set_to_none=self.config.training.disable_cuda_graphs)
#                                    ^^^^ 图启用时 disable_cuda_graphs=False → set_to_none=False
# 理由: set_to_none=True 会把梯度张量置为 None 并重新分配，改变地址 → 捕获的图失效

# trainer.py:949 —— HSDP 的 all-reduce 开关在 accum>1 且图启用时不动
if dp_replicate_enabled and (gradient_accumulation_steps == 1 or disable_cuda_graphs):
    ...  # 只有最后一个 group 才 set_requires_all_reduce(True)
# 理由: 图在第一个 group 捕获，后续 group 直接重放；toggling 会让重放的图与捕获时不符
```

**与 SDC Replay 的交互**（`_validate_sdc_replay`，`trainer.py:237`）：开启 CUDA Graph 时 `sdc_replayer.num_replays` 最多为 1，因为图捕获建立了稳定的梯度与缓冲区标识，多次重放的状态恢复还没实现（代码里有对应 TODO）。另外 SDC replay 本身要求 `debug.deterministic=True` 且 `deterministic_warn_only=False`。

### 26. 分布式工具函数

`distributed/utils.py`（740 行）提供底层原语。

#### 26.1 梯度裁剪

PyTorch 标准的 `clip_grad_norm_` 不支持跨 PP stage 的范数聚合，TorchTitan 因此自建：

```
clip_grad_norm_(parameters, max_norm, *, foreach, pp_mesh, ep_enabled)
│
├─ 非 EP 路径:
│   1. torch.nn.utils.get_total_norm(grads) → total_norm (DTensor)
│   2. total_norm.full_tensor()             ← 在 FSDP/TP 维上规约
│   3. pp_mesh 存在时再 all_reduce(SUM, pp_group)  ← 聚合所有 PP stage
│   4. clip_grads_with_norm_(...)
│
└─ EP 路径 (_clip_grad_norm_with_ep) :676
    1. 按参数的 mesh_dim_names 里是否有 "ep" 分成两组
    2. 分别算 ep_grads_total_norm / non_ep_grads_total_norm
    3. 合并: total_norm = (ep^p + non_ep^p)^(1/p)
    4. PP 聚合（同上）
    5. 分别对 ep_params 和 non_ep_params 应用裁剪
```

**为什么 EP 必须分开算？** EP 参数和稠密参数在不同 DeviceMesh 上，`full_tensor()` 会在不同的 ProcessGroup 上规约。混在一起算会出错，或至少语义不清。分开算再按范数的定义合并才是正确的。

#### 26.2 确定性设置

```python
set_determinism(parallel_dims, device, debug_config, distinct_seed_mesh_dims=["pp"])
```

```
deterministic=True 时:
  ├─ torch.use_deterministic_algorithms(True)
  ├─ cudnn.deterministic = True, cudnn.benchmark = False
  ├─ fill_uninitialized_memory = False      # 避免 DeepEP 竞争
  ├─ CUBLAS_WORKSPACE_CONFIG = ":4096:8"
  ├─ FlexAttention 禁用 max_autotune
  └─ deterministic_warn_only=True 时只警告不报错（TorchTitan 明确不推荐）

种子分配:
  ├─ 所有 rank 统一基础种子（rank 0 广播）
  ├─ PP 维: 不同 PP rank 用不同种子（seed += pp_rank × cumulative_size）
  │     理由: 各 stage 层数不同，用同种子会让不同层的初始化相关
  ├─ SPMD 维 (DP/TP/CP): 相同种子 → 保证参数初始化一致
  └─ DTensor RNG: manual_seed(seed, world_mesh)

detect_anomaly=True 时:
  torch.autograd.set_detect_anomaly(True, check_nan=False)
  └─ check_nan=False 的理由: 避免 DTensor 在无分片策略时 _is_any_true 崩溃
```

`distinct_seed_mesh_dims=["pp"]` 是一个容易忽略但很关键的设计：**PP 各 stage 必须用不同种子**，因为它们持有不同的层。

#### 26.3 训练上下文

```python
def get_spmd_context(*, parallel_dims, spmd_typechecking) -> SpmdContext:
    @contextmanager
    def context():
        with ExitStack() as stack:
            if spmd_types 后端:
                stack.enter_context(set_current_spmd_mesh(spmd_dense_mesh))
            if spmd_typechecking:
                stack.enter_context(spmd.typecheck(...))
            yield
```

`Trainer` 在 `__init__` 里构建一次，在每个 `train_step` 里 `with self.train_context():` 进入。这样 SPMD 的「当前 mesh」对整段前反向都有效。

#### 26.4 分布式初始化与通信原语

- `init_distributed(comm_config, enable_cpu_backend, base_folder)`（`:468`）：支持 NCCL 标准模式、`fake_backend`（干跑校验）、CPU backend（配合 `enable_cpu_offload`）；配置 Flight Recorder（NCCL 环形缓冲调试）；因 spmd_types 需要 TLS 存储 DeviceMesh 而禁用 autograd 多线程。
- 通信原语：`dist_sum` / `dist_max` / `dist_mean` / `dist_sum_tensor`（`:131`–`:162`），统一封装 `_dist_reduce`，接受 DeviceMesh 而非 ProcessGroup。
- `set_pg_timeouts(train_timeout, init_timeout)`（`:563`）：初始化期用较宽松的超时，之后收紧——避免慢启动被误判为 hang。

### 27. 可观测性

#### 27.1 structured_logger（`observability/structured_logger/`）

一套贯穿全流程的**结构化 trace**，在 `train.py` 里通过 `sl.init_structured_logger(source="training", output_dir=dump_folder, enable=debug.enable_structured_logging)` 初始化。

- `sl.log_trace_span("name")`：进入/退出一个 span（有嵌套渲染 `_span_stack_depth`）；
- `sl.log_trace_instant("name")`：瞬时事件；
- `sl.log_trace_scalar({"k": v})`：标量记录；
- `gantt_generator.py`：把 trace 渲染成甘特图。

代码里大量使用这些调用做**性能归因**：`fwd_bwd`、`preprocess_inputs`、`fetching_batch`、`optim`、`collect_dist_metrics`、`torch_distributed_init`、`structured_logger_started` 等。这是定位「时间花在哪」的第一手工具。

`DebugConfig.enable_structured_logging` 默认 True；关掉后所有 trace 调用变成 no-op，可完全消除开销。

#### 27.2 MetricsProcessor（`components/metrics.py`）

```
MetricsProcessor(Configurable)
├─ DeviceMemoryMonitor             # 峰值显存 / 已用显存采样
├─ LoggerContainer                 # 多后端聚合
│   ├─ TensorBoardLogger
│   └─ WandBLogger
├─ ensure_pp_loss_visible()        # 保证 PP 场景下 loss 能被看到
└─ _get_metrics_rank()             # 决定哪个 rank 负责写日志
```

`log(step, global_avg_loss, global_max_loss, grad_norm, extra_metrics=...)` 是主入口；`optimizers` / `model_parts` 在 `Trainer.__init__` 里被挂上，用于计算梯度范数与参数量相关指标。

#### 27.3 SDC Replayer（`observability/sdc_replayer.py`）

静默数据损坏（Silent Data Corruption）检测：`SDCReplayer` 通过 `run_fwd_bwd(fn, step=...)` 把第一个梯度累积组的 forward/backward **重放若干次**，比较结果是否一致。配置项 `sdc_num_steps` / `sdc_num_replays` 控制检查频率和重放次数。设计文档见 `observability/silent_data_corruption.md`。

---

## 第六部分：运行时视图

### 28. 进程拓扑与线程模型

#### 28.1 进程拓扑

每个 GPU 一个 Python 进程。以 `pp=8, dp_replicate=1, dp_shard=2, tp=4`（64 GPU）为例：

```
World Group (64 processes)
│
├── PP Group (8 stages × 8 ranks/stage)
│     └─ 每个 stage: 8 个 rank
│
├── DP Group (dp_shard=2 × 32 ranks/shard)
│
├── TP Group (4 GPU/group × 16 groups)
│
├── CP Group (未启用)
│
└── EP Group (未启用；efsdp = dp_shard × cp × tp / ep)

设备网格展开（build_mesh）:

  world[64]
    └─ _unflatten → [pp=8, batch=2, cp=1, tp=4]      dataloading_mesh
         ├─ loss_mesh   = dataloading["batch","cp"]._flatten("loss")
         ├─ dense       = _unflatten → [pp, dp_replicate, dp_shard, cp, tp]   (spmd_types)
         │                            或 [pp, dp_replicate, fsdp, tp]         (partial_dtensor)
         ├─ spmd_fwdbwd = _unflatten → [pp, dp, cp, tp]，再取 ["dp","cp","tp"]
         └─ sparse      = _unflatten → [pp, dp_replicate, efsdp, ep]
```

#### 28.2 Rank 与设备

```python
rank       = int(os.environ["RANK"])          # 全局 rank
local_rank = int(os.environ["LOCAL_RANK"])    # 节点内 rank
world_size = int(os.environ["WORLD_SIZE"])    # 总进程数

# 各维度的本地 rank 通过 DeviceMesh 获取
mesh = parallel_dims.get_mesh("batch")        # 或 "tp" / "pp" / ...
dp_rank = mesh.get_local_rank()

device = utils.get_local_device()             # 由 torchtitan 统一处理 CUDA / XPU / ...
device_module.set_device(device)              # 必须在创建 TorchFT manager 之前
```

#### 28.3 线程与流

```
主进程（每个 GPU 一个）
│
├── Python 主线程
│   ├── 取数据 → 前向 → 损失 → 反向 → 优化器 → 指标
│   └── 受 GarbageCollection 管理器控制（gc_freq 步一次 gc.collect）
│
├── CUDA 默认流
│   └── 主要计算 kernel
│
├── NCCL 通信流（由 PyTorch / NCCL 内部管理）
│   ├── FSDP all-gather / reduce-scatter
│   ├── TP all-reduce / reduce-scatter（Async TP 时可与计算重叠）
│   └── PP P2P send / recv
│
├── 检查点后台线程
│   ├── 异步状态字典序列化（AsyncMode.ASYNC / ASYNC_WITH_PINNED_MEM）
│   └── maybe_wait_for_staging() 在优化器步之前同步
│
├── purge_thread()
│   └── 删除过期检查点
│
└── DataLoader worker 进程（grain 管理）
```

**异步通信的关键**：所有 `redistribute(..., async_op=True)` 与 Async TP 的目标都是让通信发生在 CUDA 通信流上，从而与默认流上的计算重叠，而不是阻塞主线程。

### 29. 通信原语与内存管理

#### 29.1 FSDP2 通信模式

```
前向: 参数 all-gather
  for layer in model.layers:
      params = all_gather(sharded_params, group=dp_shard_group)
      output = layer.forward(input, params)
      # reshard_after_forward 决定是否立即释放全量参数
      #   "always" → 立即释放，省内存
      #   "never"  → 保留，省通信（PP 场景默认）

反向: 梯度 reduce-scatter
  for layer in reversed(model.layers):
      grad_output = layer.backward()
      sharded_grad = reduce_scatter(grad_output, group=dp_shard_group)
```

#### 29.2 Tensor Parallel 通信模式

```
列并行 (Column Parallel):
  input  [B, S, hidden]        (完整输入)
  weight [hidden, hidden/tp]   (按 out_features 切)
  → 本地 matmul，无通信
  → output [B, S, hidden/tp]

行并行 (Row Parallel):
  input  [B, S, hidden/tp]     (已分片输入)
  weight [hidden/tp, hidden]   (按 in_features 切)
  → 本地 matmul
  → output [B, S, hidden] 各 rank 持有 Partial
  → all-reduce（Replicate 布局）或 reduce-scatter（Sequence Parallel 布局）
```

`distributed/linear.py` 里的 `AllGatherLinear` / `LinearReduceScatter` 把上述通信融合进 GEMM。

#### 29.3 Pipeline Parallel 通信模式

```
forward_step(microbatch_id):
    非首 stage: activation = p2p_recv(src=prev_stage_rank, tag=microbatch_id)
    output = model(activation)
    非末 stage: p2p_send(output, dst=next_stage_rank, tag=microbatch_id)

backward_step(microbatch_id):
    非末 stage: grad_output = p2p_recv(src=next_stage_rank, tag=microbatch_id)
    grad_input = autograd(grad_output)
    非首 stage: p2p_send(grad_input, dst=prev_stage_rank, tag=microbatch_id)
```

实际实现由 PyTorch 的 `torch.distributed.pipelining` 提供，TorchTitan 只负责切分模型和构建 schedule。

#### 29.4 Context Parallel 通信模式

```
Ring Attention 风格（由 DTensor 的 _context_parallel_dispatcher 自动处理）:
  Q 保持分片 [B, S/cp, H, D]
  K/V 沿 ring 依次交换，每轮:
      attn_out += scaled_dot_product_attention(q_local, k_current, v_current)
      (k, v) = ring_exchange(send=(k,v) → cp_rank+1, recv=(k,v) ← cp_rank-1)
  最终输出 [B, S/cp, H, D]
```

#### 29.5 显存管理

以 `tp=4` 训练一个 70B 级模型为例（数量级示意，非精确值）：

```
初始化分配:
├─ 模型参数       (~参数总数 / TP)          bf16
├─ 梯度缓冲       (同参数量)                bf16
└─ 优化器状态     (Adam 的 m/v，2×参数)     fp32
   总计 ≈ 参数数 × (2 + 2 + 8) 字节 / TP     ← 优化器状态是大头

训练中峰值:
├─ 基线(上)
├─ 激活 (前向保存)                ← 与 batch × seq × layers 成正比
├─ FSDP all-gather 临时缓冲        ← 瞬时
└─ 峰值 ≈ 基线 + 激活 + 临时

优化手段（按收益排序）:
1. 激活重计算 (FullAC / SelectiveAC)     ← 用计算换内存，收益最大
2. CPU Offload (enable_cpu_offload)      ← 参数/梯度/优化器状态搬到 CPU
3. 混合精度 (mixed_precision_param)      ← 参数用 bf16，归约用 fp32
4. 量化 (float8 / mxfp8 / nvfp4)         ← 权重与激活的低精度表示
5. ChunkedLossWrapper                    ← 避免物化完整 logits
6. reshard_after_forward="always"        ← 前向后立即释放全量参数
```

**为什么激活重计算收益最大？** 优化器状态和参数的大小是固定的（与 batch/seq 无关），而激活内存随 batch × seq × layers **线性增长**。长序列训练时激活往往是峰值内存的主导项。

---

## 第七部分：工程实践

### 30. 实验性功能

`torchtitan/experiments/__init__.py` 的 `_supported_experiments` 是权威列表：

| 实验 | 目录 | 说明 |
|------|------|------|
| **Graph Trainer** | `experiments/graph_trainer/` | 图级训练：FX trace + 自动并行。子模块包括 `llama3` / `deepseek_v3` / `qwen3` / `muse_glimmer`，以及 `graph_pp`（图级流水线，`backward_requires_autograd=False`，跑显式反向图） |
| **AutoParallel** | `experiments/autoparallel/` | 自动并行探索（`llama3` / `local_map_deepseek_v3`） |
| **RL** | `experiments/rl/` | 后训练/强化学习。示例：`alphabet_sort` / `dapo_math` / `search_r1` / `verifiers.dapo_math` |
| **TorchFT** | `experiments/torchft/` | 容错训练（`llama3`） |
| **Transformers Backend** | `experiments/transformers_modeling_backend/` | 用 HuggingFace Transformers 的模型定义替换 TorchTitan 的模型实现 |

**实验目录的三条纪律**（写在项目规则里）：

1. 必须通过 `pre-commit run --all-files`，**没有例外**；
2. **不能为了实验改核心代码**——需要什么就在实验目录里绕过去；
3. 必须用 TorchTitan 的既有配置系统，不能另起一套参数解析。

`ConfigManager._load_config` 里对实验模块的查找顺序（models → experiments → rl/examples）就是对这套组织的直接支持。

### 31. 测试、验证与数值正确性

#### 31.1 常用命令

```bash
# 开发依赖
pip install -r requirements.txt -r requirements-dev.txt

# 提交前必跑
pre-commit run --all-files

# 单元测试
pytest tests/ -x

# GPU 集成测试
#   tests/integration_tests/ 里通过 OverrideDefinitions 覆盖 Llama 3 调试模型配置
```

#### 31.2 数值正确性验证

这是 TorchTitan 最严格的一条工程纪律：

> **非计算类改动**（激活检查点、重构、并行策略调整）必须用 `--debug.seed=42` + `--debug.deterministic` 产出**逐位相同（bit-wise identical）**的 loss 和 grad_norm。
>
> **计算类改动**需要在代表性数据集（如 C4）上验证 loss 收敛。

实际操作：

```bash
python -m torchtitan.train --module llama3 --config llama3_debugmodel \
    --debug.seed 42 --debug.deterministic --training.steps 10
```

**注意**：stdout 只打印 5 位有效数字，**不足以**判断逐位一致。必须用 `scripts/loss_compare.py` 打开 profiling，从 TensorBoard 结果里对比 loss 与 grad_norm 的完整精度。

**绝对不要**使用 `--debug.deterministic_warn_only`——它会把确定性违规降级成警告，使比较失去意义。

#### 31.3 性能测试

跑性能时至少用 10 个训练步（`--training.steps 10`），否则启动与 warmup 开销会主导测量结果。

#### 31.4 相关工具

| 工具 | 位置 | 用途 |
|------|------|------|
| `Profiler` | `tools/profiler.py` | PyTorch Profiler 封装，可导出 trace |
| `scripts/loss_compare.py` | `scripts/` | 逐位对比两次运行的 loss / grad_norm |
| `SDCReplayer` | `observability/sdc_replayer.py` | 运行中检测静默数据损坏 |
| structured logger | `observability/structured_logger/` | 时间归因与甘特图 |
| `docs/` | `docs/*.md` | `fsdp.md` / `checkpoint.md` / `remat.md` / `debugging.md` / `composability.md` 等专题文档 |
| `models/common/MOE_SHARDING.md` | `models/common/` | MoE 分片专题 |
| `distributed/flex_shard/README.md` | `distributed/flex_shard/` | FlexShard / DistMuon 设计 |

---

## 附录：常见误区与勘误

对照旧版文档与早期理解，以下是**当前代码已经变化**或**常被误解**的点：

| 误区 | 当前事实 |
|------|---------|
| `TrainSpec` 包含 optimizer / dataloader / loss 等 builder | **已收敛为 `ModelSpec`**，只保留 5 个字段：`model` / `parallelize_fn` / `pipelining_fn` / `post_optimizer_build_fn` / `state_dict_adapter`。其余全部回到 `Trainer.Config` 顶级字段 |
| 数据加载器是 `ParallelAwareDataloader` / `HuggingFaceTextDataLoader` | 现在是 `components/data/` 包：`BaseDataLoader` + `GrainDataLoader`，配合 `DatasetConfig` / `Source` / `Packing` / `Collator` 的可组合结构 |
| CheckpointManager 在 `components/checkpoint.py`（单文件 1092 行） | 现在是 `components/checkpoint **er**/` 包（base / dcp / torch_checkpointing / utils） |
| `models/common/moe.py` 是 MoE 的全部 | 现在拆成 `moe.py`（结构）+ `moe_sharding.py`（分片）+ `token_dispatcher.py`（dispatch）+ `aux_loss.py`（辅助损失） |
| TP 策略（ColwiseParallel / RowwiseParallel）在 `distributed/tensor_parallel.py` | 该文件现在只有 `NoParallel`（95 行）。真正的切分声明在 `models/common/decoder_sharding.py` 与各模型 `sharding.py` |
| 只有 `default`（DTensor）一种后端 | 现在有 `partial_dtensor` 与 `spmd_types` 两种，**默认已是 `spmd_types`**；`partial_dtensor` 正在被弃用 |
| 并行轴就是 `dp/pp/tp/cp/ep` | 存储网格还区分 `dp_replicate`（复制）与 `dp_shard`（分片）；另有 `efsdp`（EP 区的 FSDP）、`batch`（DP 合计）、`loss`（DP×CP）等派生轴 |
| world_size = `dp_replicate × dp_shard × cp × tp × pp` | 仍然正确，但**必须叠加 EP 约束**：`ep` 必须整除 `dp_shard × cp × tp`，且 `efsdp = dp_shard × cp × tp / ep` |
| AC 只有 selective / full 两种 | 现在是四种：`selective` / `region` / `full` / `memory-budget`，外加 `none` |
| 激活重计算是 `apply_ac()` 函数 | 现在是策略对象：`ac_config.build(dump_folder).apply(model)`，基类 `ActivationCheckpointing` |
| Context Parallel 无条件可用 | **CP 要求 `spmd_backend="spmd_types"`**，`validate_cp_backend` 会硬校验 |
| 模型列表只有 llama3 / llama4 / deepseek_v3 / qwen3 / gpt_oss / flux | 当前为：llama3、deepseek_v3、deepseek_v4、gpt_oss、qwen3、qwen3_5、qwen3_6、qwen3_8、kimi_k2_7、kimi_k3、muse_glimmer、flux（**llama4 已不在列表**） |
| 每个模型需要写 `parallelize.py` + `config_registry.py` + 模型文件 | 现在标准结构是 `model.py` + `config_registry.py` + `sharding.py` + `parallelize.py` + `state_dict_adapter.py`（后两者可选）。分片逻辑尽量下沉到 `common/decoder_sharding.py` |
| `components/` 是一个 Python 包 | **`components/` 下没有 `__init__.py`**，它是命名空间目录；子包（`data/`、`optimizer/`、`checkpointer/`、`quantization/`）各自是包 |

---

## 结语：三句话记住 TorchTitan

1. **配置先行**：一切皆 `Config`，`build()` 即构造；CLI 覆盖默认值，`@override` 在不改代码的前提下替换任意组件。
2. **网格即抽象**：把 world 组织成多维 `DeviceMesh`，让每个并行策略只关心自己那一根轴；三种后端（`spmd_types` 为当前默认）决定这些轴如何暴露给模型代码。
3. **顺序即正确性**：`parallelize → AC → compile → FSDP` 的顺序不可调换；`sum loss + global_valid_tokens 归一化 + 关闭 FSDP 梯度除法` 三者必须一致，否则梯度会错。

理解这三句话，剩下的都是细节。
