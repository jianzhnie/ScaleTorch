# ScaleTorch 代码优化与调试指南

## 项目概述

ScaleTorch 实现了 4D 并行分布式训练框架，进程网格 `[DP, PP, CP, TP]`。参考 Nanotron、Picotron、Megatron-LM 等框架设计。

## 编码规范

- **Linter**: flake8（忽略 W503、W504、E251、E501、E126）
- **Formatter**: yapf + isort（双引号、绝对导入、LF 换行）
- **Pre-commit**: `pre-commit run --all-files`
- **Python >= 3.10**, PyTorch + HuggingFace Transformers

## 测试

```bash
python run_tests.py                                        # 全部测试
python -m unittest tests.test_dist -v                      # 单模块
python -m unittest discover -s tests -p "test_*.py" -v     # 手动发现
```

测试框架: **unittest**（非 pytest）。基类: `tests/test_base.py` — `BaseTestCase` 提供 mock 进程组和分布式操作辅助方法。

## 优化原则

- 从入口文件 `train.py` 开始，按模块逐个优化
- 目标：更简洁、高效、易读、易维护
- 遵循 PyTorch 最新 API 和最佳实践
- 减少内存占用
- 添加类型提示和文档字符串
- 分布式代码使用 `scaletorch.dist` 工具，不直接调用 `torch.distributed`

## 优化任务清单

对每个模块执行以下操作：

1. **修复错误** — 修复代码中的 bug、逻辑错误、边界条件问题，确保模块可正常运行
2. **添加类型提示** — 为函数签名添加完整的类型注解（参数和返回值），使用 `typing` 模块（`Optional`、`Union`、`Tuple`、`Dict` 等）
3. **添加文档和注释** — 为模块、类、公共函数添加文档字符串，说明用途、参数含义和返回值；仅对不明显的逻辑添加行内注释
4. **改善代码结构和可读性** — 提取重复逻辑为函数，合理拆分过长函数，优化变量命名，减少嵌套层级
5. **遵循最佳实践和编码规范** — 符合 flake8 / yapf / isort 规范，避免常见反模式，使用 PyTorch 惯用写法
6. **运行测试验证** — 修改后运行相关单元测试，确保功能正确

## 推荐优化顺序

1. `train.py` — 主训练入口
2. `scaletorch/trainer/` — 配置 (`config.py`)、学习率调度 (`lr_scheduler.py`)
3. `scaletorch/utils/` — 工具函数（checkpoint、device、logger、monitor）
4. `scaletorch/data/` — 数据加载
5. `scaletorch/dist/` — 分布式通信原语
6. `scaletorch/parallel/` — 并行策略（TP/PP/CP/DP + `pg_manager.py`）
7. `scaletorch/models/` — 模型定义（LLaMA、MoE、attention 变体）

## 项目结构

```
ScaleTorch/
├── train.py                    # 主训练入口：config → dist init → 模型创建(TP/PP/CP/DP) → 训练循环
├── scaletorch/
│   ├── data/                   # 数据集与数据加载器
│   ├── dist/                   # 分布式通信原语（all_gather, all_reduce, all_to_all 等）
│   ├── models/
│   │   ├── attention/          # MHA / MQA / GQA / MLA
│   │   ├── model_llama.py      # LLaMA（支持 GQA/KV-head）
│   │   └── moe_model.py        # Mixture-of-Experts
│   ├── parallel/
│   │   ├── pg_manager.py       # ProcessGroupManager：管理 4D 进程网格
│   │   ├── tensor_parallel/    # TP：列/行并行线性层、embedding 分片
│   │   ├── pipeline_parallel/  # PP：1F1B / AFAB 调度
│   │   ├── context_parallel/   # CP：Ring Attention 序列并行
│   │   └── data_parallel/      # DP：梯度桶化 all-reduce
│   ├── trainer/
│   │   ├── config.py           # ScaleTorchArguments（HfArgumentParser dataclass）
│   │   └── lr_scheduler.py     # 学习率调度器
│   └── utils/                  # checkpoint、device、logger、monitor
├── tests/                      # unittest 测试（17 个测试文件 + 1 个 benchmark）
├── examples/                   # MNIST / FSDP / ImageNet / minGPT / picotron 示例
├── scripts/torch_dist/         # 单节点 & 多节点启动脚本
└── doc/                        # 技术文档（中文）：并行策略、通信原语详解
```
