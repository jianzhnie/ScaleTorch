# ScaleTorch 代码优化与调试指南

## 项目概述

ScaleTorch 是一个基于 PyTorch 的分布式训练工具库，专注于实现和演示多种分布式训练策略（DP、TP、PP、CP 等）。该项目参考了 Nanotron、Picotron、Megatron-LM 等优秀的开源训练框架。

## 项目结构

```
ScaleTorch/
├── scaletorch/
│   ├── data/              # 数据加载模块
│   │   ├── dataloader.py  # 微批次数据加载器
│   │   ├── dataset.py     # 数据集定义
│   │   └── pretrain_dataset.py  # 预训练数据集
│   ├── dist/              # 分布式通信模块
│   │   ├── dist.py        # 分布式操作
│   │   └── utils.py       # 分布式工具函数
│   ├── models/            # 模型定义
│   │   ├── attention/     # 注意力机制实现（MHA、GQA、MQA、MLA）
│   │   ├── model_llama.py # Llama 模型
│   │   └── moe_model.py   # MoE 模型
│   ├── parallel/          # 并行策略实现
│   │   ├── context_parallel/  # 上下文并行
│   │   ├── data_parallel/     # 数据并行（含梯度桶）
│   │   ├── pipeline_parallel/ # 流水线并行
│   │   ├── tensor_parallel/   # 张量并行
│   │   └── pg_manager.py      # 进程组管理器
│   ├── trainer/           # 训练器模块
│   │   ├── config.py      # 配置定义
│   │   └── lr_scheduler.py # 学习率调度器
│   └── utils/             # 工具函数
│       ├── checkpoint.py  # 检查点管理
│       ├── device.py      # 设备管理
│       ├── logger_utils.py # 日志工具
│       ├── monitor.py     # 性能监控
│       └── utils.py       # 通用工具
├── examples/              # 示例脚本
├── tests/                 # 单元测试
└── train.py               # 主训练入口
```

## 编码规范

- **Linter**: flake8（忽略 W503、W504、E251、E501、E126）
- **Formatter**: yapf + isort
- **字符串**: 使用双引号 `"`
- **导入**: 优先使用绝对导入
- **预提交**: 使用 pre-commit 钩子

## 优化与调试任务

### 1. 代码优化原则
- 从入口文件 `train.py` 开始，逐个模块优化代码
- 使代码更简洁、高效、易读、易维护
- 遵循 PyTorch 最佳实践
- 使用 PyTorch 新 API
- 优化内存占用
- 添加类型提示
- 完善文档字符串

### 2. 验证与测试
- 确保每个模块代码能正常运行
- 运行现有的单元测试验证功能
- 确保训练流程正确且高效

### 3. 优化顺序（推荐）
1. `train.py` - 主训练入口
2. `scaletorch/trainer/` - 训练配置与调度
3. `scaletorch/utils/` - 工具函数
4. `scaletorch/data/` - 数据加载
5. `scaletorch/dist/` - 分布式通信
6. `scaletorch/parallel/` - 并行策略
7. `scaletorch/models/` - 模型定义

### 4. 测试命令

```bash
# 运行训练
python train.py

# 运行所有测试
python run_tests.py

# 或使用 unittest
python -m unittest discover -s tests -p "test_*.py" -v
```


