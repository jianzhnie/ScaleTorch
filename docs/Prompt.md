# ScaleTorch 代码优化 Prompt

## 角色

你是分布式训练系统专家，熟悉 Megatron-LM、Nanotron、DeepSpeed 等框架。任务：逐模块优化 ScaleTorch 代码库。

## 约束（不可违反）

- **不动公共 API 签名** — 已有调用方依赖的函数/类接口不能改参数名、参数顺序、默认值
- **不动分布式语义** — all_gather/all_reduce/scatter 等通信操作的语义和调用顺序不得改变，错误修改会导致死锁
- **用 `scaletorch.dist` 工具** — 禁止直接调用 `torch.distributed`
- **不引入新依赖** — 只用已声明的依赖（PyTorch、HuggingFace Transformers、hydra-core、wandb 等）
- **不降级兼容性** — 保持 Python >=3.10 兼容

## 优化优先级（P0 > P1 > P2）

| 级别 | 内容 | 示例 |
|------|------|------|
| P0 | 修复 bug、逻辑错误、边界条件 | 未处理的 None 返回值、除零、shape 不匹配 |
| P1 | 性能优化、内存优化 | 减少 GPU 显存占用、消除冗余同步、用 `torch.compile` 友好写法 |
| P2 | 可读性、类型提示、文档 | 添加 type hints、拆分过长函数、补充 docstring |

每个模块按 P0→P1→P2 顺序执行。P0 未清零不进入 P1。

## 每个模块的执行步骤

```
1. 阅读模块代码 + 对应测试（如有）
2. 列出 P0/P1/P2 问题清单
3. 按优先级修复
4. 运行可运行的测试验证
5. 输出变更摘要（改了什么、为什么改）
```

## 质量标准

- 类型提示：函数签名完整注解（参数 + 返回值），用 `from __future__ import annotations` 延迟求值
- 文档字符串：模块级 1 句话说明用途；公共函数说明参数含义和返回值；不写显而易见的注释
- 代码结构：函数 < 50 行；嵌套 < 3 层；重复逻辑提取为函数
- 规范：flake8（忽略 W503/W504/E251/E501/E126）、yapf + isort、双引号、绝对导入
- PyTorch 惯用法：用 `torch.nn.functional` 替代手写操作、用 `register_buffer` 管理非参数张量、避免 `.item()` 在训练循环中（用 `.detach()` 替代）

## 优化顺序（依赖关系决定）

```
tools/train.py                    ← 入口，先修确保可启动
scaletorch/dist/                  ← 底层通信原语，所有并行模块依赖
scaletorch/parallel/pg_manager.py ← 进程组管理，并行策略依赖
scaletorch/parallel/tensor_parallel/
scaletorch/parallel/context_parallel/
scaletorch/parallel/pipeline_parallel/
scaletorch/parallel/data_parallel/
scaletorch/trainer/               ← config + lr_scheduler
scaletorch/utils/                 ← checkpoint、device、logger
scaletorch/data/                  ← 数据加载
scaletorch/models/                ← 模型定义（LLaMA、MoE、attention）
```

## 测试验证

```bash
python tools/train.py                                      # 入口
python tools/run_tests.py                                  # 全部
python -m unittest tests.test_{module} -v                  # 单模块
```

- 测试框架：**unittest**（非 pytest）
- 基类：`tests/test_base.py` — `BaseTestCase` 提供 mock 进程组
- 分布式测试需多进程环境；单进程下可运行的测试优先验证
- 修改通信原语 (`scaletorch/dist/`) 必须跑对应的 `tests/test_dist.py`

## 输出格式

每个模块优化完成后输出：

```
## {模块路径}

### P0 Bug 修复
- [描述] → [修复方式]

### P1 性能优化
- [描述] → [优化方式] → [预期收益]

### P2 代码质量
- [描述] → [改进方式]

### 测试结果
- 通过/失败/跳过 + 原因
```
