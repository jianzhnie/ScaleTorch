# Pytorch Training Toolkit

PyTorch Training Toolkit is an open-source library for using PyTorch to train all kinds of deeplearning models.

\`\`

## Reference

### Training Framework

[**Nanotron**](https://github.com/huggingface/nanotron)

Our framework for training large language models, featuring various parallelism strategies.

[**Picotron**](https://github.com/huggingface/picotron)

Minimalistic 4D-parallelism distributed training framework for education purpose.

[**Megatron-LM**](https://github.com/NVIDIA/Megatron-LM)

NVIDIA's framework for training large language models, featuring various parallelism strategies.

[**DeepSpeed**](https://www.deepspeed.ai/)

Microsoft's deep learning optimization library, featuring ZeRO optimization stages and various parallelism strategies.

[**FairScale**](https://github.com/facebookresearch/fairscale/tree/main)

A PyTorch extension library for large-scale training, offering various parallelism and optimization techniques.

[**Colossal-AI**](https://colossalai.org/)

An integrated large-scale model training system with various optimization techniques.

[**`torchtitan`**](https://github.com/pytorch/torchtitan)

A PyTorch native library for large model training.

[**GPT-NeoX**](https://github.com/EleutherAI/gpt-neox)

EleutherAI's framework for training large language models, used to train GPT-NeoX-20B.

[**LitGPT**](https://github.com/Lightning-AI/litgpt)

Lightning AI's implementation of 20+ state-of-the-art open source LLMs, with a focus on reproducibility.

[**OpenDiLoCo**](https://github.com/PrimeIntellect-ai/OpenDiLoCo)

An open source framework for training language models across compute clusters with DiLoCo.

[**torchgpipe**](https://github.com/kakaobrain/torchgpipe)

A GPipe implementation in PyTorch.

[**OSLO**](https://github.com/EleutherAI/oslo)

The Open Source for Large-scale Optimization framework for large-scale modeling.

### Distribution techniques

[**Data parallelism**](https://siboehm.com/articles/22/data-parallel-training)

Comprehensive explanation of data parallel training in deep learning.

[**ZeRO**](https://arxiv.org/abs/1910.02054)

Introduces the Zero Redundancy Optimizer for training large models with memory optimization.

[**FSDP**](https://arxiv.org/abs/2304.11277)

Fully Sharded Data Parallel training implementation in PyTorch.

[**Tensor and sequence parallelism + selective recomputation**](https://arxiv.org/abs/2205.05198)

Advanced techniques for efficient large-scale model training combining different parallelism strategies.

[**Pipeline parallelism**](https://developer.nvidia.com/blog/scaling-language-model-training-to-a-trillion-parameters-using-megatron/#pipeline_parallelism)

NVIDIA's guide to implementing pipeline parallelism for large model training.

[**Breadth-first pipeline parallelism**](https://arxiv.org/abs/2211.05953)

Includes broad discussions of PP schedules.

[**Ring all-reduce**](https://andrew.gibiansky.com/blog/machine-learning/baidu-allreduce/)

Detailed explanation of the ring all-reduce algorithm used in distributed training.

[**Ring Flash Attention**](https://github.com/zhuzilin/ring-flash-attention)

Implementation of the Ring Attention mechanism combined with FlashAttention for efficient training.

[**Ring Attention tutorial**](https://coconut-mode.com/posts/ring-attention/)

Tutorial explaining the concepts and implementation of Ring Attention.

[**ZeRO and 3D**](https://www.deepspeed.ai/tutorials/large-models-w-deepspeed/#understanding-performance-tradeoff-between-zero-and-3d-parallelism)

DeepSpeed's guide to understanding the trade-offs between ZeRO and 3D parallelism strategies.

[**Mixed precision training**](https://arxiv.org/abs/1710.03740)

Introduces mixed precision training techniques for deep learning models.

[**Visualizing 6D mesh parallelism**](https://main-horse.github.io/posts/visualizing-6d/)

Explains the collective communication involved in a 6D parallel mesh.

### Landmark LLM scaling papers

[**Megatron-LM**](https://arxiv.org/abs/1909.08053)

Introduces tensor parallelism and efficient model parallelism techniques for training large language models.

[**Megatron-Turing NLG 530B**](https://developer.nvidia.com/blog/using-deepspeed-and-megatron-to-train-megatron-turing-nlg-530b-the-worlds-largest-and-most-powerful-generative-language-model/)

Describes the training of a 530B parameter model using a combination of DeepSpeed and Megatron-LM frameworks.

[**PaLM**](https://arxiv.org/abs/2204.02311)

Introduces Google's Pathways Language Model, demonstrating strong performance across hundreds of language tasks and reasoning capabilities.

[**Gemini**](https://arxiv.org/abs/2312.11805)

Presents Google's multimodal model architecture capable of processing text, images, audio, and video inputs.

[**Llama 3**](https://arxiv.org/abs/2407.21783)

Introduces the Llama 3 herd of models.

[**DeepSeek-V3**](https://arxiv.org/abs/2412.19437v1)

DeepSeek's report on the architecture and training of the DeepSeek-V3 model.

## Contributing

Please read CONTRIBUTING.md for details on our code of conduct, and the process for submitting pull requests to us.

License
PyTorch Training Toolkit is licensed under the Apache 2.0 License.
