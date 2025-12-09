#!/bin/bash
# 加载 CANN 环境变量（路径需根据实际安装位置调整）
install_path=/home/jianzhnie/llmtuner/Ascend/CANN8.2.RC1
source $install_path/ascend-toolkit/set_env.sh
source $install_path/nnal/atb/set_env.sh

# 激活conda环境
source /home/jianzhnie/llmtuner/software/miniconda3/bin/activate rlhf
