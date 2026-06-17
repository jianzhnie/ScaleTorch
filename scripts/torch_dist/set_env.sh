#!/bin/bash
# 加载 CANN 环境变量（路径需根据实际安装位置调整）
CANN_INSTALL_PATH=${CANN_INSTALL_PATH:-"/usr/local/Ascend"}

if [ -f "$CANN_INSTALL_PATH/ascend-toolkit/set_env.sh" ]; then
    source "$CANN_INSTALL_PATH/ascend-toolkit/set_env.sh"
fi

if [ -f "$CANN_INSTALL_PATH/nnal/atb/set_env.sh" ]; then
    source "$CANN_INSTALL_PATH/nnal/atb/set_env.sh"
fi

# 激活 conda 环境（根据实际环境名调整）
CONDA_ENV_NAME=${CONDA_ENV_NAME:-"rlhf"}
if command -v conda &>/dev/null; then
    conda activate "$CONDA_ENV_NAME" 2>/dev/null || true
fi
