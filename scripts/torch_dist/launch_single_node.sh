#!/bin/bash
#==============================================================#
#   Filename    : run_single_node.sh
#   Description : 启动单节点多卡（NPU/GPU）分布式训练任务
#   Usage       : bash run_single_node.sh <NNODES> <NODE_RANK> <DEVICES_PER_NODE> <MASTER_ADDR> <MASTER_PORT>
#==============================================================#

set -euo pipefail

# =============================================
# 📦 环境变量 (从 launch_dpo.sh 传递)
# =============================================
# NUM_NODES           : 节点总数
# NODE_RANK           : 当前节点的排名
# DEVICES_PER_NODE    : 每个节点的设备数
# MASTER_ADDR         : 主节点 IP
# MASTER_PORT         : 主节点端口
# CKPT_LOAD_DIR       : 检查点加载目录
# CKPT_SAVE_DIR       : 检查点保存目录
# DATA_PATH           : 数据集路径
# TOKENIZER_PATH      : 分词器路径
# LOG_DIR             : 日志目录
# PROJECT_DIR         : 项目根目录
# TRAIN_SCRIPT        : Pytorch 训练脚本

# 校验数值参数
for var in NUM_NODES NODE_RANK DEVICES_PER_NODE MASTER_PORT; do
    if ! [[ ${!var} =~ ^[0-9]+$ ]]; then
        echo "❌ 错误：$var 必须是正整数，当前值: ${!var}"
        exit 1
    fi
done

# 放宽 MASTER_ADDR 校验（支持 IP 和主机名）
if [[ -z "$MASTER_ADDR" ]]; then
    echo "❌ 错误：MASTER_ADDR 不能为空"
    exit 1
fi


# =============================================
# ⚙️ 变量计算与初始化
# =============================================
WORLD_SIZE=$((DEVICES_PER_NODE * NUM_NODES))
FILE_LOCK_DIR="$LOG_DIR/FILE_LOCK"
mkdir -p $FILE_LOCK_DIR
LOCKFILE="$FILE_LOCK_DIR/.training.lock_${NODE_RANK}"

# =============================================
# 🚨 流程
# =============================================
echo "=========================================="
echo "🚀 Starting training on node rank: $NODE_RANK"
echo "   MASTER_ADDR: $MASTER_ADDR"
echo "   MASTER_PORT: $MASTER_PORT"
echo "   DEVICES_PER_NODE: $DEVICES_PER_NODE"
echo "   WORLD_SIZE: $WORLD_SIZE"
echo "=========================================="

# 检查并创建锁文件，防止重复启动
if [ -f "$LOCKFILE" ]; then
    echo "Lock file exists, training might already be running."
    exit 1
fi

touch "$LOCKFILE"
# 确保在退出时删除锁文件
trap "rm -f \"$LOCKFILE\"; echo 'Lock file removed on exit.'" EXIT

# 启动训练
echo "✅ Training started with PID $$"


#----------------------------------------
# 输出运行配置（便于调试）
#----------------------------------------
echo "========================================"
echo "🚀 节点 RANK $NODE_RANK 开始执行训练任务"
echo "   总节点数: $NUM_NODES"
echo "   本节点设备数: $DEVICES_PER_NODE"
echo "   主节点: $MASTER_ADDR:$MASTER_PORT"
echo "   主机: $(hostname)"
echo "   Python 训练脚本: $TRAIN_SCRIPT"
echo "========================================"

#----------------------------------------
# 启动 torchrun 分布式任务
#----------------------------------------
echo "▶️ 正在启动 torchrun..."

# 所有输出（stdout/stderr）都将直接打印，并由上层 launch_distributed.sh 捕获到日志文件
# 无需在此处进行任何日志重定向
torchrun \
    --nproc-per-node="$DEVICES_PER_NODE" \
    --nnodes="$NUM_NODES" \
    --node-rank="$NODE_RANK" \
    --master-addr="$MASTER_ADDR" \
    --master-port="$MASTER_PORT" \
    --max-restarts=0 \
    "$TRAIN_SCRIPT"

#----------------------------------------
# 检查执行结果
#----------------------------------------
TORCHRUN_EXIT_CODE=$?

if [ $TORCHRUN_EXIT_CODE -eq 0 ]; then
    echo "✅ 节点 RANK $NODE_RANK 训练任务成功完成"
else
    echo "❌ 节点 RANK $NODE_RANK 训练失败，退出码: $TORCHRUN_EXIT_CODE"
fi

# 以 torchrun 的退出码退出，这样上层脚本可以捕获到成功或失败状态
exit $TORCHRUN_EXIT_CODE
