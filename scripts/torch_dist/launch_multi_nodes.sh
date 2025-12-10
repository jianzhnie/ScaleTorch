#!/bin/bash
#==============================================================#
#   Filename    : launch_distributed.sh
#   Description : 多节点分布式训练启动脚本
#                 - 支持 Ascend NPU / NVIDIA GPU
#                 - 支持通过命令行参数指定节点列表文件
#                 - 支持 Ctrl+C 优雅中断所有远程任务
#                 - 将各节点日志分别保存，便于调试
#   Usage       : bash launch_distributed.sh [path/to/node_list.txt]
#==============================================================#

# --- 脚本安全设置 ---
# -e: 命令执行失败时立即退出
# -u: 尝试使用未定义的变量时立即退出
# -o pipefail: 管道中的命令失败时，将整个管道的退出码设为失败

set -euo pipefail

#----------------------------------------
# 帮助信息和参数解析
#----------------------------------------
usage() {
    echo "Usage: $0 [NODE_LIST_FILE]"
    echo
    echo "启动多节点分布式训练。"
    echo
    echo "Arguments:"
    echo "  NODE_LIST_FILE    包含节点 IP 或主机名的文件路径 (默认为: ./node_list.txt)"
    exit 1
}

# 如果提供了参数，则使用第一个参数作为节点列表文件路径
if [ "$#" -gt 1 ]; then
    echo "❌ 错误: 参数过多。"
    usage
fi

# 如果提供了参数，则使用第一个参数作为节点列表文件路径，否则使用默认值
NODE_LIST_FILE="${1:-"./node_list.txt"}"

#----------------------------------------
# 分布式训练全局配置 (可被环境变量覆盖)
#----------------------------------------
# 检查节点文件是否存在
if [ ! -f "$NODE_LIST_FILE" ]; then
    echo "❌ 错误: 节点列表文件 '$NODE_LIST_FILE' 不存在！"
    usage
fi

# 从文件读取节点列表到数组 (使用 < "$VAR" 语法，并忽略空行和注释行)
mapfile -t NODE_HOSTS < <(grep -v -e '^\s*$' -e '^\s*#' "$NODE_LIST_FILE")

# 检查节点列表是否为空
if [ ${#NODE_HOSTS[@]} -eq 0 ]; then
    echo "❌ 错误: 节点列表 '$NODE_LIST_FILE' 为空。"
    exit 1
fi

# --- 训练相关参数，来自你原始脚本的配置 ---
PROJECT_DIR="/home/jianzhnie/llmtuner/llm/ScaleTorch/scripts/torch_dist"
DATA_PATH=""
TOKENIZER_PATH=""
CKPT_LOAD_DIR=""

# --- 分布式配置 ---
MASTER_ADDR="${NODE_HOSTS[0]}"
MASTER_PORT="29500"
DEVICES_PER_NODE=8
SSH_USER="jianzhnie"
SSH_TIMEOUT=30

# --- 远程脚本和日志配置 ---
OUTPUT_DIR="$PROJECT_DIR/work_dir"
REMOTE_MAIN_SCRIPT="$PROJECT_DIR/launch_multi_nodes.sh"
REMOTE_SCRIPT="$PROJECT_DIR/launch_single_node.sh"
TRAIN_SCRIPT="$PROJECT_DIR/run_dist_demo.py"
DATETIME=$(date +%Y-%m-%d_%H-%M-%S)
LOG_DIR="$OUTPUT_DIR/logs/$DATETIME"
CKPT_SAVE_DIR=""

# --- 复制脚本和配置 ---
mkdir -p $LOG_DIR
cp $REMOTE_MAIN_SCRIPT $LOG_DIR
cp $REMOTE_SCRIPT $LOG_DIR
cp $TRAIN_SCRIPT $LOG_DIR

# --- 只读常量 ---
readonly NUM_NODES=${#NODE_HOSTS[@]}

#----------------------------------------
# 信号处理 (优雅退出)
#----------------------------------------
# 全局存储所有远程任务的 PID
PIDS=()
# 当脚本接收到 INT (Ctrl+C), TERM, EXIT 信号时，执行 cleanup 函数
trap cleanup INT TERM EXIT

cleanup() {
    local -r exit_code=$?
    echo -e "\n⚠️  接收到中断信号或脚本退出，正在清理所有远程节点任务..."

    if [ ${#PIDS[@]} -gt 0 ]; then
        echo "   -> 正在发送 SIGTERM 信号..."
        for pid in "${PIDS[@]}"; do
            if ps -p "$pid" > /dev/null; then
                kill "$pid" 2>/dev/null || true
            fi
        done

        sleep 1.0 # 给1秒钟的优雅退出时间

        echo "   -> 正在检查并强制终止未退出的进程..."
        for pid in "${PIDS[@]}"; do
            if ps -p "$pid" > /dev/null; then
                echo "      - 强制终止进程 $pid..."
                kill -9 "$pid" 2>/dev/null || true
            fi
        done
    fi

    echo "✅ 清理完成。"
    # 如果是因为中断信号退出，则返回 130
    if [ $exit_code -eq 130 ]; then
        exit 130
    else
        # 否则，返回原始退出码
        exit "$exit_code"
    fi
}
#----------------------------------------
# 主逻辑函数
#----------------------------------------

# 打印配置信息
print_config() {
    echo "========================================================"
    echo "🚀 开始启动多节点分布式训练"
    echo "--------------------------------------------------------"
    echo "  总节点数量      : $NUM_NODES"
    echo "  节点列表        : ${NODE_HOSTS[*]}"
    echo "  每节点设备数    : $DEVICES_PER_NODE"
    echo "  主节点 (Master) : $MASTER_ADDR:$MASTER_PORT"
    echo "  SSH 用户        : $SSH_USER"
    echo "  远程项目作目录    : $PROJECT_DIR"
    echo "  远程执行脚本    : $REMOTE_SCRIPT"
    echo "  PyTorch训练脚本   : $TRAIN_SCRIPT"
    echo "  日志保存目录    : $LOG_DIR"
    echo "  检查点保存目录  : $CKPT_SAVE_DIR"
    echo "========================================================"
}

# 启动所有节点
launch_nodes() {
    echo "⏳ 正在并行启动所有节点的任务..."

    for i in "${!NODE_HOSTS[@]}"; do
        local node_host=${NODE_HOSTS[$i]}
        local node_rank=$i
        # 为每个节点定义清晰的日志文件路径
        local log_file="$LOG_DIR/rank-${node_rank}_host-${node_host}.log"

        echo "  -> 启动节点 [Rank $node_rank] @ $node_host (日志: $log_file)"

        # 使用 SSH 执行远程命令，并将该节点的标准输出和错误都重定向到其独立的日志文件
        ssh \
            -o StrictHostKeyChecking=no \
            -o ConnectTimeout="$SSH_TIMEOUT" \
            -o BatchMode=yes \
            -o ServerAliveInterval=30 \
            -o ServerAliveCountMax=3 \
            "$SSH_USER@$node_host" "

                # 这是在远程节点上执行的命令块
                set -euo pipefail;
            cd '$PROJECT_DIR' || exit 1;

            export NUM_NODES='$NUM_NODES';
            export NODE_RANK='$node_rank';
            export DEVICES_PER_NODE='$DEVICES_PER_NODE';
            export MASTER_ADDR='$MASTER_ADDR';
            export MASTER_PORT='$MASTER_PORT';
            export CKPT_LOAD_DIR='$CKPT_LOAD_DIR';
            export CKPT_SAVE_DIR='$CKPT_SAVE_DIR';
            export DATA_PATH='$DATA_PATH';
            export TOKENIZER_PATH='$TOKENIZER_PATH';
            export LOG_DIR='$LOG_DIR';
            export PROJECT_DIR='$PROJECT_DIR'
            export TRAIN_SCRIPT='$TRAIN_SCRIPT'

            # 加载环境变量
            set +u
            source set_env.sh
            # 使用 exec 确保远程脚本的退出码被正确传递
            exec nohup bash '$REMOTE_SCRIPT'
        " > "$log_file" 2>&1 &

        PIDS+=($!)
        sleep 0.1
    done
}

# 等待所有任务完成并汇总结果
wait_for_completion() {
    echo "--------------------------------------------------------"
    echo "✅ 所有节点任务已启动，正在等待其完成..."
    echo "   你可以使用 'tail -f $LOG_DIR/*' 来实时查看所有节点的日志。"

    local success_count=0
    local failed_count=0

    for i in "${!PIDS[@]}"; do
        local pid=${PIDS[$i]}
        local node_host=${NODE_HOSTS[$i]}
        local node_rank=$i
        local log_file="$LOG_DIR/rank-${node_rank}_host-${node_host}.log"

        # `wait` 命令会返回子进程的退出码
        if wait "$pid"; then
            echo "   [✔️] 节点 $node_rank ($node_host) 任务成功完成。"
            success_count=$((success_count + 1))
        else
            local exit_code=$?
            echo "   [❌] 节点 $node_rank ($node_host) 任务失败！(退出码: $exit_code)"
            echo "       详情请检查日志: $log_file"
            failed_count=$((failed_count + 1))
        fi
    done

    echo "========================================================"
    if [ $failed_count -eq 0 ]; then
        echo "🎉🎉🎉 所有 $success_count 个节点任务全部成功完成！"
    else
        echo "💥 任务总结: $success_count 个成功, $failed_count 个失败。"
        echo "   请检查上述失败节点的日志文件进行排查。"
        # 以失败状态码退出
        exit 1
    fi
    echo "========================================================"
}

#----------------------------------------
# 主执行流程
#----------------------------------------
main() {
    print_config
    launch_nodes
    # 任务已全部启动，现在等待它们完成。
    # 在此期间如果用户按 Ctrl+C，INT/TERM trap 会被触发。
    # 如果任务正常结束，我们需要禁用 EXIT trap，防止 cleanup 被错误调用。
    trap - EXIT
    if wait_for_completion; then
        exit 0
    else
        exit 1
    fi
}

# 执行 main 函数
main
