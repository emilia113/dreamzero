#!/bin/bash
# 启动 BI 分析 server（替代正常 server 启动脚本）
# 用法: bash analysis/launch_bi_server.sh [model_path] [port] [nproc]
#
# 启动后正常运行评测 client 即可。Ctrl+C 后自动保存 BI 分析结果。

MODEL_PATH=${1:-"./checkpoints/dreamzero"}
PORT=${2:-8000}
NPROC=${3:-1}
MASTER_PORT=${MASTER_PORT:-29500}
BI_SAVE_DIR=${BI_SAVE_DIR:-"analysis/bi_results"}

mkdir -p $BI_SAVE_DIR

echo "============================================"
echo "  DreamZero Block Influence 分析"
echo "  Model: $MODEL_PATH"
echo "  Port: $PORT"
echo "  nproc: $NPROC"
echo "  BI 结果保存至: $BI_SAVE_DIR"
echo "============================================"

torchrun --nproc_per_node $NPROC \
    --master_port $MASTER_PORT \
    analysis/block_bi_analysis.py \
    --model-path $MODEL_PATH \
    --port $PORT \
    --bi-save-dir $BI_SAVE_DIR
