#!/bin/bash
# ============================================================
# 魔搭云服务器运行脚本
# 改动：模型直接从 ModelScope 加载，不需要本地下载
# 支持 --resume 断点续训
# ============================================================
set -e

# 环境变量
# 注意：请通过 --api_key 或环境变量注入，切勿硬编码真实密钥
export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-}"
export PYTHON=python

# 云上路径
WORK_DIR=$(pwd)
DATA_DIR="$WORK_DIR/data"
COT_DIR="$DATA_DIR/cot"

echo "============================================================"
echo "CoT Medical Distillation - ModelScope Cloud"
echo "GPU:" 
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "============================================================"

# ============================================
# Step 1: 上传 CoT 数据（如果没有）
# ============================================
if [ ! -f "$COT_DIR/medqa_cot_train.jsonl" ]; then
    echo "CoT 数据不存在！请先从本地上传 data/cot/ 文件夹到云服务器。"
    exit 1
fi
echo "CoT data: $(wc -l < $COT_DIR/medqa_cot_train.jsonl) lines"

# ============================================
# Step 2: 训练（模型自动从 ModelScope 下载，支持断点续训）
# ============================================
echo ""
echo "Starting training..."

python src/train.py \
    --model_name Qwen/Qwen3.5-9B \
    --train_data "$COT_DIR/medqa_cot_train.jsonl" \
    --output_dir checkpoints/medqa-distill \
    --exp_name main-cot-distill \
    --batch_size 4 --grad_accum 4 --epochs 5 \
    --max_length 2048 \
    --eval_data "$DATA_DIR/raw/medqa_validation.json" \
    --eval_dataset_name medqa \
    --eval_max_samples 300 \
    --resume

echo "Training complete!"

# ============================================
# Step 3: 评估
# ============================================
python src/eval.py \
    --model_path Qwen/Qwen3.5-9B \
    --lora_path checkpoints/medqa-distill/epoch_5 \
    --test_data "$DATA_DIR/raw/medqa_test.json" \
    --exp_name main-cot-distill \
    --dataset_name medqa
