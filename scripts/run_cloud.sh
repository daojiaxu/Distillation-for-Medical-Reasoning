#!/bin/bash
# ============================================================
# ModelScope cloud server run script
# Change: the model is loaded directly from ModelScope, no local download needed
# Supports --resume for checkpoint continuation
# ============================================================
set -e

# Environment variables
# Note: inject via --api_key or environment variable; never hardcode a real key
export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-}"
export PYTHON=python

# Cloud paths
WORK_DIR=$(pwd)
DATA_DIR="$WORK_DIR/data"
COT_DIR="$DATA_DIR/cot"

echo "============================================================"
echo "CoT Medical Distillation - ModelScope Cloud"
echo "GPU:" 
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "============================================================"

# ============================================
# Step 1: Upload CoT data (if missing)
# ============================================
if [ ! -f "$COT_DIR/medqa_cot_train.jsonl" ]; then
    echo "CoT data not found! Please upload the data/cot/ folder from local to the cloud server first."
    exit 1
fi
echo "CoT data: $(wc -l < $COT_DIR/medqa_cot_train.jsonl) lines"

# ============================================
# Step 2: Training (model auto-downloads from ModelScope, supports resume)
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
# Step 3: Evaluation
# ============================================
python src/eval.py \
    --model_path Qwen/Qwen3.5-9B \
    --lora_path checkpoints/medqa-distill/epoch_5 \
    --test_data "$DATA_DIR/raw/medqa_test.json" \
    --exp_name main-cot-distill \
    --dataset_name medqa
