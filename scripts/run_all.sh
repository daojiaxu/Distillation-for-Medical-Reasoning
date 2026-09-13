#!/bin/bash
# ============================================================
# CoT medical reasoning distillation - 3-dataset simplified plan
# MedQA (main experiment + ablation) + CMExam (cross-lingual) + MedMCQA (cross-domain generalization)
# ============================================================
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"
CONDA_PYTHON="E:/Anaconda3/envs/med-KD/python.exe"
export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-}"

# -----------------------------------------------------------
# Configurable parameters
# -----------------------------------------------------------
MODEL_NAME="Qwen/Qwen3.5-9B"
EPOCHS=5
BATCH_SIZE=1
GRAD_ACCUM=8
LORA_R=16

echo "============================================================"
echo "CoT Medical Distillation Pipeline (3-Dataset Simplified)"
echo "============================================================"
echo "Datasets: MedQA(10K en) + CMExam(16K zh) + MedMCQA(4K en)"
echo "Model: $MODEL_NAME"
echo "Config: epochs=$EPOCHS batch=$BATCH_SIZE grad_accum=$GRAD_ACCUM"
echo "============================================================"

# -----------------------------------------------------------
# Step 0: Check API key
# -----------------------------------------------------------
if [ -z "$DEEPSEEK_API_KEY" ]; then
    echo ""
    echo "ERROR: DEEPSEEK_API_KEY not set!"
    echo "  export DEEPSEEK_API_KEY=\"sk-xxxxxxxx\""
    exit 1
fi

# -----------------------------------------------------------
# Step 1: Generate MedQA CoT distillation data (English)
# -----------------------------------------------------------
echo ""
echo "[Step 1] Generating MedQA CoT data (English, structured)..."
$CONDA_PYTHON scripts/generate_cot_data.py \
    --input "data/raw/medqa_train.json" \
    --output "data/cot/medqa_cot_train.jsonl" \
    --lang en \
    --model deepseek-v4-flash \
    --cot_format structured \
    --sleep 0.5

echo "  MedQA CoT done: data/cot/medqa_cot_train.jsonl"

# -----------------------------------------------------------
# Step 2: Generate CMExam CoT distillation data (Chinese)
# -----------------------------------------------------------
echo ""
echo "[Step 2] Generating CMExam CoT data (Chinese, structured)..."
$CONDA_PYTHON scripts/generate_cot_data.py \
    --input "data/raw/cmexam_train_16k.json" \
    --output "data/cot/cmexam_cot_train.jsonl" \
    --lang zh \
    --model deepseek-v4-flash \
    --cot_format structured \
    --sleep 0.5

echo "  CMExam CoT done: data/cot/cmexam_cot_train.jsonl"

# -----------------------------------------------------------
# Step 3: Training - MedQA main experiment (English distillation)
# -----------------------------------------------------------
echo ""
echo "[Step 3] Distillation training on MedQA..."
$CONDA_PYTHON src/train.py \
    --model_name "$MODEL_NAME" \
    --train_data "data/cot/medqa_cot_train.jsonl" \
    --val_data "data/cot/medqa_cot_val.jsonl" \
    --output_dir "checkpoints/medqa-distill" \
    --batch_size $BATCH_SIZE \
    --grad_accum $GRAD_ACCUM \
    --epochs $EPOCHS \
    --lora_r $LORA_R

# -----------------------------------------------------------
# Step 4: Evaluation - MedQA test
# -----------------------------------------------------------
echo ""
echo "[Step 4] Evaluating on MedQA test..."
$CONDA_PYTHON src/eval.py \
    --model_path "$MODEL_NAME" \
    --lora_path "checkpoints/medqa-distill/epoch_${EPOCHS}" \
    --test_data "data/raw/medqa_test.json" \
    --output "logs/medqa_test_results.json"

# -----------------------------------------------------------
# Step 5: Evaluation - MedMCQA zero-shot cross-domain generalization
# -----------------------------------------------------------
echo ""
echo "[Step 5] Zero-shot cross-domain on MedMCQA..."
$CONDA_PYTHON src/eval.py \
    --model_path "$MODEL_NAME" \
    --lora_path "checkpoints/medqa-distill/epoch_${EPOCHS}" \
    --test_data "data/raw/medmcqa_test.json" \
    --output "logs/medmcqa_zeroshot_results.json"

# -----------------------------------------------------------
# Step 6: Evaluation - CMExam cross-lingual validation
# -----------------------------------------------------------
echo ""
echo "[Step 6] Cross-lingual validation on CMExam..."
$CONDA_PYTHON src/eval.py \
    --model_path "$MODEL_NAME" \
    --lora_path "checkpoints/medqa-distill/epoch_${EPOCHS}" \
    --test_data "data/raw/cmexam_test.json" \
    --output "logs/cmexam_crosslingual_results.json" \
    --max_samples 2000

echo ""
echo "============================================================"
echo "Pipeline complete! Summary:"
echo "  logs/medqa_test_results.json        - MedQA (main result)"
echo "  logs/medmcqa_zeroshot_results.json   - MedMCQA (cross-domain)"
echo "  logs/cmexam_crosslingual_results.json - CMExam (cross-lingual)"
echo "  checkpoints/medqa-distill/           - LoRA weights"
echo "============================================================"


