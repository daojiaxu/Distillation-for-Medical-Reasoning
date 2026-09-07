"""
CoT 推理蒸馏训练器
支持：Logit-KD + Sequence-KD (CoT) + Label Loss 联合蒸馏
集成 SwanLab 实验追踪
"""
import os
import re
import sys
import json
import time
import glob
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel

# 添加 src 目录到 path，以便导入同目录的 eval 模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval import load_questions, build_prompt, extract_answer, generate_answer, evaluate as run_qa_eval

import swanlab


# ============================================================
# 数据集
# ============================================================

class CoTDistillDataset(Dataset):
    """
    每条数据包含：
      - input_text: 完整输入 prompt（题目 + 选项）
      - label_text: 正确答案文本
      - cot_text: 教师的 CoT 推理链
    """

    def __init__(self, data_path: str, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    self.data.append(item)

    def __len__(self):
        return len(self.data)

    def _format_input(self, item: dict) -> str:
        """构建输入 prompt。"""
        q = item["question"]
        options = item.get("options", {})

        text = f"Question: {q}\n"
        if options:
            for k, v in options.items():
                text += f"{k}. {v}\n"
        text += "\nLet's think step by step:\n"
        return text

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = self._format_input(item)
        cot_text = item["cot_reasoning"]
        label = item["ground_truth"]

        # Tokenize input + CoT 作为训练序列
        full_text = input_text + cot_text

        encodings = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encodings["input_ids"].squeeze(0)
        attention_mask = encodings["attention_mask"].squeeze(0)

        # Labels: 只对 CoT 部分计算 loss (input 部分 mask 掉)
        input_len = len(self.tokenizer(input_text)["input_ids"])
        labels = input_ids.clone()
        labels[:input_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "correct_label": label,           # 正确答案文本 (用于 logit loss)
            "input_len": input_len,
        }


# ============================================================
# 联合蒸馏损失
# ============================================================

class JointDistillationLoss(nn.Module):
    """CoT 序列蒸馏损失。教师推理链本身就包含了正确答案，Seq-KD 已足够。"""
    def __init__(self):
        super().__init__()

    def forward(self, seq_loss: torch.Tensor) -> dict:
        return {
            "loss_total": seq_loss,
            "loss_seq": seq_loss,
        }


# ============================================================
# 自定义 Trainer
# ============================================================

class CoTDistillTrainer:
    """
    轻量级训练循环，避免 HuggingFace Trainer 的复杂性。
    适合单卡 3090 + LoRA 场景。
    """

    def __init__(
        self,
        model,
        tokenizer,
        distill_loss: JointDistillationLoss,
        train_dataset: CoTDistillDataset,
        val_dataset: Optional[CoTDistillDataset] = None,
        eval_questions: Optional[list[dict]] = None,
        eval_max_samples: Optional[int] = None,
        eval_dataset_name: str = "eval",
        batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 2e-4,
        warmup_ratio: float = 0.1,
        num_epochs: int = 5,
        max_grad_norm: float = 1.0,
        fp16: bool = True,
        output_dir: str = "./checkpoints",
        log_interval: int = 50,
        device: str = "cuda",
        seed: int = 42,
        save_every_n_steps: int = 0,
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.distill_loss = distill_loss
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.eval_questions = eval_questions
        self.eval_max_samples = eval_max_samples
        self.eval_dataset_name = eval_dataset_name
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_epochs = num_epochs
        self.max_grad_norm = max_grad_norm
        self.fp16 = fp16
        self.output_dir = Path(output_dir)
        self.log_interval = log_interval
        self.device = device
        self.seed = seed
        self.save_every_n_steps = save_every_n_steps

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # DataLoader (recreated per epoch for deterministic shuffle on resume)
        self._train_loader = None

        # Optimizer & Scheduler
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=0.01,
        )
        total_steps = (len(train_dataset) // (batch_size * gradient_accumulation_steps)) * num_epochs
        self.total_steps = max(total_steps, 1)
        warmup_steps = int(self.total_steps * warmup_ratio)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=self.total_steps,
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=fp16)

    def _get_train_loader(self, epoch: int):
        """创建确定性 shuffle 的 DataLoader，保证 resume 时数据顺序一致。"""
        g = torch.Generator()
        g.manual_seed(self.seed + epoch)
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size,
            shuffle=True, num_workers=0, pin_memory=True,
            generator=g,
        )

    def save_checkpoint(self, epoch: int, global_step: int, step_in_epoch: int = 0):
        """保存完整训练 checkpoint（模型权重 + 训练状态）。"""
        ckpt_path = self.output_dir / f"epoch_{epoch+1}"
        self.model.save_pretrained(str(ckpt_path))
        self.tokenizer.save_pretrained(str(ckpt_path))

        # 保存训练状态
        state = {
            'epoch': epoch,
            'global_step': global_step,
            'step_in_epoch': step_in_epoch,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.fp16 else None,
            'rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        state_path = self.output_dir / 'training_state.pt'
        torch.save(state, str(state_path))

        # 记录最新 checkpoint 路径，方便 resume 自动发现
        with open(str(self.output_dir / 'latest_checkpoint.txt'), 'w') as f:
            f.write(str(ckpt_path))

        print(f"  Checkpoint saved: {ckpt_path}")
        return ckpt_path

    @staticmethod
    def find_latest_checkpoint(output_dir: str) -> Optional[dict]:
        """查找最新的 checkpoint，返回 epoch 和路径信息。"""
        output_dir = Path(output_dir)
        state_path = output_dir / 'training_state.pt'

        # 按 training_state.pt 恢复
        if state_path.exists():
            state = torch.load(str(state_path), map_location='cpu', weights_only=False)
            epoch_dirs = sorted(glob.glob(str(output_dir / 'epoch_*')), key=os.path.getmtime)
            if epoch_dirs:
                return {
                    'epoch': state['epoch'],
                    'global_step': state['global_step'],
                    'step_in_epoch': state.get('step_in_epoch', 0),
                    'ckpt_path': epoch_dirs[-1],
                    'state': state,
                }

        # 回退：仅通过 epoch_N 目录判断
        epoch_dirs = sorted(glob.glob(str(output_dir / 'epoch_*')), key=os.path.getmtime)
        if epoch_dirs:
            match = re.search(r'epoch_(\d+)', os.path.basename(epoch_dirs[-1]))
            epoch = int(match.group(1)) - 1 if match else 0
            return {'epoch': epoch, 'global_step': 0, 'step_in_epoch': 0, 'ckpt_path': epoch_dirs[-1], 'state': None}

        return None

    def train(self, experiment_name: str = "cot-med-distill",
              resume_epoch: int = 0, resume_global_step: int = 0,
              resume_step_in_epoch: int = 0):
        """训练循环，集成 SwanLab 记录。

        Args:
            experiment_name: SwanLab 实验名称
            resume_epoch: 从哪个 epoch 开始（已完成的最大 epoch 编号）
            resume_global_step: 已完成的 global step 数
            resume_step_in_epoch: 当前 epoch 内已完成的 batch 数
        """
        import shutil as _shutil

        start_time = time.time()

        # 清理旧的 SwanLab 残留，避免 "DataPorter already exists" 错误
        for _dir in ['swanlog', 'scripts/swanlog']:
            _p = Path(_dir)
            if _p.exists():
                _shutil.rmtree(str(_p), ignore_errors=True)

        # 重新加载 swanlab 模块以清除单例
        import importlib
        importlib.reload(swanlab)

        # 始终创建新 SwanLab 实验
        print(f"  [SwanLab] 创建新实验: {experiment_name}")
        swanlab.init(
            project="cot-medical-distillation",
            experiment_name=experiment_name,
            config={
                "model": "Qwen3.5-9B",
                "batch_size": self.batch_size,
                "grad_accum": self.gradient_accumulation_steps,
                "learning_rate": self.optimizer.param_groups[0]["lr"],
                "epochs": self.num_epochs,
                "max_length": 2048,
                "fp16": self.fp16,
                "resumed": resume_epoch > 0,
            },
        )

        self.model.train()
        global_step = resume_global_step
        start_epoch = resume_epoch + 1 if resume_step_in_epoch == 0 else resume_epoch

        for epoch in range(start_epoch, self.num_epochs):
            total_loss = 0.0
            total_logit = 0.0
            total_seq = 0.0
            self.optimizer.zero_grad()

            # 创建确定性 DataLoader（保证 resume 时 shuffle 顺序一致）
            train_loader = self._get_train_loader(epoch)

            # resume 时跳过已处理的 batch
            skip_batches = resume_step_in_epoch if epoch == resume_epoch else 0

            for step, batch in enumerate(train_loader):
                if step < skip_batches:
                    continue

                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                input_len = batch["input_len"]

                with torch.cuda.amp.autocast(enabled=self.fp16):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        output_hidden_states=False,
                    )
                    seq_loss = outputs.loss

                    losses = self.distill_loss(seq_loss)
                    loss = losses["loss_total"] / self.gradient_accumulation_steps

                if self.fp16:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                total_loss += loss.item()
                total_seq += losses.get("loss_seq", torch.tensor(0)).item()

                if (step + 1) % self.gradient_accumulation_steps == 0:
                    if self.fp16:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    if self.fp16:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    # 必须在 optimizer.step() 之后、下一轮 zero_grad 之前调用
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1

                    # SwanLab 记录
                    swanlab.log({
                        "train/loss": losses["loss_total"].item(),
                        "train/learning_rate": self.scheduler.get_last_lr()[0],
                        "train/epoch": epoch + (step + 1) / len(train_loader),
                    }, step=global_step)

                    # 进度打印（每步）
                    pct = global_step / self.total_steps * 100
                    bar_len = 25
                    filled = int(bar_len * min(pct / 100, 1))
                    bar = "#" * filled + "-" * (bar_len - filled)

                    # GPU 显存 (PyTorch reserved；nvidia-smi 会高出 5-7G 因 bitsandbytes 量化)
                    vram = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0

                    # ETA
                    elapsed = time.time() - start_time
                    speed = global_step / elapsed if elapsed > 0 else 0
                    eta = (self.total_steps - global_step) / speed if speed > 0 else 0

                    print(
                        f"\r  [{bar}] {pct:5.1f}% | "
                        f"Step {global_step}/{self.total_steps} | "
                        f"Loss {losses['loss_total'].item():.4f} | "
                        f"Seq {losses.get('loss_seq', torch.tensor(0)).item():.3f} | "
                        f"LR {self.scheduler.get_last_lr()[0]:.1e} | "
                        f"VRAM {vram:.1f}G | "
                        f"ETA {eta/60:.0f}m{eta%60:.0f}s     ",
                        end="", flush=True,
                    )

                    # 按步数保存 checkpoint（用于更细粒度的断点续训）
                    if self.save_every_n_steps > 0 and global_step % self.save_every_n_steps == 0:
                        self.save_checkpoint(epoch, global_step, step + 1)

            avg_loss = total_loss / len(train_loader)
            print(f"\nEpoch {epoch+1}/{self.num_epochs} done | Avg Loss: {avg_loss:.4f}")

            # SwanLab 记录 epoch 汇总
            swanlab.log({
                "epoch/avg_loss": avg_loss,
                "epoch/avg_seq_loss": total_seq / len(train_loader),
            }, step=global_step)

            # 每个 epoch 结束保存 checkpoint
            self.save_checkpoint(epoch, global_step, step_in_epoch=0)

            # Validation (loss-based)
            if self.val_dataset:
                val_loss = self.evaluate()
                swanlab.log({"val/loss": val_loss}, step=global_step)
                print(f"  Val Loss: {val_loss:.4f}")

            # QA Accuracy 评估（每个 epoch 结束后）
            if self.eval_questions:
                qa_result = self.evaluate_qa(
                    self.eval_questions, self.eval_max_samples, self.eval_dataset_name
                )
                swanlab.log({
                    f"eval/{self.eval_dataset_name}_accuracy": qa_result["accuracy"],
                    f"eval/{self.eval_dataset_name}_correct": qa_result["correct"],
                    f"eval/{self.eval_dataset_name}_total": qa_result["total"],
                }, step=global_step)

                # 保存评估结果到文件
                eval_log_path = self.output_dir / f"eval_epoch_{epoch+1}.json"
                with open(str(eval_log_path), "w", encoding="utf-8") as f:
                    json.dump({
                        "epoch": epoch + 1,
                        "accuracy": qa_result["accuracy"],
                        "correct": qa_result["correct"],
                        "total": qa_result["total"],
                        "results": qa_result["results"],
                    }, f, ensure_ascii=False, indent=2)
                print(f"  Eval results saved: {eval_log_path}")

        swanlab.finish()

    def evaluate(self) -> float:
        self.model.eval()
        total_loss = 0.0

        val_loader = DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False,
        )

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                total_loss += outputs.loss.item()

        self.model.train()
        return total_loss / len(val_loader)

    def evaluate_qa(self, eval_questions: list[dict], max_samples: int = None,
                    dataset_name: str = "eval") -> dict:
        """QA 准确率评估（每个 epoch 结束后调用）。

        Args:
            eval_questions: 评估问题列表
            max_samples: 最大评估样本数（None = 全部）
            dataset_name: 数据集名称，用于 SwanLab 日志标识

        Returns:
            {"accuracy": float, "correct": int, "total": int}
        """
        self.model.eval()
        print(f"\n  --- QA Evaluation: {dataset_name} ---")

        result = run_qa_eval(self.model, self.tokenizer, eval_questions, max_samples)

        acc = result["accuracy"]
        correct = result["correct"]
        total = result["total"]
        print(f"  {dataset_name} Acc: {acc:.4f} ({correct}/{total})")

        self.model.train()
        return result


# ============================================================
# 主函数
# ============================================================

def load_model_and_tokenizer(model_name: str, lora_config: dict, apply_lora: bool = True):
    """加载模型并应用 LoRA + 4-bit 量化。

    Args:
        model_name: 模型名称或路径
        lora_config: LoRA 配置字典
        apply_lora: 是否应用 LoRA（resume 时设为 False，手动加载 adapter）
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    if not apply_lora:
        return model, tokenizer

    lora_cfg = LoraConfig(
        r=lora_config.get("r", 16),
        lora_alpha=lora_config.get("lora_alpha", 32),
        target_modules=lora_config.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        lora_dropout=lora_config.get("lora_dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--train_data", required=True, help="CoT 蒸馏数据 JSONL")
    parser.add_argument("--val_data", default=None)
    parser.add_argument("--output_dir", default="./checkpoints/cot-distill")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max_length", type=int, default=1536)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exp_name", default="cot-distill", help="SwanLab 实验名称")
    parser.add_argument("--resume", action="store_true", default=False,
                        help="从最新 checkpoint 恢复训练")
    parser.add_argument("--save_every_n_steps", type=int, default=0,
                        help="每 N 步保存一次 checkpoint（0=仅 epoch 结束时保存）")
    parser.add_argument("--eval_data", default=None,
                        help="每个 epoch 结束后评估的测试集 (JSON/JSONL)")
    parser.add_argument("--eval_max_samples", type=int, default=None,
                        help="评估最大样本数（None=全部，建议设 200~500 加速）")
    parser.add_argument("--eval_dataset_name", default="eval",
                        help="评估数据集名称标识（如 medqa, cmexam）")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ============================================
    # Resume 逻辑
    # ============================================
    resume_epoch = 0
    resume_global_step = 0
    resume_step_in_epoch = 0

    if args.resume:
        ckpt_info = CoTDistillTrainer.find_latest_checkpoint(args.output_dir)
        if ckpt_info is None:
            print(f"[WARNING] --resume 已指定，但 {args.output_dir} 中未找到 checkpoint，从头开始训练。")
        else:
            resume_epoch = ckpt_info['epoch']
            resume_global_step = ckpt_info['global_step']
            resume_step_in_epoch = ckpt_info.get('step_in_epoch', 0)
            ckpt_path = ckpt_info['ckpt_path']
            state = ckpt_info.get('state')

            next_epoch = resume_epoch + 1 if resume_step_in_epoch == 0 else resume_epoch
            if next_epoch >= args.epochs:
                print(f"[INFO] 所有 {args.epochs} 个 epoch 已完成，无需继续训练。")
                return

            print(f"[RESUME] 从 checkpoint 恢复:")
            print(f"  Checkpoint: {ckpt_path}")
            print(f"  已完成 epoch: {resume_epoch + 1}/{args.epochs}")
            print(f"  已完成 global step: {resume_global_step}")
            if resume_step_in_epoch > 0:
                print(f"  epoch 内已完成 batch: {resume_step_in_epoch}")
            print(f"  将从 epoch {next_epoch + 1} 继续训练")

    # Load model (resume 时加载 base model + LoRA adapter)
    print(f"\nLoading model: {args.model_name}")
    lora_config = {
        "r": args.lora_r,
        "lora_alpha": args.lora_r * 2,
    }

    if args.resume and ckpt_info is not None and ckpt_info.get('ckpt_path'):
        # 正常加载 base model + LoRA 结构，再从 checkpoint 覆写权重
        model, tokenizer = load_model_and_tokenizer(args.model_name, lora_config, apply_lora=True)
        print(f"  Loading LoRA weights from: {ckpt_info['ckpt_path']}")
        model = PeftModel.from_pretrained(model, ckpt_info['ckpt_path'], is_trainable=True)
        model.print_trainable_parameters()
        print(f"  LoRA adapter loaded successfully")
    else:
        model, tokenizer = load_model_and_tokenizer(args.model_name, lora_config)

    # Load data
    print(f"Loading data: {args.train_data}")
    train_dataset = CoTDistillDataset(args.train_data, tokenizer, args.max_length)
    print(f"  Training samples: {len(train_dataset)}")

    val_dataset = None
    if args.val_data:
        val_dataset = CoTDistillDataset(args.val_data, tokenizer, args.max_length)
        print(f"  Validation samples: {len(val_dataset)}")

    # 加载 QA 评估数据
    eval_questions = None
    if args.eval_data:
        print(f"Loading eval data: {args.eval_data}")
        eval_questions = load_questions(args.eval_data)
        print(f"  Eval questions: {len(eval_questions)}")
        if args.eval_max_samples:
            print(f"  Will evaluate on first {args.eval_max_samples} samples")

    # Distillation loss
    distill_loss = JointDistillationLoss()

    # Trainer
    trainer = CoTDistillTrainer(
        model=model,
        tokenizer=tokenizer,
        distill_loss=distill_loss,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        eval_questions=eval_questions,
        eval_max_samples=args.eval_max_samples,
        eval_dataset_name=args.eval_dataset_name,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        fp16=True,
        output_dir=args.output_dir,
        device=device,
        seed=args.seed,
        save_every_n_steps=args.save_every_n_steps,
    )

    # Resume 时恢复训练状态
    if args.resume and ckpt_info is not None and ckpt_info.get('state') is not None:
        state = ckpt_info['state']
        trainer.optimizer.load_state_dict(state['optimizer_state_dict'])
        trainer.scheduler.load_state_dict(state['scheduler_state_dict'])
        if trainer.fp16 and state.get('scaler_state_dict') is not None:
            trainer.scaler.load_state_dict(state['scaler_state_dict'])
        if state.get('rng_state') is not None:
            torch.set_rng_state(state['rng_state'])
        if state.get('cuda_rng_state') is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state['cuda_rng_state'])
        print("  Optimizer / Scheduler / Scaler state restored")

    print("\n" + "=" * 60)
    if args.resume and resume_epoch > 0:
        print(f"Resuming CoT Distillation Training from epoch {resume_epoch + 2}")
    else:
        print("Starting CoT Distillation Training")
    print(f"  Method: Seq-KD (CoT Distillation)")
    print("=" * 60 + "\n")

    trainer.train(
        experiment_name=args.exp_name,
        resume_epoch=resume_epoch,
        resume_global_step=resume_global_step,
        resume_step_in_epoch=resume_step_in_epoch,
    )

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
