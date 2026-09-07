"""
CoT 推理蒸馏训练器 v2
改进：
  1. max_length 默认 2048（覆盖 95%+ CoT）
  2. 训练 prompt 使用 chat template（与评估一致）
  3. LoRA 扩大模块：q/k/v/o + gate/up/down_proj，rank=32
  4. 支持过滤后的 CoT 数据
集成 SwanLab 实验追踪 + 断点续传
"""
import os
import re
import sys
import json
import time
import glob
import shutil
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

import swanlab


# ============================================================
# 数据集：训练 prompt 使用 chat template，与评估对齐
# ============================================================

class CoTDistillDataset(Dataset):
    """
    v2 改进：使用 apply_chat_template 构建 prompt，与评估阶段一致。
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
        """v2: 使用 chat template 构建 prompt，与 eval.py 的 build_prompt 一致。"""
        q = item["question"]
        options = item.get("options", {})
        opts_str = "\n".join([f"{k}. {v}" for k, v in options.items()])

        messages = [
            {"role": "system", "content": (
                "You are a medical AI. Briefly reason, then output ONLY: Answer: X "
                "(X = A/B/C/D/E). No extra text after the answer."
            )},
            {"role": "user", "content": f"Question: {q}\n{opts_str}"},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        return prompt

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = self._format_input(item)
        cot_text = item["cot_reasoning"]

        # 拼接 input + CoT 作为完整训练序列
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
            "correct_label": item.get("ground_truth", ""),
            "input_len": input_len,
        }


# ============================================================
# 蒸馏损失
# ============================================================

class JointDistillationLoss(nn.Module):
    """CoT 序列蒸馏损失。"""
    def __init__(self):
        super().__init__()

    def forward(self, seq_loss: torch.Tensor) -> dict:
        return {
            "loss_total": seq_loss,
            "loss_seq": seq_loss,
        }


# ============================================================
# 模型加载：v2 LoRA 扩大模块 + rank=32
# ============================================================

def load_model_and_tokenizer(model_name: str, lora_config: dict, apply_lora: bool = True):
    """v2: LoRA 覆盖 q/k/v/o + gate/up/down_proj，rank=32。"""
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

    # v2: 扩大 LoRA 目标模块
    lora_cfg = LoraConfig(
        r=lora_config.get("r", 32),
        lora_alpha=lora_config.get("lora_alpha", 64),
        target_modules=lora_config.get("target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]),
        lora_dropout=lora_config.get("lora_dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_cfg)

    return model, tokenizer


# ============================================================
# 工具函数
# ============================================================

def load_questions(path: str) -> list[dict]:
    data = []
    path = Path(path)
    if path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif path.suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    return data


def extract_answer(text: str) -> str:
    clean = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    if len(clean) < 5:
        clean = text
    patterns = [
        r'\[Final Answer\]\s*\n?\s*([A-E])',
        r'(?:Final Answer|最终答案|answer|Answer)\s*[:：]\s*([A-Ea-e])',
        r'Final Answer\]\s*\n?\s*([A-E])',
        r'(?:^|\n)\s*([A-E])\s*$',
    ]
    for pattern in patterns:
        match = re.search(pattern, clean, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).upper()
    return ""


def normalize_answer(ans) -> str:
    if isinstance(ans, int):
        return chr(ord('A') + ans)
    return str(ans).strip().upper()


# ============================================================
# Trainer
# ============================================================

class CoTDistillTrainer:
    def __init__(
        self,
        model,
        tokenizer,
        distill_loss: JointDistillationLoss,
        train_dataset: CoTDistillDataset,
        val_dataset: Optional[CoTDistillDataset] = None,
        eval_questions: Optional[list[dict]] = None,
        eval_max_samples: int = 0,
        eval_dataset_name: str = "eval",
        batch_size: int = 1,
        gradient_accumulation_steps: int = 8,
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
        eval_batch_size: int = 8,
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
        self.eval_batch_size = eval_batch_size

        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=0, pin_memory=True,
        )

        total_steps = (len(train_dataset) // (batch_size * gradient_accumulation_steps)) * num_epochs
        self.total_steps = max(total_steps, 1)
        warmup_steps = int(total_steps * warmup_ratio)

        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=learning_rate, betas=(0.9, 0.999), weight_decay=0.01,
        )
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
        )
        self.scaler = torch.amp.GradScaler('cuda', enabled=fp16)

    @staticmethod
    def find_latest_checkpoint(output_dir):
        output_dir = Path(output_dir)
        latest_file = output_dir / "latest_checkpoint.txt"
        if latest_file.exists():
            with open(latest_file) as f:
                ckpt_name = f.read().strip()
            ckpt_path = output_dir / ckpt_name
            state_file = output_dir / "training_state.pt"
            state = None
            if state_file.exists():
                state = torch.load(str(state_file), map_location="cpu", weights_only=False)
            match = re.search(r'epoch_(\d+)', ckpt_name)
            if match:
                epoch = int(match.group(1))
                global_step = state.get('global_step', 0) if state else 0
                step_in_epoch = state.get('step_in_epoch', 0) if state else 0
                return {'ckpt_path': str(ckpt_path), 'epoch': epoch,
                        'global_step': global_step, 'step_in_epoch': step_in_epoch,
                        'state': state}
        return None

    def save_checkpoint(self, epoch, global_step, step_in_epoch=0):
        ckpt_dir = self.output_dir / f"epoch_{epoch}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(ckpt_dir))
        self.tokenizer.save_pretrained(str(ckpt_dir))

        state = {
            'epoch': epoch,
            'global_step': global_step,
            'step_in_epoch': step_in_epoch,
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'scaler_state': self.scaler.state_dict(),
            'rng_state': torch.get_rng_state(),
        }
        torch.save(state, str(self.output_dir / "training_state.pt"))
        with open(self.output_dir / "latest_checkpoint.txt", "w") as f:
            f.write(f"epoch_{epoch}")
        print(f"  Checkpoint saved: {ckpt_dir}")

    def evaluate_qa(self):
        """QA 准确率评估（批量推理，显著加速）。"""
        if not self.eval_questions:
            return None
        questions = self.eval_questions
        if self.eval_max_samples > 0:
            questions = questions[:self.eval_max_samples]

        self.model.eval()
        self.tokenizer.padding_side = "left"  # CausalLM batch generation 需要左填充
        correct = 0
        total = 0
        t0 = time.time()
        bs = self.eval_batch_size

        for batch_start in range(0, len(questions), bs):
            batch = questions[batch_start:batch_start + bs]
            prompts = []
            for q in batch:
                opts_str = "\n".join([f"{k}. {v}" for k, v in q.get("options", {}).items()])
                messages = [
                    {"role": "system", "content": (
                        "You are a medical AI. Briefly reason, then output ONLY: Answer: X "
                        "(X = A/B/C/D/E). No extra text after the answer."
                    )},
                    {"role": "user", "content": f"Question: {q['question']}\n{opts_str}"},
                ]
                prompts.append(self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False,
                ))

            inputs = self.tokenizer(
                prompts, return_tensors="pt", padding=True,
                truncation=True, max_length=3072,
            ).to(self.model.device)

            with torch.no_grad():
                gen = self.model.generate(
                    **inputs, max_new_tokens=2048,
                    do_sample=False, pad_token_id=self.tokenizer.eos_token_id,
                )

            for j, q in enumerate(batch):
                text = self.tokenizer.decode(
                    gen[j][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
                )
                pred = extract_answer(text)
                gt = normalize_answer(q.get("answer", ""))
                if pred == gt:
                    correct += 1
                total += 1

            done = batch_start + len(batch)
            pct = done / len(questions) * 100
            bar_len = 20
            filled = int(bar_len * done / len(questions))
            bar = "#" * filled + "-" * (bar_len - filled)
            elapsed = time.time() - t0
            speed = done / elapsed if elapsed > 0 else 0
            eta = (len(questions) - done) / speed if speed > 0 else 0
            print(f"\r  [{bar}] {pct:5.1f}% | {correct}/{done} | "
                  f"{speed:.1f}q/s | ETA: {eta:.0f}s  ", end="", flush=True)

        self.tokenizer.padding_side = "right"  # 恢复默认
        print()
        acc = correct / total if total > 0 else 0
        print(f"  {self.eval_dataset_name} Acc: {acc:.4f} ({correct}/{total})")
        self.model.train()
        return acc

    def train(self, experiment_name: str = "cot-med-distill",
              resume_epoch: int = 0, resume_global_step: int = 0,
              resume_step_in_epoch: int = 0):
        start_time = time.time()

        # 清理旧 SwanLab 残留
        for _dir in ['swanlog', 'scripts/swanlog']:
            _p = Path(_dir)
            if _p.exists():
                shutil.rmtree(str(_p), ignore_errors=True)
        import importlib
        importlib.reload(swanlab)

        print(f"  [SwanLab] 创建新实验: {experiment_name}")
        swanlab.init(
            project="cot-medical-distillation",
            experiment_name=experiment_name,
            config={
                "model": "Qwen3.5-9B",
                "version": "v2",
                "batch_size": self.batch_size,
                "grad_accum": self.gradient_accumulation_steps,
                "learning_rate": self.optimizer.param_groups[0]["lr"],
                "epochs": self.num_epochs,
                "max_length": 2048,
                "lora_modules": "q/k/v/o + gate/up/down_proj",
                "lora_r": 32,
                "fp16": self.fp16,
                "resumed": resume_epoch > 0,
            },
        )

        self.model.train()
        global_step = resume_global_step

        for epoch in range(resume_epoch, self.num_epochs):
            total_loss = 0.0
            total_seq = 0.0
            self.optimizer.zero_grad()

            start_step = resume_step_in_epoch if epoch == resume_epoch else 0

            for step, batch in enumerate(self.train_loader):
                if step < start_step:
                    continue

                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                with torch.amp.autocast('cuda', enabled=self.fp16):
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
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1

                    # 进度打印
                    pct = global_step / self.total_steps * 100
                    bar_len = 25
                    filled = int(bar_len * min(pct / 100, 1))
                    bar = "#" * filled + "-" * (bar_len - filled)
                    vram = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
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

                    swanlab.log({
                        "train/loss": losses["loss_total"].item(),
                        "train/learning_rate": self.scheduler.get_last_lr()[0],
                        "train/epoch": epoch + (step + 1) / len(self.train_loader),
                    }, step=global_step)

                    if self.save_every_n_steps > 0 and global_step % self.save_every_n_steps == 0:
                        self.save_checkpoint(epoch, global_step, step + 1)

            print()
            avg_loss = total_loss / len(self.train_loader)
            print(f"Epoch {epoch}/{self.num_epochs - 1} done | Avg Loss: {avg_loss:.4f}")

            swanlab.log({
                "epoch/avg_loss": avg_loss,
                "epoch/avg_seq_loss": total_seq / len(self.train_loader),
            }, step=global_step)

            self.save_checkpoint(epoch, global_step)

            if self.eval_questions:
                print(f"\n  --- QA Evaluation: {self.eval_dataset_name} ---")
                acc = self.evaluate_qa()
                swanlab.log({f"eval/{self.eval_dataset_name}_accuracy": acc}, step=global_step)

                eval_result = {
                    "epoch": epoch,
                    "accuracy": acc,
                    "correct": int(acc * (self.eval_max_samples if self.eval_max_samples > 0 else len(self.eval_questions))),
                    "total": self.eval_max_samples if self.eval_max_samples > 0 else len(self.eval_questions),
                }
                eval_path = self.output_dir / f"eval_epoch_{epoch}.json"
                with open(eval_path, "w", encoding="utf-8") as f:
                    json.dump(eval_result, f, ensure_ascii=False, indent=2)
                print(f"  Eval results saved: {eval_path}")

            resume_step_in_epoch = 0

        swanlab.finish()
        print(f"\nTraining complete! Total time: {(time.time()-start_time)/3600:.1f}h")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="CoT Distillation Training v2")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--output_dir", default="./checkpoints/medqa-distill-v2")
    parser.add_argument("--val_data", default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exp_name", default="cot-distill-v2")
    parser.add_argument("--save_every_n_steps", type=int, default=0)
    parser.add_argument("--eval_data", default=None)
    parser.add_argument("--eval_max_samples", type=int, default=0,
                        help="评估抽样数量，0=全量评估")
    parser.add_argument("--eval_dataset_name", default="eval")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Resume
    resume_epoch = 0
    resume_global_step = 0
    resume_step_in_epoch = 0
    ckpt_info = None

    # 非 resume 模式：清理旧 checkpoints，从 epoch_0 干净开始
    if not args.resume:
        output_path = Path(args.output_dir)
        if output_path.exists():
            for old in output_path.iterdir():
                if old.is_dir() and old.name.startswith("epoch_"):
                    shutil.rmtree(str(old), ignore_errors=True)
                elif old.name in ("latest_checkpoint.txt", "training_state.pt"):
                    old.unlink(missing_ok=True)
            print(f"  [CLEAN] 已清理旧 checkpoints: {args.output_dir}")

    if args.resume:
        ckpt_info = CoTDistillTrainer.find_latest_checkpoint(args.output_dir)
        if ckpt_info is None:
            print(f"[WARNING] --resume 已指定，但未找到 checkpoint，从头开始。")
        else:
            resume_epoch = ckpt_info['epoch']
            resume_global_step = ckpt_info['global_step']
            resume_step_in_epoch = ckpt_info.get('step_in_epoch', 0)
            next_epoch = resume_epoch + 1 if resume_step_in_epoch == 0 else resume_epoch
            if next_epoch >= args.epochs:
                print(f"[INFO] 所有 {args.epochs} 个 epoch 已完成。")
                return
            print(f"[RESUME] 从 checkpoint 恢复:")
            print(f"  Checkpoint: {ckpt_info['ckpt_path']}")
            print(f"  已完成 epoch: {resume_epoch}/{args.epochs - 1}")
            print(f"  已完成 global step: {resume_global_step}")
            print(f"  将从 epoch {next_epoch} 继续训练")

    # Load model
    print(f"\nLoading model: {args.model_name}")
    lora_config = {
        "r": args.lora_r,
        "lora_alpha": args.lora_r * 2,
    }

    if args.resume and ckpt_info is not None and ckpt_info.get('ckpt_path'):
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

    # Eval data
    eval_questions = None
    if args.eval_data:
        print(f"Loading eval data: {args.eval_data}")
        eval_questions = load_questions(args.eval_data)
        print(f"  Eval questions: {len(eval_questions)}")
        if args.eval_max_samples > 0:
            print(f"  Will evaluate on first {args.eval_max_samples} samples")
        else:
            print(f"  Will evaluate on ALL {len(eval_questions)} samples (full)")

    # Distillation loss
    distill_loss = JointDistillationLoss()

    # Trainer
    trainer = CoTDistillTrainer(
        model=model,
        tokenizer=tokenizer,
        distill_loss=distill_loss,
        train_dataset=train_dataset,
        eval_questions=eval_questions,
        eval_max_samples=args.eval_max_samples,
        eval_dataset_name=args.eval_dataset_name,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
        max_grad_norm=1.0,
        fp16=True,
        output_dir=args.output_dir,
        log_interval=50,
        device=device,
        seed=args.seed,
        save_every_n_steps=args.save_every_n_steps,
        eval_batch_size=8,
    )

    # Restore optimizer state if resuming
    if args.resume and ckpt_info and ckpt_info.get('state'):
        state = ckpt_info['state']
        if 'optimizer_state' in state:
            trainer.optimizer.load_state_dict(state['optimizer_state'])
        if 'scheduler_state' in state:
            trainer.scheduler.load_state_dict(state['scheduler_state'])
        if 'scaler_state' in state:
            trainer.scaler.load_state_dict(state['scaler_state'])
        if 'rng_state' in state:
            torch.set_rng_state(state['rng_state'])
        print("  Optimizer / Scheduler / Scaler state restored")

    print(f"\n{'='*60}")
    if resume_epoch > 0:
        print(f"Resuming CoT Distillation Training v2 from epoch {next_epoch}")
    else:
        print(f"Starting CoT Distillation Training v2")
    print(f"  Method: Seq-KD (CoT Distillation)")
    print(f"  Max Length: {args.max_length} tokens (v1=768)")
    print(f"  LoRA: r={args.lora_r}, modules=q/k/v/o+gate/up/down (v1=r16, q/k/v/o only)")
    print(f"  Train Prompt: chat template (aligned with eval)")
    print(f"{'='*60}\n")

    trainer.train(
        experiment_name=args.exp_name,
        resume_epoch=resume_epoch,
        resume_global_step=resume_global_step,
        resume_step_in_epoch=resume_step_in_epoch,
    )


if __name__ == "__main__":
    main()
