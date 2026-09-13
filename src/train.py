"""
CoT reasoning distillation trainer.
Supports joint distillation with Logit-KD + Sequence-KD (CoT) + label loss.
Integrated with SwanLab experiment tracking.
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

# Add src to the path so the eval module in the same directory can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval import load_questions, build_prompt, extract_answer, generate_answer, evaluate as run_qa_eval

import swanlab


# ============================================================
# Dataset
# ============================================================

class CoTDistillDataset(Dataset):
    """
    Each sample contains:
      - input_text: the full input prompt (question + options)
      - label_text: the correct answer text
      - cot_text: the teacher CoT reasoning chain
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
        """Build the input prompt."""
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

        # Tokenize input + CoT as the training sequence
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

        # Labels: compute loss only over the CoT part (mask the input part)
        input_len = len(self.tokenizer(input_text)["input_ids"])
        labels = input_ids.clone()
        labels[:input_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "correct_label": label,           # correct answer text (used for the logit loss)
            "input_len": input_len,
        }


# ============================================================
# Joint distillation loss
# ============================================================

class JointDistillationLoss(nn.Module):
    """CoT sequence distillation loss. The teacher reasoning chain already contains the correct answer, so Seq-KD is sufficient."""
    def __init__(self):
        super().__init__()

    def forward(self, seq_loss: torch.Tensor) -> dict:
        return {
            "loss_total": seq_loss,
            "loss_seq": seq_loss,
        }


# ============================================================
# Custom trainer
# ============================================================

class CoTDistillTrainer:
    """
    A lightweight training loop that avoids the complexity of the HuggingFace Trainer.
    Suitable for a single RTX 3090 + LoRA setting.
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
        """Create a DataLoader with deterministic shuffle so the data order is consistent on resume."""
        g = torch.Generator()
        g.manual_seed(self.seed + epoch)
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size,
            shuffle=True, num_workers=0, pin_memory=True,
            generator=g,
        )

    def save_checkpoint(self, epoch: int, global_step: int, step_in_epoch: int = 0):
        """Save a full training checkpoint (model weights + training state)."""
        ckpt_path = self.output_dir / f"epoch_{epoch+1}"
        self.model.save_pretrained(str(ckpt_path))
        self.tokenizer.save_pretrained(str(ckpt_path))

        # Save the training state
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

        # Record the latest checkpoint path for automatic discovery on resume
        with open(str(self.output_dir / 'latest_checkpoint.txt'), 'w') as f:
            f.write(str(ckpt_path))

        print(f"  Checkpoint saved: {ckpt_path}")
        return ckpt_path

    @staticmethod
    def find_latest_checkpoint(output_dir: str) -> Optional[dict]:
        """Find the latest checkpoint and return epoch and path information."""
        output_dir = Path(output_dir)
        state_path = output_dir / 'training_state.pt'

        # Resume from training_state.pt
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

        # Fallback: infer only from the epoch_N directories
        epoch_dirs = sorted(glob.glob(str(output_dir / 'epoch_*')), key=os.path.getmtime)
        if epoch_dirs:
            match = re.search(r'epoch_(\d+)', os.path.basename(epoch_dirs[-1]))
            epoch = int(match.group(1)) - 1 if match else 0
            return {'epoch': epoch, 'global_step': 0, 'step_in_epoch': 0, 'ckpt_path': epoch_dirs[-1], 'state': None}

        return None

    def train(self, experiment_name: str = "cot-med-distill",
              resume_epoch: int = 0, resume_global_step: int = 0,
              resume_step_in_epoch: int = 0):
        """Training loop with SwanLab logging.

        Args:
            experiment_name: SwanLab experiment name
            resume_epoch: the epoch to start from (the highest completed epoch index)
            resume_global_step: the number of completed global steps
            resume_step_in_epoch: the number of completed batches within the current epoch
        """
        import shutil as _shutil

        start_time = time.time()

        # Clean up stale SwanLab state to avoid the "DataPorter already exists" error
        for _dir in ['swanlog', 'scripts/swanlog']:
            _p = Path(_dir)
            if _p.exists():
                _shutil.rmtree(str(_p), ignore_errors=True)

        # Reload the swanlab module to clear the singleton
        import importlib
        importlib.reload(swanlab)

        # Always create a new SwanLab experiment
        print(f"  [SwanLab] Creating new experiment: {experiment_name}")
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

            # Create a deterministic DataLoader (consistent shuffle order on resume)
            train_loader = self._get_train_loader(epoch)

            # On resume, skip already-processed batches
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
                    # Must be called after optimizer.step() and before the next zero_grad
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1

                    # SwanLab logging
                    swanlab.log({
                        "train/loss": losses["loss_total"].item(),
                        "train/learning_rate": self.scheduler.get_last_lr()[0],
                        "train/epoch": epoch + (step + 1) / len(train_loader),
                    }, step=global_step)

                    # Progress printing (per step)
                    pct = global_step / self.total_steps * 100
                    bar_len = 25
                    filled = int(bar_len * min(pct / 100, 1))
                    bar = "#" * filled + "-" * (bar_len - filled)

                    # GPU memory (PyTorch reserved; nvidia-smi reports 5-7G higher due to bitsandbytes quantization)
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

                    # Save checkpoints by step (for finer-grained resume)
                    if self.save_every_n_steps > 0 and global_step % self.save_every_n_steps == 0:
                        self.save_checkpoint(epoch, global_step, step + 1)

            avg_loss = total_loss / len(train_loader)
            print(f"\nEpoch {epoch+1}/{self.num_epochs} done | Avg Loss: {avg_loss:.4f}")

            # Log epoch summary to SwanLab
            swanlab.log({
                "epoch/avg_loss": avg_loss,
                "epoch/avg_seq_loss": total_seq / len(train_loader),
            }, step=global_step)

            # Save a checkpoint at the end of each epoch
            self.save_checkpoint(epoch, global_step, step_in_epoch=0)

            # Validation (loss-based)
            if self.val_dataset:
                val_loss = self.evaluate()
                swanlab.log({"val/loss": val_loss}, step=global_step)
                print(f"  Val Loss: {val_loss:.4f}")

            # QA accuracy evaluation (after each epoch)
            if self.eval_questions:
                qa_result = self.evaluate_qa(
                    self.eval_questions, self.eval_max_samples, self.eval_dataset_name
                )
                swanlab.log({
                    f"eval/{self.eval_dataset_name}_accuracy": qa_result["accuracy"],
                    f"eval/{self.eval_dataset_name}_correct": qa_result["correct"],
                    f"eval/{self.eval_dataset_name}_total": qa_result["total"],
                }, step=global_step)

                # Save evaluation results to a file
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
        """QA accuracy evaluation (called after each epoch).

        Args:
            eval_questions: list of evaluation questions
            max_samples: maximum number of evaluation samples (None = all)
            dataset_name: dataset name, used as the SwanLab log identifier

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
# Main function
# ============================================================

def load_model_and_tokenizer(model_name: str, lora_config: dict, apply_lora: bool = True):
    """Load the model and apply LoRA + 4-bit quantization.

    Args:
        model_name: model name or path
        lora_config: LoRA configuration dict
        apply_lora: whether to apply LoRA (set to False on resume, when the adapter is loaded manually)
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
    parser.add_argument("--train_data", required=True, help="CoT distillation data in JSONL format")
    parser.add_argument("--val_data", default=None)
    parser.add_argument("--output_dir", default="./checkpoints/cot-distill")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max_length", type=int, default=1536)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exp_name", default="cot-distill", help="SwanLab experiment name")
    parser.add_argument("--resume", action="store_true", default=False,
                        help="Resume training from the latest checkpoint")
    parser.add_argument("--save_every_n_steps", type=int, default=0,
                        help="Save a checkpoint every N steps (0 = only at epoch end)")
    parser.add_argument("--eval_data", default=None,
                        help="Test set (JSON/JSONL) to evaluate after each epoch")
    parser.add_argument("--eval_max_samples", type=int, default=None,
                        help="Maximum number of evaluation samples (None = all; 200-500 is recommended for speed)")
    parser.add_argument("--eval_dataset_name", default="eval",
                        help="Dataset name identifier for evaluation (e.g., medqa, cmexam)")
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
    # Resume logic
    # ============================================
    resume_epoch = 0
    resume_global_step = 0
    resume_step_in_epoch = 0

    if args.resume:
        ckpt_info = CoTDistillTrainer.find_latest_checkpoint(args.output_dir)
        if ckpt_info is None:
            print(f"[WARNING] --resume specified, but no checkpoint found in {args.output_dir}; starting training from scratch.")
        else:
            resume_epoch = ckpt_info['epoch']
            resume_global_step = ckpt_info['global_step']
            resume_step_in_epoch = ckpt_info.get('step_in_epoch', 0)
            ckpt_path = ckpt_info['ckpt_path']
            state = ckpt_info.get('state')

            next_epoch = resume_epoch + 1 if resume_step_in_epoch == 0 else resume_epoch
            if next_epoch >= args.epochs:
                print(f"[INFO] All {args.epochs} epochs are complete; no further training needed.")
                return

            print(f"[RESUME] Resuming from checkpoint:")
            print(f"  Checkpoint: {ckpt_path}")
            print(f"  Completed epochs: {resume_epoch + 1}/{args.epochs}")
            print(f"  Completed global steps: {resume_global_step}")
            if resume_step_in_epoch > 0:
                print(f"  Completed batches in epoch: {resume_step_in_epoch}")
            print(f"  Will resume training from epoch {next_epoch + 1}")

    # Load model (on resume, load base model + LoRA adapter)
    print(f"\nLoading model: {args.model_name}")
    lora_config = {
        "r": args.lora_r,
        "lora_alpha": args.lora_r * 2,
    }

    if args.resume and ckpt_info is not None and ckpt_info.get('ckpt_path'):
        # Load base model + LoRA structure normally, then overwrite weights from the checkpoint
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

    # Load QA evaluation data
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

    # Restore training state on resume
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
