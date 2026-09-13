"""
Medical QA evaluation script with SwanLab logging.
Supports: MedQA, MedMCQA, PubMedQA, CMExam
"""
import json
import argparse
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

import swanlab


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


def build_prompt(q: dict, tokenizer) -> str:
    """Build the prompt with the chat template, with thinking mode disabled."""
    opts_str = "\n".join([f"{k}. {v}" for k, v in q.get("options", {}).items()])
    messages = [
        {"role": "system", "content": (
            "You are a medical AI. Briefly reason, then output ONLY: Answer: X "
            "(X = A/B/C/D/E). No extra text after the answer."
        )},
        {"role": "user", "content": f"Question: {q['question']}\n{opts_str}"},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )


def extract_answer(text: str) -> str:
    """Extract the option letter (A/B/C/D/E) from the model output. Compatible with Qwen3.5 think tags."""
    # Remove <think>...</think>
    clean = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    if len(clean) < 5:
        clean = text

    patterns = [
        # "[Final Answer]\nA" or "[Final Answer] A" — common 2B model output
        r'\[Final Answer\]\s*\n?\s*([A-E])',
        # "Final Answer: A" or "Answer: A" (also matches the Chinese answer label)
        r'(?:Final Answer|最终答案|answer|Answer)\s*[:：]\s*([A-Ea-e])',  # matches English and Chinese answer labels
        # "Final Answer]\nA" — some truncated cases
        r'Final Answer\]\s*\n?\s*([A-E])',
        # A standalone letter at the end
        r'(?:^|\n)\s*([A-E])\s*$',
    ]
    for pattern in patterns:
        match = re.search(pattern, clean, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).upper()
    return ""


def normalize_answer(ans) -> str:
    """Normalize the answer format: int -> letter, str -> upper. Compatible with MedMCQA (0-3) and MedQA (A-E)."""
    if isinstance(ans, int):
        return chr(ord('A') + ans)
    return str(ans).strip().upper()


def generate_answer(model, tokenizer, prompt: str, max_new_tokens: int = 2048) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def evaluate(model, tokenizer, questions: list[dict], max_samples: int = None,
             batch_size: int = 1) -> dict:
    """Batch evaluation; significantly faster when batch_size > 1."""
    if max_samples:
        questions = questions[:max_samples]

    correct = 0
    total = 0
    results = []
    t0 = time.time()
    device = model.device

    for batch_start in range(0, len(questions), batch_size):
        batch = questions[batch_start:batch_start + batch_size]
        prompts = [build_prompt(q, tokenizer) for q in batch]

        # Batch tokenization (left padding, required by CausalLM)
        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True,
            padding_side="left", truncation=True, max_length=3072,
        ).to(device)

        # Batch generation
        with torch.no_grad():
            gen = model.generate(
                **inputs, max_new_tokens=2048,
                do_sample=False, pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        # Decode one by one
        for j, q in enumerate(batch):
            i = batch_start + j
            output = tokenizer.decode(
                gen[j][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
            )
            predicted = extract_answer(output)
            ground_truth = normalize_answer(q.get("answer", ""))

            is_correct = (predicted == ground_truth)
            if is_correct:
                correct += 1
            total += 1

            results.append({
                "id": i,
                "question": q["question"][:100],
                "ground_truth": ground_truth,
                "predicted": predicted,
                "output": output[:300],
                "correct": is_correct,
            })

        # Progress bar
        done = batch_start + len(batch)
        pct = done / len(questions) * 100
        bar_len = 20
        filled = int(bar_len * done / len(questions))
        bar = "#" * filled + "-" * (bar_len - filled)
        elapsed = time.time() - t0
        speed = done / elapsed if elapsed > 0 else 0
        eta = (len(questions) - done) / speed if speed > 0 else 0
        print(
            f"\r  [{bar}] {pct:5.1f}% | {correct}/{done} | "
            f"speed: {speed:.1f}q/s | ETA: {eta:.0f}s  ",
            end="", flush=True,
        )

    print()
    accuracy = correct / total if total > 0 else 0
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Model path or HuggingFace model name")
    parser.add_argument("--lora_path", default=None, help="LoRA weights path (optional)")
    parser.add_argument("--test_data", required=True, help="Test set in JSON/JSONL format")
    parser.add_argument("--output", default=None, help="Output JSON path for evaluation results")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Inference batch size; larger is faster (default 8)")
    parser.add_argument("--exp_name", default=None, help="SwanLab experiment name (optional, for logging results)")
    parser.add_argument("--dataset_name", default="unknown", help="Dataset name identifier")
    args = parser.parse_args()

    # Optional SwanLab logging
    use_swanlab = args.exp_name is not None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading model: {args.model_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    if args.lora_path:
        print(f"Loading LoRA weights: {args.lora_path}")
        model = PeftModel.from_pretrained(model, args.lora_path)
        model = model.merge_and_unload()

    model.eval()

    print(f"Loading test data: {args.test_data}")
    questions = load_questions(args.test_data)
    print(f"  Total questions: {len(questions)}")

    print("\nEvaluating...")
    result = evaluate(model, tokenizer, questions, args.max_samples, args.batch_size)

    print(f"\n{'='*50}")
    print(f"Accuracy: {result['accuracy']:.4f} ({result['correct']}/{result['total']})")
    print(f"{'='*50}")

    # Log evaluation metrics to SwanLab
    if use_swanlab:
        swanlab.init(
            project="cot-medical-distillation",
            experiment_name=args.exp_name,
        )
        swanlab.log({
            f"eval/{args.dataset_name}_accuracy": result["accuracy"],
            f"eval/{args.dataset_name}_correct": result["correct"],
            f"eval/{args.dataset_name}_total": result["total"],
        })
        swanlab.finish()
        print(f"  SwanLab logged: eval/{args.dataset_name}_accuracy = {result['accuracy']:.4f}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
