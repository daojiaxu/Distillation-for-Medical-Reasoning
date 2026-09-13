"""
Generate CoT distillation data by calling the DeepSeek API to produce medical reasoning chains.
Output format: {question, options, ground_truth, cot_reasoning, teacher_model}.
"""
import os
import json
import time
import argparse
from pathlib import Path
from openai import OpenAI


def load_questions(data_path: str) -> list[dict]:
    """Load questions in the unified JSON/JSONL format."""
    data_path = Path(data_path)
    if data_path.suffix == ".json":
        with open(data_path, "r", encoding="utf-8") as f:
            return json.load(f)
    elif data_path.suffix == ".jsonl":
        questions = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    questions.append(json.loads(line))
        return questions
    else:
        raise ValueError(f"Unsupported format: {data_path.suffix}")


def build_prompt(question: dict, lang: str, cot_format: str = "structured") -> str:
    """
    Build the prompt that asks the teacher model to generate a structured reasoning chain.

    question fields:
      - question: the question stem
      - options: {A: ..., B: ..., C: ..., D: ...} or None (open-ended)
      - answer: ground-truth answer (used as a reference during generation, not exposed at inference)
    """
    q_text = question["question"]
    options = question.get("options", None)
    answer = question.get("answer", "")

    if lang == "zh":
        sys_prompt = (
            "You are a senior clinician. Please reason through the following medical "
            "multiple-choice question step by step and provide the final answer.\n\n"
            "Strictly follow this output format:\n"
            "1. [Question Analysis] Analyze the key information in the question stem "
            "(symptoms, signs, examination results)\n"
            "2. [Option Analysis] Analyze the correctness/incorrectness of each option one by one\n"
            "3. [Final Answer] Output the answer letter\n\n"
            "The final answer line must contain only a single English letter, e.g., D\n"
            "Do not write the option text, do not add a period, only write the letter."
        )
    else:
        sys_prompt = (
            "You are a senior clinician. Reason through the medical question step by step.\n\n"
            "Strict output format:\n"
            "1. [Question Analysis] Analyze key clinical info\n"
            "2. [Option Analysis] Evaluate each option\n"
            "3. [Final Answer] Output the correct LETTER only\n\n"
            "The final answer line must contain ONLY a single letter (A/B/C/D/E). "
            "No period, no explanation on that line. Just the letter."
        )

    user_content = f"Question: {q_text}\n\n"

    if options:
        for key, val in options.items():
            user_content += f"{key}. {val}\n"

    return sys_prompt, user_content


def generate_cot(
    client: OpenAI,
    model: str,
    question: dict,
    lang: str,
    cot_format: str = "structured",
    temperature: float = 0.3,
) -> dict:
    """Call the API to generate one CoT reasoning chain."""
    sys_prompt, user_content = build_prompt(question, lang, cot_format)

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_content},
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=1024,
        extra_body={"thinking": {"type": "disabled"}},
    )

    cot_text = response.choices[0].message.content or ""

    # Treat empty or too-short responses as failures
    if not cot_text or len(cot_text.strip()) < 20:
        return None

    return {
        "question": question["question"],
        "options": question.get("options", None),
        "ground_truth": question.get("answer", ""),
        "cot_reasoning": cot_text,
        "teacher_model": model,
    }


def _print_progress(i, total, start_time, start_idx, success_count, fail_count):
    """Print a progress bar with percentage, speed, and ETA."""
    done = start_idx + success_count + fail_count
    pct = done / total * 100
    elapsed = time.time() - start_time
    speed = (success_count + fail_count) / elapsed if elapsed > 0 else 0
    eta = (total - done) / speed if speed > 0 else 0

    bar_len = 30
    filled = int(bar_len * done / total)
    bar = "#" * filled + "-" * (bar_len - filled)

    print(
        f"\r[{bar}] {pct:5.1f}% | {done}/{total} | "
        f"OK:{start_idx + success_count} FAIL:{fail_count} | "
        f"{speed:.1f} q/s | ETA: {eta/60:.0f}m{eta%60:.0f}s   ",
        end="", flush=True,
    )
    # Start a new line every 200 questions to avoid terminal buffering issues
    if done % 200 == 0:
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to the input JSON/JSONL file")
    parser.add_argument("--output", required=True, help="Path to the output JSONL file")
    parser.add_argument("--lang", default="zh", choices=["zh", "en"])
    parser.add_argument("--model", default="deepseek-v4-flash",
                        help="deepseek-v4-flash / deepseek-v4-pro")
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--base_url", default="https://api.deepseek.com")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--cot_format", default="structured",
                        choices=["structured", "freeform", "concise", "verbose"])
    parser.add_argument("--sleep", type=float, default=0.5, help="Interval between API calls (seconds)")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("Set the DEEPSEEK_API_KEY environment variable or pass --api_key")

    client = OpenAI(api_key=api_key, base_url=args.base_url)

    questions = load_questions(args.input)
    if args.max_samples:
        questions = questions[:args.max_samples]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume support: check for existing progress
    start_idx = 0
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            existing = sum(1 for _ in f)
        if existing > 0:
            start_idx = existing
            print(f"Found {existing} existing records, resuming from question {start_idx + 1}")

    if start_idx >= len(questions):
        print(f"All {len(questions)} questions are already done. Nothing to do.")
        return

    if start_idx == 0:
        print(f"Total {len(questions)} questions, generating CoT reasoning chains...")
    else:
        print(f"{len(questions) - start_idx} questions remaining, appending...")

    success_count = 0
    skip_count = start_idx
    fail_count = 0
    total = len(questions)
    start_time = time.time()
    mode = "a" if start_idx > 0 else "w"
    with open(output_path, mode, encoding="utf-8") as f:
        for i, q in enumerate(questions):
            if i < start_idx:
                continue

            try:
                result = None
                for retry in range(3):
                    result = generate_cot(client, args.model, q, args.lang, args.cot_format)
                    if result is not None:
                        break
                    time.sleep(3)
                if result is None:
                    fail_count += 1
                    time.sleep(1)
                    if (i + 1) % 20 == 0:
                        _print_progress(i, total, start_time, start_idx, success_count, fail_count)
                    continue
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
                success_count += 1
            except Exception as e:
                fail_count += 1
                if (i + 1) % 20 == 0:
                    _print_progress(i, total, start_time, start_idx, success_count, fail_count)
                time.sleep(5)

            # Print progress every 10 questions
            done = start_idx + success_count + fail_count
            if done % 10 == 0:
                _print_progress(i, total, start_time, start_idx, success_count, fail_count)

            time.sleep(args.sleep)

    elapsed = time.time() - start_time
    print(f"\n{'='*55}")
    print(f"Done! Success {success_count} | Failed {fail_count} | Total {start_idx + success_count}")
    print(f"Elapsed: {elapsed/60:.1f} min | Average: {elapsed/success_count:.1f}s/question" if success_count else "")


if __name__ == "__main__":
    main()
