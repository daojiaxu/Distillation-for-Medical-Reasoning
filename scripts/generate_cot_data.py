"""
生成 CoT 蒸馏数据：调用 DeepSeek API 生成医学推理链。
输出格式：{question, options, answer, cot_reasoning, teacher_logits}
"""
import os
import json
import time
import argparse
from pathlib import Path
from openai import OpenAI


def load_questions(data_path: str) -> list[dict]:
    """加载统一格式的题目数据。"""
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
    构建 prompt，要求教师模型生成结构化推理链。

    question 字段：
      - question: 题干
      - options: {A: ..., B: ..., C: ..., D: ...} 或 None (open-ended)
      - answer: 正确答案 (用于生成阶段的 few-shot 参考，实际推理时不暴露)
    """
    q_text = question["question"]
    options = question.get("options", None)
    answer = question.get("answer", "")

    if lang == "zh":
        sys_prompt = (
            "你是一位资深临床医生。请对以下医学选择题进行逐步推理，最终给出答案。\n\n"
            "严格按以下格式输出：\n"
            "1. 【题干分析】分析题干关键信息（症状、体征、检查结果）\n"
            "2. 【选项分析】逐一分析每个选项的正确/错误原因\n"
            "3. 【最终答案】输出答案字母\n\n"
            "最终答案那行必须只输出一个英文字母，例如：D\n"
            "不要写选项文字，不要加句号，只写字母。"
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

    user_content = f"题目：{q_text}\n\n" if lang == "zh" else f"Question: {q_text}\n\n"

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
    """调用 API 生成一条 CoT 推理链。"""
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

    # 空响应判定为失败
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
    """打印进度条，含百分比、速度、ETA。"""
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
    # 每 200 题换行一次，避免终端缓冲问题
    if done % 200 == 0:
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="输入 JSON/JSONL 文件路径")
    parser.add_argument("--output", required=True, help="输出 JSONL 文件路径")
    parser.add_argument("--lang", default="zh", choices=["zh", "en"])
    parser.add_argument("--model", default="deepseek-v4-flash",
                        help="deepseek-v4-flash / deepseek-v4-pro")
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--base_url", default="https://api.deepseek.com")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--cot_format", default="structured",
                        choices=["structured", "freeform", "concise", "verbose"])
    parser.add_argument("--sleep", type=float, default=0.5, help="API 调用间隔 (秒)")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("请设置 DEEPSEEK_API_KEY 环境变量或通过 --api_key 传入")

    client = OpenAI(api_key=api_key, base_url=args.base_url)

    questions = load_questions(args.input)
    if args.max_samples:
        questions = questions[:args.max_samples]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 断点续传：检查已有进度
    start_idx = 0
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            existing = sum(1 for _ in f)
        if existing > 0:
            start_idx = existing
            print(f"检测到已有 {existing} 条记录，从第 {start_idx + 1} 题续传")

    if start_idx >= len(questions):
        print(f"全部 {len(questions)} 题已完成！无事可做。")
        return

    if start_idx == 0:
        print(f"共 {len(questions)} 道题，开始生成 CoT 推理链...")
    else:
        print(f"剩余 {len(questions) - start_idx} 道题，追加生成...")

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
                    self._print_progress(i, total, start_time, start_idx, success_count, fail_count)
                time.sleep(5)

            # 每 10 题打印一次进度
            done = start_idx + success_count + fail_count
            if done % 10 == 0:
                _print_progress(i, total, start_time, start_idx, success_count, fail_count)

            time.sleep(args.sleep)

    elapsed = time.time() - start_time
    print(f"\n{'='*55}")
    print(f"完成! 成功 {success_count} | 失败 {fail_count} | 总计 {start_idx + success_count}")
    print(f"耗时: {elapsed/60:.1f} min | 平均: {elapsed/success_count:.1f}s/题" if success_count else "")


if __name__ == "__main__":
    main()
