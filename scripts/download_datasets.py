"""
下载所有实验需要的医学 QA 数据集到 data/raw/
"""
import os
import json
from pathlib import Path
from datasets import load_dataset

RAW_DIR = Path(r"C:\Users\armstrong\Desktop\知识蒸馏研究\data\raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)


def download_medqa():
    """MedQA (USMLE) - 英文医学考试题"""
    print("\n" + "=" * 60)
    print("[1/4] Downloading MedQA (USMLE)...")
    print("=" * 60)

    try:
        ds = load_dataset("bigbio/med_qa", trust_remote_code=True)
        print(f"  Splits: {list(ds.keys())}")
        for split_name, split_data in ds.items():
            questions = []
            for item in split_data:
                q = {
                    "question": item.get("question", ""),
                    "answer": item.get("answer", ""),
                    "options": {},
                }
                # 构建选项
                if "options" in item and item["options"]:
                    for k, v in item["options"].items():
                        q["options"][k] = v
                elif "choices" in item and item["choices"]:
                    for i, choice in enumerate(item["choices"]):
                        q["options"][chr(65 + i)] = choice
                questions.append(q)

            out_path = RAW_DIR / f"medqa_{split_name}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(questions, f, ensure_ascii=False, indent=2)
            print(f"  {split_name}: {len(questions)} questions -> {out_path}")
        return True
    except Exception as e:
        print(f"  bigbio/med_qa failed: {e}")
        print("  Trying alternative: med_qa...")
        try:
            ds = load_dataset("med_qa", trust_remote_code=True)
            for split_name, split_data in ds.items():
                questions = []
                for item in split_data:
                    q = {"question": item["question"], "answer": item.get("answer_idx", ""), "options": {}}
                    if "options" in item:
                        for k, v in item["options"].items():
                            q["options"][k] = v
                    questions.append(q)
                out_path = RAW_DIR / f"medqa_{split_name}.json"
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(questions, f, ensure_ascii=False, indent=2)
                print(f"  {split_name}: {len(questions)} questions -> {out_path}")
            return True
        except Exception as e2:
            print(f"  All MedQA attempts failed: {e2}")
            return False


def download_medmcqa():
    """MedMCQA - 印度医学入学考试，英文"""
    print("\n" + "=" * 60)
    print("[2/4] Downloading MedMCQA...")
    print("=" * 60)

    try:
        ds = load_dataset("medmcqa", trust_remote_code=True)
        print(f"  Splits: {list(ds.keys())}")

        for split_name, split_data in ds.items():
            questions = []
            for item in split_data:
                q = {
                    "question": item["question"],
                    "answer": item.get("cop", ""),  # correct option
                    "options": {},
                }
                # MedMCQA 的选项在 opa/opb/opc/opd 字段
                for label, key in [("A", "opa"), ("B", "opb"), ("C", "opc"), ("D", "opd")]:
                    if key in item and item[key]:
                        q["options"][label] = item[key]
                questions.append(q)

            # MedMCQA 训练集很大 (~180k)，取子集
            if "train" in split_name and len(questions) > 4000:
                questions = questions[:4000]
                print(f"  {split_name}: sampled {len(questions)} from full set")
            out_path = RAW_DIR / f"medmcqa_{split_name}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(questions, f, ensure_ascii=False, indent=2)
            print(f"  {split_name}: {len(questions)} questions -> {out_path}")
        return True
    except Exception as e:
        print(f"  Failed: {e}")
        return False


def download_pubmedqa():
    """PubMedQA - 基于 PubMed 摘要的 yes/no/maybe 问答"""
    print("\n" + "=" * 60)
    print("[3/4] Downloading PubMedQA...")
    print("=" * 60)

    try:
        ds = load_dataset("bigbio/pubmed_qa", "pubmed_qa_labeled_fold0_bigbio_qa", trust_remote_code=True)
        print(f"  Splits: {list(ds.keys())}")

        for split_name, split_data in ds.items():
            questions = []
            for item in split_data:
                q = {
                    "question": item["question"],
                    "answer": item.get("answer", [""])[0] if isinstance(item.get("answer"), list) else item.get("answer", ""),
                    "options": {"A": "yes", "B": "no", "C": "maybe"},
                }
                questions.append(q)

            out_path = RAW_DIR / f"pubmedqa_{split_name}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(questions, f, ensure_ascii=False, indent=2)
            print(f"  {split_name}: {len(questions)} questions -> {out_path}")
        return True
    except Exception as e:
        print(f"  bigbio/pubmed_qa failed: {e}")
        print("  Trying alternative: qiaojin/PubMedQA...")
        try:
            ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", trust_remote_code=True)
            for split_name, split_data in ds.items():
                questions = []
                for item in split_data:
                    q = {
                        "question": item["question"],
                        "answer": item.get("final_decision", ""),
                        "options": {"A": "yes", "B": "no", "C": "maybe"},
                    }
                    questions.append(q)
                out_path = RAW_DIR / f"pubmedqa_{split_name}.json"
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(questions, f, ensure_ascii=False, indent=2)
                print(f"  {split_name}: {len(questions)} questions -> {out_path}")
            return True
        except Exception as e2:
            print(f"  All PubMedQA attempts failed: {e2}")
            return False


def download_cblue():
    """CBLUE - 中文医疗 NLP 基准 (含 CMExam 执业医师考试题)"""
    print("\n" + "=" * 60)
    print("[4/4] Downloading CBLUE (Chinese Medical NLP Benchmark)...")
    print("=" * 60)

    try:
        ds = load_dataset("CBLUE/cmexam", trust_remote_code=True)
        print(f"  CBLUE-CMExam splits: {list(ds.keys())}")
        for split_name, split_data in ds.items():
            questions = []
            for item in split_data:
                q = {
                    "question": item.get("question", ""),
                    "answer": item.get("answer", ""),
                    "options": {},
                }
                if "choices" in item and item["choices"]:
                    for i, choice in enumerate(item["choices"]):
                        q["options"][chr(65 + i)] = choice
                questions.append(q)

            out_path = RAW_DIR / f"cblue_cmexam_{split_name}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(questions, f, ensure_ascii=False, indent=2)
            print(f"  {split_name}: {len(questions)} questions -> {out_path}")
        return True
    except Exception as e:
        print(f"  CBLUE/cmexam failed: {e}")
        print("  Trying CBLUE/KUAKE-QIC...")
        try:
            ds = load_dataset("CBLUE/KUAKE-QIC", trust_remote_code=True)
            for split_name, split_data in ds.items():
                questions = []
                for item in split_data:
                    q = {
                        "question": item.get("query", ""),
                        "answer": item.get("label", ""),
                    }
                    questions.append(q)
                out_path = RAW_DIR / f"cblue_kuake_qic_{split_name}.json"
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(questions, f, ensure_ascii=False, indent=2)
                print(f"  KUAKE-QIC {split_name}: {len(questions)} questions -> {out_path}")
            return True
        except Exception as e2:
            print(f"  All CBLUE attempts failed: {e2}")
            return False


def main():
    results = {}

    results["medqa"] = download_medqa()
    results["medmcqa"] = download_medmcqa()
    results["pubmedqa"] = download_pubmedqa()
    results["cblue"] = download_cblue()

    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    for name, ok in results.items():
        status = "OK" if ok else "FAILED"
        print(f"  {name}: {status}")

    total_files = sum(1 for f in RAW_DIR.glob("*.json"))
    print(f"\n  Total JSON files in data/raw/: {total_files}")


if __name__ == "__main__":
    main()
