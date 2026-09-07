"""
通过 ModelScope (魔搭) 下载 Qwen3.5-9B 到本地，再用 transformers 加载。
"""
import os
import shutil
import time
from pathlib import Path

MODEL_NAME = "Qwen/Qwen3.5-9B"
LOCAL_DIR = Path(r"C:\Users\armstrong\Desktop\知识蒸馏研究\models\Qwen3.5-9B")


def download_from_modelscope():
    """从魔搭社区下载模型到本地。"""
    from modelscope import snapshot_download

    print(f"[1/2] 从魔搭下载 {MODEL_NAME}...")
    print(f"      目标目录: {LOCAL_DIR}")

    # 如果已存在则跳过
    if LOCAL_DIR.exists() and any(LOCAL_DIR.iterdir()):
        existing = list(LOCAL_DIR.glob("*.safetensors")) + list(LOCAL_DIR.glob("*.bin"))
        if existing:
            print(f"      已有模型文件，跳过下载")
            return

    t0 = time.time()
    snapshot_download(
        MODEL_NAME,
        cache_dir=str(LOCAL_DIR),
        local_dir=str(LOCAL_DIR),
    )
    print(f"      下载完成 ({time.time() - t0:.0f}s)")

    # 打印文件大小
    total_size = sum(f.stat().st_size for f in LOCAL_DIR.rglob("*") if f.is_file())
    print(f"      总大小: {total_size / 1e9:.1f} GB")


def load_and_verify():
    """用 transformers 加载本地模型验证。"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n[2/2] 加载本地模型验证...")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(
        str(LOCAL_DIR), trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(LOCAL_DIR),
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    params = sum(p.numel() for p in model.parameters()) / 1e9
    elapsed = time.time() - t0
    print(f"      加载完成 ({elapsed:.1f}s)")
    print(f"      模型参数: {params:.1f}B")
    print(f"      词表大小: {tokenizer.vocab_size}")
    print(f"      模型路径: {LOCAL_DIR}")


def main():
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)

    download_from_modelscope()
    load_and_verify()

    print(f"\n完成! 模型已保存到: {LOCAL_DIR}")
    print(f"后续训练使用: --model_name {LOCAL_DIR}")


if __name__ == "__main__":
    main()
