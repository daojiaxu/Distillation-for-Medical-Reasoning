"""
Download Qwen3.5-9B locally via ModelScope, then load it with transformers.
"""
import os
import shutil
import time
from pathlib import Path

MODEL_NAME = "Qwen/Qwen3.5-9B"
LOCAL_DIR = Path(__file__).resolve().parent.parent / "models" / "Qwen3.5-9B"


def download_from_modelscope():
    """Download the model from the ModelScope community."""
    from modelscope import snapshot_download

    print(f"[1/2] Downloading {MODEL_NAME} from ModelScope...")
    print(f"      Target directory: {LOCAL_DIR}")

    # Skip if the model already exists locally
    if LOCAL_DIR.exists() and any(LOCAL_DIR.iterdir()):
        existing = list(LOCAL_DIR.glob("*.safetensors")) + list(LOCAL_DIR.glob("*.bin"))
        if existing:
            print(f"      Model files already exist, skipping download")
            return

    t0 = time.time()
    snapshot_download(
        MODEL_NAME,
        cache_dir=str(LOCAL_DIR),
        local_dir=str(LOCAL_DIR),
    )
    print(f"      Download finished ({time.time() - t0:.0f}s)")

    # Print the total file size
    total_size = sum(f.stat().st_size for f in LOCAL_DIR.rglob("*") if f.is_file())
    print(f"      Total size: {total_size / 1e9:.1f} GB")


def load_and_verify():
    """Load the local model with transformers to verify it."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n[2/2] Loading the local model for verification...")
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
    print(f"      Load finished ({elapsed:.1f}s)")
    print(f"      Model parameters: {params:.1f}B")
    print(f"      Vocabulary size: {tokenizer.vocab_size}")
    print(f"      Model path: {LOCAL_DIR}")


def main():
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)

    download_from_modelscope()
    load_and_verify()

    print(f"\nDone! Model saved to: {LOCAL_DIR}")
    print(f"For subsequent training, use: --model_name {LOCAL_DIR}")


if __name__ == "__main__":
    main()
