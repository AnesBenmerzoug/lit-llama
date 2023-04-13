import os
from pathlib import Path

from transformers import LlamaTokenizer, LlamaForCausalLM


def download_hf_model(
    model_name: str = "decapoda-research/llama-7b-hf",
    model_dir: Path = Path("checkpoints/llama-7b-hf"),
) -> None:
    print("Downloading model weights and tokenzier from pretrained LLaMA from HuggingFace")

    LlamaTokenizer.from_pretrained(model_name, cache_dir=os.fspath(model_dir))

    LlamaForCausalLM.from_pretrained(model_name, cache_dir=os.fspath(model_dir))


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(download_hf_model)
