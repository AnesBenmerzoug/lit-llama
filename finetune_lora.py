"""
Instruction-tuning with LoRA on the Alpaca dataset.

Note: If you run into a CUDA error "Expected is_sm80 to be true, but got false", uncomment the line
`torch.backends.cuda.enable_flash_sdp(False)` in the script below (see https://github.com/Lightning-AI/lit-llama/issues/101).
"""
import logging
import os
import time
from pathlib import Path
from typing import Optional

import lightning as L
import numpy as np
import torch
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from lightning.fabric.loggers import TensorBoardLogger
from lightning.fabric.strategies import DeepSpeedStrategy

from generate import generate
from lit_llama.lora import mark_only_lora_as_trainable, lora, lora_state_dict
from lit_llama.model import Block, LLaMA, LLaMAConfig
from lit_llama.tokenizer import Tokenizer
from lit_llama.utils import save_model_checkpoint
from scripts.prepare_alpaca import generate_prompt

logging.basicConfig(level=logging.INFO)

eval_interval = 10
save_interval = 20
eval_iters = 100
log_interval = 2

# Hyperparameters
learning_rate = 3e-4
batch_size = 128
micro_batch_size = 4
gradient_accumulation_steps = batch_size // micro_batch_size
max_iters = 50000 * 3 // micro_batch_size
weight_decay = 0.0
block_size = 256
lora_r = 8
lora_alpha = 16
lora_dropout = 0.05
warmup_steps = 100


def main(
    output_dir: Path = Path("out/lora/llama-dolly"),
    checkpoint_path: Path = Path("checkpoints/lit-llama/lit_llama_7b.ckpt"),
    tokenizer_path: Path = Path("checkpoints/llama/tokenizer.model"),
    dataset_dir: Path = Path("data/dolly"),
    offload_optimizer: bool = False,
):
    logger = TensorBoardLogger(root_dir="out/logs")

    strategy = DeepSpeedStrategy(stage=3, offload_optimizer=offload_optimizer)

    fabric = L.Fabric(
        strategy=strategy,
        accelerator="gpu",
        devices=1,
        precision="32",
        loggers=logger,
    )
    fabric.launch()
    fabric.seed_everything(1337 + fabric.global_rank)

    if fabric.global_rank == 0:
        os.makedirs(output_dir, exist_ok=True)

    train_data, val_data = load_datasets(dataset_dir)

    config = LLaMAConfig.from_name("7B")
    config.block_size = block_size

    with fabric.device, lora(
        r=lora_r, alpha=lora_alpha, dropout=lora_dropout, enabled=True
    ):
        # torch.set_default_tensor_type(torch.HalfTensor)
        model = LLaMA(config)  # .bfloat16()
        torch.set_default_tensor_type(torch.FloatTensor)
        # strict=False because missing keys due to LoRA weights not contained in checkpoint state
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint, strict=False)

    mark_only_lora_as_trainable(model)

    num_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print(f"Number of trainable parameters: {num_params}")

    if offload_optimizer:
        optimizer = DeepSpeedCPUAdam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        # optimizer = FusedAdam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model, optimizer = fabric.setup(model, optimizer)
    train(
        fabric,
        model,
        optimizer,
        train_data,
        val_data,
        output_dir=output_dir,
        tokenizer_path=tokenizer_path,
    )

    # Save the final LoRA checkpoint at the end of training
    checkpoint = lora_state_dict(model)
    fabric.save(os.path.join(output_dir, "lit-llama-lora-finetuned.pth"), checkpoint)


def train(
    fabric: L.Fabric,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_data: np.ndarray,
    val_data: np.ndarray,
    *,
    output_dir: Path,
    tokenizer_path: Path,
) -> None:
    """The training loop.

    Loosely based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT.
    """
    step_count = 0

    for iter_num in range(max_iters):

        if step_count <= warmup_steps:
            # linear warmup
            lr = learning_rate * step_count / warmup_steps
            lr = max(lr, 1e-9)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        fabric.log("lr", lr, step=iter_num)

        t0 = time.time()

        is_accumulating = (iter_num + 1) % gradient_accumulation_steps != 0

        input_ids, targets = get_batch(fabric, train_data)
        logits = model(input_ids)
        loss = loss_fn(logits, targets)
        fabric.log("training_loss", loss, step=iter_num)

        with fabric.no_backward_sync(model, enabled=is_accumulating):
            fabric.backward(loss / gradient_accumulation_steps)

        if not is_accumulating:
            optimizer.step()
            optimizer.zero_grad()
            step_count += 1
            fabric.log("step_count", step_count, step=iter_num)

            if step_count % eval_interval == 0:
                val_loss = validate(
                    fabric, model, val_data, tokenizer_path=tokenizer_path, step=iter_num,
                )
                fabric.log("validation_loss", val_loss, step=iter_num)
                fabric.print(f"step {iter_num}: val loss {val_loss:.4f}")
                fabric.barrier()

            if step_count % save_interval == 0:
                print(f"Saving LoRA weights to {output_dir}")
                save_model_checkpoint(fabric, model, os.path.join(output_dir, f"iter-{iter_num:06d}.ckpt"))

        dt = time.time() - t0
        if iter_num % log_interval == 0:
            fabric.print(
                f"iter {iter_num}: loss {loss.item():.4f}, time: {dt*1000:.2f}ms"
            )


def generate_response(model, instruction, *, tokenizer_path: Path):
    tokenizer = Tokenizer(tokenizer_path)
    sample = {"instruction": instruction, "input": ""}
    prompt = generate_prompt(sample)
    encoded = tokenizer.encode(prompt, bos=True, eos=False)
    encoded = encoded[None, :]  # add batch dimension
    encoded = encoded.to(model.device)

    output = generate(
        model,
        idx=encoded,
        max_seq_length=block_size,
        max_new_tokens=100,
    )
    output = tokenizer.decode(output[0].cpu())
    return output  # output.split("### Response:")[1].strip()


@torch.no_grad()
def validate(
    fabric: L.Fabric,
    model: torch.nn.Module,
    val_data: np.ndarray,
    *,
    tokenizer_path: Path,
    step: int,
) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        input_ids, targets = get_batch(fabric, val_data)
        logits = model(input_ids)
        loss = loss_fn(logits, targets)
        losses[k] = loss.item()
    out = losses.mean()

    # produce an example:
    instruction = (
        "Recommend a movie for me to watch during the weekend and explain the reason."
    )

    output = generate_response(model, instruction, tokenizer_path=tokenizer_path)
    log_text(fabric, "validation_response", output, step=step)

    fabric.print("=" * 32)
    fabric.print(instruction)
    fabric.print(output)
    fabric.print("=" * 32)

    model.train()
    return out.item()


def loss_fn(logits, targets):
    # shift the targets such that output n predicts token n+1
    logits = logits[..., :-1, :].contiguous()
    targets = targets[..., 1:].contiguous()
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
    )
    return loss


def get_batch(fabric: L.Fabric, data: list):
    ix = torch.randint(len(data), (micro_batch_size,))

    input_ids = [data[i]["input_ids"].type(torch.int64) for i in ix]
    labels = [data[i]["labels"].type(torch.int64) for i in ix]

    max_len = max(len(s) for s in input_ids)

    def pad_right(x, pad_id):
        # pad right based on the longest sequence
        n = max_len - len(x)
        return torch.cat((x, torch.full((n,), pad_id, dtype=x.dtype)))

    x = torch.stack([pad_right(x, pad_id=0) for x in input_ids])
    y = torch.stack([pad_right(x, pad_id=-1) for x in labels])
    x, y = fabric.to_device((x.pin_memory(), y.pin_memory()))
    return x, y


def load_datasets(data_dir: Path = "data/dolly"):
    train_data = torch.load(os.path.join(data_dir, "train.pt"))
    val_data = torch.load(os.path.join(data_dir, "test.pt"))
    return train_data, val_data


def log_text(fabric: L.Fabric, tag: str, text_string: str, step: Optional[int] = None) -> None:
    if fabric.global_rank == 0:
        for logger in fabric.loggers:
            if isinstance(logger, TensorBoardLogger):
                logger.experiment.add_text(tag, text_string, global_step=step)


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    CLI(main)
