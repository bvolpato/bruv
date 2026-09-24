"""Measure direct-option accuracy for a base, adapter, or merged checkpoint."""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from bruv.data import encode_file, label_token_ids
from bruv.train import evaluate


@contextmanager
def cpu_reference_kernels(model):
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    if base.config.model_type != "qwen3_5_text":
        yield
        return

    # Transformers selects installed FLA kernels at import time, even for CPU tensors.
    # The decorated functions retain their PyTorch reference implementations.
    from transformers.models.qwen3_5 import modeling_qwen3_5

    names = ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule")
    original = {name: getattr(modeling_qwen3_5, name) for name in names}
    try:
        for name, function in original.items():
            reference = function
            while hasattr(reference, "__wrapped__"):
                reference = reference.__wrapped__
            if reference is function:
                raise RuntimeError(f"PyTorch reference kernel unavailable: {name}")
            setattr(modeling_qwen3_5, name, reference)
        yield
    finally:
        for name, function in original.items():
            setattr(modeling_qwen3_5, name, function)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, default=Path("recipes/bruv1-0.8b.json"))
    parser.add_argument(
        "--data", type=Path, required=True, help="JSONL structured decision records"
    )
    parser.add_argument(
        "--model",
        type=Path,
        help="Merged checkpoint directory; omit for the base model",
    )
    parser.add_argument("--adapter", type=Path, help="LoRA adapter directory")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--offset", type=int, default=0, help="Skip records before scoring a slice"
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--batch-size", type=int, help="Override the recipe's evaluation batch size"
    )
    parser.add_argument("--threads", type=int, help="Limit CPU inference threads")
    args = parser.parse_args()
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.offset < 0:
        parser.error("--offset must be nonnegative")
    if args.threads is not None and args.threads < 1:
        parser.error("--threads must be positive")
    if args.threads is not None:
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
    selected = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )
    device = torch.device(selected)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available")
    recipe = json.loads(args.recipe.read_text())
    if args.model and args.adapter:
        parser.error("Use --model or --adapter, not both")
    source = str(args.model) if args.model else recipe["base_model"]
    revision = None if args.model else recipe["base_revision"]
    tokenizer = AutoTokenizer.from_pretrained(source, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    rows = encode_file(
        args.data,
        tokenizer,
        recipe["max_length"],
        args.limit,
        recipe["prompt_format"],
        args.offset,
    )
    model = AutoModelForCausalLM.from_pretrained(
        source, revision=revision, dtype=torch.bfloat16
    ).to(device)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    labels = torch.tensor(label_token_ids(tokenizer), device=device)
    kernels = cpu_reference_kernels(model) if device.type == "cpu" else nullcontext()
    with kernels:
        metrics = evaluate(
            model,
            rows,
            args.batch_size or recipe["batch_size"],
            tokenizer.pad_token_id,
            labels,
            device,
        )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
