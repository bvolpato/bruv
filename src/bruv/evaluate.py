"""Measure direct-option accuracy for a base, adapter, or merged checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from bruv.data import encode_file, label_token_ids
from bruv.train import evaluate


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
    args = parser.parse_args()
    recipe = json.loads(args.recipe.read_text())
    if args.model and args.adapter:
        parser.error("Use --model or --adapter, not both")
    source = str(args.model) if args.model else recipe["base_model"]
    revision = None if args.model else recipe["base_revision"]
    tokenizer = AutoTokenizer.from_pretrained(source, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    rows = encode_file(
        args.data, tokenizer, recipe["max_length"], args.limit, recipe["prompt_format"]
    )
    model = AutoModelForCausalLM.from_pretrained(
        source, revision=revision, dtype=torch.bfloat16
    ).to("cuda")
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    labels = torch.tensor(label_token_ids(tokenizer), device="cuda")
    metrics = evaluate(
        model,
        rows,
        recipe["batch_size"],
        tokenizer.pad_token_id,
        labels,
        torch.device("cuda"),
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
