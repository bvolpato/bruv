"""Train a LoRA decision model locally and export a merged HF checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from bruv.data import encode_file, label_token_ids
from bruv.model import collate, score_batch


def evaluate(model, rows, batch_size, pad_id, label_ids, device):
    model.eval()
    correct = total = 0
    negative_log_likelihood = chance = 0.0
    by_source = {}
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            group = rows[start : start + batch_size]
            batch = collate(group, pad_id, device)
            log_probabilities = F.log_softmax(
                score_batch(model, batch, label_ids), dim=-1
            )
            predicted = log_probabilities.argmax(-1).tolist()
            nll = (
                -log_probabilities.gather(1, batch["answers"][:, None]).squeeze(1)
            ).tolist()
            for row, choice, row_nll in zip(group, predicted, nll):
                hit = choice == row["answer"]
                correct += hit
                total += 1
                negative_log_likelihood += row_nll
                chance += 1 / row["option_count"]
                source = by_source.setdefault(row["source"], [0, 0, 0.0])
                source[0] += hit
                source[1] += 1
                source[2] += row_nll
    model.train()
    return {
        "accuracy": correct / total,
        "chance_accuracy": chance / total,
        "log_loss": negative_log_likelihood / total,
        "correct": correct,
        "total": total,
        "macro_accuracy": sum(value[0] / value[1] for value in by_source.values())
        / len(by_source),
        "by_source": {
            key: {
                "correct": value[0],
                "total": value[1],
                "accuracy": value[0] / value[1],
                "log_loss": value[2] / value[1],
            }
            for key, value in sorted(by_source.items())
        },
    }


def batches(rows, batch_size, seed):
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    for start in range(0, len(indices), batch_size * 64):
        pool = indices[start : start + batch_size * 64]
        pool.sort(key=lambda index: len(rows[index]["input_ids"]))
        groups = [pool[i : i + batch_size] for i in range(0, len(pool), batch_size)]
        random.Random(seed + start).shuffle(groups)
        for group in groups:
            yield [rows[index] for index in group]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, default=Path("recipes/bruv1-0.8b.json"))
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Directory of structured records/train.jsonl and dev.jsonl",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--limit", type=int, help="Use a short pilot sample of training records"
    )
    parser.add_argument(
        "--dev-limit", type=int, help="Use a short pilot sample of development records"
    )
    parser.add_argument(
        "--max-steps", type=int, help="Limit optimizer steps for a smoke run"
    )
    args = parser.parse_args()
    recipe = json.loads(args.recipe.read_text())
    if not torch.cuda.is_available():
        parser.error("A CUDA GPU is required for local training")
    torch.manual_seed(recipe["seed"])
    random.seed(recipe["seed"])
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(
        recipe["base_model"], revision=recipe["base_revision"]
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    labels = torch.tensor(label_token_ids(tokenizer), device=device)
    train_rows = encode_file(
        args.data / "train.jsonl",
        tokenizer,
        recipe["max_length"],
        args.limit,
        recipe["prompt_format"],
    )
    dev_rows = encode_file(
        args.data / "dev.jsonl",
        tokenizer,
        recipe["max_length"],
        args.dev_limit,
        recipe["prompt_format"],
    )
    print(
        f"Data: {len(train_rows)} train, {len(dev_rows)} dev; longest train {max(len(r['input_ids']) for r in train_rows)}",
        flush=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        recipe["base_model"], revision=recipe["base_revision"], dtype=torch.bfloat16
    ).to(device)
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=recipe["lora_rank"],
            lora_alpha=recipe["lora_alpha"],
            lora_dropout=0,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        ),
    )
    if recipe.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model.print_trainable_parameters()
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=recipe["learning_rate"],
        weight_decay=0,
    )
    steps_per_epoch = math.ceil(
        math.ceil(len(train_rows) / recipe["batch_size"])
        / recipe["gradient_accumulation"]
    )
    planned_steps = steps_per_epoch * recipe["epochs"]
    steps = min(planned_steps, args.max_steps) if args.max_steps else planned_steps
    warmup = max(1, int(steps * recipe["warmup_ratio"]))

    def factor(step):
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "run.json").write_text(
        json.dumps(
            {
                "recipe": recipe,
                "train_rows": len(train_rows),
                "dev_rows": len(dev_rows),
                "planned_steps": planned_steps,
                "steps": steps,
                "gpu": torch.cuda.get_device_name(),
            },
            indent=2,
        )
        + "\n"
    )
    start_time = time.monotonic()
    optimizer.zero_grad(set_to_none=True)
    step = accumulation = logged_batches = 0
    running_loss = 0.0
    model.train()
    for epoch in range(recipe["epochs"]):
        for group in batches(train_rows, recipe["batch_size"], recipe["seed"] + epoch):
            batch = collate(group, tokenizer.pad_token_id, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                scores = score_batch(model, batch, labels)
                loss = F.cross_entropy(scores, batch["answers"])
            (loss / recipe["gradient_accumulation"]).backward()
            running_loss += loss.item()
            accumulation += 1
            logged_batches += 1
            if accumulation < recipe["gradient_accumulation"]:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            accumulation = 0
            if step % 25 == 0 or step == steps:
                elapsed = time.monotonic() - start_time
                print(
                    f"step={step}/{steps} loss={running_loss / logged_batches:.4f} elapsed_s={elapsed:.1f} peak_gib={torch.cuda.max_memory_allocated() / 2**30:.2f}",
                    flush=True,
                )
                running_loss = 0.0
                logged_batches = 0
            if step >= steps:
                break
        if step >= steps:
            break
    if accumulation:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
    metrics = evaluate(
        model, dev_rows, recipe["batch_size"], tokenizer.pad_token_id, labels, device
    )
    metrics.update(
        {
            "optimizer_steps": step,
            "elapsed_seconds": time.monotonic() - start_time,
            "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
        }
    )
    (args.output / "dev.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics), flush=True)
    model.save_pretrained(args.output / "adapter")
    merged = model.merge_and_unload()
    merged.config.use_cache = True
    merged.save_pretrained(args.output / "merged", safe_serialization=True)
    tokenizer.save_pretrained(args.output / "merged")


if __name__ == "__main__":
    main()
