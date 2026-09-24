"""The direct option scorer used during training and evaluation."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def score_batch(model, batch: dict, label_ids: torch.Tensor) -> torch.Tensor:
    """Score only the option tokens consumed by Kevala's direct-options reader."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    hidden = base.model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
    ).last_hidden_state
    positions = batch["lengths"] - 1
    last = hidden[torch.arange(hidden.shape[0], device=hidden.device), positions]
    label_rows = base.lm_head.weight.index_select(0, label_ids).float()
    scores = F.linear(last.float(), label_rows)
    columns = torch.arange(len(label_ids), device=scores.device)
    return scores.masked_fill(columns >= batch["option_counts"][:, None], float("-inf"))


def collate(rows: list[dict], pad_id: int, device: torch.device) -> dict:
    length = max(len(row["input_ids"]) for row in rows)
    ids = torch.full((len(rows), length), pad_id, dtype=torch.long)
    mask = torch.zeros((len(rows), length), dtype=torch.long)
    for index, row in enumerate(rows):
        n = len(row["input_ids"])
        ids[index, :n] = torch.tensor(row["input_ids"], dtype=torch.long)
        mask[index, :n] = 1
    return {
        "input_ids": ids.to(device),
        "attention_mask": mask.to(device),
        "lengths": torch.tensor([len(row["input_ids"]) for row in rows], device=device),
        "option_counts": torch.tensor(
            [row["option_count"] for row in rows], device=device
        ),
        "answers": torch.tensor([row["answer"] for row in rows], device=device),
    }
