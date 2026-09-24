"""Structured decision records rendered for the Kevala direct-options reader."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

LETTERS = "ABCDEFGHIJKLMNOP"
DIRECT_SYSTEM = (
    "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)


def read_records(path: Path):
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{line_number}: {error}") from error


def messages(
    record: dict, prompt_format: str = "kevala-direct-options-v1"
) -> list[dict]:
    if prompt_format != "kevala-direct-options-v1":
        raise ValueError(f"Unsupported prompt format: {prompt_format}")
    options = record["options"]
    if not 2 <= len(options) <= len(LETTERS):
        raise ValueError(f"{record.get('id')}: expected 2-16 options")
    labels = [option["label"] for option in options]
    if labels != list(LETTERS[: len(options)]):
        raise ValueError(f"{record.get('id')}: labels must be consecutive A-P")
    if record["answer"] not in labels:
        raise ValueError(f"{record.get('id')}: answer is not an option")
    payload = {
        "evidence": record["state"],
        "criterion": record["question"],
        "options": [
            {"letter": option["label"], "description": option["description"]}
            for option in options
        ],
    }
    return [
        {"role": "system", "content": DIRECT_SYSTEM},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def fit_options(record: dict) -> dict:
    """Preserve the answer while adapting wider source tasks to 16 runtime slots."""
    options = record["options"]
    if len(options) <= len(LETTERS):
        return record
    gold = next(option for option in options if option["label"] == record["answer"])
    seed = int.from_bytes(
        hashlib.sha256(str(record["id"]).encode()).digest()[:8], "big"
    )
    rng = random.Random(seed)
    selected = [
        gold,
        *rng.sample(
            [option for option in options if option is not gold], len(LETTERS) - 1
        ),
    ]
    rng.shuffle(selected)
    answer = LETTERS[selected.index(gold)]
    return {
        **record,
        "options": [
            {**option, "label": LETTERS[index]} for index, option in enumerate(selected)
        ],
        "answer": answer,
    }


def label_token_ids(tokenizer) -> list[int]:
    ids = []
    for letter in LETTERS:
        encoded = tokenizer.encode(letter, add_special_tokens=False)
        if len(encoded) != 1 or tokenizer.decode(encoded) != letter:
            raise ValueError(f"{letter}: label is not one round-trip token")
        ids.append(encoded[0])
    if len(ids) != len(set(ids)):
        raise ValueError("label tokens are not distinct")
    return ids


def encode_record(
    record: dict,
    tokenizer,
    max_length: int,
    prompt_format: str = "kevala-direct-options-v1",
) -> dict:
    record = fit_options(record)
    prompt = tokenizer.apply_chat_template(
        messages(record, prompt_format),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(ids) > max_length:
        raise ValueError(f"{record.get('id')}: {len(ids)} tokens exceed {max_length}")
    answer = LETTERS.index(record["answer"])
    label_id = tokenizer.encode(record["answer"], add_special_tokens=False)[0]
    if tokenizer.encode(prompt + record["answer"], add_special_tokens=False) != ids + [
        label_id
    ]:
        raise ValueError(f"{record.get('id')}: answer token changes at prompt boundary")
    return {
        "id": record.get("id"),
        "source": record.get("source", "unknown"),
        "input_ids": ids,
        "answer": answer,
        "option_count": len(record["options"]),
    }


def encode_file(
    path: Path,
    tokenizer,
    max_length: int,
    limit: int | None = None,
    prompt_format: str = "kevala-direct-options-v1",
) -> list[dict]:
    rows = []
    for record in read_records(path):
        rows.append(encode_record(record, tokenizer, max_length, prompt_format))
        if limit is not None and len(rows) >= limit:
            break
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows
