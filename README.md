# Bruv

Bruv trains local decision models from structured classification records. Its first recipes adapt [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) and [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) to choose one option letter. They are designed for [Kevala](https://github.com/bvolpato/kevala)'s direct option scorer. Training, evaluation, and export run on a local CUDA GPU; no Together account or hosted training job is needed.

This project takes its initial data mixture and LoRA starting settings from Together's [Tev1 recipe](https://github.com/togethercomputer/tev1), released under MIT. Bruv is a separate training implementation. Tev1's `train_together.py` uploads data to Together; Bruv trains locally. The dataset, model, and prompt are selected separately, so later recipes can use different data or base checkpoints. A new inference prompt format needs an explicit renderer that matches its target runtime.

## Reproduce the experiments

The starting dataset is Tev1's pinned `new-v1` mix. Its builders download public sources and generate 37,840 train and 4,568 development records without a Together account. They also produce separate held-out and transfer records. Source datasets have their own terms; see [Tev1's provenance table](https://github.com/togethercomputer/tev1/blob/main/DATA_SOURCES.md). Bruv does not copy or publish the raw records.

The 37,840 training records cover MultiNLI (5,000), BoolQ (3,000), Banking77 (3,000), AG News (1,500), SST-5 (2,000), synthetic policies (13,500), routing (6,000), and research classification (3,840). The article's prose says 38,340, but its table and the pinned builder both total 37,840. Both Bruv recipes process every training record once in one epoch. Tev1's [starting training script](https://github.com/togethercomputer/tev1/blob/main/examples/train_together.py) also specifies one epoch; this is not a claim about the unpublished settings of every Together production run.

```bash
git clone https://github.com/togethercomputer/tev1.git ../tev1
git -C ../tev1 checkout 57399714e6b5ef215c9c821cab35a7d5fbc5482b
(cd ../tev1 && uv sync --locked && uv run python fetch_sources.py && uv run python build_all.py)
uv sync --locked
uv run bruv-train --data ../tev1/data/new-v1/records --output runs/bruv1-0.8b-tev1
uv run bruv-train --recipe recipes/bruv1-4b.json \
  --data ../tev1/data/new-v1/records --output runs/bruv1-4b-tev1
```

`bruv-train` requires a CUDA GPU with BF16 support. The [0.8B recipe](recipes/bruv1-0.8b.json) and [4B recipe](recipes/bruv1-4b.json) pin their base revisions and the data builder revision. Train one model at a time. Each command writes run metadata, development accuracy, a PEFT adapter, and a merged Hugging Face checkpoint under the ignored `runs/` directory. `--limit`, `--dev-limit`, and `--max-steps` are available for a short smoke run; those results are not a full training evaluation.

The local objective scores only valid A–P answer tokens at the end of Kevala's exact non-thinking chat prompt. This matches Kevala's readout and avoids building vocabulary logits for every input token. It differs from Tev1's completion-plus-EOS SFT objective and does not use packing. Tev1 research questions with 24 options are deterministically reduced to 16, always retaining the correct option. The original record remains unchanged.

## Evaluate and export

```bash
uv run bruv-eval --data ../tev1/data/v2/records/test.jsonl
uv run bruv-eval --data ../tev1/data/v2/records/test.jsonl \
  --model runs/bruv1-0.8b-tev1/merged
uv run bruv-eval --recipe recipes/bruv1-4b.json \
  --data ../tev1/data/v2/records/test.jsonl \
  --model runs/bruv1-4b-tev1/merged
```

The first command scores the frozen Qwen checkpoint with the same prompt and reader. Evaluate the separate test and transfer splits in addition to development data. Report source-level accuracy and any changes to the model or prompt before comparing numbers.

The pinned training records have no `group_id` overlap or exact `(state, question, options)` duplicate with the development, test, transfer, or research challenge records. These splits still share task families and construction methods, and Tev1 describes its saved evaluations as reused development benchmarks. Use a fresh, independently collected holdout before making deployment claims.

The merged checkpoints are standard Hugging Face Qwen3.5 text models. Kevala can convert either with its architecture-based command:

```bash
kevala convert runs/bruv1-0.8b-tev1/merged -o runs/bruv1-0.8b-q8.kevala
```

Use Kevala's parity and browser checks before publishing a pack. The `.kevala` binary and source checkpoint are release artifacts, not Git files.

## Published 0.8B result

The [Bruv1-0.8B checkpoint and Q8 pack](https://huggingface.co/bvolpato/bruv1-0.8b) were trained and converted on an NVIDIA GeForce RTX 5070 Ti. All 37,840 records were processed in 4,730 optimizer steps, with effective batch 8. Training took 1,879 seconds and peaked at 6.74 GiB of allocated GPU memory.

| Split | Records | Frozen Qwen3.5-0.8B | Bruv1-0.8B |
|---|---:|---:|---:|
| Development | 4,568 | 43.4% | 84.9% |
| Separate test | 2,800 | 37.7% | 80.1% |
| Transfer policies and routing | 1,800 | 30.3% | 79.1% |
| Research classification challenge | 768 | 63.7% | 95.4% |

These are conditional option-choice accuracies for the BF16 merged checkpoint. The [model card](https://huggingface.co/bvolpato/bruv1-0.8b) records provenance, evaluation files, and Q8 conversion parity. Kevala's [decision benchmark](https://github.com/bvolpato/kevala/blob/main/BENCHMARK.md) evaluates the browser pack on different tasks and rotating option orders.

On Kevala's separate 864-decision Firefox WebGPU suite, the released Q8 pack answered 652 correctly (75.46%) with no invalid responses. Its per-suite scores were 98/108 Kevala-authored, 344/432 SemIf-authored, and 210/324 SemIf perturbation decisions. The full pack hash was verified after the run.

## 4B browser result and remaining evaluation

The local 4B run completed all 37,840 training records in one epoch on the RTX 5070 Ti. Its
exported BF16 checkpoint scored **4,194/4,568 (91.81%)** on development records. Kevala
converted the merged checkpoint to a 4,751,303,168-byte Q8 pack. Three BF16-to-Q8 reference
cases matched exact prompt tokens and answer choices, with maximum absolute option-score
difference 0.019503.

On Kevala's separate 864-decision Firefox WebGPU suite, the Q8 pack answered **799/864 (92.48%)**
correctly with no invalid responses. Its per-suite scores were 100/108 Kevala-authored,
403/432 SemIf-authored, and 296/324 SemIf perturbation decisions. The complete local pack
SHA-256 is `fa50a0998428bb3cf575f12e4965a38c8b52ea98caf80204850146c231fd2edb`.
See Kevala's [benchmark method and raw results](https://github.com/bvolpato/kevala/blob/main/BENCHMARK.md)
for fixture provenance, option-order checks, and latency limitations.

The frozen-base comparison and Tev test, transfer, and research challenge splits remain in
progress. Development and browser fixture scores do not establish general reasoning or safety
performance. The 4B checkpoint and pack have not yet been published to Hugging Face.

### Low-power evaluation

`bruv-eval` can score the merged checkpoint on CPU with one record per batch. On a shared machine,
limit inference threads and use `--offset` with `--limit` to save separate, resumable slices:

```bash
CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 uv run bruv-eval \
  --recipe recipes/bruv1-4b.json \
  --data ../tev1/data/v2/records/test.jsonl \
  --model runs/bruv1-4b-tev1/merged \
  --device cpu --threads 2 --batch-size 1 --offset 0 --limit 20
```

The CPU path uses PyTorch reference kernels for Qwen3.5's recurrent layers, even when the CUDA
FLA package is installed. It is slower than CUDA scoring. Record each slice's offset and size,
and combine only nonoverlapping slices from the same checkpoint and evaluation file.

## Record contract

Each JSONL record provides `id`, `state` (text or structured JSON), `question`, `options` with ordered `label` and `description`, and `answer` as the correct letter. `source` enables per-source reporting. The data loader is independent of a named model family. A recipe selects the base checkpoint, revision, prompt format, LoRA settings, and training budget. `kevala-direct-options-v1` is the implemented prompt renderer for this first run; unsupported formats fail explicitly.

The code is MIT licensed. The Qwen base checkpoint is Apache 2.0. Tev1's code license does not replace the source datasets' terms. Model weights should carry their own model card and provenance record.
