# CPU LoRA pilot results

These are pipeline diagnostics, not clinical extraction accuracy scores.
The task was silver-label treatment-name extraction, which includes non-drug
therapies and does not implement the full medication schema.

Model: `HuggingFaceTB/SmolLM2-135M-Instruct`, revision
`12fd25f77366fa6b3b4b768ec3050bf629380bac`. CPU float32, four threads,
460,800 trainable LoRA parameters, rank 8, alpha 16, dropout 0.05, query/value
projections, learning rate 0.0002, batch size 1, seed 42.

The candidate training pool contains 85 exact-text-deduplicated positive examples
from the first 200 Augmented Clinical Notes rows. Exclusions: 34 rows matching
annotation candidates, 29 additional duplicates, 52 unusable silver examples.
The annotation set was not modified. This isolation is based on full-note hashes,
not patients or source articles.

| Diagnostic | 20-step smoke run | 200-step pilot |
|---|---:|---:|
| Training-pool probe loss before | 1.8925 | 1.8925 |
| Training-pool probe loss after | 1.0675 | 0.0615 |
| Generated probe response follows JSON shape | No | Yes |
| Adapter reload reproduces recorded loss | Yes | Yes |

The 20-step run visits 20 examples; the 200-step run visits all 85 examples and
cycles through the same shuffled order. Both runs start from the base model;
the longer run is not a continuation of the first adapter.
The longer run took approximately 134 seconds including saving and its generation
probe, excluding model download. All 200 training losses were finite.

The 200-step probe generation contained a treatment string rather than invented
metadata, and followed the requested array-of-strings JSON shape. This is one
training-pool example, not a JSON-validity rate or an independent accuracy result.
The short run's malformed response remains preserved in its run record.

Local artifacts (ignored by Git):

- `artifacts/pilot-lora/`: 20-step adapter, tokenizer, run and verification records.
- `artifacts/pilot-lora-200/`: 200-step adapter, tokenizer, run and verification records.

Package versions: torch 2.6.0+cpu, transformers 4.51.3, peft 0.15.2,
accelerate 1.6.0. The full tested CPU environment is pinned in
`requirements-training.lock`. Raw data hashes and all candidate training IDs and
source hashes are recorded in each run's `run.json`.

Before scaling: resolve article/case grouping and upstream terms, add independently
reviewed medication labels (including negative cases), and compare identical
held-out prompts before and after fine-tuning. This pilot is not evidence that
fine-tuning improves clinical accuracy or that the model is ready for deployment.
