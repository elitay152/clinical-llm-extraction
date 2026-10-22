# First live hosted distillation pilot

The API credential worked. The teacher returned
`gpt-4.1-mini-2025-04-14`; the student was the pinned
SmolLM2-135M-Instruct base model with a fresh LoRA adapter.

## Data and execution

- Prefix pilot v1: four hosted responses, all empty with no denial evidence;
  all rejected. No student trained. These inputs were mostly surgical history.
- Dosage-focused pilot v2: four hosted responses, all mechanically accepted.
  Three training excerpts contain seven teacher-labeled medication records;
  one held-out excerpt contains one record.
- Eight real API responses total across the two separate four-request pilots.
  Raw responses, usage, inputs, selection rules and failures are preserved.
- Source dosage-window selection is intentionally enriched and is not a
  representative sample. Labels apply only to each supplied excerpt.
- Exact normalized full-note groups are separated; patient/article isolation
  remains unresolved. No manual annotation or clinical gold labels were used.

The fresh student completed 20 optimizer steps on three excerpts. All losses
were finite. Fixed training-probe loss fell from 0.5472 to 0.3687. The adapter was
saved and reloaded for evaluation. This loss reduction is a training diagnostic,
not evidence of better extraction.

## Held-out teacher agreement

One held-out teacher-labeled excerpt was evaluated using the same prompt and
greedy decoding for both checkpoints.

| Check | Base student | Fine-tuned student |
|---|---:|---:|
| Valid JSON outputs | 0/1 | 0/1 |
| Valid schema and evidence outputs | 0/1 | 0/1 |
| Exact-document agreement | 0/1 | 0/1 |
| Exact medication-record F1 | 0 | 0 |

Both checkpoints returned prose instead of the requested JSON. Neither matched
the teacher record under the strict evaluation. The adapter did not demonstrate
an improvement on this example. One example cannot establish general performance.
No held-out prompting or decoding tweaks were used to obtain a better score.

## Artifacts and checks

- `data/silver/teacher-pilot-v1/`: preserved failed prefix run.
- `data/silver/teacher-pilot-v2/`: frozen dosage-window inputs, raw teacher responses,
  accepted train/validation labels, and rejection/acceptance report.
- `artifacts/distillation-pilot-v2/`: adapter, tokenizer, and training run record.
- `artifacts/distillation-comparison-v2/`: baseline/adapted predictions and comparison.

25 unit tests passed after adding the dosage-window selection check. The live
execution also verified the API route, optimizer, persistence, reload, and scoring.
Data and model artifacts remain local and ignored by Git.

## Decision

The pipeline is executable, but this 135M student trained on three examples is
not a usable medication extractor. The next experiment should use substantially
more teacher-labeled training examples, establish output-format behavior on
separate development inputs, and consider a more capable student with suitable
compute. Keep a fresh held-out set for that experiment. Generated-reference
agreement must continue to be distinguished from clinical accuracy.
