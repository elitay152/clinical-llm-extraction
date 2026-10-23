# Expanding the training pool

Select new training sources without changing the development reference version
or the locked test membership:

```bash
.venv/bin/python scripts/prepare_training_expansion.py prepare \
  data/silver/my-reviewed-references data/silver/my-expanded-training --count 40
.venv/bin/python scripts/label_medication_names.py label data/silver/my-expanded-training \
  --split train --model gpt-4.1-mini-2025-04-14 \
  --base-url https://api.openai.com/v1 --limit 40
.venv/bin/python scripts/label_medication_names.py collect data/silver/my-expanded-training
.venv/bin/python scripts/prepare_training_expansion.py seal data/silver/my-expanded-training
```

Preparation uses the cached source pool and its original exclusion pool. It
reconstructs the frozen similarity clusters and excludes every cluster assigned
to an existing experiment candidate, including rejected and unlabeled candidates.
The exclusion pool's clusters are also blocked. Pool changes that alter frozen
cluster IDs fail explicitly. One dosage excerpt is selected per remaining cluster,
with deterministic ordering. Selection does not use model predictions or scores.

Only the new training IDs can receive teacher requests. Collection verifies the
source snapshot and preserves its training labels, development labels, and empty
test-label file. New labels use the same name protocol and source checks as the
existing training pool. Rejections do not trigger replacement calls. Sealing
records hashes for the complete label files before any training starts.

## Paired comparison

```bash
HF_HOME=/path/to/huggingface-cache .venv/bin/python scripts/run_training_expansion.py \
  data/silver/my-expanded-training --baseline-batch artifacts/my-iteration \
  --negative-controls artifacts/my-synthetic-controls.json \
  --output artifacts/my-expansion-comparison
```

The runner rechecks the baseline predictions against identical development
references, then trains the expanded pool at the same three seeds and settings.
It records its plan before training, including a one-percentage-point mean F1
promotion threshold. A candidate must also retain all synthetic-control passes.

The epoch budget is fixed, so a larger pool receives more optimizer updates.
Record both sample counts and step counts; this comparison does not isolate data
quantity from compute. Inspect paired seed changes and the per-run disagreement
files. All labels, checkpoints, ledgers, and comparison files remain in ignored
data/artifact directories. Published code describes the procedure, not experiment
reports or source excerpts.
