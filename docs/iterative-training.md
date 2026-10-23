# Iterative classifier experiments

Use one frozen development reference version for a complete experiment batch.
Train only on training notes, choose one hypothesis at a time, and retain rejected
experiments in the local ledger. Development scores guide iteration; they are not
an unbiased final evaluation.

## Reference audit

```bash
.venv/bin/python scripts/review_name_references.py data/silver/my-name-labels \
  --output data/silver/my-reviewed-references \
  --model gpt-4.1-2025-04-14 --base-url https://api.openai.com/v1
```

This makes paid hosted requests using `TEACHER_API_KEY`. A blind teacher receives
source text and the naming protocol, without student predictions. Differences
from existing labels receive a separate adjudication pass. Invalid review outputs
retain the original labels with an unresolved flag. Agreement and adjudication
remain AI-generated supervision, not independently reviewed clinical gold.

The snapshot preserves training labels and development membership. Its manifest
seals label hashes and audit provenance; modifications invalidate loading. Raw
review responses and per-note decisions remain beside the ignored dataset.

## Training loop

```bash
HF_HOME=/path/to/huggingface-cache .venv/bin/python scripts/run_classifier_experiments.py \
  data/silver/my-reviewed-references --output artifacts/my-iteration \
  --baseline-run artifacts/my-classifier \
  --negative-controls artifacts/my-synthetic-controls.json
```

The runner uses cached models and refuses existing output directories. It records
the plan before scoring and runs:

1. A four-note training-only overfit diagnostic. Below 98% exact span F1, stop and
   inspect alignment, masking, and decoding before optimizing generalization.
2. Nested 25%, 50%, and 100% training subsets with a fixed epoch budget. The same
   synthetic negatives are included at every size. Both data quantity and update
   count grow; this is not a compute-matched comparison.
3. A class-weight ablation: uniform versus square-root inverse training frequency,
   holding data, encoder, seed, epochs, and other settings fixed.
4. Baseline repeats at two additional seeds. Repeat the candidate at those same
   seeds only if its initial exact span F1 improves by at least one percentage
   point. Keep the baseline unless the paired-seed mean supports the candidate.

A matching saved baseline can be reused after checking configuration and training
label hashes. Evaluation uses `--reference-directory` to score identical source
membership against the batch's reference version. `--training-diagnostic` explicitly
marks training-set probes and prevents them from being reported as development
results. Development and diagnostic metrics have separate keys.

Inspect each run's `disagreements.json` and predictions against its source references: distinguish wrong
names, boundary mismatches, omitted mentions, and incomplete reference labels.
Do not revise references midway through a batch or select examples after scoring.
A corrected reference requires a new sealed snapshot and rescoring all compared
models. Keep `plan.json`, `ledger.jsonl`, checkpoints, logs, and `decision.json` in
ignored artifact directories. Test notes stay locked until model selection ends.
