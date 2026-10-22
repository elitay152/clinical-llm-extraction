# Experiment design

Question: how closely can a LoRA-tuned student reproduce a teacher's medication
extraction, compared with the same student before fine-tuning?

## Frozen data splits

```bash
python scripts/prepare_data.py --limit 1000
python scripts/prepare_experiment.py data/silver/my-experiment
```

Preparation freezes 60 training, 20 development, and 20 test excerpts with seed 42.
It excludes the cached 200-note pool and any text-similarity cluster containing
those notes. Five-word shingle Jaccard similarity of at least 0.8 joins notes into
connected components. One dosage-containing excerpt is selected per eligible
component, and components are assigned to only one split.

The manifest records input hashes, grouping rules, source revisions, and student
configuration. Its checksum is verified before labeling or loading split files.
Text similarity does not establish article- or patient-disjointness, and dosage
selection does not represent all notes.

## Label and develop

```bash
.venv/bin/python scripts/pseudo_label.py label data/silver/my-experiment \
  --backend hosted --model gpt-4.1-mini-2025-04-14 \
  --base-url https://api.openai.com/v1 --split train --limit 60 --max-tokens 2000
.venv/bin/python scripts/pseudo_label.py label data/silver/my-experiment \
  --backend hosted --model gpt-4.1-mini-2025-04-14 \
  --base-url https://api.openai.com/v1 --split development --limit 20 --max-tokens 2000
python scripts/pseudo_label.py collect data/silver/my-experiment
```

Test labeling, implicit split selection, and collection of test responses are
blocked for frozen experiments. Keep the test set unlabeled until model, prompt,
and training choices are finalized. Development data supports those choices;
only training rows reach the optimizer. See [label acceptance rules](distillation.md).

The student is pinned in `configs/student.json`; its prompt is in
`configs/student-prompt.txt`. Training snapshots the prompt and model revision.

```bash
.venv/bin/python scripts/train_distillation.py data/silver/my-experiment \
  --output artifacts/my-student --steps 100
.venv/bin/python scripts/evaluate_distillation.py artifacts/my-student \
  --output artifacts/my-development-comparison
```

## Evaluation

Both checkpoints use the saved training prompt and greedy decoding. Report JSON
validity, schema/evidence validity, exact-document agreement, and medication-record
precision/recall/F1 against teacher references. Record equality is case-sensitive
and includes name, strength, dose, route, frequency, and status, retaining
multiplicity. Evidence offsets are validated separately from record equality.
Invalid responses count as empty predictions and receive no exact-document
credit. All-empty record comparisons have undefined F1; zero-reference runs fail.
Generation timings include first-use overhead and are not production benchmarks.

Teacher agreement is not clinical accuracy. Evidence matching cannot prove
completeness, temporal reasoning, or medication–attribute linking. Clinical
accuracy claims require independent reviewed labels. Report MTS-Dialog transfer
results separately and resolve upstream terms before releasing data or weights.
