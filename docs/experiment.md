# Experiment design

Question: how well can biomedical token classification and prompted or fine-tuned
language models reproduce source-grounded medication-name references?

## Frozen data splits

```bash
python scripts/prepare_data.py --limit 200
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
.venv/bin/python scripts/label_medication_names.py prepare \
  data/silver/my-experiment data/silver/my-name-labels
.venv/bin/python scripts/label_medication_names.py label data/silver/my-name-labels \
  --model gpt-4.1-mini-2025-04-14 --base-url https://api.openai.com/v1 \
  --split train --limit 60
.venv/bin/python scripts/label_medication_names.py label data/silver/my-name-labels \
  --model gpt-4.1-mini-2025-04-14 --base-url https://api.openai.com/v1 \
  --split development --limit 20
python scripts/label_medication_names.py collect data/silver/my-name-labels
.venv/bin/python scripts/train_token_classifier.py data/silver/my-name-labels \
  --output artifacts/my-classifier
.venv/bin/python scripts/evaluate_token_classifier.py artifacts/my-classifier \
  --output artifacts/my-development-comparison
```

The label commands make up to 80 paid teacher requests, preserve raw responses,
and resume by ID. Rejections do not trigger replacements. Naming and boundary
rules are in [the name-labeling protocol](name-labeling.md); classifier settings
are pinned in `configs/token-classifier.json`.

Test labeling, implicit split selection, and collection of test responses are
blocked for frozen experiments. Keep test notes unlabeled until model selection
ends. Development data supports those choices; only training rows reach the
optimizer. Use [iterative experiments](iterative-training.md) to audit and freeze
a reference version before controlled comparisons.

For prompted and LoRA-tuned language-model comparisons, use the same name-label
directory with [the generative workflow](distillation.md). Attribute/status
extraction requires the full-regimen label format described there.

## Evaluation

Name extraction reports exact source-span and unique name-text precision,
recall, and F1, plus exact-document agreement. Names are case-sensitive; repeated
occurrences remain distinct in span metrics. The classifier decodes BIO labels;
generative models use a saved prompt and greedy decoding. All compared models
must use the same source membership and reference version.

Full-schema modes report medication-record precision/recall/F1. Record equality is case-sensitive
and includes name, strength, dose, route, frequency, and status, retaining
multiplicity. Evidence IDs are resolved to source spans and validated separately from record equality.
Invalid responses count as empty predictions and receive no exact-document
credit. All-empty record comparisons have undefined F1; zero-reference runs fail.
Generation timings include first-use overhead and are not production benchmarks.

Teacher agreement is not clinical accuracy. Evidence matching cannot prove
completeness, temporal reasoning, or medication–attribute linking. Clinical
accuracy claims require independent reviewed labels. Report MTS-Dialog transfer
results separately and resolve upstream terms before releasing data or weights.
