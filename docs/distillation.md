# Distillation workflow

Follow the [quick start](../README.md#quick-start) for installation and credentials.
The student initializes from the pinned SmolLM2-135M-Instruct base checkpoint. Evaluation definitions and limitations are in [experiment design](experiment.md).

## Inputs and labeling

Preparation selects 12 dosage-containing excerpts of up to 1,800 characters from
the cached clinical notes. It excludes reserved annotation notes and exact
normalized duplicates, then freezes 10 training and 2 validation source groups.
Labels apply only to the supplied excerpt.

Individual stages, as an alternative to the end-to-end runner:

```bash
python scripts/pseudo_label.py prepare data/silver/my-pilot --limit 12 --window dosage
.venv/bin/python scripts/pseudo_label.py label data/silver/my-pilot \
  --backend hosted --model gpt-4.1-mini --base-url https://api.openai.com/v1 \
  --split train --limit 3 --max-tokens 1200
.venv/bin/python scripts/pseudo_label.py label data/silver/my-pilot \
  --backend hosted --model gpt-4.1-mini --base-url https://api.openai.com/v1 \
  --split validation --limit 1 --max-tokens 1200
python scripts/pseudo_label.py collect data/silver/my-pilot
```

The runner caps each pilot at three training and one validation response,
across resumed calls. Labeling preserves raw responses and skips completed
IDs. Changing the teacher, prompt, or generation settings requires a new run.
Rejections do not trigger replacement API calls.

Acceptance requires strict JSON, verbatim medication names and attributes in
supporting quotes, and quotes that occur exactly once in the input. Python
computes Unicode offsets. Empty outputs require quoted denial evidence.
Accepted records retain `teacher_generated` provenance; rejected responses and
reasons remain available for inspection.

## Train and compare

```bash
.venv/bin/python scripts/train_distillation.py data/silver/my-pilot \
  --steps 20 --output artifacts/my-adapter
.venv/bin/python scripts/evaluate_distillation.py artifacts/my-adapter \
  --output artifacts/my-comparison
```

Training requires two accepted examples, masks prompt tokens, and skips
overlength examples without truncating answers. Evaluation requires at least one
accepted validation reference and reloads the adapter before inference.

Each run retains inputs, source revisions, hashes, teacher metadata, and rejection
reports under `data/silver/`. Adapter and comparison records go under `artifacts/`.
These directories are local and ignored by Git.

## Other teachers

Hosted providers must support HTTPS `/chat/completions`, temperature 0,
`max_tokens`, and JSON-object response mode. Configure the model, API base,
credential, and allowed destination for your provider. The labeling command also
supports `--backend local --model MODEL --revision EXACT_REVISION`; local inference
requires downloading that model and uses CPU float32.
