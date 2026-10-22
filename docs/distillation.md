# Distillation workflow

Follow the [quick start](../README.md#quick-start) for installation and credentials.
The student checkpoint and evidence mode are configured in `configs/student.json`.
Evaluation definitions and limitations are in [experiment design](experiment.md).

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

## Medication-name spans

The default `name_spans` task uses `configs/student-name-prompt.txt`. The student
receives indexed source tokens and returns `medication_name_spans`: inclusive
`[first_token_id, last_token_id]` ranges. Code copies the original text between
those token boundaries, preserving Unicode offsets and whitespace. Unknown,
reversed, duplicate, and non-integer ranges are rejected. An empty array means
no medication names were selected; no attributes or status are predicted.

Targets include exact, token-aligned occurrences of teacher medication names
inside their supporting evidence. Identical spans are deduplicated across
medication records. Unalignable training rows are excluded and recorded in the
local run metadata; unalignable development references fail evaluation.
`synthetic_lab_negatives` adds eight fixed lab-only denial examples to training,
with separate synthetic provenance. They never enter development metrics. Source
indexing guarantees copied text, not that a selected term is a medication.

Evaluation reports exact name-span precision/recall/F1, unique case-sensitive
name-text precision/recall/F1 per document, JSON/range validity, and exact-document
agreement. These metrics evaluate a narrower task than full medication records.
Teacher evidence may omit repeated mentions, so span recall is agreement with
the attested reference occurrences, not a completeness measure.

## Sentence-ID evidence

With `evidence_mode: sentence_ids`, the student receives numbered source sentences and returns integer
`evidence_sentence_ids` for each medication, plus `negative_evidence_sentence_ids`
for an explicit denial. Code resolves IDs to exact text and Unicode offsets;
unknown, duplicate, or non-integer IDs are rejected. Medication names and populated
attributes must still occur verbatim in the selected evidence.

Teacher quotes are converted to overlapping sentence IDs without changing their
medication fields. Adjacent selected sentences resolve to a contiguous source span,
including the original whitespace. Repeated sentences remain distinguishable by ID.
Segmentation uses deterministic punctuation/line boundaries with abbreviation
handling; it is a source indexing rule, not a clinical sentence parser.

`evidence_mode: sentence_ids` uses `configs/student-sentence-prompt.txt`.
Quote-based output is supported by `evidence_mode: quotes` and
`configs/student-prompt.txt`. The run saves the mode, prompt, and segmentation
version for evaluation. Exact source retrieval does not prove that the selected
sentence supports the medication or its attributes.

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
