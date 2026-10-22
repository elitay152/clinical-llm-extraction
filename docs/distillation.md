# Medication extraction by teacher–student distillation

The active experiment uses generated supervision instead of requiring manual
annotation. A teacher extracts medication records and supporting quotations; a
fresh small student is trained on mechanically accepted outputs. The headline
measurement is **agreement with the teacher**, not clinical accuracy.

The earlier treatment-name pilot and optional annotation tool remain available.
The new student starts from the original base checkpoint, not the adapter trained
on treatment names. Teacher labels never receive human-reviewed or gold status.

## Teacher and inputs

Selected teacher: OpenAI **gpt-4.1-mini**, using its OpenAI-compatible HTTPS API.
The cloud setup used `TEACHER_API_KEY` restricted to `api.openai.com`,
`TEACHER_API_BASE_URL=https://api.openai.com/v1`, and `TEACHER_MODEL=gpt-4.1-mini`.
A live request succeeded in the development environment. Fresh checkouts require
your own credential in the process environment or secure environment settings.
No credential or cloud configuration is bundled with the repository.
A model alias can change; each raw response records the returned model identifier.

Live hosted labeling has run; see the latest results below. Local caches and raw
run artifacts are excluded from Git and are not required to reproduce a new run.

To run a new bounded pilot, choose new output directories (existing runs are preserved):

```bash
.venv/bin/python scripts/run_distillation_pilot.py \
  --silver data/silver/NEW_RUN --output artifacts/NEW_ADAPTER \
  --comparison artifacts/NEW_COMPARISON
```

It generates at most **three training and one validation response** in total,
including responses already recorded on an earlier attempt. It collects labels,
requires at least two accepted training examples and one accepted validation
example, trains a fresh 20-step student, and compares the baseline and adapter on
the held-out teacher reference. Rejections do not trigger extra paid calls.
No API calls are made during installation or startup.

Twelve candidate notes are selected from the existing 200-row source sample.
Reserved annotation candidates and exact duplicate full notes are excluded. The
split is frozen before labeling: 10 training and 2 validation source-text groups.
Article- and patient-level isolation remain unresolved. Inputs are up to 1,800
characters from a normalized source window containing a dosage expression,
explicitly marked as excerpts. Selection uses source text only and deliberately
enriches medication-related content; it is not a representative sample. Labels
and metrics apply only to this supplied text, not omitted portions of the note.

```bash
# Existing checkout; run from /workspace/clinical-llm-extraction.
python scripts/pseudo_label.py prepare data/silver/NEW_RUN --limit 12 --window dosage

# Optional individual stages, instead of the bounded end-to-end command:
.venv/bin/python scripts/pseudo_label.py label data/silver/NEW_RUN \
  --backend hosted --model gpt-4.1-mini --base-url https://api.openai.com/v1 \
  --split train --limit 3 --max-tokens 1200

python scripts/pseudo_label.py collect data/silver/NEW_RUN
```

Preparation refuses to overwrite a run. Labeling skips already recorded responses
and preserves raw generations and teacher configuration. Changing the model,
revision, prompt or generation settings requires a new directory. `--limit` limits
new requests; `--split validation` can label held-out candidates separately.
A response failing validation remains rejected rather than being silently fixed
or retried until it passes.

## Mechanical acceptance

- Strict JSON with the specified medication fields and status vocabulary.
- Verbatim medication names and non-null attributes in attached evidence.
- Every evidence quotation occurs exactly once in the supplied input. Offsets are
  computed in Python Unicode characters; the teacher does not guess them.
- Empty medication outputs require quoted denial evidence. Such examples still
  represent teacher assertions, not independently confirmed negative labels.
- Unsupported, ambiguous or malformed responses are recorded as rejections.

These checks cannot establish completeness, negation/temporal interpretation,
correct medication-to-attribute linking, or clinical correctness. A plausible but
incorrect status can pass them. Generated supervision should retain that label.

Accepted rows are stored in separate `train.jsonl` and `validation.jsonl` files.
Raw responses, rejection reasons, input hashes, source revisions and the exact
prompt are preserved alongside them. No model calls or source data are published.

## Train and compare

Training requires at least two accepted training examples. Validation is never
used by the optimizer. Overlength examples are rejected rather than truncating
answers; input prompt tokens are masked from the training loss.

```bash
.venv/bin/python scripts/train_distillation.py \
  data/silver/teacher-pilot-v2 --steps 20 --output artifacts/distillation-pilot-v2

# First obtain/collect validation labels if needed, then compare both checkpoints.
.venv/bin/python scripts/evaluate_distillation.py \
  artifacts/distillation-pilot-v2 --output artifacts/distillation-comparison-v2
```

The comparison uses the same prompt and greedy decoding for the unmodified base
student and the saved adapter. It reports JSON validity, schema/evidence validity,
exact-document agreement, and exact medication-record precision/recall/F1 against
teacher references. Invalid responses receive no exact-document credit. Exact
record scores compare name, strength, dose, route, frequency and status, retaining
record multiplicity. Evidence positions are not part of record equality, but
source matching is checked. Scores are case-sensitive and surface-form exact.
All-empty record comparisons report undefined record F1, alongside document
agreement. A zero-reference run fails rather than producing reassuring metrics.

A tiny pilot can establish an executable pipeline, not a useful extractor. Before
scaling, inspect rejection rates and teacher outputs, increase input/domain
coverage, and revisit the teacher or prompt if it cannot follow the schema. Use independent labels if later claiming clinical accuracy.

## Other hosted teachers

The hosted backend uses standard HTTPS `/chat/completions`, temperature 0, and
`max_tokens`, plus JSON-object response mode. Provider compatibility must be checked. Configure a provider URL
and model explicitly; the pipeline does not choose a paid provider for you.

```bash
# Set TEACHER_API_KEY securely in environment settings, never in source or chat.
# Configure TEACHER_API_BASE_URL to the provider's HTTPS API base (e.g. ending /v1).
.venv/bin/python scripts/pseudo_label.py label data/silver/NEW_RUN \
  --backend hosted --model PROVIDER_MODEL_ID --limit 3
```

A provider domain and usable credential binding are prerequisites for this route.
Existing credentials should be reused. The optional `--backend local` labeling
route requires a separately downloaded model and exact revision, and no API key. Source and provider data-use terms must be resolved before external
processing or redistribution. No automated API calls happen during setup.

## Checks

```bash
python -m unittest discover -s tests -v
```

Tests exercise filtering, quote ambiguity, Unicode spans, split separation,
prompt masking, and agreement scoring using synthetic fixtures. Fixture tests
are not real teacher outputs or evidence that a clinical model works.

## Live pilot history

The preserved v1 run used arbitrary source prefixes. All four responses were
empty without explicit denial evidence and were rejected; no student trained.
The v2 run uses dosage-focused source windows. Its four teacher responses passed
mechanical validation: three training excerpts (seven medication records) and one
validation excerpt (one record). Eight real API responses were generated across
the two separate diagnostic runs; each used at most four requests.

The v1 failure remains in `data/silver/teacher-pilot-v1/`. The v2 labels and raw
responses are in `data/silver/teacher-pilot-v2/`. These runs are selection-biased
pipeline pilots, not a controlled clinical benchmark.

The live run completed training and evaluation. Both student checkpoints failed
the JSON requirement on the one held-out example. See
[distillation-results.md](distillation-results.md) for the measured outcome.
