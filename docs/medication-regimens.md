# Medication routes and frequencies

The demo combines the name/dose classifier with source-candidate routes and
frequencies and conservative attribute linking. Joint and separate BIO models
are available for comparison; neither replaces the names-only or dose checkpoint.
Fields copy exact source text and Unicode offsets without normalization,
calculated schedules, or inferred defaults. Medication status is outside this task.

## Annotation

Start from a frozen [dose-label dataset](medication-doses.md), preserving its
names, doses, source groups, and locked test membership:

```bash
.venv/bin/python scripts/label_medication_doses.py prepare \
  data/silver/my-dose-labels data/silver/my-regimen-labels --task regimen
.venv/bin/python scripts/label_medication_doses.py label data/silver/my-regimen-labels \
  --split train --limit 8 --model gpt-4.1-2025-04-14
.venv/bin/python scripts/label_medication_doses.py collect data/silver/my-regimen-labels
```

Inspect the ignored label report, then label remaining training notes and,
separately, development notes using the same command and explicit split. Each
note receives one request with at most two schema-error retries. The teacher
selects marked phrases and medication occurrences, excluding disease sites,
cancer stages, follow-up intervals, symptom timing, and unstated attributes.
Freeze the labels with `collect ... --freeze` after all selected notes validate.
No test notes are submitted.

## Model comparisons

Train an independent route/frequency classifier alongside the fixed dose model:

```bash
.venv/bin/python scripts/train_token_classifier.py data/silver/my-regimen-labels \
  --config configs/attribute-classifier.json --output artifacts/my-attributes
.venv/bin/python scripts/evaluate_medication_doses.py artifacts/my-dose-classifier \
  --attribute-model artifacts/my-attributes --dose-baseline artifacts/my-dose-classifier \
  --output artifacts/my-regimen-development
```

Use `--training-limit 4` with a separate config containing additional epochs and
no synthetic negatives for an overfit probe, then `--training-diagnostic` when
evaluating the paired checkpoints. Only the attribute model is trained by that
probe. The names/doses always come from the frozen base model.

For a joint nine-label baseline, train with `configs/regimen-classifier.json` and
evaluate that checkpoint directly with `--dose-baseline`. The evaluator records
exact field spans, medication–attribute pairs, and complete records; comparisons
include all source candidates and linking with reference entities. Raw model
metrics are retained alongside results constrained to supported candidate formats.
These are candidate-limited teacher-reference development comparisons, not an
exhaustive annotation or clinical accuracy estimate. Detailed reports stay outside Git.

The deployed candidate baseline can be evaluated without training a new model:

```bash
.venv/bin/python scripts/evaluate_medication_doses.py artifacts/my-dose-classifier \
  --attribute-candidates --reference-directory data/silver/my-regimen-labels \
  --output artifacts/my-candidate-development
```

## Inference and deployment

```bash
printf 'Take aspirin 81 mg orally once daily.' | \
  .venv/bin/python scripts/predict_medication_names.py artifacts/my-dose-classifier \
  --attribute-candidates
```

Use `--attribute-model artifacts/my-attributes` instead to compare the trained
attribute model. Both options preserve base name/dose predictions. Output adds
`route_spans`, `frequency_spans`, per-medication `routes`/`frequencies`, and unlinked
attributes. Local linking abstains across other medications, clause boundaries,
and ambiguous coordinated lists. A candidate phrase can be unrelated to a drug;
matching source text does not establish a correct medication association.

Add `--regimen-rules` with `--dose-run` to [the deployment command](deployment.md)
for the source-candidate demo mode. To deploy a trained attribute or joint model
instead, use `--regimen-run`. Optional weights stay in the private model repository.

## Relation classifier experiment

The optional relation experiment freezes the name/dose encoder and trains a small
residual classifier using contextual medication/attribute spans and source-layout
features. Its initial decisions match the rules. Each candidate pair receives an
independent score, allowing shared attributes or no association; selected reference
attributes without a resolved medication are excluded from negative supervision.

```bash
.venv/bin/python scripts/train_relation_classifier.py \
  --encoder artifacts/my-dose-classifier --data data/silver/my-regimen-labels \
  --predictions artifacts/my-candidate-development/predictions.json \
  --output artifacts/my-relations
```

The command checks frozen labels and source-group separation, caches encoder
features, and requires a six-note memorization check before full training. The
fixed configuration and threshold apply to both reference-entity and frozen
pipeline-prediction comparisons. Reports include pair precision/recall/F1 and
exact complete records. Checkpoints, feature caches, and reports stay under the
ignored output directory. This is an offline experiment; the demo uses the rules.

Training-only error review and label expansion use the reserved, unlabeled training
notes in the frozen manifest. Selection prioritizes source-layout signals such as
parentheses, coordination words, and attribute-rich text; these are sampling
heuristics, not verified clinical relationships.

```bash
.venv/bin/python scripts/expand_relation_training.py audit \
  data/silver/my-regimen-labels artifacts/my-relations artifacts/training-errors.json
.venv/bin/python scripts/expand_relation_training.py prepare \
  data/silver/my-regimen-labels data/silver/my-relation-selection --count 19
.venv/bin/python scripts/expand_relation_training.py label data/silver/my-relation-selection
.venv/bin/python scripts/expand_relation_training.py collect \
  data/silver/my-relation-selection data/silver/my-expanded-regimen-labels
```

The teacher labels names, doses, routes, and frequencies with existing source checks
and bounded retries. Collection revalidates responses, reports rejected notes, and
copies development/test files byte-for-byte. Source candidates omitted by the
teacher provide negative pairs; unresolved selected attributes remain excluded.
Run the relation training command with the expanded dataset, the same configuration,
and the same saved development predictions to compare the resulting model.

## Abstention thresholds

Select separate dose, route, and frequency thresholds with five source-grouped
training folds, then evaluate the frozen thresholds on development data:

```bash
.venv/bin/python scripts/calibrate_relation_thresholds.py \
  --run artifacts/my-relations --data data/silver/my-expanded-regimen-labels \
  --predictions artifacts/my-candidate-development/predictions.json \
  --output artifacts/my-relation-thresholds
```

Each fold refits the relation head and holds out complete source groups. The encoder
and cached span features remain fixed; the encoder has seen name/dose labels from
some held-out training groups. This measures head-level threshold selection, not
cross-validation of the full training pipeline. Ambiguous pairs remain excluded.

Selection maximizes out-of-fold pair F1 while keeping false-link counts no higher
and precision no lower than the rules, with ties favoring fewer false links and
higher thresholds. The fixed grid includes abstaining from every link in a field
(`null`). Thresholds act on sigmoid scores, not calibrated correctness probabilities.
The command saves its plan, fold assignments, and selected thresholds before the
development comparison. Detailed artifacts stay outside Git; deployment is separate.

## Final held-out comparison

Freeze the rules, both relation checkpoints, the existing encoder, and selected
thresholds before creating test references. `--source` is the expanded frozen
training dataset; `--original-data` is the original relation training snapshot.

```bash
.venv/bin/python scripts/final_relation_comparison.py freeze \
  --source data/silver/my-expanded-regimen-labels \
  --original-data data/silver/my-regimen-labels \
  --encoder artifacts/my-dose-classifier \
  --original artifacts/my-original-relations --expanded artifacts/my-relations \
  --calibration artifacts/my-relation-thresholds --output artifacts/my-final-test
.venv/bin/python scripts/final_relation_comparison.py label artifacts/my-final-test
.venv/bin/python scripts/final_relation_comparison.py evaluate artifacts/my-final-test
```

The sealed plan hashes checkpoints, thresholds, code, prompts, and source snapshots.
Labeling uses the same teacher protocols with exact-source validation; every
reserved test note must pass before scoring. References and predictions remain in
the ignored output directory, leaving source snapshots unchanged. All approaches
receive identical predicted entities. The primary metric is exact complete-record
F1; secondary metrics are field-specific pair F1 and false-link counts.

This final comparison describes the frozen approaches, rather than selecting a
new demo model from test results. Further model or threshold selection needs a
fresh holdout. Only aggregate chart values are published in the README.
