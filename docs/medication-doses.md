# Medication names and doses

The dose task adds `B-DOSE` and `I-DOSE` to the medication-name BIO labels.
The classifier selects exact source spans; a separate conservative rule links
nearby doses to names. Names, dose spans, and complete name–dose pairs are scored
separately. For route and frequency extraction, use the [regimen workflow](medication-regimens.md).
Medication status is outside these tasks.

## Labels

Start from a frozen name-label dataset with locked test membership:

```bash
.venv/bin/python scripts/label_medication_doses.py prepare \
  data/silver/my-name-labels data/silver/my-dose-labels
.venv/bin/python scripts/label_medication_doses.py label data/silver/my-dose-labels \
  --split train --limit 8 --model gpt-4.1-2025-04-14
.venv/bin/python scripts/label_medication_doses.py collect data/silver/my-dose-labels
```

Inspect the ignored label report before continuing with the remaining training
notes and the development notes. The teacher selects marked source candidates
and links them to frozen medication occurrences. Candidate amounts include mass,
volume, units, counted forms, and weight/body-area-adjusted amounts. They exclude
concentrations and daily/infusion rates. The teacher must distinguish doses from
strength-only mentions and laboratory values. Missing fields remain absent;
amounts are never calculated. This is candidate-limited teacher supervision:
source validation checks membership and spans, not clinical correctness or
exhaustive coverage. Unsupported formats cannot enter the reference labels.

Each selected note receives one request, with at most two additional attempts
for schema/selection errors. Attempts are saved; completed notes are not relabeled
on resume. No test notes are submitted. A teacher key is required for labeling.
The model, prompt, candidate version, membership, and source hashes are recorded.
All selected records must pass validation before freezing:

```bash
.venv/bin/python scripts/label_medication_doses.py label data/silver/my-dose-labels \
  --split train --limit 100 --model gpt-4.1-2025-04-14
.venv/bin/python scripts/label_medication_doses.py label data/silver/my-dose-labels \
  --split development --limit 100 --model gpt-4.1-2025-04-14
.venv/bin/python scripts/label_medication_doses.py collect data/silver/my-dose-labels --freeze
```

## Training and evaluation

Use a training-only subset that covers every BIO class for a small overfit check
before the full run. `--training-limit 4` selects such a subset deterministically;
use a separate config with additional epochs and synthetic negatives disabled.
Evaluate it with `--training-diagnostic`, measuring name and dose entity recovery.
The fixed linking rule is not trained by this diagnostic.

```bash
.venv/bin/python scripts/train_token_classifier.py data/silver/my-dose-labels \
  --config configs/dose-classifier.json --output artifacts/my-dose-classifier
.venv/bin/python scripts/evaluate_medication_doses.py artifacts/my-dose-classifier \
  --name-baseline artifacts/my-name-classifier --output artifacts/my-dose-development
printf 'The patient takes aspirin 81 mg daily.' | \
  .venv/bin/python scripts/predict_medication_names.py artifacts/my-dose-classifier
```

The evaluator compares name extraction with the supplied names-only checkpoint
on identical notes and unchanged name references. It also measures a baseline
using the names-only model plus all dose candidates, and the linking rule with
reference entities. The latter separates association failures from entity errors. Use
`--linking-rule conservative-local-rule-v1` to evaluate the baseline linking rule;
the default is `conservative-local-rule-v2`.
These are development-set comparisons against teacher labels, not test or clinical
accuracy estimates. Detailed metrics and predictions stay in ignored artifacts.

Inference returns names, dose spans, per-name `doses` lists, and `unlinked_doses`.
Offsets count Unicode code points with exclusive ends. The rule handles local
name–dose phrases, explicit dose introductions
(such as "at a dose of"), and route text between an amount and name. It abstains
on ambiguous candidates and bare coordinated lists; it cannot reliably handle
distant references or complex schedules. An
unlinked dose stays visible instead of being assigned to a guessed medication.
The existing names-only checkpoints and commands retain their output format.
