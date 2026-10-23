# Medication-name token classifier

BiomedBERT labels source tokens with `O`, `B-MED`, or `I-MED`. The model, revision,
training settings, and compute precision are pinned in
[`configs/token-classifier.json`](../configs/token-classifier.json).

Using the [frozen split workflow](experiment.md) and the training dependencies:

```bash
.venv/bin/python scripts/train_token_classifier.py data/silver/my-experiment \
  --output artifacts/my-classifier
.venv/bin/python scripts/evaluate_token_classifier.py artifacts/my-classifier \
  --output artifacts/my-classifier-development
```

Training accepts [native medication-name mentions](name-labeling.md) or names
located in regimen evidence, plus optional synthetic lab-only
negatives. Unalignable training rows are recorded and excluded. Subword
continuations, special tokens, and padding are masked from the loss. Class
weights are square-root inverse training frequencies, normalized to mean one.
All encoder weights and the classification head are trained. Epochs are fixed;
development data is not used for early stopping or class weights.

Inference uses the first subword prediction for each source token. An orphan
`I-MED` starts a span; adjacent `B-MED` labels start separate spans. Decoded ranges
copy exact source text and Unicode offsets through the shared name-span resolver.
Inputs exceeding the configured context limit fail without truncation.

Evaluation reloads the saved checkpoint and reports the same exact name-span and
unique name-text metrics as the generative name extractor. Attributes and status
are outside this task. Matching source text does not establish that a selected
term is a medication. Teacher labels may omit mentions; teacher agreement is not
clinical accuracy, and pretraining overlap cannot be excluded.

An optional `--negative-controls PATH` accepts a JSON array of separate synthetic
records with empty `medications`, IDs, text, `input_sha256`, and
`label_origin: synthetic`. Their results are reported separately. Data, weights,
and evaluation files stay in ignored `data/` and `artifacts/` directories.

## Inference and local demo

A saved classifier directory needs its weights, tokenizer, and `run.json` only;
training labels and a teacher API key are not required. With the training
requirements installed, extract names from a UTF-8 file or stdin:

```bash
.venv/bin/python scripts/predict_medication_names.py artifacts/my-classifier --input note.txt
printf 'The patient takes aspirin daily.' | \
  .venv/bin/python scripts/predict_medication_names.py artifacts/my-classifier
```

JSON contains `medication_names` with exact `text`, zero-based `start`, and
exclusive `end` offsets measured in Unicode code points, plus elapsed extraction
time (tokenization and inference, excluding model loading). Repeated mentions
remain separate; no names returns an empty list. Blank and overlong inputs fail
with a nonzero exit code. No input or output files are saved automatically.

Launch the browser demo with the same checkpoint:

```bash
.venv/bin/python scripts/serve_demo.py artifacts/my-classifier --port 8000
```

Open `http://127.0.0.1:8000`. The demo includes editable synthetic notes,
highlighted source evidence, and JSON output. It runs on localhost with no model
API calls or note storage. Requests are limited to 64 KiB; the checkpoint's
wordpiece context limit also applies. Use synthetic text for demonstrations.
