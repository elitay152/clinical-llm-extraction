# Clinical Medication Extraction

Train a small medication extractor using LLM-generated labels, then compare the
base and fine-tuned model against held-out teacher outputs. The schema covers
medication name, strength, dose, route, frequency, status, and source evidence.

## Quick start

Python 3.11+; training uses CPU with precision configured in `configs/student.json`. From the repository root:

```bash
python -m venv .venv
.venv/bin/python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements-training.lock
.venv/bin/python scripts/prepare_data.py --limit 200
```

Supply `TEACHER_API_KEY` securely in your environment. Defaults are
`TEACHER_MODEL=gpt-4.1-mini` and `TEACHER_API_BASE_URL=https://api.openai.com/v1`.
The following command makes up to four paid teacher requests. Use new paths for
each run; existing outputs are preserved.

```bash
.venv/bin/python scripts/run_distillation_pilot.py \
  --silver data/silver/my-pilot --output artifacts/my-adapter \
  --comparison artifacts/my-comparison
```

Network access is needed to `huggingface.co`, `*.huggingface.co`, `*.hf.co`,
`api.github.com`, `raw.githubusercontent.com`, and `api.openai.com`.
Data, credentials, labels, predictions, and weights stay outside Git. Review
upstream data terms before redistribution.

For a frozen train/development/test experiment, use the
[split and labeling workflow](docs/experiment.md). The student checkpoint and
prompt are configured in `configs/student.json`. The default student selects
medication-name token ranges; code retrieves exact source text and offsets.

## Tests

Unit tests need only Python and make no network or model API calls:

```bash
python -m unittest discover -s tests -v
```

## Documentation

- [Distillation workflow](docs/distillation.md) and [experiment design](docs/experiment.md)
- [Medication schema](schemas/medications.schema.json) and [annotation protocol](docs/annotation.md)
- [Optional offline annotation tool](docs/annotation-tool.md)
- [Iterative classifier experiments](docs/iterative-training.md)
- [Medication-name token classifier](docs/token-classifier.md) and [name-labeling rules](docs/name-labeling.md)
- [Treatment-name training](docs/training.md)
