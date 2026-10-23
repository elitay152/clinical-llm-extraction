# Clinical Medication Extraction

Train medication-name extractors from LLM-generated, source-checked labels.
Compare a biomedical token classifier with prompted and LoRA-tuned language
models using frozen data splits and controlled experiments.

## Quick start

Python 3.11+; training runs on CPU. From the repository root:

```bash
python -m venv .venv
.venv/bin/python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements-training.lock
```

Add `TEACHER_API_KEY` securely to your environment for hosted labeling. Follow
[the experiment workflow](docs/experiment.md) to prepare frozen splits, label
medication names, train the token classifier, and evaluate teacher-reference
agreement. Model revisions and training settings are pinned in
`configs/token-classifier.json`. The task predicts names and source spans;
regimen attributes and status use a separate [medication schema](schemas/medications.schema.json).

Use [iterative experiments](docs/iterative-training.md) for overfit checks,
learning curves, ablations, and seed repeats. The [generative workflow](docs/distillation.md)
provides optional prompted and fine-tuned language-model comparisons.

Network access is needed to `huggingface.co`, `*.huggingface.co`, `*.hf.co`,
`api.github.com`, `raw.githubusercontent.com`, and your teacher provider.
Data, credentials, labels, predictions, reports, and weights stay outside Git.
Review upstream data terms before redistribution.

## Tests

Unit tests need only Python and make no network or model API calls:

```bash
python -m unittest discover -s tests -v
```

## Documentation

- [Distillation workflow](docs/distillation.md) and [experiment design](docs/experiment.md)
- [Medication schema](schemas/medications.schema.json) and [annotation protocol](docs/annotation.md)
- [Optional offline annotation tool](docs/annotation-tool.md)
- [Iterative classifier experiments](docs/iterative-training.md) and [training expansion](docs/training-expansion.md)
- [Medication-name token classifier](docs/token-classifier.md) and [name-labeling rules](docs/name-labeling.md)
- [Treatment-name training](docs/training.md)
