# Clinical Medication Extraction

Compare an encoder baseline, a prompted small language model, and the same LLM
with LoRA fine-tuning on evidence-grounded medication extraction.

**Current stage:** teacher–student medication extraction with generated supervision.
See the [distillation workflow](docs/distillation.md). The earlier CPU LoRA pilot
covers treatment names only. No independent clinical accuracy evaluation has run.

## Run the data audit

Python 3.11+ is sufficient; this milestone has no third-party dependencies.
From the repository root:

```bash
python -m unittest discover -s tests -v
python scripts/prepare_data.py --limit 200
```

The command pins upstream revisions on first use, caches the first 200 Augmented
Clinical Notes rows, downloads the four main MTS-Dialog splits and source
metadata, and writes `artifacts/data_audit.json`. It also creates up to 30 blind
clinical-note annotation candidates at `data/annotation/acn-candidates.jsonl`.
Existing annotation files are never overwritten. An unannotated empty medication
list must not be treated as a negative example.

Rerunning uses cached data. Use `--limit 0` to audit the full 30K dataset after
reviewing its source terms and resource needs. This prefix sample is not random
or representative. No automatic training split is created. Keep the cached
`data/raw/revisions.json` with experiment records to reproduce upstream versions.
Checksums record downloaded bytes; they are not publisher-supplied signatures.
Data and generated artifacts are excluded from Git; do not redistribute them
until upstream permissions are resolved.

Required network hosts: `huggingface.co`, its dataset/download endpoints under
`*.huggingface.co` and `*.hf.co`, `api.github.com`, and
`raw.githubusercontent.com`. No API key is needed for these public sources.

## Run a hosted distillation pilot

Prepare a CPU training environment from the repository root:

```bash
python -m venv .venv
.venv/bin/python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements-training.lock
.venv/bin/python scripts/prepare_data.py --limit 200
```

Configure these variables in your shell or environment settings:

- `TEACHER_API_KEY`: your OpenAI API credential, supplied securely.
- `TEACHER_API_BASE_URL`: `https://api.openai.com/v1` (the default).
- `TEACHER_MODEL`: `gpt-4.1-mini` (the default).

Allow HTTPS to `api.openai.com` in addition to the dataset download hosts. No
credential is bundled in the repository. The following command makes up to four
paid teacher requests, trains on accepted labels, and compares held-out teacher
agreement. Use new output paths for each experiment:

```bash
.venv/bin/python scripts/run_distillation_pilot.py \
  --silver data/silver/my-pilot --output artifacts/my-adapter \
  --comparison artifacts/my-comparison
```

Training uses CPU float32; there is no automatic GPU provisioning. Each pilot
prepares dosage-enriched source excerpts and separates exact source-text groups.
No API calls run during installation or unit tests. See the [workflow](docs/distillation.md)
for validation rules and limitations. Raw data, labels, predictions and weights
are generated locally; only code and aggregate result summaries are versioned.

## Annotation pilot

Run `python scripts/prepare_annotation_pilot.py` after the data audit. Download
and open `data/annotation/pilot-tool.html` in your browser to annotate 20 development
documents (15 clinical notes and 5 training dialogues). Source text and evidence
selection work offline; export a workspace backup to preserve your annotations.
The documents begin unannotated. See the [annotation tool guide](docs/annotation-tool.md).

## Design and findings

- [Medication JSON schema](schemas/medications.schema.json)
- [Annotation protocol](docs/annotation.md)
- [Experiment design and data gates](docs/experiment.md)
- [Initial audit findings](docs/data-audit.md)
- [Completed CPU LoRA pilot results](docs/pilot-results.md)
- [Medication teacher–student distillation](docs/distillation.md)
- [First live distillation results](docs/distillation-results.md)

The live pilot verified labeling, training, and adapter reload, but both student
checkpoints returned invalid JSON on the single held-out excerpt. No extraction
improvement or clinical accuracy is established. Next steps are a larger silver
training set, source/article grouping, and a more capable student. Independent
annotation is optional for distillation and required for future clinical accuracy
claims. MTS-Dialog remains available for separately annotated dialogue evaluation.
