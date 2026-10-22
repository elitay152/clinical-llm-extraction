# First LoRA training pilot

This is a CPU pipeline smoke run, not the final medication extraction experiment.
It trains SmolLM2-135M-Instruct (Apache-2.0), pinned to revision
`12fd25f77366fa6b3b4b768ec3050bf629380bac`, using LoRA on query/value projections.

The silver target is a JSON array of treatment names whose strings occur in an
excerpt of `full_note`. Treatments can include procedures; this is deliberately
not labeled a medication-only task. Unsupported names and examples without a
usable target are excluded. Silver labels can still omit facts or misinterpret
context. This positive-only pilot supplies no supervision for notes without
treatments and must not be used to judge negative-note behavior. Dose, route, frequency, status, and evidence prediction are not trained
in this pilot. No clinically meaningful performance claim follows from its loss.

The completed 20- and 200-step runs are documented in [pilot-results.md](pilot-results.md).

## Reproduce

From the existing checkout (cloud tasks are already isolated; no worktree needed):

```bash
python -m venv .venv
.venv/bin/python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements-training.lock
python scripts/prepare_data.py --limit 200
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/train_pilot.py --steps 20
.venv/bin/python scripts/verify_pilot.py artifacts/pilot-lora
```

CPU torch is intentional for this environment; use a matching official CUDA torch
build on a separately provisioned GPU machine. The current script uses CPU float32.
An alternate run needs a new output directory, e.g. `--output artifacts/pilot-lora-2`.
The default output refuses to overwrite an existing run, even a partial one.

Raw data, adapters, caches, and generated predictions remain local/ignored.
Do not publish adapters or redistribute data before upstream terms are resolved.
The run and verification records contain model/data revisions, package versions, training IDs,
source hashes, exclusions, losses, and a training-example generation.

## Boundaries and validation

- All current annotation-candidate full-note hashes and exact duplicate training
  full notes are excluded. This is not article- or patient-level isolation.
- The reserved annotation candidate pool remains unannotated and untouched.
- The fixed loss probe and generated example are selected from the training pool, not validation. A short run may not visit every example in that pool.
- Prompt tokens have loss masked; overlength examples are skipped rather than
  silently dropping target tokens. Batch size is 1, seed 42, up to four CPU threads.
- Check that the run completed all requested steps with finite losses, adapter
  files exist, and saved-adapter inference works. Inspect `run.json` for whether
  the example generation has the expected JSON structure; valid JSON alone does
  not establish source support or extraction accuracy.

The active medication-schema experiment uses [teacher-generated supervision](distillation.md).
Independent gold annotation remains a future option for clinical accuracy claims.
