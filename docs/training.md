# Treatment-name training

This CPU workflow extracts source-supported treatment names from generated
summaries, including non-drug therapies. Use the [distillation workflow](distillation.md)
for medication attributes, status, and evidence prediction.

After the [quick start](../README.md#quick-start) installation and data preparation:

```bash
.venv/bin/python scripts/train_pilot.py --steps 20 --output artifacts/name-pilot
.venv/bin/python scripts/verify_pilot.py artifacts/name-pilot
```

SmolLM2-135M-Instruct is pinned to `12fd25f77366fa6b3b4b768ec3050bf629380bac`.
The script uses CPU float32, batch size 1, seed 42, and up to four threads. It masks
prompt tokens, skips overlength examples, and excludes reserved notes and exact
duplicates. Training is positive-only and lacks supervision for negative notes.
Output directories cannot be overwritten, including partial runs.

The probe and generation use the training pool; short runs may not visit every
example. `verify_pilot.py` reloads the adapter and checks loss and generation
against the recorded run.
