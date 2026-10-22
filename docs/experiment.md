# Current direction

The active workflow uses teacher-generated medication supervision and measures
student–teacher agreement; manual annotation is optional for this stage. See
[distillation.md](distillation.md). Independent labels remain necessary for future
clinical accuracy claims. The original clinical benchmark plan below is retained
as a future extension, not a prerequisite for the distillation pilot.

# Experiment contract

Question: does LoRA fine-tuning improve evidence-grounded medication extraction
relative to prompting the same small open-weight model?

## Data gates

1. Audit Augmented Clinical Notes: JSON parseability, treatment field coverage,
   non-drug treatments, unsupported attributes, and note/full-note differences.
   Its card tags MIT; verify PMC-Patients, NoteChat, and underlying article terms
   before redistributing data, training artifacts, or deploying an API.
2. Map NoteChat identifiers back to original PMC articles/cases where possible.
   Group all variants and cases from the same source article before splitting.
   Audit exact and near duplicates across groups. Patient-disjointness cannot be
   assumed from distinct rows or IDs. The current script checks exact normalized
   text duplicates only and intentionally does not create training splits.
3. Reserve source groups for independent annotation before training. Never use
   their generated labels for training, prompt selection, or few-shot examples.
4. Use MTS-Dialog only as a separately reported dialogue transfer evaluation after
   medication annotation. Its source is not a substitute for clinical note gold.

## Comparisons

- Encoder token classification plus medication–attribute linking; score only
  fields it supports. Generated summaries lack token spans, so encoder training
  requires audited span annotation/alignment, not a direct conversion to BIO.
- Zero-shot and fixed few-shot prompts for one small LLM.
- LoRA on that same LLM, with identical inference schema and decoding settings.

Choose model/license/context length and GPU budget after the data audit. Record
model revisions, dataset revisions, seeds, prompts, hyperparameters, and hardware.
No GPU training or external model API calls are part of initial preparation.

## Evaluation

Primary: medication-linked field precision/recall/F1 and complete-record exact
match. Predeclare normalization and one-to-one record matching; report each field
separately, including status. Score unsupported facts by manual review of source
support; a missing reference label is not by itself proof of hallucination.
Report JSON/schema validity, evidence-offset validity and support, negative-note
false positives, negation/status error slices, latency (including warmup policy,
batch size and input/output lengths), and peak accelerator memory. Use grouped
bootstrap confidence intervals where the evaluation size supports them.

The data milestone prepares annotation. The subsequent CPU pilot in
[training.md](training.md) exercises LoRA on silver treatment-name labels only.
No benchmark scores, patient-disjoint splits, or production readiness is claimed.
