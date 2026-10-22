# Experiment design

Question: how closely can a small LoRA-tuned student reproduce a teacher's
medication extraction, compared with the same student before fine-tuning?

## Evaluation

Both checkpoints use identical prompts and greedy decoding. Report JSON validity,
schema/evidence validity, exact-document agreement, and medication-record
precision/recall/F1 against teacher references. Record equality is case-sensitive
and includes name, strength, dose, route, frequency, and status, retaining
multiplicity. Evidence offsets are validated separately from record equality.
Invalid responses count as empty predictions and receive no exact-document
credit. All-empty record comparisons have undefined F1; zero-reference runs fail.
Generation timings include first-use overhead and are not production benchmarks.

## Interpretation and next steps

- Teacher agreement is not clinical accuracy. Exact evidence matches cannot prove
  completeness, temporal reasoning, medication–attribute linking, or correctness.
- The pilot is dosage-enriched and tiny. Exact-text grouping does not establish
  article- or patient-disjoint splits. Map source articles/cases and check near
  duplicates before scaling; preserve fresh held-out groups.
- Independent clinical evaluation would require separately reviewed labels,
  medication-linked field scores, negative cases, and negation/status error slices.
  Keep evaluation labels out of training, prompt selection, and demonstrations.
- An encoder baseline remains future work: generated summaries lack token spans
  and require audited alignment for token classification and attribute linking.
- Report MTS-Dialog transfer results separately from clinical-note results.
  Resolve upstream data and model terms before releasing data or trained weights.
