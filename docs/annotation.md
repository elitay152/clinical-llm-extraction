# Medication annotation protocol (v1)

Annotate only facts supported by the supplied `text`, not the reference summary,
medical knowledge, or other versions of the note. Use the schema in
`schemas/medications.schema.json`. Preserve text exactly while annotating.

- One record per distinct medication regimen or temporally distinct medication
  event. Do not combine an old dose with a later dose. Exclude procedures,
  supplements mentioned only generically, and medications belonging to someone
  other than the patient. Include named supplements used as treatments.
- Preserve the medication name as written. Separate strength (`5 mg` per tablet)
  from dose (`two tablets`). In `5 mg daily`, dose is `5 mg`; strength is unknown
  unless the formulation strength is explicit. Never infer route from formulation.
- Use null for absent or ambiguous attributes. Preserve frequency wording.
- Status is relative to the note's reference time: current (explicitly ongoing),
  discontinued (explicitly stopped), historical (past use without an explicit
  discontinuation), hypothetical (proposed/conditional future use), unknown
  (insufficient context). Negated use is not current. `Never took aspirin` is
  excluded; a proposed aspirin regimen is hypothetical. An allergy alone is not
  medication use. Document uncertain cases for adjudication.
- Evidence uses zero-based Python Unicode character offsets, end-exclusive:
  `text[start:end] == evidence.text`. Include enough spans to support all populated
  attributes and status. A matching span alone does not establish correctness.
- An empty medication list is a valid reviewed negative. An unannotated empty list
  is never a negative example for training or evaluation.

First independently annotate a small pilot, reconcile disagreements, and revise
this protocol before scaling. Have a second reviewer independently annotate the
headline evaluation set, then adjudicate disagreements. Record annotator IDs,
protocol version, and decisions separately from model-visible inputs. Do not
show GPT-generated labels to evaluation annotators before their first pass.

MTS-Dialog is a separate dialogue-domain evaluation. Its prose summaries are not
exhaustive medication labels. Review every selected dialogue, including sections
other than MEDICATIONS. Preserve official split membership. Do not use its test
examples for prompt tuning or demonstrations.

For Augmented Clinical Notes, labels are generated silver labels, not human gold.
Audit the source scope (`note` versus `full_note`) before choosing model inputs.
The preparation script exports full notes as annotation candidates, without
copying generated summaries into annotation files. Sampling the first N rows is
an engineering smoke check, not a representative benchmark.

The first-pass browser workflow and 20-document development pilot are described
in [annotation-tool.md](annotation-tool.md). Review stage is self-declared;
independent adjudication and headline test construction are separate future steps.
