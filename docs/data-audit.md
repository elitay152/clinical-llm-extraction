# Initial dataset feasibility audit

This is a prefix sample of 200 Augmented Clinical Notes examples, not an estimate
of full-dataset prevalence. The audit was executed, not inferred from the card.

Pinned sources:

- Augmented Clinical Notes: `c1cb55e3fb2a08bef749c0de59500ef97ca582b6`
- MTS-Dialog: `3ff0801933608d6f570468c13125125fb5cabdea`

## Augmented Clinical Notes

| Check | Result |
|---|---:|
| Rows inspected | 200 |
| Rows with shorter `note` than `full_note` | 197 |
| Duplicate full-note rows, whitespace normalized | 33 |
| Summaries failing strict JSON parsing | 57 |
| Summaries recovered by accepting literal control characters | 53 |
| Summaries still unparseable | 4 |
| Parsed summaries missing a list-valued `treatments` field | 8 |
| Treatment entries including placeholders | 258 |
| Treatment entries with populated dosage | 32 |
| Treatment entries with populated frequency | 33 |

The four remaining parse failures include unescaped quotation marks. The parser
does not guess corrections for these. Recovery only permits literal control
characters or repairs escaped whitespace outside JSON strings, and is reported
explicitly. This does not validate clinical correctness.

Treatment entries include non-medication treatments and placeholders. Counts of
populated fields are not annotation accuracy or recall. Route, medication status,
and evidence spans are not dedicated fields in the published template. Strength
and dose are conflated under dosage.

**Decision:** suitable for exploring silver-label training, subject to upstream
terms and label audit; not currently suitable as independent gold evaluation.
Exact duplicate rows demonstrate why row-wise random splitting is inappropriate.
Distinct row IDs do not prove distinct patients or articles.

## MTS-Dialog

All four main CSV files downloaded successfully. Columns are `ID`,
`section_header`, `section_text`, and `dialogue`.

| Split | Rows | MEDICATIONS section rows |
|---|---:|---:|
| Training | 1,201 | 54 |
| Validation | 100 | 7 |
| Test 1 | 200 | 10 |
| Test 2 | 200 | 9 |

These are section counts, not counts of medication-containing dialogues. Other
sections can contain medications; medication sections can contain no medication
use. No entity/attribute/evidence annotations are supplied in these CSVs.
The repository publishes CC BY 4.0 terms and requests citation of Ben Abacha et al.,
*An Empirical Study of Clinical Note Generation from Doctor-Patient Encounters*
(EACL 2023), https://aclanthology.org/2023.eacl-main.168.

**Decision:** use as a separately annotated dialogue transfer evaluation, not as
an already labeled medication benchmark.
