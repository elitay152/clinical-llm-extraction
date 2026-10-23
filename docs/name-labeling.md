# Medication-name mention labels

The name-only protocol in [`configs/name-teacher-prompt.txt`](../configs/name-teacher-prompt.txt)
labels every explicitly named medicinal-agent occurrence, regardless of regimen,
status, or person. It includes historical, discontinued, proposed, negated, allergy,
and background mentions. Patient/regimen linking is a separate task.

Copy core brand or generic names with their original spelling and case. Keep
chemical salts and meaningful multiword names. Exclude dose forms, routes,
strengths, doses, frequency, trademarks, and surrounding punctuation. Annotate
brand and generic names separately, and separate individually named ingredients
in mixtures. Do not infer names from descriptions or verbs. Exclude laboratory
analytes mentioned only as measurements, unnamed drug classes, and procedures.

The teacher returns names with unique verbatim context passages and confirms a
coverage review. Code resolves exact, token-aligned occurrences within each
context to Unicode source spans. Overlapping spans and selected boundary noise
are rejected. Repeated source occurrences remain distinct; identical spans are
deduplicated. An empty list means no named agents were selected, not that the
patient takes no medication. Teacher self-review and source matching do not
establish completeness or clinical correctness.

Follow [the experiment workflow](experiment.md) to label a new frozen split.
To restrict a new label version to an existing classifier's training membership
and accepted development notes, add `--membership-run artifacts/my-classifier`
to the `prepare SOURCE OUTPUT` command.

Label commands make paid hosted requests using `TEACHER_API_KEY`. They resume by
ID, preserve raw responses, and do not automatically replace rejected responses.
The manifest seals the prompt, source membership, original manifest hash, and
classifier-run hash when a membership run is supplied. Test labeling is blocked. Source labels remain unchanged;
all generated data and reports belong in ignored `data/` and `artifacts/` paths.

Train with the [token classifier workflow](token-classifier.md). Native name
references use `name_mentions` spans and `name_label_protocol`; no regimen
attributes or status values are fabricated. For label comparisons, use identical
accepted source membership and score both models against the same reference
version. Scores from different reference versions are not directly comparable.
