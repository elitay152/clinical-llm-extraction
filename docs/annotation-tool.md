# Annotate the development pilot

Open the generated `data/annotation/pilot-tool.html` in a desktop browser. It is a
self-contained file: it requires no server, package installation, model, API key,
or network connection. Download it from the workspace and open it locally if your
cloud browser blocks local-file navigation. It contains source text; keep it with
your research data rather than publishing it as a website.

## First session

1. Enter an annotator alias. Read the entire source and the expandable rules.
2. Add a medication record. Select its name in the source and click **Use selection
   as name**. Select a supporting passage and click **Attach selected evidence**.
   Attach further passages as needed to support all filled attributes and status.
3. Fill only source-supported attributes. Leave absent/ambiguous values blank,
   which exports as null. Use separate records for distinct regimens or events.
4. Save a draft while working. Once complete, check the review confirmation and
   mark the document reviewed. For an empty result, explicitly confirm that no
   in-scope medication records were found. Drafts require a name and evidence for
   every record; remove incomplete records if necessary before saving.
5. **Export workspace backup** before closing the tab or switching devices.
   This backup contains all documents, provenance, notes, review confirmations,
   aliases, and progress. Import it to resume. Browser-tab saves are temporary;
   there is no automatic disk save or local-storage backup.

Start with three to five documents and record ambiguous cases in the notes field.
We can refine the protocol before you finish all 20. These are development
annotations, so discussing ambiguities does not compromise a headline test set.
An alias and review checkbox record a self-declared review, not independently
verified clinical correctness or inter-annotator agreement.

## Contents and isolation

- 15 clinical-note full texts from the existing reserved candidate pool, selected
  deterministically with text-only screening for temporal/conditional language.
- 5 MTS-Dialog **training** dialogues: 12, 19, 109, 279, and 1051. These supply clear
  medication denials and regimen/status examples. No reference summary is shown.
- 20 blank medication lists with `unannotated` status. None is a confirmed negative
  or a completed human annotation at preparation time.

The set is deliberately selected, small, and mixed-domain; it is not a random
sample or a headline evaluation set. Report clinical notes and dialogues
separately. Exact full-note hashes of the 15 clinical sources were checked against
the completed training pool: no overlap. This does not establish article-level or
patient-level isolation. MTS dialogues were not part of the existing training run.

Clinical notes may include literature discussion. Annotate only the described
patient's regimens, not medications mentioned solely as background examples.
Literal escaped paragraph separators and CRLF were converted to newlines before
annotation. Provenance records raw and normalized text SHA-256 hashes and the
normalization rule. Offsets refer to normalized text, so future model inputs and
scoring must use that same text. The original reserved candidates remain unchanged.

## Export and validate

**Export reviewed JSONL** contains only reviewed documents with medication records
and evidence. It excludes drafts and unannotated documents; it does not include
review notes or provenance. Always keep the full workspace backup alongside it.

After copying a downloaded backup into the workspace, validate it against the
frozen pilot source:

```bash
python scripts/annotation_data.py /path/to/medication-pilot-workspace.json
```

The checker rejects changed source text or membership, malformed records, invalid
Unicode evidence offsets, missing annotator aliases, and unconfirmed empty
reviewed results. These mechanical checks do not assess clinical completeness or
whether evidence supports every attribute. Independent adjudication is deferred
to a later workflow; this tool cannot mark records adjudicated.

## Rebuild and test

From `/workspace/clinical-llm-extraction`:

```bash
python scripts/prepare_data.py --limit 200
python scripts/prepare_annotation_pilot.py
python -m unittest discover -s tests -v
```

Preparation preserves an existing `pilot-v1.json` and all previous annotations.
It rebuilds the HTML from that workspace; it never imports downloaded annotation
backups automatically. Store downloaded backups separately and use the tool's
import control to resume. There is no multi-user merge or synchronization.

Optional browser checks require Playwright and Chromium. They were run using the
cloud machine's existing system Python/Chromium:

```bash
python -m unittest discover -s tests/browser -v
```

The cloud Chromium policy blocks `file:` navigation, so browser tests load the
same HTML into an empty page through Playwright. They exercise selection, review,
backup export/import, source-change rejection, Unicode offsets, and offline
operation. The desktop layout and mobile overflow were also checked. No local
web preview or running service is required.
