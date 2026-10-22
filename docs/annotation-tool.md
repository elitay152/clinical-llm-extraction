# Offline annotation tool

From the repository root, prepare the data and build the self-contained HTML:

```bash
python scripts/prepare_data.py --limit 200
python scripts/prepare_annotation_pilot.py
```

Download `data/annotation/pilot-tool.html` and open it locally in a desktop browser.
No server or network connection is required. The development pilot contains 15
reserved clinical notes and 5 MTS-Dialog training dialogues (12, 19, 109, 279, 1051).
It begins unannotated and hides generated labels. Follow the [protocol](annotation.md).

## Annotate and save

1. Enter an annotator alias and read the entire source.
2. Add a medication record. Select its name and click **Use selection as name**;
   select supporting passages and click **Attach selected evidence**.
3. Fill supported attributes, leaving unknown values blank. Record ambiguities
   in the notes field. Draft records require a name and evidence.
4. Save a draft or confirm review. Empty reviewed results need explicit negative
   confirmation. A review checkbox records self-declared review, not adjudication.
5. **Export workspace backup before closing the tab.** There is no automatic disk
   save. Import that backup to resume; it includes provenance, notes, and progress.
   **Export reviewed JSONL** contains reviewed documents only, without metadata.

Start with three to five documents and reconcile ambiguities before scaling.
This selected, mixed-domain pilot is development data; report the domains separately.

## Source integrity

Literal escaped paragraph separators and CRLF are normalized to newlines before
annotation. Offsets refer to this normalized text, which must also be used for
model inputs. Provenance retains raw/normalized hashes; original source files
remain unchanged. Exact clinical-note hashes were excluded from the training
pool; broader isolation limits are described in [experiment design](experiment.md).

Validate an exported backup against the frozen source:

```bash
python scripts/annotation_data.py /path/to/medication-pilot-workspace.json
```

Rebuilding preserves `pilot-v1.json` and does not import downloaded backups or
merge users' work. Browser tests require Playwright and `/usr/bin/chromium`:

```bash
python -m unittest discover -s tests/browser -v
```

Tests load HTML through Playwright because cloud Chromium blocks `file:` navigation.
