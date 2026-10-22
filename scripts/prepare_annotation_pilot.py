"""Build a blind 20-document development pilot and self-contained browser tool."""
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_data import text_hash
from scripts.annotation_data import FORMAT, validate_workspace


def canonical_text(text):
    # Normalize upstream escaped paragraphs; retain raw-text hashes in provenance.
    return text.replace('\\\\n', '\n').replace('\\n', '\n').replace('\r\n', '\n')


def build_workspace(reserved, mts_rows, revisions):
    unique = {}
    for doc in reserved:
        unique.setdefault(text_hash(doc['text']), doc)
    def rank(doc):
        t = doc['text'].lower()
        challenges = bool(re.search(r'discontinu|stopped|withdraw', t)) + bool(re.search(r'\bif\b|would|consider', t))
        return (-challenges, doc['id'])
    chosen = sorted(unique.values(), key=rank)[:15]
    if len(chosen) != 15:
        raise ValueError('Need at least 15 unique reserved notes')
    docs, provenance = [], {}
    def add(identifier, raw, source, revision, split):
        text = canonical_text(raw)
        docs.append({'id': identifier, 'text': text, 'medications': [], 'annotation_status': 'unannotated'})
        provenance[identifier] = {'source': source, 'revision': revision, 'source_split': split,
                                  'original_text_sha256': hashlib.sha256(raw.encode()).hexdigest(),
                                  'annotation_text_sha256': hashlib.sha256(text.encode()).hexdigest(),
                                  'normalization': 'literal escaped newline separators and CRLF converted to LF'}
    for doc in chosen:
        add(doc['id'], doc['text'], 'AGBonnet/augmented-clinical-notes', revisions['augmented_clinical_notes'], 'reserved_from_train')
    mts = {row['ID']: row for row in mts_rows}
    # Text-only screening for denial and regimen complexity; no reference labels exported.
    for identifier in ('12', '19', '109', '279', '1051'):
        add('mts:train:' + identifier, mts[identifier]['dialogue'], 'abachaa/MTS-Dialog', revisions['mts_dialog'], 'train')
    workspace = {'format': FORMAT, 'role': 'development_pilot', 'protocol_version': '1',
                 'documents': docs, 'metadata': {}, 'provenance': provenance}
    return validate_workspace(workspace)


def main():
    directory = ROOT / 'data/annotation'
    path = directory / 'pilot-v1.json'
    if path.exists():
        workspace = json.loads(path.read_text())
        validate_workspace(workspace)
    else:
        reserved = [json.loads(line) for line in (directory / 'acn-candidates.jsonl').read_text().splitlines()]
        with (ROOT / 'data/raw/MTS-Dialog-TrainingSet.csv').open(encoding='utf-8-sig', newline='') as handle:
            mts = list(csv.DictReader(handle))
        revisions = json.loads((ROOT / 'data/raw/revisions.json').read_text())
        workspace = build_workspace(reserved, mts, revisions)
        with path.open('x') as output:
            json.dump(workspace, output, indent=2, ensure_ascii=False)
    html = (ROOT / 'annotation/app.html').read_text()
    payload = json.dumps(workspace, ensure_ascii=False).replace('<', '\\u003c')
    if html.count('__PILOT_DATA__') != 1:
        raise ValueError('Tool template must contain exactly one data marker')
    output = directory / 'pilot-tool.html'
    output.write_text(html.replace('__PILOT_DATA__', payload))
    print(json.dumps({'documents': len(workspace['documents']), 'workspace': str(path), 'tool': str(output)}, indent=2))


if __name__ == '__main__':
    main()
