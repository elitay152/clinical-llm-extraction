"""Download revision-pinned public samples and audit them without ML dependencies."""
import argparse
import collections
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
HF = 'https://huggingface.co'
REPO = 'AGBonnet/augmented-clinical-notes'
MTS = 'abachaa/MTS-Dialog'
FILES = {
    'train': 'MTS-Dialog-TrainingSet.csv',
    'validation': 'MTS-Dialog-ValidationSet.csv',
    'test1': 'MTS-Dialog-TestSet-1-MEDIQA-Chat-2023.csv',
    'test2': 'MTS-Dialog-TestSet-2-MEDIQA-Sum-2023.csv',
}


def request(url):
    return urllib.request.urlopen(urllib.request.Request(
        url, headers={'User-Agent': 'clinical-llm-extraction-data-audit/0.1'}), timeout=60)


def get_json(url):
    with request(url) as response:
        return json.load(response)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def parse_summary(text):
    """Accept literal control characters or repair outer whitespace; never eval."""
    try:
        return json.loads(text), False
    except json.JSONDecodeError:
        pass
    try:
        return json.loads(text, strict=False), True
    except json.JSONDecodeError:
        pass
    output = []
    quoted = escaped = False
    i = 0
    while i < len(text):
        char = text[i]
        if not quoted and text[i:i + 2] in ('\\n', '\\r', '\\t'):
            output.append(' ')
            i += 2
            continue
        output.append(char)
        if quoted:
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        i += 1
    return json.loads(''.join(output), strict=False), True


def text_hash(text):
    return hashlib.sha256(re.sub(r'\s+', ' ', text).strip().encode()).hexdigest()


def audit(rows):
    counts = collections.Counter()
    fields = collections.Counter()
    hashes = collections.Counter()
    for row in rows:
        counts['rows'] += 1
        hashes[text_hash(row['full_note'])] += 1
        counts['note_differs_from_full_note'] += row['note'] != row['full_note']
        counts['note_shorter_than_full_note'] += len(row['note']) < len(row['full_note'])
        try:
            json.loads(row['summary'])
        except (json.JSONDecodeError, TypeError):
            counts['summary_strict_json_failures'] += 1
        try:
            summary, repaired = parse_summary(row['summary'])
        except (json.JSONDecodeError, TypeError):
            counts['summary_parse_failures'] += 1
            continue
        counts['summary_parse_successes'] += 1
        counts['summary_syntax_repairs'] += repaired
        treatments = summary.get('treatments') if isinstance(summary, dict) else None
        if not isinstance(treatments, list):
            counts['invalid_or_missing_treatment_list'] += 1
            continue
        for treatment in treatments:
            if not isinstance(treatment, dict):
                counts['invalid_treatment_entries'] += 1
                continue
            counts['treatment_entries_including_placeholders'] += 1
            for key, value in treatment.items():
                if value is not None and str(value).strip().lower() not in ('', 'none', 'null', 'n/a', 'unknown'):
                    fields[key] += 1
    return {'counts': dict(counts), 'populated_treatment_fields': dict(fields),
            'duplicate_full_note_rows': sum(n - 1 for n in hashes.values()),
            'limitations': ['Prefix sample, not a random sample, unless entire dataset requested.',
                           'Exact whitespace-normalized duplicates only; no source/patient grouping yet.',
                           'Field presence does not establish correctness or medication-only content.']}


def download(path, url, limit=None):
    if path.exists():
        return
    temporary = path.with_suffix(path.suffix + '.partial')
    try:
        with request(url) as source, temporary.open('wb') as target:
            if limit is None:
                while chunk := source.read(1024 * 1024):
                    target.write(chunk)
            else:
                for _ in range(limit):
                    line = source.readline()
                    if not line:
                        break
                    json.loads(line)  # Detect invalid rows before retaining the sample.
                    target.write(line)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int, default=200, help='Prefix sample size; 0 downloads all 30K rows.')
    args = parser.parse_args()
    if args.limit < 0:
        parser.error('--limit must be nonnegative')
    raw = ROOT / 'data/raw'
    raw.mkdir(parents=True, exist_ok=True)
    provenance = raw / 'revisions.json'
    if not provenance.exists():
        write_json(provenance, {
            'augmented_clinical_notes': get_json(f'{HF}/api/datasets/{REPO}')['sha'],
            'mts_dialog': get_json(f'https://api.github.com/repos/{MTS}/commits/main')['sha'],
            'sources': [f'{HF}/datasets/{REPO}', f'https://github.com/{MTS}'],
        })
    revisions = json.loads(provenance.read_text())
    hf_revision = revisions['augmented_clinical_notes']
    mts_revision = revisions['mts_dialog']
    for name in ('README.md', 'template_definitions.json'):
        download(raw / ('acn-' + name), f'{HF}/datasets/{REPO}/resolve/{hf_revision}/{name}')
    for name in ('README.md', 'LICENSE.txt'):
        download(raw / ('mts-' + name), f'https://raw.githubusercontent.com/{MTS}/{mts_revision}/{name}')
    acn = raw / f'acn-{args.limit or "all"}.jsonl'
    download(acn, f'{HF}/datasets/{REPO}/resolve/{hf_revision}/augmented_notes_30K.jsonl', args.limit or None)
    rows = [json.loads(line) for line in acn.read_text().splitlines()]
    report = {'revisions': revisions, 'requested_prefix_rows': args.limit or 'all', 'acn': audit(rows), 'mts': {}}
    for split, filename in FILES.items():
        path = raw / filename
        download(path, f'https://raw.githubusercontent.com/{MTS}/{mts_revision}/Main-Dataset/{filename}')
        records = list(csv.DictReader(io.StringIO(path.read_text(encoding='utf-8-sig'))))
        report['mts'][split] = {'rows': len(records), 'section_counts': dict(collections.Counter(r['section_header'].lower() for r in records))}
    report['cached_file_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in raw.iterdir() if p.is_file()}
    write_json(ROOT / 'artifacts/data_audit.json', report)
    # Blind annotation candidates: generated summaries are deliberately absent.
    target = ROOT / 'data/annotation/acn-candidates.jsonl'
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        with target.open('x') as output:
            seen = set()
            for row in rows:
                digest = text_hash(row['full_note'])
                if digest in seen:
                    continue
                if len(seen) >= 30:
                    break
                seen.add(digest)
                output.write(json.dumps({'id': 'acn:' + str(row['idx']), 'text': row['full_note'],
                                         'medications': [], 'annotation_status': 'unannotated'}, ensure_ascii=False) + '\n')
    print(json.dumps(report, indent=2))
    print('Annotation candidates preserved at', target.relative_to(ROOT))


if __name__ == '__main__':
    main()
