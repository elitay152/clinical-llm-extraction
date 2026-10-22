"""Freeze train/development/test source groups before generating teacher labels."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_data import text_hash
from scripts.prepare_annotation_pilot import canonical_text
from scripts.pseudo_label import SYSTEM, PROMPT_VERSION, select_excerpt, write_json


def shingles(text):
    words = re.findall(r'\w+', text.casefold())
    return {tuple(words[i:i+5]) for i in range(max(1, len(words)-4))}


def group_notes(texts, threshold=0.8):
    parent = list(range(len(texts)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    sets = [shingles(t) for t in texts]
    inverted = collections.defaultdict(list)
    for i, current in enumerate(sets):
        overlaps = collections.Counter(j for token in current for j in inverted[token])
        for j, common in overlaps.items():
            if common / (len(current) + len(sets[j]) - common) >= threshold:
                parent[find(i)] = find(j)
        for token in current:
            inverted[token].append(i)
    members = collections.defaultdict(list)
    for i, text in enumerate(texts):
        members[find(i)].append(text_hash(text))
    identifiers = {root: hashlib.sha256(''.join(sorted(values)).encode()).hexdigest() for root, values in members.items()}
    return [identifiers[find(i)] for i in range(len(texts))]


def build_manifest(rows, excluded_texts, train_count, dev_count, test_count, max_chars, seed):
    if min(train_count, dev_count, test_count) < 1 or max_chars < 200:
        raise ValueError('Positive split counts and max_chars >= 200 required')
    unique = {}
    for row in rows:
        text = canonical_text(row['full_note'])
        unique.setdefault(text_hash(text), (row, text))
    pairs = list(unique.values())
    excluded = {text_hash(canonical_text(text)) for text in excluded_texts}
    # Include exclusion sources in clustering, including sources outside this sample.
    all_texts = [text for _, text in pairs] + [canonical_text(t) for t in excluded_texts]
    groups = group_notes(all_texts)
    excluded_groups = set(groups[len(pairs):])
    chosen_by_group = {}
    for (row, text), group in zip(pairs, groups):
        if group in excluded_groups or text_hash(text) in excluded:
            continue
        excerpt = select_excerpt(text, max_chars, 'dosage')
        if excerpt is None:
            continue
        selected, start = excerpt
        chosen_by_group.setdefault(group, {'id': 'acn:' + str(row['idx']), 'text': selected,
            'source_group': group, 'source_text_hash': text_hash(text),
            'input_sha256': hashlib.sha256(selected.encode()).hexdigest(),
            'scope': 'dosage_excerpt', 'source_start': start, 'full_text_chars': len(text)})
    ordered = sorted(chosen_by_group.values(), key=lambda row: hashlib.sha256((str(seed)+row['source_group']).encode()).hexdigest())
    total = train_count + dev_count + test_count
    if len(ordered) < total:
        raise ValueError(f'Need {total} eligible groups; found {len(ordered)}')
    selected = ordered[:total]
    for i, row in enumerate(selected):
        row['split'] = 'train' if i < train_count else 'development' if i < train_count + dev_count else 'test'
    return {'format': 'silver-medications-v1', 'protocol': 'frozen-three-way-v1', 'seed': seed,
            'prompt_version': PROMPT_VERSION, 'system_prompt': SYSTEM, 'candidates': selected,
            'test_locked': True, 'grouping': {'method': 'connected_components_word_5gram_jaccard', 'threshold': 0.8},
            'selection': 'one dosage excerpt per nonexcluded source-text cluster',
            'limitations': ['Text similarity groups do not establish article- or patient-disjointness.',
                           'Dosage-enriched excerpts are not representative of all notes.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--source', type=Path, default=ROOT/'data/raw/acn-1000.jsonl')
    parser.add_argument('--train', type=int, default=60)
    parser.add_argument('--development', type=int, default=20)
    parser.add_argument('--test', type=int, default=20)
    parser.add_argument('--max-chars', type=int, default=1800)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Frozen experiment directory already exists')
    rows = [json.loads(line) for line in args.source.read_text().splitlines()]
    excluded = [json.loads(line)['full_note'] for line in (ROOT/'data/raw/acn-200.jsonl').read_text().splitlines()]
    manifest = build_manifest(rows, excluded, args.train, args.development, args.test, args.max_chars, args.seed)
    manifest['source_sha256'] = hashlib.sha256(args.source.read_bytes()).hexdigest()
    manifest['source_revisions'] = json.loads((ROOT/'data/raw/revisions.json').read_text())
    manifest['student'] = json.loads((ROOT/'configs/student.json').read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output/'manifest.json', manifest)
    (args.output/'manifest.sha256').write_text(hashlib.sha256((args.output/'manifest.json').read_bytes()).hexdigest()+'\n')
    print(json.dumps({'splits': dict(collections.Counter(row['split'] for row in manifest['candidates'])), 'test_locked': True}))


if __name__ == '__main__':
    main()
