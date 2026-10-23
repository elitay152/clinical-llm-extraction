"""Create sealed training views and deterministic subsets for classifier experiments."""
import hashlib
import json
from pathlib import Path
import random

from scripts import name_spans
from scripts.pseudo_label import load_manifest, write_json
from scripts.train_distillation import read_splits


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def nested_membership(rows, seed):
    ids = [r['id'] for r in rows]
    if len(set(ids)) != len(ids) or any(r['split'] != 'train' for r in rows):
        raise ValueError('Learning curves require unique training rows')
    random.Random(seed).shuffle(ids)
    return {fraction: ids[:max(2, len(ids)*fraction//100)] for fraction in [25,50,100]}


def make_view(source, output, training_ids, development=True):
    source, output = Path(source), Path(output)
    manifest = load_manifest(source)
    train, references = read_splits(source)
    selected = set(training_ids)
    if len(selected) != len(training_ids) or not selected <= {r['id'] for r in train}:
        raise ValueError('Training view membership must be a unique subset of training IDs')
    output.mkdir(parents=True, exist_ok=False)
    if selected == {r['id'] for r in train}:
        (output/'train.jsonl').write_bytes((source/'train.jsonl').read_bytes())
    else:
        (output/'train.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in train if r['id'] in selected))
    (output/'development.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in references) if development else '')
    (output/'test.jsonl').write_text('')
    manifest.update(experiment_view={'source_manifest_sha256':digest(source/'manifest.json'),
                                    'training_ids':training_ids, 'training_diagnostic_only':not development},
                    frozen_label_hashes={n:digest(output/n) for n in ['train.jsonl','development.jsonl','test.jsonl']})
    write_json(output/'manifest.json',manifest)
    (output/'manifest.sha256').write_text(digest(output/'manifest.json')+'\n')
    read_splits(output)
    return output


def span_errors(references, predictions):
    by_id = {p['id']:p for p in predictions}
    if len(by_id) != len(predictions) or set(by_id) != {r['id'] for r in references}:
        raise ValueError('Error analysis membership differs from references')
    errors = []
    for record in references:
        expected = name_spans.resolve_response(record, json.dumps(name_spans.reference_payload(record)))
        actual = name_spans.resolve_response(record, by_id[record['id']]['response'])
        for kind, source, other in [('extra',actual,expected),('missed',expected,actual)]:
            positions = {(s['start'],s['end']) for s in other}
            for span in source:
                if (span['start'],span['end']) not in positions:
                    overlap = any(s['start'] < span['end'] and s['end'] > span['start'] for s in other)
                    errors.append({'id':record['id'], 'kind':'boundary_'+kind if overlap else kind, 'span':span})
    return errors
