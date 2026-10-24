"""Label reserved training notes and audit medication relation errors."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import label_medication_doses as attributes, label_medication_names as names
from scripts import medication_doses, medication_regimens, medication_relations
from scripts.pseudo_label import hosted_response, load_manifest, write_json
from scripts.train_distillation import read_splits


def signals(text):
    return {'parenthetical': bool(re.search(r'\([^)]{1,100}\)', text)),
            'shared': bool(re.search(r'\b(?:both|each|and|or)\b', text, re.I)),
            'attributes': len(medication_doses.dose_candidates(text)) + len(medication_regimens._ROUTE.findall(text)) + len(medication_regimens._FREQUENCY.findall(text)),
            'distractor': bool(re.search(r'\b(?:stage IV|oral cavity|follow.up|symptoms? daily|daily activities)\b', text, re.I))}


def select_targeted(candidates, count):
    if type(count) is not int or count < 1 or len(candidates) < count:
        raise ValueError('Insufficient candidates or invalid count')
    def priority(record):
        value = signals(record['text'])
        return (-(3 * value['parenthetical'] + 2 * value['shared'] +
                  3 * value['distractor'] + min(8, value['attributes'])), record['source_group'])
    ranked = sorted(candidates, key=priority)
    return ranked[:count]


def prepare(source, output, count):
    manifest = load_manifest(source)
    train, dev = read_splits(source)
    if not manifest.get('test_locked') or not manifest.get('frozen_label_hashes'):
        raise ValueError('Frozen labels and locked test membership required')
    trained = {r['id'] for r in train}
    candidates = [r for r in manifest['candidates'] if r['split'] == 'train' and r['id'] not in trained]
    selected = select_targeted(candidates, count)
    output.mkdir(parents=True, exist_ok=False)
    plan = dict(source=str(source.resolve()), source_hashes={n: attributes.digest(source/n) for n in
                ('manifest.json', 'train.jsonl', 'development.jsonl', 'test.jsonl')},
                selection_version='reserved-training-layout-priority-v1',
                candidates=selected, teacher=dict(model='gpt-4.1-2025-04-14', base_url='https://api.openai.com/v1',
                max_tokens=3500, max_attempts=3), prompt_hashes={p.name: attributes.digest(p) for p in
                (names.PROMPT, attributes.PROMPT, ROOT/'configs/regimen-teacher-prompt.txt')})
    write_json(output/'selection.json', plan)
    (output/'selection.sha256').write_text(attributes.digest(output/'selection.json')+'\n')
    return dict(base_notes=len(train), new_notes=len(selected), development_notes=len(dev),
                signals=dict(Counter(k for r in selected for k, v in signals(r['text']).items() if v)))


def checked(directory):
    path = directory/'selection.json'
    if attributes.digest(path) != (directory/'selection.sha256').read_text().strip():
        raise ValueError('Selection plan changed')
    plan = json.loads(path.read_text()); source = Path(plan['source'])
    if any(attributes.digest(source/n) != h for n, h in plan['source_hashes'].items()):
        raise ValueError('Source snapshot changed')
    if any(attributes.digest(ROOT/'configs'/n) != h for n, h in plan['prompt_hashes'].items()):
        raise ValueError('Annotation prompt changed')
    manifest = load_manifest(source)
    selected = plan['candidates']; reserved = {r['id']: r for r in manifest['candidates']}
    train, _ = read_splits(source)
    trained = {r['id'] for r in train}
    if (len({r['id'] for r in selected}) != len(selected) or
            len({r['source_group'] for r in selected}) != len(selected) or
            any(r['id'] in trained or r != reserved.get(r['id']) or r['split'] != 'train' or
                hashlib.sha256(r['text'].encode()).hexdigest() != r['input_sha256'] for r in selected)):
        raise ValueError('Invalid training-only selection')
    return plan


def annotate(directory, workers=2):
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError('Use 1–4 workers')
    plan = checked(directory); path = directory/'annotations.jsonl'
    existing = [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
    ids = {r['id'] for r in existing}
    if len(ids) != len(existing) or not ids <= {r['id'] for r in plan['candidates']}:
        raise ValueError('Unexpected or duplicate annotations')
    def request(candidate):
        record = candidate; history = []
        for task in ('names', 'dose', 'regimen'):
            if task == 'names':
                prompt = names.PROMPT.read_text(); content = candidate['text']
                validate = lambda response: names.validate_response(candidate, response)
            else:
                module, _, _, prompt_path, validator = attributes.annotation_spec(task)
                prompt = prompt_path.read_text()
                marked, meds, options = module.marked_source(record)
                payload = dict(marked_note=marked, medication_indices=list(range(len(meds))))
                if task == 'dose': payload['dose_indices'] = list(range(len(options)))
                else: payload.update({f'{f}_indices': list(range(len(spans))) for f, spans in options.items()})
                content = json.dumps(payload)
                validate = lambda response: validator(record, response)
            for attempt in range(plan['teacher']['max_attempts']):
                response, metadata = hosted_response({'text': content}, plan['teacher']['model'],
                    plan['teacher']['base_url'], plan['teacher']['max_tokens'], system_prompt=prompt)
                try:
                    revised = validate(response)
                except (ValueError, TypeError, KeyError) as error:
                    reason = str(error)
                else:
                    reason = None
                history.append(dict(task=task, response=response, metadata=metadata, validation_error=reason))
                if reason is None:
                    record = revised; break
                content = json.dumps(dict(source=content, previous_output=response, validation_error=reason,
                                          request='Return corrected complete JSON under the same protocol.'))
            else:
                return dict(id=candidate['id'], input_sha256=candidate['input_sha256'], record=None, attempts=history)
        return dict(id=candidate['id'], input_sha256=candidate['input_sha256'], record=record, attempts=history)
    pending = [r for r in plan['candidates'] if r['id'] not in ids]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(request, r): r['id'] for r in pending}
        for future in as_completed(futures):
            result = future.result()
            with path.open('a') as stream: stream.write(json.dumps(result, ensure_ascii=False)+'\n')
            print(json.dumps(dict(id=result['id'], accepted=result['record'] is not None)), flush=True)


def collect(directory, output):
    plan = checked(directory); source = Path(plan['source'])
    rows = [json.loads(s) for s in (directory/'annotations.jsonl').read_text().splitlines()]
    candidates = {r['id']: r for r in plan['candidates']}
    if len(rows) != len(candidates) or {r['id'] for r in rows} != set(candidates):
        raise ValueError('Collect requires every planned annotation exactly once')
    additions = []; rejected = []
    for row in rows:
        candidate = candidates[row['id']]; record = row['record']
        if row['input_sha256'] != candidate['input_sha256']:
            raise ValueError('Annotation input hash changed')
        if record is None:
            rejected.append(dict(id=row['id'], reason=row['attempts'][-1]['validation_error']))
            continue
        replay = candidate
        for task in ('names', 'dose', 'regimen'):
            attempts = [a for a in row['attempts'] if a['task'] == task]
            if not attempts or attempts[-1]['validation_error'] is not None:
                raise ValueError('Accepted annotation is missing a valid stage')
            if task == 'names': replay = names.validate_response(candidate, attempts[-1]['response'])
            else: replay = attributes.annotation_spec(task)[-1](replay, attempts[-1]['response'])
        if replay != record:
            raise ValueError('Stored record differs from validated teacher responses')
        if any(record[k] != candidate[k] for k in ('id', 'text', 'split', 'source_group', 'input_sha256')):
            raise ValueError('Annotation failed or changed source membership')
        medication_regimens.reference_spans(record)
        payload = medication_relations.reference_payload(record, True)
        medication_relations.targets(record, payload, medication_relations.pairs(record['text'], payload))
        additions.append(record)
    if not additions:
        raise ValueError('No valid training additions')
    additions.sort(key=lambda r: list(candidates).index(r['id']))
    output.mkdir(parents=True, exist_ok=False)
    for split in ('development', 'test'): (output/f'{split}.jsonl').write_bytes((source/f'{split}.jsonl').read_bytes())
    (output/'train.jsonl').write_bytes((source/'train.jsonl').read_bytes() + ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in additions).encode())
    manifest = dict(load_manifest(source))
    manifest.update(relation_expansion=dict(selection_sha256=attributes.digest(directory/'selection.json'), annotation_sha256=attributes.digest(directory/'annotations.jsonl')),
                    frozen_label_hashes={f'{s}.jsonl': attributes.digest(output/f'{s}.jsonl') for s in ('train','development','test')})
    write_json(output/'manifest.json', manifest)
    (output/'manifest.sha256').write_text(attributes.digest(output/'manifest.json')+'\n')
    train, _ = read_splits(output)
    report = dict(training_notes=len(train), additions=len(additions), rejected=rejected,
                  development_unchanged=True, test_locked=True,
                  positive_links={f: sum(len(r[f'{f}_relations']) for r in additions) for f in medication_relations.FIELDS})
    write_json(output/'expansion-report.json', report)
    return report


def audit_training(source, run, output):
    import torch
    from safetensors.torch import load_file
    train, _ = read_splits(source)
    report = json.loads((run/'run.json').read_text())
    if report['train_sha256'] != attributes.digest(source/'train.jsonl'):
        raise ValueError('Baseline training labels differ')
    metadata = json.loads((run/'features.json').read_text())['train']
    tensors = load_file(str(run/'features.safetensors'))
    torch.set_num_threads(4)
    model = medication_relations.make_model(tensors['train.0.semantic'].shape[1], report['config'])
    model.load_state_dict(load_file(str(run/'model.safetensors'))); model.eval()
    counts = Counter(); errors = []
    for i, record in enumerate(train):
        payload = medication_relations.reference_payload(record, True)
        rows = medication_relations.pairs(record['text'], payload)
        if rows != metadata[i]: raise ValueError('Cached training pair features differ')
        target = medication_relations.targets(record, payload, rows)
        with torch.no_grad():
            probabilities = model(tensors[f'train.{i}.semantic'], tensors[f'train.{i}.layout'],
                torch.tensor([r['kind'] for r in rows], dtype=torch.long)).sigmoid().tolist()
        meds, attrs = medication_relations.entities(record['text'], payload)
        for row, gold, probability in zip(rows, target, probabilities, strict=True):
            if gold is None or (probability >= report['config']['threshold']) == bool(gold): continue
            field = medication_relations.FIELDS[row['kind']]
            attr = attrs[field][row['attribute_index']]; med = meds[row['medication_index']]
            shared = sum(r[f'{field}_index'] == next((j for j, s in enumerate(record[f'{field}_mentions']) if s == attr), -1) for r in record[f'{field}_relations']) > 1
            left, right = sorted((med, attr), key=lambda span: span['start'])
            gap = record['text'][left['end']:right['start']]
            category = ('unrelated_candidate' if attr not in record[f'{field}_mentions'] else
                        'shared_instruction' if shared else 'distant' if len(gap) > 120 else
                        'parenthetical' if '(' in gap or ')' in gap else 'local_assignment')
            mistake = 'missed' if gold else 'false_link'
            counts[f'{field}:{category}:{mistake}'] += 1
            errors.append(dict(id=record['id'], field=field, category=category, mistake=mistake,
                               medication_index=row['medication_index'], attribute_index=row['attribute_index']))
    write_json(output, dict(training_sha256=report['train_sha256'], counts=dict(counts), errors=errors))
    return dict(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare'); p.add_argument('source', type=Path); p.add_argument('output', type=Path)
    p.add_argument('--count', type=int, default=19)
    p = sub.add_parser('audit'); p.add_argument('source', type=Path); p.add_argument('run', type=Path); p.add_argument('output', type=Path)
    p = sub.add_parser('label'); p.add_argument('directory', type=Path); p.add_argument('--workers', type=int, default=2)
    p = sub.add_parser('collect'); p.add_argument('directory', type=Path); p.add_argument('output', type=Path)
    a = parser.parse_args()
    if a.command == 'prepare': result = prepare(a.source, a.output, a.count)
    elif a.command == 'audit': result = audit_training(a.source, a.run, a.output)
    elif a.command == 'label': result = annotate(a.directory, a.workers)
    else: result = collect(a.directory, a.output)
    print(json.dumps(result))


if __name__ == '__main__': main()
