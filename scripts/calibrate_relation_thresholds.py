"""Select relation abstention thresholds with source-grouped training folds."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import medication_relations as relations
from scripts.train_relation_classifier import fit, metrics, score
from scripts.train_distillation import read_splits

VERSION = 'grouped-relation-thresholds-v1'
THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99, None)


def group_folds(records, count=5, seed=42):
    groups = {r['source_group'] for r in records}
    if type(count) is not int or not 2 <= count <= len(groups):
        raise ValueError('Fold count must be between two and the source-group count')
    ordered = sorted(groups, key=lambda g: hashlib.sha256(f'{seed}:{g}'.encode()).hexdigest())
    assignments = {g: i % count for i, g in enumerate(ordered)}
    return [assignments[r['source_group']] for r in records]


def accepted(probability, threshold):
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError('Invalid relation probability')
    if threshold is not None and (not math.isfinite(threshold) or not 0 <= threshold <= 1):
        raise ValueError('Invalid decision threshold')
    return threshold is not None and probability >= threshold


def pair_metrics(labels, predictions):
    if len(labels) != len(predictions) or any(y not in (0, 1) for y in labels):
        raise ValueError('Expected matching binary labels and decisions')
    return metrics([sum(bool(p) and y == 1 for y, p in zip(labels, predictions)),
                    sum(bool(p) and y == 0 for y, p in zip(labels, predictions)),
                    sum(not p and y == 1 for y, p in zip(labels, predictions))])


def select_threshold(labels, probabilities, rule_decisions):
    if not labels or len(probabilities) != len(labels):
        raise ValueError('Threshold selection requires labeled out-of-fold predictions')
    baseline = pair_metrics(labels, rule_decisions)
    candidates = []
    for threshold in THRESHOLDS:
        result = pair_metrics(labels, [accepted(p, threshold) for p in probabilities])
        candidates.append(dict(threshold=threshold, metrics=result))
    eligible = [c for c in candidates if c['metrics']['fp'] <= baseline['fp'] and
                (baseline['precision'] is None or c['metrics']['precision'] is None or
                 c['metrics']['precision'] >= baseline['precision'])]
    selected = max(eligible, key=lambda c: (c['metrics']['f1'] or 0, -c['metrics']['fp'],
                                           1.1 if c['threshold'] is None else c['threshold']))
    return dict(selected=selected, rules=baseline, candidates=candidates)


def decode(rows, probabilities, thresholds):
    if set(thresholds) != set(relations.FIELDS) or len(rows) != len(probabilities):
        raise ValueError('Expected three thresholds and one score per candidate pair')
    links = {f: [] for f in relations.FIELDS}
    for row, probability in zip(rows, probabilities, strict=True):
        field = relations.FIELDS[row['kind']]
        if accepted(probability, thresholds[field]):
            links[field].append(dict(medication_index=row['medication_index'], **{f'{field}_index': row['attribute_index']}))
    return links


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import torch
    from safetensors.torch import load_file, save_file
    torch.set_num_threads(4)
    train, dev = read_splits(args.data)
    run = json.loads((args.run/'run.json').read_text())
    expected = {'train_sha256': args.data/'train.jsonl', 'development_sha256': args.data/'development.jsonl',
                'predictions_sha256': args.predictions, 'feature_code_sha256': Path(relations.__file__)}
    if any(run[k] != relations.fingerprint(p) for k, p in expected.items()):
        raise ValueError('Frozen experiment inputs differ from cached run')
    metadata = json.loads((args.run/'features.json').read_text())
    tensors = load_file(str(args.run/'features.safetensors'))
    fold_ids = group_folds(train)
    args.output.mkdir(parents=True, exist_ok=False)
    plan = dict(version=VERSION, folds=5, seed=42, thresholds=THRESHOLDS,
                selection='Maximize out-of-fold pair F1 per field with FP no greater and precision no lower than rules; ties prefer fewer FP then higher threshold.',
                cv_scope='Relation-head calibration conditional on the existing frozen encoder, which saw name/dose labels from some held-out training groups.',
                inputs={p.name: relations.fingerprint(p) for p in (args.run/'run.json', args.run/'features.json', args.run/'features.safetensors', args.run/'model.safetensors', args.predictions)},
                calibration_code_sha256=relations.fingerprint(Path(__file__)),
                training_code_sha256=relations.fingerprint(ROOT/'scripts/train_relation_classifier.py'),
                fold_assignments=[dict(id=r['id'], source_group=r['source_group'], fold=f) for r, f in zip(train, fold_ids)],
                config=run['config'])
    (args.output/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    def training_parts(indices):
        parts = [[], [], [], []]
        for i in indices:
            record = train[i]; rows = metadata['train'][i]
            payload = relations.reference_payload(record, True)
            if rows != relations.pairs(record['text'], payload):
                raise ValueError('Cached training pairs differ')
            targets = relations.targets(record, payload, rows)
            keep = [j for j, t in enumerate(targets) if t is not None]
            parts[0].append(tensors[f'train.{i}.semantic'][keep]); parts[1].append(tensors[f'train.{i}.layout'][keep])
            parts[2].append(torch.tensor([rows[j]['kind'] for j in keep], dtype=torch.long))
            parts[3].append(torch.tensor([targets[j] for j in keep], dtype=torch.float))
        return tuple(torch.cat(p) for p in parts)
    oof = []
    for fold in range(5):
        model = fit(training_parts([i for i, f in enumerate(fold_ids) if f != fold]), run['config'])
        save_file(model.state_dict(), str(args.output/f'fold-{fold}.safetensors'))
        for i, f in enumerate(fold_ids):
            if f != fold: continue
            semantic, layout, kinds, labels = training_parts([i])
            with torch.no_grad(): probabilities = model(semantic, layout, kinds).sigmoid().tolist()
            oof.extend(dict(id=train[i]['id'], fold=fold, kind=k, label=int(y), probability=p, rule=bool(rule))
                       for k, y, p, rule in zip(kinds.tolist(), labels.tolist(), probabilities, layout[:, 0].tolist(), strict=True))
        print(f'Completed training fold {fold+1}/5', flush=True)
    selections = {}
    for kind, field in enumerate(relations.FIELDS):
        rows = [r for r in oof if r['kind'] == kind]
        selections[field] = select_threshold([r['label'] for r in rows], [r['probability'] for r in rows], [r['rule'] for r in rows])
    thresholds = {f: s['selected']['threshold'] for f, s in selections.items()}
    frozen = dict(thresholds=thresholds, selection=selections, plan_sha256=relations.fingerprint(args.output/'plan.json'))
    (args.output/'thresholds.json').write_text(json.dumps(frozen, indent=2)+'\n')
    (args.output/'out-of-fold.json').write_text(json.dumps(oof, indent=2)+'\n')
    # Development scores are read only after the training-derived thresholds are saved.
    model = relations.make_model(tensors['train.0.semantic'].shape[1], run['config'])
    model.load_state_dict(load_file(str(args.run/'model.safetensors'))); model.eval()
    payloads = json.loads(args.predictions.read_text())
    if [p['id'] for p in payloads] != [r['id'] for r in dev]:
        raise ValueError('Development prediction membership differs')
    result = dict(thresholds=thresholds, comparisons={})
    for view, sources in [('reference', [relations.reference_payload(r) for r in dev]), ('pipeline', payloads)]:
        calibrated = []; standard = []
        for i, (record, payload) in enumerate(zip(dev, sources, strict=True)):
            rows = metadata[view][i]
            if rows != relations.pairs(record['text'], payload): raise ValueError('Cached development pairs differ')
            with torch.no_grad(): probabilities = model(tensors[f'{view}.{i}.semantic'], tensors[f'{view}.{i}.layout'],
                torch.tensor([r['kind'] for r in rows], dtype=torch.long)).sigmoid().tolist()
            calibrated.append(decode(rows, probabilities, thresholds))
            standard.append(relations.decode(rows, probabilities, run['config']['threshold']))
        rules = [relations.rule_links(r['text'], p) for r, p in zip(dev, sources, strict=True)]
        result['comparisons'][view] = dict(rules=score(dev, sources, rules), uncalibrated=score(dev, sources, standard), calibrated=score(dev, sources, calibrated))
        (args.output/f'{view}-relations.json').write_text(json.dumps(calibrated, indent=2)+'\n')
    (args.output/'run.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
