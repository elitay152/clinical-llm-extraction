"""Train and compare a frozen-encoder relation head against fixed linking rules."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import medication_relations as relations, name_spans


def metrics(counts):
    tp, fp, fn = counts
    return dict(tp=tp, fp=fp, fn=fn, precision=tp/(tp+fp) if tp+fp else None,
                recall=tp/(tp+fn) if tp+fn else None, f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None)


def score(records, payloads, predictions):
    counts = {f: [0, 0, 0] for f in (*relations.FIELDS, 'complete_records')}
    key = lambda s: (s['start'], s['end'])
    for record, payload, links in zip(records, payloads, predictions, strict=True):
        names, attrs = relations.entities(record['text'], payload)
        gold_names = name_spans.resolve_response(record, json.dumps(name_spans.reference_payload(record)))
        gold_records = {key(n): {f: [] for f in relations.FIELDS} for n in gold_names}
        actual_records = {key(n): {f: [] for f in relations.FIELDS} for n in names}
        for field in relations.FIELDS:
            expected = {(key(gold_names[r['medication_index']]), key(record[f'{field}_mentions'][r[f'{field}_index']])) for r in record[f'{field}_relations']}
            actual = {(key(names[r['medication_index']]), key(attrs[field][r[f'{field}_index']])) for r in links[field]}
            for n, a in expected: gold_records[n][field].append(a)
            for n, a in actual: actual_records[n][field].append(a)
            for i, count in enumerate((len(expected & actual), len(actual - expected), len(expected - actual))): counts[field][i] += count
        def complete(mapping):
            return {(n, *(tuple(sorted(v[f])) for f in relations.FIELDS)) for n, v in mapping.items()}
        expected, actual = complete(gold_records), complete(actual_records)
        for i, count in enumerate((len(expected & actual), len(actual - expected), len(expected - actual))): counts['complete_records'][i] += count
    return {f: metrics(c) for f, c in counts.items()}


def validate_config(config):
    if any(type(config[k]) is not int or config[k] < 1 for k in ('epochs', 'batch_size', 'semantic_width', 'layout_width')):
        raise ValueError('Training sizes must be positive integers')
    if not 0 <= config['dropout'] < 1 or not 0 < config['threshold'] < 1:
        raise ValueError('Invalid dropout or decision threshold')
    if config['learning_rate'] <= 0 or config['weight_decay'] < 0 or config['rule_prior_logit'] <= 0:
        raise ValueError('Invalid optimizer or rule prior')


def fit(tensors, config, epochs=None):
    import torch
    torch.manual_seed(config['seed'])
    semantic, layout, kinds, labels = tensors
    model = relations.make_model(semantic.shape[1], config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    for _ in range(epochs or config['epochs']):
        model.train()
        for indices in torch.randperm(len(labels)).split(config['batch_size']):
            optimizer.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(semantic[indices], layout[indices], kinds[indices]), labels[indices])
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--encoder', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('configs/relation-classifier.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import torch
    from safetensors.torch import load_file, save_file
    from scripts.predict_medication_names import MedicationNameExtractor
    args.output.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text())
    validate_config(config)
    identity = dict(version=relations.VERSION, layout=relations.LAYOUT, config=config,
                    feature_code_sha256=relations.fingerprint(Path(relations.__file__)),
                    dose_rule=relations.doses.LINKING_RULE, attribute_rule=relations.regimens.LINKING_RULE,
                    encoder_files={p.name: relations.fingerprint(p) for p in sorted(args.encoder.iterdir()) if p.suffix == '.json' or p.name == 'vocab.txt'},
                    encoder_sha256=relations.fingerprint(args.encoder/'model.safetensors'),
                    train_sha256=relations.fingerprint(args.data/'train.jsonl'),
                    development_sha256=relations.fingerprint(args.data/'development.jsonl'),
                    predictions_sha256=relations.fingerprint(args.predictions))
    plan = args.output/'plan.json'
    if plan.exists() and json.loads(plan.read_text()) != json.loads(json.dumps(identity)):
        raise ValueError('Experiment inputs changed; choose a new output directory')
    plan.write_text(json.dumps(identity, indent=2)+'\n')
    from scripts.pseudo_label import load_manifest
    from scripts.train_distillation import read_splits
    if not load_manifest(args.data).get('frozen_label_hashes'):
        raise ValueError('Freeze relation references before training')
    train, dev = read_splits(args.data)
    if {r['id'] for r in train} & {r['id'] for r in dev}:
        raise ValueError('Training/development IDs overlap')
    served = json.loads(args.predictions.read_text())
    if [p['id'] for p in served] != [r['id'] for r in dev]:
        raise ValueError('Frozen prediction IDs differ from development references')
    extractor = MedicationNameExtractor(args.encoder)
    cache = args.output/'features.safetensors'
    metadata_path = args.output/'features.json'
    views = [('train', train, [relations.reference_payload(r, True) for r in train]),
             ('reference', dev, [relations.reference_payload(r) for r in dev]), ('pipeline', dev, served)]
    if cache.exists():
        tensors = load_file(str(cache)); metadata = json.loads(metadata_path.read_text())
    else:
        tensors = {}; metadata = {}
        for view, records, payloads in views:
            metadata[view] = []
            for i, (record, payload) in enumerate(zip(records, payloads, strict=True)):
                rows, semantic, layout = relations.features(extractor, record['text'], payload)
                prefix = f'{view}.{i}'
                tensors[prefix+'.semantic'] = semantic.contiguous()
                tensors[prefix+'.layout'] = layout.contiguous()
                metadata[view].append(rows)
                print(f'Encoded {view} {i+1}/{len(records)}', flush=True)
        save_file(tensors, str(cache)); metadata_path.write_text(json.dumps(metadata))
    def training_tensors(indices):
        parts = [[], [], [], []]
        for i in indices:
            rows = metadata['train'][i]
            targets = relations.targets(train[i], views[0][2][i], rows)
            keep = [j for j, t in enumerate(targets) if t is not None]
            parts[0].append(tensors[f'train.{i}.semantic'][keep])
            parts[1].append(tensors[f'train.{i}.layout'][keep])
            parts[2].append(torch.tensor([rows[j]['kind'] for j in keep], dtype=torch.long))
            parts[3].append(torch.tensor([targets[j] for j in keep], dtype=torch.float))
        return tuple(torch.cat(p) for p in parts)
    selected = []; covered = set()
    for i, record in enumerate(train):
        kinds = {f for f in relations.FIELDS if record[f'{f}_relations']}
        if kinds - covered:
            selected.append(i); covered.update(kinds)
    selected.extend(i for i in range(len(train)) if i not in selected)
    tiny = training_tensors(selected[:6])
    overfit_config = dict(config, dropout=0.0, learning_rate=0.005)
    tiny_model = fit(tiny, overfit_config, epochs=300)
    with torch.no_grad():
        predicted = tiny_model(*tiny[:3]).sigmoid() >= config['threshold']
    overfit = {}
    for k, field in enumerate(relations.FIELDS):
        mask = tiny[2] == k; gold = tiny[3].bool()
        overfit[field] = metrics([int((mask & predicted & gold).sum()), int((mask & predicted & ~gold).sum()), int((mask & ~predicted & gold).sum())])
    (args.output/'overfit.json').write_text(json.dumps(overfit, indent=2)+'\n')
    if any(v['f1'] is None or v['f1'] < .98 for v in overfit.values()):
        raise ValueError('Tiny training-set memorization check failed')
    print('Overfit check passed', flush=True)
    all_training = training_tensors(range(len(train)))
    initial = relations.make_model(all_training[0].shape[1], config).eval()
    with torch.no_grad():
        if not torch.equal(initial(*all_training[:3]) >= 0, all_training[1][:, 0].bool()):
            raise ValueError('Untrained head differs from rules')
    model = fit(all_training, config)
    save_file(model.state_dict(), str(args.output/'model.safetensors'))
    report = dict(identity, overfit=overfit, overfit_ids=[train[i]['id'] for i in selected[:6]],
                  training_pairs=len(all_training[3]), training_positives=int(all_training[3].sum()),
                  reference_type='candidate-limited teacher agreement', comparisons={})
    for view, records, payloads in views[1:]:
        predicted = []; elapsed = 0
        for i, rows in enumerate(metadata[view]):
            started = time.monotonic()
            with torch.no_grad():
                probabilities = model(tensors[f'{view}.{i}.semantic'], tensors[f'{view}.{i}.layout'], torch.tensor([r['kind'] for r in rows], dtype=torch.long)).sigmoid().tolist()
            predicted.append(relations.decode(rows, probabilities, config['threshold']))
            elapsed += time.monotonic()-started
        rules = [relations.rule_links(r['text'], p) for r, p in zip(records, payloads, strict=True)]
        report['comparisons'][view] = dict(rules=score(records, payloads, rules), learned=score(records, payloads, predicted), head_seconds=elapsed)
        (args.output/f'{view}-relations.json').write_text(json.dumps(predicted, indent=2)+'\n')
    (args.output/'run.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['comparisons'], indent=2), flush=True)


if __name__ == '__main__':
    main()
