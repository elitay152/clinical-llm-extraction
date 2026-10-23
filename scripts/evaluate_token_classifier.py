"""Evaluate a saved medication-name BIO classifier on development references."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import name_spans, token_classifier
from scripts.train_distillation import read_splits
from scripts.predict_medication_names import MedicationNameExtractor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--negative-controls', type=Path)
    parser.add_argument('--reference-directory', type=Path)
    parser.add_argument('--training-diagnostic', action='store_true')
    args = parser.parse_args()
    if args.training_diagnostic and args.reference_directory:
        raise ValueError('Training diagnostics cannot use alternate references')
    if args.output.exists():
        raise ValueError('Output exists; choose a new directory')
    report = json.loads((args.run/'run.json').read_text())
    if report['alignment_version'] != token_classifier.VERSION or report['source_token_version'] != name_spans.VERSION:
        raise ValueError('Classifier source alignment version differs from training')
    config = report['config']
    silver = Path(report['silver_directory'])
    training, references = read_splits(silver)
    if args.reference_directory:
        _, alternate = read_splits(args.reference_directory)
        if {r['id']:r['input_sha256'] for r in alternate} != {r['id']:r['input_sha256'] for r in references}:
            raise ValueError('Alternate reference source membership differs')
        references = alternate
    if args.training_diagnostic:
        references = training
    if not references:
        raise ValueError('No references for requested split')
    if hashlib.sha256((silver/'train.jsonl').read_bytes()).hexdigest() != report['train_file_sha256']:
        raise ValueError('Training source labels changed since training')
    if not args.training_diagnostic and set(report['source_groups']) & {r['source_group'] for r in references}:
        raise ValueError('Development groups overlap trained examples')
    controls = json.loads(args.negative_controls.read_text()) if args.negative_controls else []
    if controls and (len({r['id'] for r in controls}) != len(controls)
                     or any(r['label_origin'] != 'synthetic' or r['medications']
                            or hashlib.sha256(r['text'].encode()).hexdigest() != r['input_sha256'] for r in controls)
                     or {r['input_sha256'] for r in training} & {r['input_sha256'] for r in controls}
                     or set(report['training_ids']) & {r['id'] for r in controls}
                     or {r['input_sha256'] for r in name_spans.synthetic_negatives()} & {r['input_sha256'] for r in controls}):
        raise ValueError('Negative controls must be separate synthetic negatives')
    extractor = MedicationNameExtractor(args.run)
    def predict(records):
        predictions = []
        for record in records:
            payload, elapsed = extractor.predict_payload(record['text'])
            predictions.append({'id': record['id'], 'response': json.dumps(payload),
                                'elapsed_seconds': elapsed})
        return predictions
    predictions = predict(references)
    reference_root = args.reference_directory or silver
    label_file = 'train.jsonl' if args.training_diagnostic else ('development.jsonl' if (reference_root/'development.jsonl').exists() else 'validation.jsonl')
    result = {'training_diagnostic' if args.training_diagnostic else 'development': name_spans.score(references, predictions), 'reference_origin': 'teacher_generated',
              'reference_split': 'train' if args.training_diagnostic else label_file.removesuffix('.jsonl'),
              'reference_labels_sha256': hashlib.sha256((reference_root/label_file).read_bytes()).hexdigest(),
              'diagnostic_only': args.training_diagnostic,
              'reference_ids': [r['id'] for r in references], 'model': config['model'], 'revision': config['revision'],
              'alignment_version': token_classifier.VERSION, 'device': 'cpu', 'threads': report['threads'],
              'compute_dtype': config['compute_dtype'],
              'limitations': [('Training-set diagnostic; no generalization claim.' if args.training_diagnostic else 'Small teacher-labeled development set; not clinical accuracy.'),
                              'Labels cover teacher-attested names, not verified exhaustive mentions.',
                              'Source grouping does not establish patient independence or exclude pretraining overlap.',
                              'Names only; medication attributes and status are not predicted.']}
    control_predictions = predict(controls) if controls else []
    if controls:
        result['synthetic_negative_controls'] = name_spans.score(controls, control_predictions)
        result['negative_controls_sha256'] = hashlib.sha256(args.negative_controls.read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'predictions.json').write_text(json.dumps(predictions,indent=2)+'\n')
    if controls:
        (args.output/'synthetic-control-predictions.json').write_text(json.dumps(control_predictions,indent=2)+'\n')
    (args.output/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
