"""Evaluate a saved medication-name BIO classifier on development references."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import name_spans, token_classifier
from scripts.train_distillation import read_splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--negative-controls', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Output exists; choose a new directory')
    report = json.loads((args.run/'run.json').read_text())
    if report['alignment_version'] != token_classifier.VERSION or report['source_token_version'] != name_spans.VERSION:
        raise ValueError('Classifier source alignment version differs from training')
    config = report['config']
    silver = Path(report['silver_directory'])
    training, references = read_splits(silver)
    if not references:
        raise ValueError('No development references')
    if hashlib.sha256((silver/'train.jsonl').read_bytes()).hexdigest() != report['train_file_sha256']:
        raise ValueError('Training source labels changed since training')
    if set(report['source_groups']) & {r['source_group'] for r in references}:
        raise ValueError('Development groups overlap trained examples')
    controls = json.loads(args.negative_controls.read_text()) if args.negative_controls else []
    if controls and (len({r['id'] for r in controls}) != len(controls)
                     or any(r['label_origin'] != 'synthetic' or r['medications']
                            or hashlib.sha256(r['text'].encode()).hexdigest() != r['input_sha256'] for r in controls)
                     or {r['input_sha256'] for r in training} & {r['input_sha256'] for r in controls}
                     or set(report['training_ids']) & {r['id'] for r in controls}
                     or {r['input_sha256'] for r in name_spans.synthetic_negatives()} & {r['input_sha256'] for r in controls}):
        raise ValueError('Negative controls must be separate synthetic negatives')
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    torch.set_num_threads(report['threads'])
    tokenizer = AutoTokenizer.from_pretrained(args.run, local_files_only=True, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(args.run, local_files_only=True)
    if model.config.id2label != dict(enumerate(token_classifier.LABELS)):
        raise ValueError('Saved model labels differ from BIO schema')
    model.eval()
    def predict(records):
        predictions = []
        for record in records:
            encoded, words = token_classifier.encode_record(record, tokenizer, config['max_length'], labeled=False)
            started = time.monotonic()
            with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16, enabled=config['compute_dtype']=='bfloat16'):
                logits = model(**{k:torch.tensor([v]) for k,v in encoded.items()}).logits
            payload = token_classifier.decode_predictions(words, logits.argmax(-1)[0].tolist(), len(name_spans.source_tokens(record['text'])))
            predictions.append({'id': record['id'], 'response': json.dumps(payload),
                                'elapsed_seconds': time.monotonic()-started})
        return predictions
    predictions = predict(references)
    result = {'development': name_spans.score(references, predictions), 'reference_origin': 'teacher_generated',
              'reference_split': 'development' if (silver/'development.jsonl').exists() else 'validation',
              'reference_ids': [r['id'] for r in references], 'model': config['model'], 'revision': config['revision'],
              'alignment_version': token_classifier.VERSION, 'device': 'cpu', 'threads': report['threads'],
              'compute_dtype': config['compute_dtype'],
              'limitations': ['Small teacher-labeled development set; not clinical accuracy.',
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
