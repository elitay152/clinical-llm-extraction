"""Fine-tune a biomedical encoder on training-only medication-name BIO labels."""
import argparse
import collections
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import name_spans, token_classifier
from scripts.train_distillation import read_splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('silver_directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/token-classifier.json')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Output exists; choose a new directory')
    config = json.loads(args.config.read_text())
    revision = config['revision']
    if len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision):
        raise ValueError('Encoder revision must be an exact commit SHA')
    if config['epochs'] < 1 or config['batch_size'] < 1:
        raise ValueError('Epochs and batch size must be positive')
    if config['compute_dtype'] not in ('float32', 'bfloat16'):
        raise ValueError('Unsupported compute precision')
    train, development = read_splits(args.silver_directory)
    if any(r['label_origin'] != 'teacher_generated' for r in train):
        raise ValueError('Expected teacher-generated training references')
    import torch
    import torch.nn.functional as functional
    from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, set_seed
    set_seed(config['seed'])
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    tokenizer = AutoTokenizer.from_pretrained(config['model'], revision=revision, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError('A fast tokenizer with word alignment is required')
    rows = []; examples = []; skipped = []
    for record in train:
        try:
            example, _ = token_classifier.encode_record(record, tokenizer, config['max_length'])
        except ValueError as error:
            skipped.append({'id': record['id'], 'reason': str(error)})
        else:
            rows.append(record); examples.append(example)
    if len(rows) < 2:
        raise ValueError('Need at least two aligned teacher training examples')
    synthetic = name_spans.synthetic_negatives() if config['synthetic_lab_negatives'] else []
    for record in synthetic:
        example, _ = token_classifier.encode_record(record, tokenizer, config['max_length'])
        rows.append(record); examples.append(example)
    counts = collections.Counter(label for example in examples for label in example['labels'] if label != -100)
    weight_rule = config.get('class_weighting', 'sqrt_inverse_frequency')
    weights = torch.tensor(token_classifier.loss_weights(counts, weight_rule), dtype=torch.float32)
    model = AutoModelForTokenClassification.from_pretrained(
        config['model'], revision=revision, num_labels=3,
        id2label=dict(enumerate(token_classifier.LABELS)),
        label2id={label: i for i, label in enumerate(token_classifier.LABELS)},
        attn_implementation=config['attention_implementation'], torch_dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    collate = DataCollatorForTokenClassification(tokenizer)
    losses = []; epoch_losses = []; started = time.monotonic()
    for epoch in range(config['epochs']):
        order = list(range(len(examples)))
        random.Random(config['seed']+epoch).shuffle(order)
        model.train(); current_losses = []
        for position in range(0, len(order), config['batch_size']):
            batch = collate([examples[i] for i in order[position:position+config['batch_size']]])
            labels = batch.pop('labels')
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cpu', dtype=torch.bfloat16, enabled=config['compute_dtype']=='bfloat16'):
                logits = model(**batch).logits
                loss = functional.cross_entropy(logits.float().reshape(-1, 3), labels.reshape(-1), weight=weights, ignore_index=-100)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(gradient_norm):
                raise ValueError('Nonfinite gradients')
            optimizer.step()
            losses.append(loss.item()); current_losses.append(loss.item())
            print(json.dumps({'epoch': epoch+1, 'step': len(losses), 'loss': loss.item()}), flush=True)
        epoch_losses.append(sum(current_losses)/len(current_losses))
    args.output.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(args.output); tokenizer.save_pretrained(args.output)
    report = {'task': 'medication-name token classification', 'alignment_version': token_classifier.VERSION,
              'source_token_version': name_spans.VERSION, 'config': config,
              'silver_directory': str(args.silver_directory.resolve()),
              'train_file_sha256': hashlib.sha256((args.silver_directory/'train.jsonl').read_bytes()).hexdigest(),
              'training_ids': [r['id'] for r in rows], 'source_groups': [r['source_group'] for r in rows],
              'teacher_training_rows': len(rows)-len(synthetic), 'synthetic_training_rows': len(synthetic),
              'synthetic_training_inputs': [{'id': r['id'], 'input_sha256': r['input_sha256']} for r in synthetic],
              'skipped_training_rows': skipped, 'development_rows': len(development),
              'labels': token_classifier.LABELS, 'label_counts': dict(counts), 'class_weights': weights.tolist(),
              'class_weight_rule': weight_rule,
              'losses': losses, 'epoch_mean_losses': epoch_losses, 'steps': len(losses),
              'parameters': sum(p.numel() for p in model.parameters()), 'device': 'cpu',
              'parameter_dtype': 'float32', 'threads': torch.get_num_threads(), 'elapsed_seconds': time.monotonic()-started,
              'packages': {p: importlib.metadata.version(p) for p in ['torch','transformers','tokenizers']},
              'interpretation': 'Teacher-reference supervision with synthetic negatives; no clinical accuracy claim.'}
    (args.output/'run.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['losses','training_ids','source_groups']},indent=2))


if __name__ == '__main__':
    main()
