"""Train a fresh LoRA student on mechanically accepted teacher medication labels."""
import argparse
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
from scripts.pseudo_label import SYSTEM
from scripts.train_pilot import MODEL, REVISION


def response_payload(record):
    return {'medications': [dict(med, evidence=[s['text'] for s in med['evidence']]) for med in record['medications']],
            'negative_evidence': [s['text'] for s in record['negative_evidence']]}


def encode_record(record, tokenizer, max_length):
    if record['label_origin'] != 'teacher_generated' or record['split'] != 'train':
        raise ValueError('Training accepts teacher-generated training rows only')
    prompt = tokenizer.apply_chat_template([{'role': 'system', 'content': SYSTEM},
             {'role': 'user', 'content': record['text']}], tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)['input_ids']
    answer_ids = tokenizer(json.dumps(response_payload(record), ensure_ascii=False) + tokenizer.eos_token,
                           add_special_tokens=False)['input_ids']
    if len(prompt_ids) + len(answer_ids) > max_length:
        return None
    return {'input_ids': prompt_ids + answer_ids, 'labels': [-100] * len(prompt_ids) + answer_ids,
            'id': record['id'], 'source_group': record['source_group'], 'prompt_ids': prompt_ids}


def read_splits(directory):
    def load(name):
        path = directory / f'{name}.jsonl'
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    train, validation = load('train'), load('validation')
    if {r['source_group'] for r in train} & {r['source_group'] for r in validation}:
        raise ValueError('Source groups overlap between train and validation')
    if any(r['split'] != 'train' for r in train) or any(r['split'] != 'validation' for r in validation):
        raise ValueError('Split files contain misplaced records')
    if len({r['id'] for r in train + validation}) != len(train + validation):
        raise ValueError('Duplicate records across silver files')
    return train, validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('silver_directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--max-length', type=int, default=2048)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be positive')
    train, validation = read_splits(args.silver_directory)
    if len(train) < 2:
        raise ValueError('Need at least two accepted training labels; run labeling and collect first')
    if args.output.exists():
        raise ValueError('Output exists; choose a new directory')
    os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
    from peft import get_peft_model, LoraConfig
    set_seed(42)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    examples = [e for r in train if (e := encode_record(r, tokenizer, args.max_length)) is not None]
    if len(examples) < 2:
        raise ValueError('Too few examples fit context. No targets were truncated.')
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, torch_dtype=torch.float32)
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                                           target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM'))
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=2e-4)
    def batch(example):
        return {k: torch.tensor([example[k]], dtype=torch.long) for k in ('input_ids', 'labels')}
    model.eval()
    with torch.no_grad():
        before = model(**batch(examples[0])).loss.item()
    order = list(range(len(examples))); random.Random(42).shuffle(order)
    losses = []; started = time.monotonic(); model.train()
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        loss = model(**batch(examples[order[step % len(order)]])).loss
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite training loss')
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        losses.append(loss.item())
        print(json.dumps({'step': step + 1, 'loss': loss.item()}), flush=True)
    model.eval()
    with torch.no_grad():
        after = model(**batch(examples[0])).loss.item()
    args.output.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(args.output); tokenizer.save_pretrained(args.output)
    report = {'task': 'medication schema distillation from silver labels', 'model': MODEL, 'revision': REVISION,
              'silver_directory': str(args.silver_directory.resolve()), 'system_prompt': SYSTEM,
              'train_file_sha256': hashlib.sha256((args.silver_directory / 'train.jsonl').read_bytes()).hexdigest(),
              'training_ids': [e['id'] for e in examples], 'source_groups': [e['source_group'] for e in examples],
              'max_length': args.max_length, 'steps': args.steps, 'seed': 42, 'learning_rate': 2e-4,
              'batch_size': 1, 'device': 'cpu', 'threads': torch.get_num_threads(),
              'lora': {'r': 8, 'alpha': 16, 'dropout': 0.05, 'target_modules': ['q_proj', 'v_proj']},
              'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
              'accepted_train_rows': len(train), 'encoded_train_rows': len(examples),
              'skipped_overlength_rows': len(train) - len(examples), 'validation_rows': len(validation),
              'losses': losses, 'training_probe_loss_before': before, 'training_probe_loss_after': after,
              'elapsed_seconds': time.monotonic() - started,
              'packages': {p: importlib.metadata.version(p) for p in ('torch', 'transformers', 'peft', 'accelerate')},
              'interpretation': 'Training diagnostics only; no human annotation or clinical accuracy claim.'}
    (args.output / 'run.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('losses', 'system_prompt', 'source_groups')}, indent=2))


if __name__ == '__main__':
    main()
