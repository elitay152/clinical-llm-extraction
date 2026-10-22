"""Run a CPU LoRA pilot on source-supported silver treatment names."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import time

os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
ROOT = Path(__file__).resolve().parents[1]
MODEL = 'HuggingFaceTB/SmolLM2-135M-Instruct'
REVISION = '12fd25f77366fa6b3b4b768ec3050bf629380bac'


def build_examples(rows, reserved, tokenizer, max_length):
    from scripts.prepare_data import parse_summary, text_hash
    examples, seen = [], set()
    counts = {'reserved': 0, 'duplicates': 0, 'unusable': 0, 'too_long': 0}
    for row in rows:
        text = row['full_note']
        digest = text_hash(text)
        if digest in reserved:
            counts['reserved'] += 1
            continue
        if digest in seen:
            counts['duplicates'] += 1
            continue
        seen.add(digest)
        try:
            summary, _ = parse_summary(row['summary'])
            treatments = summary.get('treatments', [])
            names = [t.get('name') for t in treatments if isinstance(t, dict)]
            # Require an exact case-insensitive surface match, not guessed alignment.
            names = [n for n in names if isinstance(n, str) and n.strip().lower() not in ('', 'none', 'unknown', 'null') and n.lower() in text.lower()]
        except (ValueError, AttributeError, TypeError):
            counts['unusable'] += 1
            continue
        if not names:
            counts['unusable'] += 1
            continue
        start = max(0, text.lower().find(names[0].lower()) - 200)
        excerpt = text[start:start + 900]
        surfaces = []
        for name in names:
            pos = excerpt.lower().find(name.lower())
            if pos >= 0:
                surface = excerpt[pos:pos + len(name)]
                if surface not in surfaces:
                    surfaces.append(surface)
        if not surfaces:
            counts['unusable'] += 1
            continue
        messages = [{'role': 'user', 'content': 'Extract treatment names from this note. Return only JSON with a treatments array of strings. Use exact source wording.\n\n' + excerpt}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        target = json.dumps({'treatments': surfaces}, ensure_ascii=False) + tokenizer.eos_token
        prompt_ids = tokenizer(prompt, add_special_tokens=False)['input_ids']
        target_ids = tokenizer(target, add_special_tokens=False)['input_ids']
        if len(prompt_ids) + len(target_ids) > max_length:
            counts['too_long'] += 1
            continue
        examples.append({'id': str(row['idx']), 'source_hash': digest,
                         'input_ids': prompt_ids + target_ids,
                         'labels': [-100] * len(prompt_ids) + target_ids,
                         'prompt_ids': prompt_ids, 'silver_target': surfaces})
    return examples, counts


def main():
    import sys
    sys.path.insert(0, str(ROOT))
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from peft import LoraConfig, get_peft_model
    from scripts.prepare_data import text_hash

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--max-length', type=int, default=512)
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/pilot-lora')
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be positive')
    args.output.mkdir(parents=True, exist_ok=False)  # Never overwrite an earlier adapter.
    set_seed(42)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    rows = [json.loads(line) for line in (ROOT / 'data/raw/acn-200.jsonl').read_text().splitlines()]
    annotation_path = ROOT / 'data/annotation/acn-candidates.jsonl'
    if not annotation_path.exists():
        raise RuntimeError('Run prepare_data.py first to reserve annotation candidates.')
    reserved = {text_hash(json.loads(line)['text']) for line in annotation_path.read_text().splitlines()}
    examples, exclusions = build_examples(rows, reserved, tokenizer, args.max_length)
    if len(examples) < 2:
        raise RuntimeError(f'Insufficient usable silver examples: {len(examples)}')
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, torch_dtype=torch.float32)
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                                            target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM'))
    model.print_trainable_parameters()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=2e-4)

    def batch(example):
        return {key: torch.tensor([example[key]], dtype=torch.long) for key in ('input_ids', 'labels')}

    probe = batch(examples[0])
    model.eval()
    with torch.no_grad():
        initial_loss = model(**probe).loss.item()
    order = list(range(len(examples)))
    random.Random(42).shuffle(order)
    started = time.monotonic()
    losses = []
    model.train()
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        loss = model(**batch(examples[order[step % len(order)]])).loss
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite training loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())
        print(json.dumps({'step': step + 1, 'loss': loss.item()}), flush=True)
    model.eval()
    with torch.no_grad():
        final_loss = model(**probe).loss.item()
    if not any(p.grad is not None for p in model.parameters() if p.requires_grad):
        raise RuntimeError('No adapter gradients observed')
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    prompt = torch.tensor([examples[0]['prompt_ids']], dtype=torch.long)
    with torch.no_grad():
        generated = model.generate(input_ids=prompt, attention_mask=torch.ones_like(prompt),
                                   max_new_tokens=80, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    prediction = tokenizer.decode(generated[0, prompt.shape[1]:], skip_special_tokens=True)
    try:
        parsed = json.loads(prediction)
        valid = isinstance(parsed, dict) and set(parsed) == {'treatments'} and isinstance(parsed['treatments'], list) and all(isinstance(n, str) for n in parsed['treatments'])
    except ValueError:
        valid = False
    report = {'task': 'silver treatment-name extraction pipeline pilot; not medication benchmark',
              'model': MODEL, 'model_revision': REVISION, 'seed': 42, 'device': 'cpu',
              'threads': torch.get_num_threads(), 'steps': args.steps,
              'package_versions': {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'peft', 'accelerate')}, 'learning_rate': 2e-4,
              'lora': {'r': 8, 'alpha': 16, 'dropout': 0.05, 'targets': ['q_proj', 'v_proj']},
              'max_length': args.max_length, 'batch_size': 1,
              'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
              'training_examples': len(examples), 'unique_examples_seen': min(args.steps, len(examples)),
              'examples_seen': args.steps, 'excluded': exclusions,
              'data_sha256': hashlib.sha256((ROOT / 'data/raw/acn-200.jsonl').read_bytes()).hexdigest(),
              'training_source_hashes': [e['source_hash'] for e in examples],
              'training_ids': [e['id'] for e in examples],
              'annotation_overlap': len(reserved & {e['source_hash'] for e in examples}),
              'losses': losses, 'training_probe_loss_before': initial_loss,
              'training_probe_loss_after': final_loss, 'elapsed_seconds': time.monotonic() - started,
              'training_example_generation': prediction, 'generation_schema_valid': valid,
              'limitations': ['No held-out evaluation; probe and generation use training data.',
                              'Silver labels may omit treatments or include non-drugs.',
                              'Exact-text deduplication only; article/patient grouping unresolved.',
                              'Not the full medication schema, no status/evidence supervision.']}
    (args.output / 'run.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('training_source_hashes', 'training_ids', 'losses')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
