"""Verify saved LoRA adapter reload reproduces the recorded training probe."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from scripts.prepare_data import text_hash
    from scripts.train_pilot import build_examples

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    report = json.loads((args.run / 'run.json').read_text())
    torch.set_num_threads(report['threads'])
    raw = ROOT / 'data/raw/acn-200.jsonl'
    if hashlib.sha256(raw.read_bytes()).hexdigest() != report['data_sha256']:
        raise RuntimeError('Training data changed')
    tokenizer = AutoTokenizer.from_pretrained(args.run, local_files_only=True)
    rows = [json.loads(line) for line in raw.read_text().splitlines()]
    reserved = {text_hash(json.loads(line)['text']) for line in (ROOT / 'data/annotation/acn-candidates.jsonl').read_text().splitlines()}
    examples, _ = build_examples(rows, reserved, tokenizer, report['max_length'])
    if [e['source_hash'] for e in examples] != report['training_source_hashes']:
        raise RuntimeError('Training example selection changed')
    base = AutoModelForCausalLM.from_pretrained(report['model'], revision=report['model_revision'],
                                               torch_dtype=torch.float32, local_files_only=True)
    model = PeftModel.from_pretrained(base, args.run, local_files_only=True)
    model.eval()
    probe = {k: torch.tensor([examples[0][k]], dtype=torch.long) for k in ('input_ids', 'labels')}
    with torch.no_grad():
        loss = model(**probe).loss.item()
    if abs(loss - report['training_probe_loss_after']) > 1e-5:
        raise RuntimeError(f'Reloaded loss mismatch: {loss}')
    prompt = torch.tensor([examples[0]['prompt_ids']], dtype=torch.long)
    with torch.no_grad():
        generated = model.generate(input_ids=prompt, attention_mask=torch.ones_like(prompt),
                                   max_new_tokens=80, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    prediction = tokenizer.decode(generated[0, prompt.shape[1]:], skip_special_tokens=True)
    if prediction != report['training_example_generation']:
        raise RuntimeError('Reloaded generation differs from recorded generation')
    result = {'adapter_reload_verified': True, 'generation_reproduced': True,
              'package_versions': {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'peft', 'accelerate')}, 'reloaded_training_probe_loss': loss,
              'adapter_sha256': hashlib.sha256((args.run / 'adapter_model.safetensors').read_bytes()).hexdigest(),
              'note': 'Persistence check only; not independent evaluation.'}
    (args.run / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
