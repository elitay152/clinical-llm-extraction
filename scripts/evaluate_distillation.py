"""Compare baseline/adapted student agreement with held-out teacher labels."""
import argparse
import collections
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.pseudo_label import validate_teacher_response, SYSTEM
from scripts.train_distillation import read_splits


def record_key(record):
    # Evidence positions are validated but excluded from record equality.
    return json.dumps({key: record[key] for key in ('name', 'strength', 'dose', 'route', 'frequency', 'status')}, sort_keys=True)


def score(reference, predictions):
    if not reference:
        raise ValueError('No held-out teacher references; refusing zero-example metrics')
    by_id = {p['id']: p for p in predictions}
    if len(by_id) != len(predictions) or set(by_id) != {r['id'] for r in reference}:
        raise ValueError('Prediction membership differs from reference')
    tp = fp = fn = valid = exact = json_valid = 0
    errors = []
    for ref in reference:
        prediction = by_id[ref['id']]
        candidate = dict(ref, input_sha256=ref['input_sha256'])
        try:
            json.loads(prediction['response']); json_valid += 1
            parsed = validate_teacher_response(candidate, prediction['response'])
            valid += 1
            actual = collections.Counter(record_key(r) for r in parsed['medications'])
        except (ValueError, TypeError, KeyError) as error:
            actual = collections.Counter()
            errors.append({'id': ref['id'], 'reason': str(error)})
            # Invalid response never receives exact-document credit, including negatives.
            parsed = None
        expected = collections.Counter(record_key(r) for r in ref['medications'])
        tp += sum((actual & expected).values())
        fp += sum((actual - expected).values())
        fn += sum((expected - actual).values())
        exact += parsed is not None and actual == expected
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    return {'examples': len(reference), 'json_valid': json_valid, 'schema_and_evidence_valid': valid,
            'exact_document_agreement': exact / len(reference),
            'exact_medication_record_precision': precision, 'exact_medication_record_recall': recall,
            'exact_medication_record_f1': f1, 'tp': tp, 'fp': fp, 'fn': fn,
            'invalid_response_errors': errors,
            'interpretation': 'Agreement with generated teacher references; not clinical accuracy.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-tokens', type=int, default=1200)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Evaluation output exists; choose a new directory')
    report = json.loads((args.run / 'run.json').read_text())
    silver = Path(report['silver_directory'])
    _, references = read_splits(silver)
    if not references:
        raise ValueError('No accepted validation teacher labels; label and collect validation first')
    if set(report['source_groups']) & {r['source_group'] for r in references}:
        raise ValueError('Evaluation source groups overlap trained examples')
    if report['system_prompt'] != SYSTEM:
        raise ValueError('Prompt differs from the training run')
    os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    torch.set_num_threads(report['threads'])
    tokenizer = AutoTokenizer.from_pretrained(args.run, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(report['model'], revision=report['revision'],
                                                torch_dtype=torch.float32, local_files_only=True)
    args.output.mkdir(parents=True, exist_ok=False)
    def predict(model):
        model.eval(); predictions = []
        for ref in references:
            prompt = tokenizer.apply_chat_template([{'role': 'system', 'content': SYSTEM},
                     {'role': 'user', 'content': ref['text']}], tokenize=False, add_generation_prompt=True)
            tokens = tokenizer(prompt, return_tensors='pt', add_special_tokens=False)
            if tokens['input_ids'].shape[1] + args.max_tokens > model.config.max_position_embeddings:
                raise ValueError('Evaluation exceeds model context; refusing silent truncation')
            started = time.monotonic()
            with torch.no_grad():
                output = model.generate(**tokens, max_new_tokens=args.max_tokens, do_sample=False,
                                         pad_token_id=tokenizer.eos_token_id)
            tail = output[0, tokens['input_ids'].shape[1]:]
            predictions.append({'id': ref['id'], 'response': tokenizer.decode(tail, skip_special_tokens=True),
                                'elapsed_seconds': time.monotonic() - started, 'output_tokens': len(tail)})
        return predictions
    baseline = predict(model)
    (args.output / 'baseline.json').write_text(json.dumps(baseline, indent=2) + '\n')
    model = PeftModel.from_pretrained(model, args.run, local_files_only=True)
    adapted = predict(model)
    (args.output / 'adapted.json').write_text(json.dumps(adapted, indent=2) + '\n')
    result = {'baseline': score(references, baseline), 'adapted': score(references, adapted),
              'reference_origin': 'teacher_generated', 'model': report['model'], 'revision': report['revision'],
              'validation_ids': [r['id'] for r in references], 'max_tokens': args.max_tokens,
              'decoding': 'greedy', 'device': 'cpu', 'threads': report['threads'],
              'limitations': ['Very small silver-reference comparison; no clinical gold labels.',
                              'Exact-text grouping only; article/patient overlap remains unresolved.',
                              'Exact record scoring excludes evidence position agreement; evidence source matches are validated.',
                              'Generation times include model first-use overhead and are not production latency benchmarks.']}
    (args.output / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
