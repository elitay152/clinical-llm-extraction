"""Run a bounded hosted-labeling pilot, then train and compare a fresh student."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.pseudo_label import prepare, label, collect


def run(silver, output, comparison, model, base_url, steps):
    if not os.environ.get('TEACHER_API_KEY'):
        raise ValueError('Add the TEACHER_API_KEY OpenAI API credential securely in environment settings, save and publish, then retry. Never paste it into chat.')
    if output.exists() or comparison.exists():
        raise ValueError('Training/comparison output exists; choose new output paths before making API calls')
    if not (silver / 'manifest.json').exists():
        prepare(silver, limit=12, max_chars=1800, window='dosage')
    # Resumed calls count toward the per-split request limits.
    path = silver / 'responses.jsonl'
    records = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    manifest = json.loads((silver / 'manifest.json').read_text())
    if manifest.get('test_locked'):
        raise ValueError('Frozen experiments require explicit train/development labeling, not the pilot runner')
    split_by_id = {c['id']: c['split'] for c in manifest['candidates']}
    for split, total in [('train', 3), ('validation', 1)]:
        done = sum(split_by_id.get(record['id']) == split for record in records)
        remaining = max(0, total - done)
        if remaining:
            label(silver, model, base_url, 'hosted', None, remaining, 1200, split)
    report = collect(silver)
    print(json.dumps(report, indent=2), flush=True)
    counts = report['accepted_by_split']
    if counts.get('train', 0) < 2 or counts.get('validation', 0) < 1:
        raise ValueError('Too few teacher labels passed checks. Inspect report.json/rejected.json; no extra API calls or training were attempted.')
    subprocess.run([sys.executable, str(ROOT / 'scripts/train_distillation.py'), str(silver),
                    '--output', str(output), '--steps', str(steps)], check=True)
    subprocess.run([sys.executable, str(ROOT / 'scripts/evaluate_distillation.py'), str(output),
                    '--output', str(comparison)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--silver', type=Path, default=ROOT / 'data/silver/teacher-pilot-v2')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/distillation-pilot-v2')
    parser.add_argument('--comparison', type=Path, default=ROOT / 'artifacts/distillation-comparison-v2')
    parser.add_argument('--model', default=os.environ.get('TEACHER_MODEL', 'gpt-4.1-mini'))
    parser.add_argument('--base-url', default=os.environ.get('TEACHER_API_BASE_URL', 'https://api.openai.com/v1'))
    parser.add_argument('--steps', type=int, default=20)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be positive')
    run(args.silver, args.output, args.comparison, args.model, args.base_url, args.steps)


if __name__ == '__main__':
    main()
