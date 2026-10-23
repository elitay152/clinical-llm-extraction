"""Compare a sealed training expansion with a frozen baseline batch across paired seeds."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.classifier_experiments import digest, span_errors
from scripts.name_spans import score
from scripts.pseudo_label import load_manifest, write_json
from scripts.train_distillation import read_splits


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('expanded',type=Path)
    parser.add_argument('--baseline-batch',type=Path,required=True)
    parser.add_argument('--negative-controls',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    manifest=load_manifest(args.expanded)
    extension=manifest['training_extension']
    if not manifest.get('frozen_label_hashes') or 'train.jsonl' not in manifest['frozen_label_hashes']:
        raise ValueError('Seal the expanded labels before training')
    train,refs=read_splits(args.expanded)
    baseline_plan=json.loads((args.baseline_batch/'plan.json').read_text())
    if extension['source_hashes']['manifest.json']!=baseline_plan['reference_manifest_sha256']:
        raise ValueError('Expansion source differs from the baseline reference snapshot')
    if digest(args.expanded/'development.jsonl')!=baseline_plan['reference_labels_sha256']:
        raise ValueError('Development references differ from the baseline batch')
    if digest(args.negative_controls)!=baseline_plan['negative_controls_sha256']:
        raise ValueError('Synthetic negative controls changed')
    baseline_rows=[r for r in map(json.loads,(args.baseline_batch/'ledger.jsonl').read_text().splitlines()) if r['id'].startswith('baseline-')]
    if len(baseline_rows)!=3 or len({r['config']['seed'] for r in baseline_rows})!=3:
        raise ValueError('Require three baseline seeds for paired comparison')
    reference_config={k:v for k,v in baseline_rows[0]['config'].items() if k!='seed'}
    if any({k:v for k,v in r['config'].items() if k!='seed'}!=reference_config for r in baseline_rows):
        raise ValueError('Baseline configurations differ beyond the seed')
    base_train,_=read_splits(Path(extension['source']))
    if len(train)<=len(base_train):raise ValueError('No accepted training expansion')
    if {r['id']:r for r in base_train}!={r['id']:r for r in train if r['id'] in {b['id'] for b in base_train}}:
        raise ValueError('Original training labels changed in the expansion')
    baseline_metrics={}
    for row in baseline_rows:
        predictions=json.loads((args.baseline_batch/row['id']/'evaluation/predictions.json').read_text())
        metrics=score(refs,predictions)
        if metrics!=row['metrics'] or row['reference_labels_sha256']!=baseline_plan['reference_labels_sha256']:
            raise ValueError('Baseline predictions or reference version differ')
        baseline_metrics[row['config']['seed']]=metrics
    plan={'baseline_batch':str(args.baseline_batch.resolve()),'expanded_manifest_sha256':digest(args.expanded/'manifest.json'),
          'expanded_train_sha256':digest(args.expanded/'train.jsonl'), 'reference_labels_sha256':baseline_plan['reference_labels_sha256'],
          'negative_controls_sha256':baseline_plan['negative_controls_sha256'],
          'baseline_training_notes':len(base_train),'expanded_training_notes':len(train),
          'seeds':[r['config']['seed'] for r in baseline_rows], 'epochs':baseline_rows[0]['config']['epochs'],
          'hypothesis':'Additional source-disjoint teacher-labeled training notes improve fixed-reference name-span F1.',
          'promotion_min_mean_f1_gain':0.01,
          'comparison':'Fixed epoch budget; expanded data also increases optimizer updates.'}
    args.output.mkdir(parents=True,exist_ok=False);write_json(args.output/'plan.json',plan)
    ledger=args.output/'ledger.jsonl';entries=[]
    def execute(command,log):
        with log.open('w') as stream:
            subprocess.run([sys.executable,*map(str,command)],cwd=ROOT,env=dict(os.environ,HF_HUB_OFFLINE='1'),
                           stdout=stream,stderr=subprocess.STDOUT,check=True)
    for baseline in baseline_rows:
        config=baseline['config'];seed=config['seed'];directory=args.output/f'seed-{seed}';directory.mkdir()
        write_json(directory/'config.json',config)
        print(json.dumps({'started_seed':seed}),flush=True)
        model=directory/'model'
        execute([ROOT/'scripts/train_token_classifier.py',args.expanded,'--output',model,'--config',directory/'config.json'],directory/'train.log')
        execute([ROOT/'scripts/evaluate_token_classifier.py',model,'--output',directory/'evaluation',
                 '--negative-controls',args.negative_controls],directory/'evaluate.log')
        report=json.loads((model/'run.json').read_text())
        if report['teacher_training_rows']!=len(train) or report['skipped_training_rows']:
            raise ValueError('Training skipped expanded examples; inspect before comparing seeds')
        comparison=json.loads((directory/'evaluation/comparison.json').read_text())
        if comparison['reference_labels_sha256']!=plan['reference_labels_sha256']:
            raise ValueError('Reference version changed')
        predictions=json.loads((directory/'evaluation/predictions.json').read_text())
        write_json(directory/'disagreements.json',span_errors(refs,predictions))
        entry={'seed':seed,'baseline':baseline_metrics[seed],'expanded':comparison['development'],
               'synthetic_negative_controls':comparison['synthetic_negative_controls'],
               'training_steps':report['steps'],'teacher_training_rows':report['teacher_training_rows']}
        entry['paired_f1_change']=entry['expanded']['exact_name_spans']['f1']-entry['baseline']['exact_name_spans']['f1']
        with ledger.open('a') as stream:stream.write(json.dumps(entry)+'\n')
        entries.append(entry)
        print(json.dumps({'completed_seed':seed,'baseline_f1':entry['baseline']['exact_name_spans']['f1'],
                          'expanded_f1':entry['expanded']['exact_name_spans']['f1']}),flush=True)
    for path,key in [(args.expanded/'manifest.json','expanded_manifest_sha256'),(args.expanded/'train.jsonl','expanded_train_sha256'),
                     (args.expanded/'development.jsonl','reference_labels_sha256'),(args.negative_controls,'negative_controls_sha256')]:
        if digest(path)!=plan[key]:raise ValueError('Experiment inputs changed during training')
    means={variant:sum(e[variant]['exact_name_spans']['f1'] for e in entries)/len(entries) for variant in ['baseline','expanded']}
    gain=means['expanded']-means['baseline']
    controls=all(e['synthetic_negative_controls']['exact_document_agreement']==1.0 for e in entries)
    decision={'mean_f1':means,'paired_mean_change':gain,'all_synthetic_controls_passed':controls,
              'decision':'Prefer the expanded training pool provisionally.' if gain>=plan['promotion_min_mean_f1_gain'] and controls else
                         'Keep the current training pool; expansion did not meet the predeclared promotion rule.',
              'limits':'Ten fixed AI-generated development references; three seeds; fixed epochs do not match compute.'}
    write_json(args.output/'decision.json',decision);print(json.dumps(decision),flush=True)


if __name__=='__main__':main()
