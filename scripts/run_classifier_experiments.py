"""Run a gated overfit check, learning curve, class-weight ablation, and seed repeats."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.classifier_experiments import digest, make_view, nested_membership, span_errors
from scripts.pseudo_label import load_manifest, write_json
from scripts.token_classifier import word_labels
from scripts.train_distillation import read_splits


def normalized(config):
    return dict(config, class_weighting=config.get('class_weighting','sqrt_inverse_frequency'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('references',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--baseline-run',type=Path,required=True)
    parser.add_argument('--negative-controls',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/token-classifier.json')
    args=parser.parse_args()
    manifest=load_manifest(args.references)
    if not manifest.get('test_locked') or not manifest.get('frozen_label_hashes'):
        raise ValueError('Experiments require a sealed label snapshot and locked test set')
    train,references=read_splits(args.references)
    if len(train)<8 or not references:
        raise ValueError('Need eight training notes and development references')
    base=normalized(json.loads(args.config.read_text()))
    if base['class_weighting'] != 'sqrt_inverse_frequency':
        raise ValueError('This ablation requires a frequency-weighted baseline')
    memberships=nested_membership(train,base['seed'])
    tiny=memberships[100][:4]
    if not any(2 in word_labels(r) for r in train if r['id'] in tiny):
        multi=next(r['id'] for r in train if 2 in word_labels(r))
        tiny[-1]=multi
    plan={'reference_manifest_sha256':digest(args.references/'manifest.json'),
          'reference_labels_sha256':digest(args.references/'development.jsonl'),
          'train_sha256':digest(args.references/'train.jsonl'), 'config':base,
          'negative_controls_sha256':digest(args.negative_controls),
          'baseline_run_sha256':digest(args.baseline_run/'run.json'),
          'nested_training_ids':memberships, 'overfit_ids':tiny,
          'overfit_min_f1':0.98, 'promotion_min_f1_gain':0.01, 'repeat_seeds':[seed for seed in [7,123,42] if seed != base['seed']][:2],
          'hypothesis':'Uniform class weights reduce false positives relative to square-root inverse-frequency weights.',
          'limits':'Development silver references, including unresolved audit items; not clinical accuracy.'}
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'plan.json',plan)
    ledger=args.output/'ledger.jsonl'
    results={}
    def execute(command, log):
        with log.open('w') as stream:
            subprocess.run([sys.executable,*map(str,command)],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
                           env=dict(os.environ,HF_HUB_OFFLINE='1'),check=True)
    def run(name, ids, config, diagnostic=False, reuse=None):
        print(json.dumps({'started':name}),flush=True)
        directory=args.output/name; directory.mkdir()
        view=make_view(args.references,directory/'data',ids,development=not diagnostic)
        write_json(directory/'config.json',config)
        model=directory/'model'
        if reuse:
            saved=json.loads((reuse/'run.json').read_text())
            if normalized(saved['config'])!=config or saved['train_file_sha256']!=digest(view/'train.jsonl'):
                raise ValueError('Reusable baseline configuration or training labels differ')
            model=reuse
        else:
            execute([ROOT/'scripts/train_token_classifier.py',view,'--output',model,'--config',directory/'config.json'],directory/'train.log')
        command=[ROOT/'scripts/evaluate_token_classifier.py',model,'--output',directory/'evaluation']
        if diagnostic:command+=['--training-diagnostic']
        else:command+=['--reference-directory',args.references,'--negative-controls',args.negative_controls]
        execute(command,directory/'evaluate.log')
        comparison=json.loads((directory/'evaluation/comparison.json').read_text())
        metrics=comparison['training_diagnostic' if diagnostic else 'development']
        _, development_refs = read_splits(args.references)
        diagnostic_refs, _ = read_splits(view)
        predictions = json.loads((directory/'evaluation/predictions.json').read_text())
        write_json(directory/'disagreements.json', span_errors(diagnostic_refs if diagnostic else development_refs, predictions))
        if not diagnostic and comparison['reference_labels_sha256']!=plan['reference_labels_sha256']:
            raise ValueError('Experiment reference version changed')
        entry={'id':name,'training_notes':len(ids),'config':config,'diagnostic_only':diagnostic,
               'reused_checkpoint':str(reuse) if reuse else None,'metrics':metrics,
               'reference_labels_sha256':comparison['reference_labels_sha256']}
        if 'synthetic_negative_controls' in comparison:entry['synthetic_negative_controls']=comparison['synthetic_negative_controls']
        with ledger.open('a') as stream:stream.write(json.dumps(entry)+'\n')
        results[name]=entry
        print(json.dumps({'completed':name,'exact_span_f1':metrics['exact_name_spans']['f1']}),flush=True)
        return metrics['exact_name_spans']['f1']
    probe=dict(base,epochs=40,batch_size=4,learning_rate=0.0001,synthetic_lab_negatives=False)
    f1=run('overfit',tiny,probe,diagnostic=True)
    if f1 is None or f1<plan['overfit_min_f1']:
        write_json(args.output/'decision.json',{'decision':'stop','reason':'Overfit diagnostic below gate; investigate pipeline before optimization.'})
        raise RuntimeError('Overfit gate failed; generalization experiments were not started')
    run('learning-25',memberships[25],base)
    run('learning-50',memberships[50],base)
    baseline=run(f'baseline-{base["seed"]}',memberships[100],base,reuse=args.baseline_run)
    uniform=dict(base,class_weighting='uniform')
    candidate=run(f'uniform-{base["seed"]}',memberships[100],uniform)
    promoted=candidate is not None and candidate>=baseline+plan['promotion_min_f1_gain']
    for seed in plan['repeat_seeds']:
        run(f'baseline-{seed}',memberships[100],dict(base,seed=seed))
        if promoted:run(f'uniform-{seed}',memberships[100],dict(uniform,seed=seed))
    if digest(args.references/'manifest.json')!=plan['reference_manifest_sha256']:
        raise ValueError('Reference manifest changed during experiments')
    for file,key in [('train.jsonl','train_sha256'),('development.jsonl','reference_labels_sha256')]:
        if digest(args.references/file)!=plan[key]:raise ValueError('Frozen labels changed during experiments')
    if digest(args.negative_controls) != plan['negative_controls_sha256']:
        raise ValueError('Negative controls changed during experiments')
    summaries={}
    for variant in ['baseline','uniform']:
        values=[entry['metrics']['exact_name_spans']['f1'] for key,entry in results.items() if key.startswith(variant+'-')]
        if values:summaries[variant]={'runs':len(values),'mean_f1':sum(values)/len(values),'min_f1':min(values),'max_f1':max(values)}
    decision={'class_weight_candidate_repeated':promoted,'seed_summary':summaries,
              'decision':('Keep the baseline; the candidate did not improve the paired-seed mean by the planned margin.' if promoted else
                          'Keep frequency-based weights; uniform weighting failed the initial screen.'),
              'limitations':plan['limits']}
    if promoted and summaries['uniform']['mean_f1']>=summaries['baseline']['mean_f1']+plan['promotion_min_f1_gain']:
        decision['decision']='Uniform weighting improves mean development F1 across paired seeds; treat as provisional.'
    write_json(args.output/'decision.json',decision)
    print(json.dumps(decision),flush=True)


if __name__=='__main__':
    main()
