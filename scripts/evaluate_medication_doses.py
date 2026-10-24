"""Evaluate name spans, dose spans, and medication-dose pairs against frozen references."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import medication_doses, name_spans
from scripts.predict_medication_names import MedicationNameExtractor
from scripts.train_distillation import read_splits


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--name-baseline',type=Path)
    parser.add_argument('--training-diagnostic',action='store_true')
    parser.add_argument('--linking-rule',choices=medication_doses.LINKING_RULES,default=medication_doses.LINKING_RULE)
    args=parser.parse_args()
    if args.output.exists():
        raise ValueError('Choose a new output directory')
    report=json.loads((args.run/'run.json').read_text())
    if report['alignment_version'] != medication_doses.VERSION:
        raise ValueError('Expected a name/dose classifier')
    directory=Path(report['silver_directory'])
    train, references=read_splits(directory)
    if hashlib.sha256((directory/'train.jsonl').read_bytes()).hexdigest() != report['train_file_sha256']:
        raise ValueError('Training labels changed')
    if args.training_diagnostic:
        references=[r for r in train if r['id'] in report['training_ids']]
        if args.name_baseline:
            raise ValueError('Baseline comparison is development-only')
    elif set(report['source_groups']) & {r['source_group'] for r in references}:
        raise ValueError('Development groups overlap training')
    if not references:
        raise ValueError('No reference records')
    model=MedicationNameExtractor(args.run)
    predictions=[model.predict_payload(r['text'])[0] for r in references]
    metrics=medication_doses.score(references,predictions,linking_rule=args.linking_rule)
    oracle_predictions=[dict(name_spans.reference_payload(r),dose_spans=medication_doses.reference_spans(r)) for r in references]
    result={'split':'training_diagnostic' if args.training_diagnostic else 'development',
            'metrics':metrics, 'rule_with_reference_entities':medication_doses.score(references,oracle_predictions,linking_rule=args.linking_rule),
            'reference_labels_sha256':hashlib.sha256((directory/('train.jsonl' if args.training_diagnostic else 'development.jsonl')).read_bytes()).hexdigest(),
            'interpretation':'Exact agreement with AI-generated/source-checked references; not clinical accuracy.',
            'dose_linking':args.linking_rule}
    if args.name_baseline:
        baseline=MedicationNameExtractor(args.name_baseline)
        if baseline.classifier is medication_doses:
            raise ValueError('Name baseline must be a names-only classifier')
        if set(baseline.report['source_groups']) & {r['source_group'] for r in references}:
            raise ValueError('Name baseline trained on development groups')
        baseline_payloads=[baseline.predict_payload(r['text'])[0] for r in references]
        responses=[{'id':r['id'],'response':json.dumps(p)} for r,p in zip(references,baseline_payloads)]
        candidate_predictions=[]
        for record,payload in zip(references,baseline_payloads):
            tokens=name_spans.source_tokens(record['text'])
            starts={t['start']:i+1 for i,t in enumerate(tokens)}
            ends={t['end']:i+1 for i,t in enumerate(tokens)}
            candidates=medication_doses.dose_candidates(record['text'])
            candidate_predictions.append(dict(payload,dose_spans=[[starts[s['start']],ends[s['end']]] for s in candidates]))
        result['names_plus_candidate_baseline']=medication_doses.score(references,candidate_predictions,linking_rule=args.linking_rule)
        old=name_spans.score(references,responses)
        result['name_baseline']=old
        result['name_f1_change']=metrics['names']['f1']-old['exact_name_spans']['f1']
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    (args.output/'predictions.json').write_text(json.dumps([dict(id=r['id'],**p) for r,p in zip(references,predictions)],indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
