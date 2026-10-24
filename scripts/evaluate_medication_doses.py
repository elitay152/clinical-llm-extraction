"""Evaluate name spans, dose spans, and medication-dose pairs against frozen references."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import medication_attributes, medication_doses, medication_regimens, name_spans, token_classifier
from scripts.predict_medication_names import MedicationNameExtractor
from scripts.train_distillation import read_splits


def check_reference_compatibility(base,references):
    if {r['id']:r['input_sha256'] for r in references}!={r['id']:r['input_sha256'] for r in base}:
        raise ValueError('Attribute and base reference sources differ')
    by_id={r['id']:r for r in base}
    for record in references:
        old=by_id[record['id']]
        if (name_spans.reference_payload(old)!=name_spans.reference_payload(record)
                or any(old[k]!=record[k] for k in ('dose_mentions','dose_relations'))):
            raise ValueError('Frozen name/dose reference labels differ')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--name-baseline',type=Path)
    parser.add_argument('--dose-baseline',type=Path)
    attributes=parser.add_mutually_exclusive_group()
    attributes.add_argument('--attribute-model',type=Path)
    attributes.add_argument('--attribute-candidates',action='store_true')
    parser.add_argument('--reference-directory',type=Path,help='Frozen regimen references for candidate-only evaluation')
    parser.add_argument('--training-diagnostic',action='store_true')
    parser.add_argument('--linking-rule',choices=medication_doses.LINKING_RULES,default=medication_doses.LINKING_RULE)
    args=parser.parse_args()
    if bool(args.reference_directory)!=args.attribute_candidates:
        raise ValueError('Candidate evaluation requires --reference-directory; model references come from their run')
    if args.output.exists():
        raise ValueError('Choose a new output directory')
    base_report=json.loads((args.run/'run.json').read_text())
    report=(json.loads((args.attribute_model/'run.json').read_text()) if args.attribute_model else base_report)
    if args.attribute_model and (base_report['alignment_version']!=medication_doses.VERSION or report['alignment_version']!=medication_attributes.VERSION):
        raise ValueError('Expected dose base and separate route/frequency checkpoints')
    if args.attribute_candidates and base_report['alignment_version']!=medication_doses.VERSION:
        raise ValueError('Candidate attributes require a dose checkpoint')
    if args.attribute_model or args.attribute_candidates:
        base_directory=Path(base_report['silver_directory'])
        if hashlib.sha256((base_directory/'train.jsonl').read_bytes()).hexdigest()!=base_report['train_file_sha256']:
            raise ValueError('Base training labels changed')
    classifiers={m.VERSION:m for m in (medication_doses,medication_regimens)}
    classifier=medication_regimens if (args.attribute_model or args.attribute_candidates) else classifiers.get(report['alignment_version'])
    if classifier is None:
        raise ValueError('Expected a dose or regimen classifier')
    if args.dose_baseline and (classifier is not medication_regimens or args.training_diagnostic):
        raise ValueError('Dose baseline comparison requires regimen development evaluation')
    training_directory=Path(report['silver_directory'])
    directory=args.reference_directory or training_directory
    train,references=read_splits(directory)
    if args.attribute_model or args.attribute_candidates:
        _,base_refs=read_splits(base_directory)
        check_reference_compatibility(base_refs,references)
        if set(base_report['source_groups']) & {r['source_group'] for r in references}:
            raise ValueError('Base trained on development groups')
    if hashlib.sha256((training_directory/'train.jsonl').read_bytes()).hexdigest()!=report['train_file_sha256']:
        raise ValueError('Training labels changed')
    if args.training_diagnostic:
        references=[r for r in train if r['id'] in report['training_ids']]
        if args.name_baseline:
            raise ValueError('Baseline comparison is development-only')
    elif set(report['source_groups']) & {r['source_group'] for r in references}:
        raise ValueError('Development groups overlap training')
    if not references:
        raise ValueError('No reference records')
    model=MedicationNameExtractor(args.run,attribute_run=args.attribute_model,validate_attributes=False,attribute_candidates=args.attribute_candidates)
    predictions=[model.predict_payload(r['text'])[0] for r in references]
    raw_predictions=predictions
    if classifier is medication_regimens:
        predictions=[medication_regimens.validated_payload(r['text'],p) for r,p in zip(references,raw_predictions)]
    metrics=classifier.score(references,predictions,linking_rule=args.linking_rule)
    oracle_predictions=[dict(name_spans.reference_payload(r),dose_spans=medication_doses.reference_spans(r)) for r in references]
    if classifier is medication_regimens:
        for record,payload in zip(references,oracle_predictions):
            payload.update({f'{field}_spans':pairs for field,pairs in medication_regimens.reference_spans(record).items()})
    result={'split':'training_diagnostic' if args.training_diagnostic else 'development',
            'metrics':metrics, 'rule_with_reference_entities':classifier.score(references,oracle_predictions,linking_rule=args.linking_rule),
            'reference_labels_sha256':hashlib.sha256((directory/('train.jsonl' if args.training_diagnostic else 'development.jsonl')).read_bytes()).hexdigest(),
            'interpretation':'Exact agreement with AI-generated/source-checked references; not clinical accuracy.',
            'dose_linking':args.linking_rule,'separate_attribute_model':bool(args.attribute_model),'candidate_attributes':args.attribute_candidates}
    if classifier is medication_regimens:
        result['unvalidated_metrics']=classifier.score(references,raw_predictions,linking_rule=args.linking_rule)
        result['attribute_validation']=medication_regimens.CANDIDATE_VERSION
    if args.name_baseline:
        if classifier is medication_regimens:
            raise ValueError('Use --dose-baseline for regimen comparisons')
        baseline=MedicationNameExtractor(args.name_baseline)
        if baseline.classifier is not token_classifier:
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
    if args.dose_baseline:
        baseline=MedicationNameExtractor(args.dose_baseline)
        if baseline.classifier is not medication_doses:
            raise ValueError('Expected a dose-only baseline')
        if set(baseline.report['source_groups']) & {r['source_group'] for r in references}:
            raise ValueError('Dose baseline trained on development groups')
        payloads=[dict(baseline.predict_payload(r['text'])[0],route_spans=[],frequency_spans=[]) for r in references]
        result['dose_baseline']=classifier.score(references,payloads,linking_rule=args.linking_rule)
        candidate_payloads=[]
        for record,payload in zip(references,payloads):
            candidate_payloads.append(medication_regimens.candidate_payload(record['text'],payload))
        result['dose_plus_candidate_baseline']=classifier.score(references,candidate_payloads,linking_rule=args.linking_rule)
        result['entity_f1_changes']={key:metrics[key]['f1']-result['dose_baseline'][key]['f1'] for key in ('names','doses')}
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    if classifier is medication_regimens:
        (args.output/'raw-predictions.json').write_text(json.dumps(raw_predictions,indent=2)+'\n')
    (args.output/'predictions.json').write_text(json.dumps([dict(id=r['id'],**p) for r,p in zip(references,predictions)],indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
