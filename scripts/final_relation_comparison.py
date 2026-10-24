"""Freeze approaches, label the reserved test set, and compare medication linkers."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import medication_relations as relations
from scripts import expand_relation_training as expansion
from scripts.calibrate_relation_thresholds import decode
from scripts.pseudo_label import load_manifest
from scripts.train_distillation import read_splits
from scripts.train_relation_classifier import score


def freeze(source, encoder, original, expanded, calibration, output, original_data):
    manifest = load_manifest(source); train, dev = read_splits(source)
    if not manifest.get('test_locked') or (source/'test.jsonl').read_text().strip():
        raise ValueError('Final comparison requires an unlabeled locked test set')
    candidates = [r for r in manifest['candidates'] if r['split'] == 'test']
    if not candidates or {r['source_group'] for r in candidates} & {r['source_group'] for r in train+dev}:
        raise ValueError('Missing test membership or overlapping source groups')
    reports = {name: json.loads((run/'run.json').read_text()) for name, run in [('original', original), ('expanded', expanded)]}
    encoder_hash = relations.fingerprint(encoder/'model.safetensors')
    if any(r['encoder_sha256'] != encoder_hash or r['feature_code_sha256'] != relations.fingerprint(Path(relations.__file__)) for r in reports.values()):
        raise ValueError('Relation models do not match frozen encoder/features')
    if reports['expanded']['train_sha256'] != relations.fingerprint(source/'train.jsonl') or reports['original']['train_sha256'] != relations.fingerprint(original_data/'train.jsonl'):
        raise ValueError('Relation training snapshots differ from frozen runs')
    original_train, _ = read_splits(original_data)
    encoder_report = json.loads((encoder/'run.json').read_text())
    if ({r['source_group'] for r in candidates} & {r['source_group'] for r in original_train} or
            {r['id'] for r in candidates} & set(encoder_report.get('training_ids', [])) or
            {r['source_group'] for r in candidates} & set(encoder_report.get('source_groups', []))):
        raise ValueError('Test membership overlaps model training')
    calibrated = json.loads((calibration/'thresholds.json').read_text())
    calibration_plan = json.loads((calibration/'plan.json').read_text())
    if calibration_plan['inputs']['model.safetensors'] != relations.fingerprint(expanded/'model.safetensors'):
        raise ValueError('Thresholds belong to a different model')
    files = [original_data/'manifest.json', original_data/'train.jsonl', source/'manifest.json', source/'train.jsonl', source/'development.jsonl', source/'test.jsonl',
             calibration/'thresholds.json', calibration/'plan.json']
    for run in (encoder, original, expanded):
        files.extend(p for p in run.iterdir() if p.name in ('model.safetensors','run.json','config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json','vocab.txt'))
    files.extend(ROOT/'scripts'/n for n in ('medication_doses.py','medication_regimens.py','medication_relations.py','predict_medication_names.py','final_relation_comparison.py','expand_relation_training.py','label_medication_names.py','label_medication_doses.py','name_spans.py','token_classifier.py','train_relation_classifier.py','calibrate_relation_thresholds.py'))
    files.extend((expansion.names.PROMPT, expansion.attributes.PROMPT, ROOT/'configs/regimen-teacher-prompt.txt'))
    plan = dict(version='final-relation-comparison-v1', candidates=candidates,
                encoder=str(encoder.resolve()), original=str(original.resolve()), expanded=str(expanded.resolve()),
                thresholds=calibrated['thresholds'],
                fixed_thresholds={n:r['config']['threshold'] for n,r in reports.items()},
                files={str(p.resolve()): relations.fingerprint(p) for p in files},
                teacher=dict(model='gpt-4.1-2025-04-14',base_url='https://api.openai.com/v1',max_tokens=3500,max_attempts=3),
                approaches=['rules','original','expanded','abstention'],
                selection='Rules retained from development; test results are final descriptive comparisons, not model or threshold selection.',
                evaluation='Same predicted entities for all approaches; primary complete-record F1, secondary pair F1 and false-link counts. No test tuning.',
                labeling='All planned notes required; invalid teacher labels block scoring rather than dropping notes.')
    output.mkdir(parents=True, exist_ok=False)
    (output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    (output/'plan.sha256').write_text(relations.fingerprint(output/'plan.json')+'\n')
    return dict(test_notes=len(candidates), approaches=plan['approaches'])


def checked(directory):
    if relations.fingerprint(directory/'plan.json') != (directory/'plan.sha256').read_text().strip():
        raise ValueError('Frozen comparison plan changed')
    plan=json.loads((directory/'plan.json').read_text())
    if any(relations.fingerprint(Path(p)) != h for p,h in plan['files'].items()):
        raise ValueError('Frozen comparison inputs changed')
    return plan


def label(directory):
    plan=checked(directory); path=directory/'annotations.jsonl'
    existing=[json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
    ids={r['id'] for r in existing}
    if len(ids)!=len(existing) or not ids<={r['id'] for r in plan['candidates']}:
        raise ValueError('Duplicate or unexpected test annotations')
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures=[executor.submit(expansion.annotate_record,r,plan['teacher']) for r in plan['candidates'] if r['id'] not in ids]
        for future in as_completed(futures):
            row=future.result()
            with path.open('a') as stream: stream.write(json.dumps(row,ensure_ascii=False)+'\n')
            print(json.dumps(dict(id=row['id'],accepted=row['record'] is not None)),flush=True)


def references(plan, annotations):
    if len(annotations)!=len(plan['candidates']) or len({r['id'] for r in annotations})!=len(annotations):
        raise ValueError('All test notes require annotations exactly once')
    by_id={r['id']:r for r in annotations}; records=[]
    for candidate in plan['candidates']:
        row=by_id.get(candidate['id'])
        if not row or row['record'] is None:
            raise ValueError('Test annotation failed; no notes may be dropped')
        record=candidate
        for task in ('names','dose','regimen'):
            attempts=[a for a in row['attempts'] if a['task']==task]
            if not attempts or attempts[-1]['validation_error'] is not None:
                raise ValueError('Missing valid annotation stage')
            record=(expansion.names.validate_response(candidate,attempts[-1]['response']) if task=='names' else
                    expansion.attributes.annotation_spec(task)[-1](record,attempts[-1]['response']))
        if record!=row['record'] or row['input_sha256']!=candidate['input_sha256']:
            raise ValueError('Stored reference differs from source-checked responses')
        records.append(record)
    return records


def evaluate(directory):
    plan=checked(directory)
    if (directory/'run.json').exists(): raise ValueError('Final test comparison already completed')
    annotations=[json.loads(s) for s in (directory/'annotations.jsonl').read_text().splitlines()]
    records=references(plan,annotations)
    import torch
    from safetensors.torch import load_file
    from scripts.predict_medication_names import MedicationNameExtractor
    extractor=MedicationNameExtractor(plan['encoder'],attribute_candidates=True)
    models={}
    for name in ('original','expanded'):
        run=Path(plan[name]); config=json.loads((run/'run.json').read_text())['config']
        model=relations.make_model(extractor.model.config.hidden_size*3,config)
        model.load_state_dict(load_file(str(run/'model.safetensors'))); model.eval(); models[name]=model
    payloads=[]; predictions={n:[] for n in plan['approaches']}
    for i,record in enumerate(records):
        text=record['text'];payload,_=extractor.predict_payload(text);payloads.append(payload)
        rows,semantic,layout=relations.features(extractor,text,payload)
        kinds=torch.tensor([r['kind'] for r in rows],dtype=torch.long)
        predictions['rules'].append(relations.rule_links(text,payload))
        for name,model in models.items():
            with torch.no_grad(): probabilities=model(semantic,layout,kinds).sigmoid().tolist()
            predictions[name].append(relations.decode(rows,probabilities,plan['fixed_thresholds'][name]))
            if name=='expanded': predictions['abstention'].append(decode(rows,probabilities,plan['thresholds']))
        print(f'Evaluated test note {i+1}/{len(records)}',flush=True)
    results={n:score(records,payloads,p) for n,p in predictions.items()}
    result=dict(plan_sha256=relations.fingerprint(directory/'plan.json'),annotations_sha256=relations.fingerprint(directory/'annotations.jsonl'),
                notes=len(records),reference_type='source-checked teacher agreement; candidate-limited dose/route/frequency references',results=results)
    (directory/'predictions.json').write_text(json.dumps(dict(entities=payloads,relations=predictions),indent=2)+'\n')
    (directory/'run.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('freeze')
    for name in ('source','encoder','original','expanded','calibration','output','original-data'):p.add_argument('--'+name,type=Path,required=True)
    for command in ('label','evaluate'):
        p=sub.add_parser(command);p.add_argument('directory',type=Path)
    a=parser.parse_args()
    if a.command=='freeze': print(json.dumps(freeze(a.source,a.encoder,a.original,a.expanded,a.calibration,a.output,a.original_data)))
    elif a.command=='label': label(a.directory)
    else:evaluate(a.directory)


if __name__=='__main__':main()
