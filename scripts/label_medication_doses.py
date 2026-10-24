"""Annotate dose amounts and links while preserving frozen medication-name labels."""
import argparse
import collections
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import medication_doses, name_spans
from scripts.pseudo_label import hosted_response, load_manifest, write_json
from scripts.train_distillation import read_splits

PROMPT = ROOT/'configs/dose-teacher-prompt.txt'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(source, output):
    manifest = load_manifest(source)
    train, development = read_splits(source)
    if not manifest.get('test_locked') or not train or not development:
        raise ValueError('Dose labeling requires frozen training/development splits and a locked test')
    records = train+development
    for record in records:
        name_spans.reference_payload(record)
    revised = dict(manifest)
    for key in ('frozen_label_hashes', 'training_extension', 'labeling_ids', 'system_prompt'):
        revised.pop(key, None)
    revised.update(format='silver-medication-doses-v1', dose_protocol=medication_doses.PROTOCOL,
                   system_prompt=PROMPT.read_text(), candidate_version=medication_doses.CANDIDATE_VERSION, label_records=records,
                   labeling_ids=[r['id'] for r in records], source_directory=str(source.resolve()),
                   source_hashes={name:digest(source/name) for name in ('manifest.json','train.jsonl','development.jsonl','test.jsonl')})
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'manifest.json', revised)
    (output/'manifest.sha256').write_text(digest(output/'manifest.json')+'\n')
    return dict(collections.Counter(r['split'] for r in records))


def checked(directory):
    manifest = load_manifest(directory)
    if (manifest.get('dose_protocol') != medication_doses.PROTOCOL
            or manifest.get('candidate_version') != medication_doses.CANDIDATE_VERSION
            or manifest.get('system_prompt') != PROMPT.read_text() or not manifest.get('test_locked')):
        raise ValueError('Dose protocol or frozen prompt differs')
    rows = manifest['label_records']
    by_id = {row['id']:row for row in rows}
    if (len(by_id) != len(rows) or set(by_id) != set(manifest['labeling_ids'])
            or any(r['split'] not in ('train','development') for r in rows)):
        raise ValueError('Invalid dose-label selection')
    candidates = {r['id']:r for r in manifest['candidates']}
    for row in rows:
        if any(row[k] != candidates[row['id']][k] for k in ('text','input_sha256','split','source_group')):
            raise ValueError('Dose selection differs from source candidates')
    source = Path(manifest['source_directory'])
    if any(digest(source/name) != value for name,value in manifest['source_hashes'].items()):
        raise ValueError('Source name-label snapshot changed')
    return manifest, by_id


def validate_response(record, response):
    value = json.loads(response)
    names = name_spans.resolve_response(record, json.dumps(name_spans.reference_payload(record)))
    if (not isinstance(value, dict) or set(value) != {'reviewed_medication_indices','doses','coverage_reviewed'}
            or value['coverage_reviewed'] is not True or not isinstance(value['doses'], list)
            or not isinstance(value['reviewed_medication_indices'], list)
            or any(type(i) is not int for i in value['reviewed_medication_indices'])
            or sorted(value['reviewed_medication_indices']) != list(range(len(names)))):
        raise ValueError('Every frozen medication mention needs explicit dose coverage review')
    candidates = medication_doses.dose_candidates(record['text'])
    selected=set(); links=set(); emitted=set()
    for item in value['doses']:
        if not isinstance(item,dict) or set(item) != {'medication_index','dose_index'}:
            raise ValueError('Dose selection needs medication_index and dose_index only')
        med,dose=item['medication_index'],item['dose_index']
        if type(dose) is not int or not 0 <= dose < len(candidates):
            raise ValueError('Unknown source dose candidate')
        if med is not None and (type(med) is not int or not 0 <= med < len(names)):
            raise ValueError('Unknown medication occurrence')
        if (med,dose) in emitted:
            raise ValueError('Duplicate dose annotation')
        emitted.add((med,dose));selected.add(dose)
        if med is not None:
            links.add((med,dose))
    indices=sorted(selected)
    result=dict(record,dose_label_protocol=medication_doses.PROTOCOL,
                dose_mentions=[candidates[i] for i in indices],
                dose_relations=[{'medication_index':med,'dose_index':indices.index(dose)} for med,dose in sorted(links)],
                dose_coverage_reviewed=True)

    medication_doses.reference_spans(result)
    return result


def label(directory, split, model, base_url, limit, workers):
    manifest, by_id = checked(directory)
    if manifest.get('frozen_label_hashes'):
        raise ValueError('Dose labels are already frozen')
    if limit < 1 or not 1 <= workers <= 4:
        raise ValueError('Positive request limit and 1–4 workers required')
    config = {'model':model,'base_url':base_url,'max_tokens':3500,'protocol':medication_doses.PROTOCOL,
              'prompt_sha256':digest(PROMPT),'candidate_version':medication_doses.CANDIDATE_VERSION,'max_attempts':3}
    config_path = directory/'dose-teacher.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('Dose teacher settings changed')
    write_json(config_path,config)
    path = directory/'dose-responses.jsonl'
    rows = [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
    completed = {r['id'] for r in rows}
    if len(rows) != len(completed) or not completed <= set(by_id):
        raise ValueError('Duplicate or unexpected dose responses')
    pending = [r for r in by_id.values() if r['split']==split and r['id'] not in completed][:limit]
    def request(record):
        marked,names,candidates=medication_doses.marked_source(record)
        payload={'marked_note':marked,'medication_indices':list(range(len(names))),
                 'dose_indices':list(range(len(candidates)))}
        content=json.dumps(payload)
        attempts=[]
        for _ in range(config['max_attempts']):
            response, metadata = hosted_response({'text':content},model,base_url,3500,system_prompt=manifest['system_prompt'])
            try:
                validate_response(record,response)
            except (ValueError,TypeError,KeyError) as error:
                reason=str(error)
            else:
                reason=None
            attempts.append({'response':response,'provider_metadata':metadata,'validation_error':reason})
            if reason is None:
                break
            content=json.dumps(dict(payload,
                                previous_output=response,validation_error=reason,
                                    request='Return a corrected complete annotation under the same protocol.'))
        return {'id':record['id'],'input_sha256':record['input_sha256'],'response':response,
                'teacher':config,'attempts':attempts}

    failures=[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(request,r):r['id'] for r in pending}
        for future in as_completed(futures):
            try:
                row = future.result()
            except Exception as error:
                failures.append({'id':futures[future],'error_type':type(error).__name__})
                continue
            with path.open('a') as output:
                output.write(json.dumps(row,ensure_ascii=False)+'\n')
            print(json.dumps({'labeled':row['id'],'split':split}),flush=True)
    if failures:
        raise RuntimeError(f'Teacher calls failed; successful calls were saved: {failures}')
    return {'new_responses':len(pending)}


def collect(directory, freeze=False):
    manifest, by_id = checked(directory)
    if manifest.get('frozen_label_hashes'):
        raise ValueError('Dose snapshot is already frozen')
    config = json.loads((directory/'dose-teacher.json').read_text())
    responses = [json.loads(s) for s in (directory/'dose-responses.jsonl').read_text().splitlines()]
    accepted=[]; rejected=[]; seen=set()
    for row in responses:
        identifier=row['id']
        if (identifier in seen or identifier not in by_id or row['teacher'] != config
                or row['input_sha256'] != by_id[identifier]['input_sha256']):
            raise ValueError('Dose response differs from frozen selection')
        seen.add(identifier)
        try:
            record=validate_response(by_id[identifier],row['response'])
            record['dose_teacher']=row['teacher']
            accepted.append(record)
        except (ValueError,TypeError,KeyError) as error:
            rejected.append({'id':identifier,'reason':str(error)})
    accepted.sort(key=lambda r:manifest['labeling_ids'].index(r['id']))
    for split in ('train','development','test'):
        (directory/f'{split}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in accepted if r['split']==split))
    report = {'accepted':dict(collections.Counter(r['split'] for r in accepted)), 'rejected':rejected,
              'pending':len(by_id)-len(seen), 'dose_mentions':sum(len(r['dose_mentions']) for r in accepted),
              'dose_relations':sum(len(r['dose_relations']) for r in accepted)}
    write_json(directory/'dose-label-report.json',report)
    if freeze:
        if len(accepted) != len(by_id):
            raise ValueError('All selected records must pass dose validation before freezing')
        manifest['frozen_label_hashes'] = {f'{s}.jsonl':digest(directory/f'{s}.jsonl') for s in ('train','development','test')}
        write_json(directory/'manifest.json',manifest)
        (directory/'manifest.sha256').write_text(digest(directory/'manifest.json')+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    p=sub.add_parser('label');p.add_argument('directory',type=Path);p.add_argument('--split',choices=['train','development'],required=True)
    p.add_argument('--model',default='gpt-4.1-mini-2025-04-14');p.add_argument('--base-url',default='https://api.openai.com/v1')
    p.add_argument('--limit',type=int,default=1);p.add_argument('--workers',type=int,default=2)
    p=sub.add_parser('collect');p.add_argument('directory',type=Path);p.add_argument('--freeze',action='store_true')
    args=parser.parse_args()
    if args.command=='prepare': result=prepare(args.source,args.output)
    elif args.command=='label': result=label(args.directory,args.split,args.model,args.base_url,args.limit,args.workers)
    else: result=collect(args.directory,args.freeze)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
