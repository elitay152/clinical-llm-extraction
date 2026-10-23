"""Version, label, and collect source-grounded medication-name mention references."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import name_spans
from scripts.pseudo_label import load_manifest, hosted_response, quote_span, write_json
from scripts.train_distillation import read_splits

PROMPT = ROOT/'configs/name-teacher-prompt.txt'
PREFIX = re.compile(r'^(?:tablets?|tab\.?|capsules?|cap\.?|oral|intravenous|iv|subcutaneous|intramuscular)\s+', re.I)
SUFFIX = re.compile(r'\s+(?:tablets?|capsules?|supplements?)$', re.I)
DOSE = re.compile(r'\b\d+(?:\.\d+)?\s*(?:mg|mcg|µg|μg|g|ml|units?)\b', re.I)


def prepare(source, run, output):
    manifest = load_manifest(source)
    train, development = read_splits(source)
    if run is None:
        selected = [r for r in manifest['candidates'] if r['split'] in ('train','development')]
    else:
        report = json.loads((run/'run.json').read_text())
        if report['train_file_sha256'] != hashlib.sha256((source/'train.jsonl').read_bytes()).hexdigest():
            raise ValueError('Source training labels differ from classifier run')
        trained = set(report['training_ids'])
        selected = [r for r in train if r['id'] in trained] + development
    ids = [r['id'] for r in selected]
    if not manifest.get('test_locked') or any(r['split'] not in ('train','development') for r in selected):
        raise ValueError('Name relabeling requires locked train/development source splits')
    if not selected or len(set(ids)) != len(ids):
        raise ValueError('Missing or duplicate source membership')
    revised = dict(manifest, format='silver-medication-names-v1', prompt_version=name_spans.NAME_PROTOCOL,
                   system_prompt=PROMPT.read_text(), labeling_ids=ids,
                   source_manifest_sha256=hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest(),
                   source_classifier_run_sha256=hashlib.sha256((run/'run.json').read_bytes()).hexdigest() if run else None,
                   name_label_scope='all explicit medicinal-agent mentions, regardless of regimen/status/person')
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'manifest.json', revised)
    (output/'manifest.sha256').write_text(hashlib.sha256((output/'manifest.json').read_bytes()).hexdigest()+'\n')
    return {'selected':len(ids), 'splits':dict(collections.Counter(r['split'] for r in selected))}


def validate_response(candidate, response):
    payload = json.loads(response)
    if (not isinstance(payload, dict) or set(payload) != {'mentions','coverage_reviewed'}
            or payload['coverage_reviewed'] is not True or not isinstance(payload['mentions'], list)):
        raise ValueError('Name output needs mentions and an explicit completed coverage review')
    text = candidate['text']; tokens = name_spans.source_tokens(text)
    starts = {t['start'] for t in tokens}; ends = {t['end'] for t in tokens}
    spans = {}
    for item in payload['mentions']:
        if not isinstance(item, dict) or set(item) != {'name','context'}:
            raise ValueError('Mention needs name and context only')
        name = item['name']
        if (not isinstance(name, str) or not name.strip() or name != name.strip()
                or PREFIX.search(name) or SUFFIX.search(name) or DOSE.search(name)
                or name[0] in '([,.' or name[-1] in ')],.®™' or ' with ' in name.lower()):
            raise ValueError('Name includes a forbidden form, route, dose, or boundary marker')
        context = quote_span(text,item['context'])
        matches = []
        for match in re.finditer(re.escape(name), context['text']):
            start, end = context['start']+match.start(), context['start']+match.end()
            if start in starts and end in ends:
                matches.append((start,end))
        if not matches:
            raise ValueError('Name has no exact token-aligned occurrence in its context')
        for start,end in matches:
            spans[(start,end)] = {'start':start, 'end':end, 'text':text[start:end]}
    record = {key:candidate[key] for key in ['id','text','split','source_group','input_sha256']}
    record.update(name_mentions=[spans[key] for key in sorted(spans)],
                  name_label_protocol=name_spans.NAME_PROTOCOL, label_origin='teacher_generated',
                  coverage_reviewed=True)
    name_spans.reference_payload(record)
    return record


def checked_manifest(directory):
    manifest = load_manifest(directory)
    if manifest['prompt_version'] != name_spans.NAME_PROTOCOL or manifest['system_prompt'] != PROMPT.read_text():
        raise ValueError('Name-label protocol differs from the frozen prompt')
    selected = manifest['labeling_ids']
    by_id = {r['id']:r for r in manifest['candidates']}
    if (len(by_id) != len(manifest['candidates']) or len(set(selected)) != len(selected) or not manifest.get('test_locked')
            or any(identifier not in by_id or by_id[identifier]['split'] not in ('train','development') for identifier in selected)):
        raise ValueError('Invalid selected membership or locked test ID')
    return manifest, by_id


def label(directory, split, model, base_url, limit, max_tokens):
    if split not in ('train','development') or limit < 1 or max_tokens < 1:
        raise ValueError('Explicit train/development split and positive limits are required')
    manifest, by_id = checked_manifest(directory)
    if manifest.get('training_extension') and split != 'train':
        raise ValueError('Training extensions cannot label development notes')
    config = {'model':model, 'base_url':base_url, 'max_tokens':max_tokens,
              'prompt_version':name_spans.NAME_PROTOCOL, 'response_format':'json_object'}
    path = directory/'teacher.json'
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError('Teacher configuration changed; prepare a new label version')
    responses = directory/'responses.jsonl'
    rows = [json.loads(line) for line in responses.read_text().splitlines()] if responses.exists() else []
    completed = {r['id'] for r in rows}
    if len(completed) != len(rows) or not completed <= set(manifest['labeling_ids']):
        raise ValueError('Response membership differs from frozen name selection')
    write_json(path,config)
    pending = [by_id[i] for i in manifest['labeling_ids'] if i not in completed and by_id[i]['split']==split][:limit]
    for candidate in pending:
        started = time.monotonic()
        response, metadata = hosted_response(candidate,model,base_url,max_tokens,system_prompt=manifest['system_prompt'])
        record = {'id':candidate['id'], 'input_sha256':candidate['input_sha256'], 'response':response,
                  'teacher':config, 'provider_metadata':metadata, 'elapsed_seconds':time.monotonic()-started}
        with responses.open('a') as output:
            output.write(json.dumps(record,ensure_ascii=False)+'\n')
        print(json.dumps({'generated':candidate['id'],'split':split,'elapsed_seconds':record['elapsed_seconds']}),flush=True)
    return {'new_responses':len(pending)}


def collect(directory):
    manifest, by_id = checked_manifest(directory)
    rows = [json.loads(line) for line in (directory/'responses.jsonl').read_text().splitlines()]
    config = json.loads((directory/'teacher.json').read_text())
    accepted = []; rejected = []; seen = set()
    for row in rows:
        identifier = row['id']
        if (row['teacher'] != config or identifier in seen or identifier not in manifest['labeling_ids']
                or row['input_sha256'] != by_id[identifier]['input_sha256']):
            raise ValueError('Response is duplicated or differs from frozen name-label membership')
        seen.add(identifier)
        try:
            record = validate_response(by_id[identifier],row['response'])
        except (ValueError,TypeError,KeyError) as error:
            rejected.append({'id':identifier,'reason':str(error)})
        else:
            record['teacher'] = row['teacher']; accepted.append(record)
    base_train = []; base_development = []
    extension = manifest.get('training_extension')
    if extension:
        source = Path(extension['source'])
        for name, digest in extension['source_hashes'].items():
            if hashlib.sha256((source/name).read_bytes()).hexdigest() != digest:
                raise ValueError('Training extension source changed')
        base_train, base_development = read_splits(source)
        if any(row['split'] != 'train' for row in accepted):
            raise ValueError('Training extension received non-training labels')
    combined = base_train + base_development + accepted
    for split in ['train','development','test']:
        if extension and split != 'train':
            (directory/f'{split}.jsonl').write_bytes((source/f'{split}.jsonl').read_bytes())
        else:
            (directory/f'{split}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in combined if r['split']==split))
    report = {'responses':len(rows),'accepted':len(accepted),'accepted_by_split':dict(collections.Counter(r['split'] for r in accepted)),
              'total_training_rows':sum(r['split']=='train' for r in combined),
              'rejected':rejected, 'pending':len(manifest['labeling_ids'])-len(rows),
              'interpretation':'Source/boundary validation and teacher self-review; not independent completeness verification.'}
    write_json(directory/'report.json',report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('source',type=Path);p.add_argument('output',type=Path);p.add_argument('--membership-run',type=Path)
    p=sub.add_parser('label');p.add_argument('directory',type=Path);p.add_argument('--split',choices=['train','development'],required=True)
    p.add_argument('--model',required=True);p.add_argument('--base-url',required=True);p.add_argument('--limit',type=int,default=1);p.add_argument('--max-tokens',type=int,default=3500)
    p=sub.add_parser('collect');p.add_argument('directory',type=Path)
    args=parser.parse_args()
    if args.command=='prepare':result=prepare(args.source,args.membership_run,args.output)
    elif args.command=='label':result=label(args.directory,args.split,args.model,args.base_url,args.limit,args.max_tokens)
    else:result=collect(args.directory)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
