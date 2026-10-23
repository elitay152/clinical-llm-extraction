"""Blind-review development name references, adjudicate differences, and seal a snapshot."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.label_medication_names import PROMPT, validate_response
from scripts.pseudo_label import hosted_response, load_manifest, write_json
from scripts.train_distillation import read_splits


def span_keys(record):
    return {(s['start'], s['end'], s['text']) for s in record['name_mentions']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--base-url', required=True)
    args = parser.parse_args()
    source_manifest = load_manifest(args.source)
    _, development = read_splits(args.source)
    if not source_manifest.get('test_locked') or not development or any(r['split'] != 'development' for r in development):
        raise ValueError('Review requires locked splits and development-only references')
    args.output.mkdir(parents=True, exist_ok=False)
    prompt = PROMPT.read_text()
    frozen = {name: hashlib.sha256((args.source/name).read_bytes()).hexdigest()
              for name in ['manifest.json', 'train.jsonl', 'development.jsonl', 'test.jsonl']}
    config = {'model':args.model, 'base_url':args.base_url, 'max_tokens':3500,
              'prompt':prompt, 'source_hashes':frozen, 'source':str(args.source.resolve()),
              'development_ids':[r['id'] for r in development]}
    write_json(args.output/'review-config.json', config)
    ledger = args.output/'review-responses.jsonl'
    def request(record, phase, content, system):
        response, metadata = hosted_response(dict(record, text=content), args.model, args.base_url, 3500, system_prompt=system)
        with ledger.open('a') as output:
            output.write(json.dumps({'id':record['id'], 'phase':phase, 'input_sha256':record['input_sha256'],
                                     'response':response, 'provider_metadata':metadata}, ensure_ascii=False)+'\n')
        print(json.dumps({'reviewed':record['id'], 'phase':phase}), flush=True)
        return validate_response(record, response)
    reviewed = []; audit = []
    for record in development:
        item = {'id':record['id'], 'original_mentions':record['name_mentions']}
        chosen = dict(record)
        try:
            blind = request(record, 'blind', record['text'], prompt)
        except (ValueError, TypeError, KeyError) as error:
            item.update(status='unresolved', reason=str(error))
        else:
            item['blind_mentions'] = blind['name_mentions']
            if span_keys(blind) == span_keys(record):
                item['status'] = 'teacher_agreement'
            else:
                content = ('SOURCE:\n'+record['text']+'\n\nCANDIDATE ANNOTATIONS (not ground truth):\n'+
                           json.dumps({'first':record['name_mentions'],'second':blind['name_mentions']},ensure_ascii=False))
                system = (prompt+'\nAdjudicate these two candidate annotations against the SOURCE only. '
                          'Neither candidate is authoritative or exhaustive. Re-read the complete source, '
                          'resolve boundaries and omissions using the protocol, and return the same JSON schema. '
                          'Contexts must be copied from SOURCE, never from the candidate annotations.')
                try:
                    adjudicated = request(record, 'adjudication', content, system)
                except (ValueError, TypeError, KeyError) as error:
                    item.update(status='unresolved', reason=str(error))
                else:
                    chosen = adjudicated
                    chosen['teacher'] = {'model':args.model, 'role':'reference_adjudication'}
                    item.update(status='teacher_adjudicated', adjudicated_mentions=chosen['name_mentions'])
        chosen['reference_review'] = {'status':item['status'], 'model':args.model,
                                      'limitation':'AI review; not independent clinical gold.'}
        reviewed.append(chosen); audit.append(item)
    for name, digest in frozen.items():
        if hashlib.sha256((args.source/name).read_bytes()).hexdigest() != digest:
            raise ValueError('Source changed during reference review')
    for name in ['train.jsonl', 'test.jsonl']:
        (args.output/name).write_bytes((args.source/name).read_bytes())
    (args.output/'development.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in reviewed))
    write_json(args.output/'reference-audit.json', {'documents':audit, 'unresolved':sum(i['status']=='unresolved' for i in audit),
                                                  'interpretation':'Teacher agreement/adjudication is not independent clinical verification.'})
    manifest = dict(source_manifest, reference_review={'model':args.model, 'source_hashes':frozen,
                    'audit_sha256':hashlib.sha256((args.output/'reference-audit.json').read_bytes()).hexdigest()},
                    frozen_label_hashes={n:hashlib.sha256((args.output/n).read_bytes()).hexdigest()
                                         for n in ['train.jsonl','development.jsonl','test.jsonl']})
    write_json(args.output/'manifest.json',manifest)
    (args.output/'manifest.sha256').write_text(hashlib.sha256((args.output/'manifest.json').read_bytes()).hexdigest()+'\n')
    print(json.dumps({'references':len(reviewed),'unresolved':sum(i['status']=='unresolved' for i in audit)}))


if __name__ == '__main__':
    main()
