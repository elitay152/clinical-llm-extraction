"""Select new training clusters while preserving frozen development and test membership."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.classifier_experiments import digest
from scripts.prepare_annotation_pilot import canonical_text
from scripts.prepare_data import text_hash
from scripts.prepare_experiment import group_notes
from scripts.pseudo_label import load_manifest, select_excerpt, write_json
from scripts.train_distillation import read_splits


def select_additions(rows, excluded_texts, previous, count, seed=42, max_chars=1800):
    if count<1:raise ValueError('Positive candidate count required')
    unique={}
    for row in rows:
        text=canonical_text(row['full_note']);unique.setdefault(text_hash(text),(row,text))
    pairs=list(unique.values());texts=[text for _,text in pairs]+[canonical_text(t) for t in excluded_texts]
    groups=group_notes(texts)
    by_hash={text_hash(text):group for (_,text),group in zip(pairs,groups)}
    reserved={r['source_text_hash'] for r in previous}
    if not reserved<=set(by_hash):raise ValueError('Previous source notes are missing from the pool')
    if any(by_hash[r['source_text_hash']]!=r['source_group'] for r in previous):
        raise ValueError('Pool clustering differs from frozen source groups')
    blocked={by_hash[h] for h in reserved}|set(groups[len(pairs):])
    previous_ids={r['id'] for r in previous};chosen={}
    for (row,text),group in zip(pairs,groups):
        identifier='acn:'+str(row['idx'])
        if group in blocked or identifier in previous_ids:continue
        excerpt=select_excerpt(text,max_chars,'dosage')
        if excerpt is None:continue
        selected,start=excerpt
        chosen.setdefault(group,{'id':identifier,'text':selected,'source_group':group,
            'source_text_hash':text_hash(text),'input_sha256':hashlib.sha256(selected.encode()).hexdigest(),
            'scope':'dosage_excerpt','source_start':start,'full_text_chars':len(text),'split':'train'})
    ordered=sorted(chosen.values(),key=lambda r:hashlib.sha256((str(seed)+r['source_group']).encode()).hexdigest())
    if len(ordered)<count:raise ValueError(f'Need {count} new groups; found {len(ordered)}')
    return ordered[:count]


def prepare(source,output,pool,excluded,count,seed):
    manifest=load_manifest(source);train,dev=read_splits(source)
    if not manifest.get('test_locked') or not manifest.get('frozen_label_hashes'):
        raise ValueError('Expansion requires frozen label references and locked test membership')
    additions=select_additions([json.loads(l) for l in pool.read_text().splitlines()],
        [json.loads(l)['full_note'] for l in excluded.read_text().splitlines()],manifest['candidates'],count,seed)
    extension={'source':str(source.resolve()),'source_hashes':{n:digest(source/n) for n in ['manifest.json','train.jsonl','development.jsonl','test.jsonl']},
               'pool_sha256':digest(pool),'excluded_pool_sha256':digest(excluded),'requested_candidates':count,'seed':seed}
    revised=dict(manifest,candidates=manifest['candidates']+additions,labeling_ids=[r['id'] for r in additions],
                 training_extension=extension,frozen_label_hashes={n:digest(source/n) for n in ['development.jsonl','test.jsonl']})
    output.mkdir(parents=True,exist_ok=False)
    for name in ['train.jsonl','development.jsonl','test.jsonl']:(output/name).write_bytes((source/name).read_bytes())
    write_json(output/'manifest.json',revised);(output/'manifest.sha256').write_text(digest(output/'manifest.json')+'\n')
    read_splits(output)
    return {'base_training_rows':len(train),'new_candidates':len(additions),'development_rows':len(dev),'test_locked':True}


def seal(directory):
    manifest=load_manifest(directory);report=json.loads((directory/'report.json').read_text())
    if report['pending']:raise ValueError('Collect every planned response before sealing')
    if 'train.jsonl' in manifest.get('frozen_label_hashes',{}):
        return {'sealed':True,'total_training_rows':report['total_training_rows']}
    read_splits(directory)
    manifest['precollection_manifest_sha256']=digest(directory/'manifest.json')
    manifest['frozen_label_hashes']={n:digest(directory/n) for n in ['train.jsonl','development.jsonl','test.jsonl']}
    write_json(directory/'manifest.json',manifest);(directory/'manifest.sha256').write_text(digest(directory/'manifest.json')+'\n')
    return {'sealed':True,'total_training_rows':report['total_training_rows']}


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--pool',type=Path,default=ROOT/'data/raw/acn-1000.jsonl')
    p.add_argument('--excluded',type=Path,default=ROOT/'data/raw/acn-200.jsonl')
    p.add_argument('--count',type=int,default=40);p.add_argument('--seed',type=int,default=42)
    p=sub.add_parser('seal');p.add_argument('directory',type=Path)
    a=parser.parse_args()
    print(json.dumps(prepare(a.source,a.output,a.pool,a.excluded,a.count,a.seed) if a.command=='prepare' else seal(a.directory)))


if __name__=='__main__':main()
