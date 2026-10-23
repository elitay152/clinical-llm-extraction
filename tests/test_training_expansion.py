import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.classifier_experiments import digest
from scripts.label_medication_names import collect, validate_response, PROMPT
from scripts.name_spans import NAME_PROTOCOL
from scripts.prepare_experiment import group_notes
from scripts.prepare_training_expansion import select_additions, seal
from scripts.prepare_data import text_hash


class TrainingExpansionTests(unittest.TestCase):
    def test_existing_near_duplicate_and_reserved_clusters_are_excluded(self):
        base='aspirin 5 mg '+' '.join(f'base{i}' for i in range(80))
        reserved='aspirin 6 mg '+' '.join(f'reserved{i}' for i in range(80))
        texts=[base,base+' extra',reserved,'metformin 10 mg '+' '.join(f'newA{i}' for i in range(80)),
               'atorvastatin 20 mg '+' '.join(f'newB{i}' for i in range(80))]
        rows=[{'idx':i,'full_note':text} for i,text in enumerate(texts)]
        groups=group_notes(texts+[reserved])
        previous=[{'id':'acn:0','source_text_hash':text_hash(base),'source_group':groups[0]}]
        result=select_additions(rows,[reserved],previous,2)
        self.assertEqual({r['id'] for r in result},{'acn:3','acn:4'})
        self.assertTrue(all(r['split']=='train' for r in result))
        self.assertEqual(result,select_additions(rows,[reserved],previous,2))
        with self.assertRaises(ValueError):select_additions(rows,[reserved],previous,3)
        with self.assertRaises(ValueError):select_additions(rows,[reserved],[dict(previous[0],source_group='changed')],1)

    def test_collect_preserves_base_labels_and_frozen_development(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);base=root/'base';base.mkdir();out=root/'extension';out.mkdir()
            candidates=[];records=[]
            for i,split in enumerate(['train','development','test','train']):
                text=f'Take aspirin {i}.'
                candidate={'id':str(i),'text':text,'split':split,'source_group':str(i),
                           'input_sha256':hashlib.sha256(text.encode()).hexdigest()}
                candidates.append(candidate)
                records.append(validate_response(candidate,json.dumps({'mentions':[{'name':'aspirin','context':text}], 'coverage_reviewed':True})))
            original={'protocol':'frozen-three-way-v1','test_locked':True,'prompt_version':NAME_PROTOCOL,
                      'system_prompt':PROMPT.read_text(),'candidates':candidates[:3],'labeling_ids':['0','1']}
            (base/'manifest.json').write_text(json.dumps(original));(base/'manifest.sha256').write_text(digest(base/'manifest.json'))
            for split in ['train','development','test']:
                (base/f'{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records[:2] if r['split']==split))
                (out/f'{split}.jsonl').write_bytes((base/f'{split}.jsonl').read_bytes())
            manifest=dict(original,candidates=candidates,labeling_ids=['3'],training_extension={
                'source':str(base),'source_hashes':{n:digest(base/n) for n in ['manifest.json','train.jsonl','development.jsonl','test.jsonl']}})
            (out/'manifest.json').write_text(json.dumps(manifest));(out/'manifest.sha256').write_text(digest(out/'manifest.json'))
            config={'model':'fixture'};(out/'teacher.json').write_text(json.dumps(config))
            response={'id':'3','input_sha256':candidates[3]['input_sha256'],'teacher':config,
                      'response':json.dumps({'mentions':[{'name':'aspirin','context':candidates[3]['text']}],'coverage_reviewed':True})}
            (out/'responses.jsonl').write_text(json.dumps(response)+'\n')
            result=collect(out)
            self.assertEqual(result['total_training_rows'],2)
            self.assertEqual((out/'development.jsonl').read_bytes(),(base/'development.jsonl').read_bytes())
            self.assertEqual((out/'test.jsonl').read_bytes(),b'')
            self.assertEqual([json.loads(l)['id'] for l in (out/'train.jsonl').read_text().splitlines()],['0','3'])
            seal(out)
            sealed_hash=digest(out/'manifest.json')
            seal(out)
            self.assertEqual(digest(out/'manifest.json'),sealed_hash)
            (base/'train.jsonl').write_text('')
            with self.assertRaisesRegex(ValueError,'source changed'):collect(out)


if __name__=='__main__':unittest.main()
