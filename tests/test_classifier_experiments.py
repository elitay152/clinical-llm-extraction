import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.classifier_experiments import make_view, nested_membership, span_errors
from scripts.pseudo_label import load_manifest
from scripts.token_classifier import loss_weights
from scripts.train_distillation import read_splits


class ClassifierExperimentTests(unittest.TestCase):
    def test_class_weight_ablation_is_explicit_and_train_frequency_based(self):
        counts={0:1000,1:10,2:5}
        weights=loss_weights(counts)
        self.assertAlmostEqual(sum(weights),3)
        self.assertLess(weights[0],weights[1])
        self.assertLess(weights[1],weights[2])
        self.assertEqual(loss_weights(counts,'uniform'),[1,1,1])
        with self.assertRaises(ValueError):loss_weights(counts,'unknown')
        with self.assertRaises(ValueError):loss_weights({0:10,1:1,2:0})

    def test_learning_curve_membership_is_nested_and_deterministic(self):
        rows=[{'id':str(i),'split':'train'} for i in range(20)]
        subsets=nested_membership(rows,42)
        self.assertEqual(subsets,nested_membership(rows,42))
        self.assertEqual([len(subsets[f]) for f in [25,50,100]],[5,10,20])
        self.assertLess(set(subsets[25]),set(subsets[50]))
        self.assertLess(set(subsets[50]),set(subsets[100]))
        with self.assertRaises(ValueError):nested_membership(rows+[{'id':'dev','split':'development'}],42)
        with self.assertRaises(ValueError):nested_membership(rows+[rows[0]],42)

    def test_error_analysis_distinguishes_boundaries_and_extra_names(self):
        text='Take oral phosphorus. Review aspirin.'
        record={'id':'fixture','text':text,'name_label_protocol':'medication-name-mentions-v1',
                'name_mentions':[{'start':5,'end':20,'text':'oral phosphorus'}]}
        prediction={'id':'fixture','response':json.dumps({'medication_name_spans':[[3,3],[6,6]]})}
        errors=span_errors([record],[prediction])
        self.assertEqual([e['kind'] for e in errors],['boundary_extra','extra','boundary_missed'])
        with self.assertRaises(ValueError):span_errors([record],[prediction,prediction])

    def test_views_keep_development_frozen_and_reject_test_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source';source.mkdir()
            rows=[]
            for i,split in enumerate(['train','train','train','development','test']):
                text=f'Take aspirin {i}.'
                rows.append({'id':str(i),'split':split,'text':text,'source_group':str(i),
                             'input_sha256':hashlib.sha256(text.encode()).hexdigest()})
            for split in ['train','development','test']:
                (source/f'{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows if r['split']==split and split!='test'))
            manifest={'protocol':'frozen-three-way-v1','test_locked':True,'candidates':rows,
                      'frozen_label_hashes':{n:hashlib.sha256((source/n).read_bytes()).hexdigest()
                                             for n in ['train.jsonl','development.jsonl','test.jsonl']}}
            (source/'manifest.json').write_text(json.dumps(manifest))
            (source/'manifest.sha256').write_text(hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest())
            view=make_view(source,root/'view',['0','2'])
            training,development=read_splits(view)
            self.assertEqual([r['id'] for r in training],['0','2'])
            self.assertEqual([r['id'] for r in development],['3'])
            self.assertEqual((view/'test.jsonl').read_text(),'')
            probe=make_view(source,root/'probe',['0','1'],development=False)
            self.assertEqual(read_splits(probe)[1],[])
            with self.assertRaises(ValueError):make_view(source,root/'bad',['4'])
            with self.assertRaises(ValueError):make_view(source,root/'duplicate',['0','0'])
            (view/'development.jsonl').write_text('')
            with self.assertRaisesRegex(ValueError,'Frozen label snapshot changed'):load_manifest(view)
            self.assertEqual(len(read_splits(source)[1]),1)


if __name__=='__main__':
    unittest.main()
