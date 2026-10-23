import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts import name_spans
from scripts.label_medication_names import validate_response, prepare, checked_manifest, label
from scripts.train_distillation import read_splits


def candidate(text):
    return {'id':'fixture','text':text,'split':'train','source_group':'group',
            'input_sha256':hashlib.sha256(text.encode()).hexdigest()}


def answer(items):
    return json.dumps({'mentions':[{'name':name,'context':context} for name,context in items],
                       'coverage_reviewed':True})


class NameLabelingTests(unittest.TestCase):
    def test_repeated_mentions_and_unicode_offsets(self):
        text = '😀 Take aspirin. Stop aspirin.'
        record = validate_response(candidate(text),answer([('aspirin','Take aspirin.'),('aspirin','Stop aspirin.')]))
        self.assertEqual([s['text'] for s in record['name_mentions']],['aspirin','aspirin'])
        self.assertEqual(record['name_mentions'][0]['start'],text.index('aspirin'))
        payload = name_spans.reference_payload(record)
        self.assertEqual(len(payload['medication_name_spans']),2)
        self.assertNotIn('medications',record)

    def test_brand_generic_and_multiword_chemical_names(self):
        text = 'Selo-zok® (metoprolol) and ferrous sulfate were stopped.'
        record = validate_response(candidate(text),answer([(n,text) for n in ['Selo-zok','metoprolol','ferrous sulfate']]))
        self.assertEqual([s['text'] for s in record['name_mentions']],['Selo-zok','metoprolol','ferrous sulfate'])

    def test_one_context_can_cover_repetition_without_duplicate_spans(self):
        text = 'Take aspirin and stop aspirin tomorrow.'
        record = validate_response(candidate(text),answer([('aspirin',text),('aspirin',text)]))
        self.assertEqual(len(record['name_mentions']),2)

    def test_boundary_noise_and_inferred_substrings_are_rejected(self):
        names = ['tablet alprazolam','oral phosphorus supplements','aspirin 5 mg','Selo-zok®',
                 '(metoprolol)','Lignocaine with adrenaline','heparin']
        for name in names:
            text = name+'.' if name != 'heparin' else 'He was heparinized.'
            with self.subTest(name=name),self.assertRaises(ValueError):
                validate_response(candidate(text),answer([(name,text)]))

    def test_coverage_flag_negatives_and_ambiguous_context(self):
        c=candidate('Laboratory phosphorus 1.1 mg/dL.')
        self.assertEqual(validate_response(c,answer([]))['name_mentions'],[])
        with self.assertRaises(ValueError):
            validate_response(c,'{"mentions":[],"coverage_reviewed":false}')
        with self.assertRaises(ValueError):
            validate_response(candidate('aspirin then aspirin'),answer([('aspirin','aspirin')]))

    def test_reference_span_corruption_and_overlap_rejected(self):
        text='Take sodium chloride.'
        record=validate_response(candidate(text),answer([('sodium chloride',text)]))
        record['name_mentions'][0]['text']='chloride'
        with self.assertRaises(ValueError):name_spans.reference_payload(record)
        with self.assertRaises(ValueError):
            validate_response(candidate(text),answer([('sodium chloride',text),('chloride',text)]))

    def test_prepare_preserves_membership_and_never_selects_test(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source';source.mkdir();run=root/'run';run.mkdir()
            rows=[dict(candidate(f'Take aspirin {i}.'),id=str(i),split=split,source_group=str(i))
                  for i,split in enumerate(['train','train','development','test'])]
            manifest={'protocol':'frozen-three-way-v1','test_locked':True,'candidates':rows}
            raw=json.dumps(manifest);(source/'manifest.json').write_text(raw)
            (source/'manifest.sha256').write_text(hashlib.sha256(raw.encode()).hexdigest())
            for split in ['train','development','test']:
                (source/f'{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows if r['split']==split and split!='test'))
            report={'training_ids':['0','1','synthetic:fixture'],
                    'train_file_sha256':hashlib.sha256((source/'train.jsonl').read_bytes()).hexdigest()}
            (run/'run.json').write_text(json.dumps(report))
            output=root/'names';result=prepare(source,run,output)
            self.assertEqual(result['selected'],3)
            revised,_=checked_manifest(output)
            self.assertEqual(revised['candidates'],rows)
            self.assertNotIn('3',revised['labeling_ids'])
            self.assertEqual(read_splits(output),([],[]))
            with self.assertRaises(ValueError):label(output,'test','fixture','https://example.test',1,100)
            revised['labeling_ids'].append('3')
            (output/'manifest.json').write_text(json.dumps(revised))
            (output/'manifest.sha256').write_text(hashlib.sha256((output/'manifest.json').read_bytes()).hexdigest())
            with self.assertRaises(ValueError):checked_manifest(output)


if __name__=='__main__':
    unittest.main()
