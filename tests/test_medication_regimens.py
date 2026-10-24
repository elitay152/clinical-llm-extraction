import copy
import json
import unittest

from scripts import medication_regimens, name_spans
from tests.test_medication_doses import fixture as dose_fixture
from tests.test_token_classifier import Tokenizer


def fixture():
    record=dose_fixture('😀 Take aspirin 81 mg by mouth once daily and metformin 500 mg orally twice daily.')
    response={'reviewed_medication_indices':[0,1],'coverage_reviewed':True,
              'routes':[{'medication_index':0,'route_index':0},{'medication_index':1,'route_index':1}],
              'frequencies':[{'medication_index':0,'frequency_index':0},{'medication_index':1,'frequency_index':1}]}
    return medication_regimens.validate_response(record,json.dumps(response))


class MedicationRegimenTests(unittest.TestCase):
    def test_roundtrip_keeps_source_offsets_and_links_each_medication(self):
        record=fixture()
        labels=medication_regimens.word_labels(record)
        encoded,words=medication_regimens.encode_record(record,Tokenizer())
        predictions=[labels[word] if word is not None else 0 for word in words]
        payload=medication_regimens.decode_predictions(words,predictions,len(labels))
        result=medication_regimens.output(record['text'],payload)
        self.assertEqual(result['medications'][0]['routes'][0]['text'],'by mouth')
        self.assertEqual(result['medications'][0]['frequencies'][0]['text'],'once daily')
        self.assertEqual(result['medications'][1]['routes'][0]['text'],'orally')
        self.assertEqual(result['medications'][1]['frequencies'][0]['text'],'twice daily')
        for field in ('medication_names','dose_spans','route_spans','frequency_spans'):
            for span in result[field]:self.assertEqual(record['text'][span['start']:span['end']],span['text'])
        self.assertEqual(medication_regimens.score([record],[payload])['complete_records']['f1'],1)
        self.assertEqual(encoded['labels'].count(-100),3)

    def test_candidate_selection_preserves_frozen_name_and_dose_labels(self):
        record=fixture()
        source=dose_fixture(record['text'])
        for field in ('name_mentions','dose_mentions','dose_relations'):
            self.assertEqual(record[field],source[field])
        marked,names,candidates=medication_regimens.marked_source(source)
        self.assertIn('⟦ROUTE_0:by mouth⟧',marked)
        self.assertIn('⟦FREQUENCY_1:twice daily⟧',marked)
        self.assertEqual(len(names),2)

    def test_abbreviations_and_complete_frequency_phrases_are_candidates(self):
        source=dose_fixture('aspirin 81 mg PO every 6 hours as needed; metformin 500 mg i.v. BID for 7 days.')
        options=medication_regimens.candidates(source)
        self.assertEqual([s['text'] for s in options['route']],['PO','i.v.'])
        self.assertEqual([s['text'] for s in options['frequency']],['every 6 hours as needed','BID'])

    def test_unknown_ids_and_incomplete_reviews_are_rejected(self):
        source=dose_fixture('aspirin 81 mg oral daily and metformin 500 mg IV twice daily.')
        base={'reviewed_medication_indices':[0,1],'coverage_reviewed':True,'routes':[],'frequencies':[]}
        for field,item in [('routes',{'medication_index':0,'route_index':99}),
                           ('routes',{'medication_index':True,'route_index':0}),
                           ('frequencies',{'medication_index':3,'frequency_index':0})]:
            value=dict(base,**{field:[item]})
            with self.assertRaises(ValueError):medication_regimens.validate_response(source,json.dumps(value))
        with self.assertRaises(ValueError):
            medication_regimens.validate_response(source,json.dumps(dict(base,reviewed_medication_indices=[0])))

    def test_attributes_remain_unlinked_for_shared_lists_and_cross_sentence_context(self):
        for text in ['aspirin and metformin 81 mg oral daily and 500 mg',
                     'aspirin 81 mg. Oral daily. metformin 500 mg']:
            source=dose_fixture(text)
            options=medication_regimens.candidates(source)
            names=name_spans.resolve_response(source,json.dumps(name_spans.reference_payload(source)))
            for field in ('route','frequency'):
                links=medication_regimens.link_attributes(text,names,source['dose_mentions'],options['route'],options['frequency'],field)
                self.assertEqual(links,[],(text,field))

    def test_overlap_corrupt_offsets_and_relation_indices_are_rejected(self):
        for kind in ('overlap','text','relation'):
            record=copy.deepcopy(fixture())
            if kind=='overlap':record['route_mentions'][0]=record['name_mentions'][0]
            if kind=='text':record['route_mentions'][0]['text']='by injection'
            if kind=='relation':record['route_relations'][0]['route_index']=5
            with self.assertRaises(ValueError):medication_regimens.reference_spans(record)

    def test_empty_attributes_and_synthetic_negatives(self):
        for record in medication_regimens.synthetic_negatives():
            self.assertEqual(set(medication_regimens.word_labels(record)),{0})
        record=fixture()
        payload=dict(name_spans.reference_payload(record),dose_spans=[],route_spans=[],frequency_spans=[])
        metrics=medication_regimens.score([record],[payload])
        self.assertEqual(metrics['routes']['fn'],2)
        self.assertEqual(metrics['medication_frequency_pairs']['fn'],2)
        self.assertEqual(metrics['complete_records']['tp'],0)
        for label in [9,-1,True]:
            with self.assertRaises(ValueError):medication_regimens.decode_predictions([0],[label],1)

    def test_separate_attribute_head_preserves_frozen_base_predictions(self):
        from scripts import medication_attributes,medication_doses
        record=fixture()
        labels=medication_attributes.word_labels(record)
        encoded,words=medication_attributes.encode_record(record,Tokenizer())
        predictions=[labels[word] if word is not None else 0 for word in words]
        attributes=medication_attributes.decode_predictions(words,predictions,len(labels))
        base=dict(name_spans.reference_payload(record),dose_spans=medication_doses.reference_spans(record))
        original=copy.deepcopy(base)
        merged=medication_attributes.merge_payload(base,attributes)
        self.assertEqual(base,original)
        self.assertEqual(merged['medication_name_spans'],original['medication_name_spans'])
        self.assertEqual(merged['dose_spans'],original['dose_spans'])
        self.assertEqual(medication_regimens.score([record],[merged])['complete_records']['f1'],1)
        with self.assertRaises(ValueError):
            medication_attributes.merge_payload(base,dict(attributes,medication_name_spans=[]))
        for start,end in original['medication_name_spans']+original['dose_spans']:
            self.assertTrue(all(label==0 for label in labels[start-1:end]))

    def test_attribute_validation_rejects_unsupported_spans_and_preserves_base(self):
        from scripts import medication_doses
        record=fixture()
        base=dict(name_spans.reference_payload(record),dose_spans=medication_doses.reference_spans(record))
        spans=medication_regimens.reference_spans(record)
        payload=dict(base,route_spans=spans['route'],frequency_spans=spans['frequency'])
        payload['route_spans']=payload['route_spans']+base['medication_name_spans']
        payload['frequency_spans']=payload['frequency_spans']+base['dose_spans']
        filtered=medication_regimens.validated_payload(record['text'],payload)
        self.assertEqual(filtered['route_spans'],spans['route'])
        self.assertEqual(filtered['frequency_spans'],spans['frequency'])
        self.assertEqual(filtered['dose_spans'],base['dose_spans'])
        self.assertEqual(filtered['medication_name_spans'],base['medication_name_spans'])
        filtered=medication_regimens.validated_payload(record['text'],dict(payload,medication_name_spans=[]))
        self.assertEqual(filtered['route_spans'],[])
        self.assertEqual(filtered['frequency_spans'],[])


class RegimenSnapshotTests(unittest.TestCase):
    def test_preparation_excludes_locked_test_and_detects_source_changes(self):
        import hashlib
        from pathlib import Path
        import tempfile
        from scripts.label_medication_doses import prepare,checked
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'source';source.mkdir()
            rows=[]
            for split,prefix in [('train','Training note: '),('development','Development note: ')]:
                record=dose_fixture(prefix+'aspirin 81 mg oral daily and metformin 500 mg twice daily.')
                record.update(id=split,split=split,source_group=split,label_origin='teacher_generated',
                              input_sha256=hashlib.sha256(record['text'].encode()).hexdigest())
                rows.append(record)
                (source/f'{split}.jsonl').write_text(json.dumps(record)+'\n')
            (source/'test.jsonl').write_text('')
            locked={'id':'test','split':'test','source_group':'test','text':'Locked test note',
                    'input_sha256':hashlib.sha256(b'Locked test note').hexdigest()}
            manifest={'protocol':'frozen-three-way-v1','test_locked':True,'candidates':rows+[locked]}
            manifest['frozen_label_hashes']={f'{s}.jsonl':hashlib.sha256((source/f'{s}.jsonl').read_bytes()).hexdigest() for s in ('train','development','test')}
            (source/'manifest.json').write_text(json.dumps(manifest))
            (source/'manifest.sha256').write_text(hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest())
            output=Path(directory)/'labels'
            self.assertEqual(prepare(source,output,'regimen'),{'train':1,'development':1})
            prepared,selected=checked(output)
            self.assertEqual(set(selected),{'train','development'})
            self.assertNotIn('frozen_label_hashes',prepared)
            self.assertEqual(selected['train']['dose_mentions'],rows[0]['dose_mentions'])
            (source/'train.jsonl').write_text('changed')
            with self.assertRaisesRegex(ValueError,'snapshot changed'):checked(output)

    def test_comparison_requires_identical_frozen_names_and_doses(self):
        from scripts.evaluate_medication_doses import check_reference_compatibility
        reference=fixture()
        reference.update(id='example',input_sha256='source-hash')
        check_reference_compatibility([reference],[copy.deepcopy(reference)])
        changed=copy.deepcopy(reference)
        changed['dose_relations'][0]['medication_index']=1
        with self.assertRaisesRegex(ValueError,'reference labels differ'):
            check_reference_compatibility([reference],[changed])
        changed=copy.deepcopy(reference);changed['input_sha256']='other-source'
        with self.assertRaisesRegex(ValueError,'reference sources differ'):
            check_reference_compatibility([reference],[changed])

    def test_source_candidates_do_not_emit_attributes_without_medications(self):
        payload={'medication_name_spans':[],'dose_spans':[]}
        text='Follow-up every 2 weeks for stage IV disease.'
        result=medication_regimens.candidate_payload(text,payload)
        self.assertEqual(result['route_spans'],[])
        self.assertEqual(result['frequency_spans'],[])


if __name__=='__main__':
    unittest.main()
