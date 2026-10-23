import json
import unittest
from scripts.sentence_evidence import source_sentences, selected_spans, sentence_payload, resolve_response, render_source
from scripts.pseudo_label import validate_teacher_response
from scripts.evaluate_distillation import score


def candidate(text):
    return {'id':'fixture','text':text,'split':'development','source_group':'group','input_sha256':'fixture'}


def medication(ids):
    return {'name':'aspirin','strength':None,'dose':None,'route':None,'frequency':None,
            'status':'current','evidence_sentence_ids':ids}


def response(ids):
    return json.dumps({'medications':[medication(ids)],'negative_evidence_sentence_ids':[]})


class SentenceEvidenceTests(unittest.TestCase):
    def test_unicode_offsets_and_numbering(self):
        text='😀 No allergies. Take aspirin daily.'
        units=source_sentences(text)
        self.assertEqual(len(units),2)
        self.assertIn('[2] Take aspirin daily.',render_source(text,'sentence_ids'))
        for unit in units:
            self.assertEqual(text[unit['start']:unit['end']],unit['text'])
        self.assertEqual(selected_spans(text,[2])[0]['start'],text.index('Take'))

    def test_abbreviations_decimals_and_lines(self):
        text='Dr. Lee prescribed aspirin 2.5 mg daily.\nStopped yesterday.'
        self.assertEqual([u['text'] for u in source_sentences(text)],
                         ['Dr. Lee prescribed aspirin 2.5 mg daily.','Stopped yesterday.'])

    def test_repeated_sentences_resolve_to_the_requested_occurrence(self):
        text='Take aspirin. Take aspirin.'
        parsed=resolve_response(candidate(text),response([2]))
        span=parsed['medications'][0]['evidence'][0]
        self.assertEqual(span['start'],len('Take aspirin. '))
        self.assertEqual(span['text'],'Take aspirin.')

    def test_adjacent_ids_preserve_source_gaps(self):
        text='Take aspirin.\n  Dose is 5 mg.\nReview tomorrow.'
        spans=selected_spans(text,[2,1])
        self.assertEqual(spans,[{'start':0,'end':text.index('\nReview'),'text':text[:text.index('\nReview')]}])
        self.assertEqual(len(selected_spans(text,[1,3])),2)

    def test_invalid_or_duplicate_ids_are_rejected(self):
        for ids in [[0],[99],[True],['1'],[],[1,1]]:
            with self.subTest(ids=ids),self.assertRaises(ValueError):
                resolve_response(candidate('Take aspirin.'),response(ids))

    def test_selected_source_must_support_the_fields(self):
        with self.assertRaisesRegex(ValueError,'name'):
            resolve_response(candidate('No allergies. Take aspirin.'),response([1]))
        payload=json.loads(response([1]));payload['medications'][0]['dose']='500 mg'
        with self.assertRaisesRegex(ValueError,'dose'):
            resolve_response(candidate('Take aspirin.'),json.dumps(payload))

    def test_reference_conversion_preserves_record_scoring(self):
        c=candidate('Take aspirin daily. Review tomorrow.')
        gold=validate_teacher_response(c,json.dumps({'medications':[{
            'name':'aspirin','strength':None,'dose':None,'route':None,'frequency':'daily',
            'status':'current','evidence':['aspirin daily']}],'negative_evidence':[]}))
        payload=sentence_payload(gold)
        self.assertEqual(payload['medications'][0]['evidence_sentence_ids'],[1])
        result=score([gold],[{'id':gold['id'],'response':json.dumps(payload)}],'sentence_ids')
        self.assertEqual(result['schema_and_evidence_valid'],1)
        self.assertEqual(result['exact_medication_record_f1'],1)

    def test_explicit_negative_ids_resolve_without_quotes(self):
        c=candidate('The patient does not take medications.')
        result=resolve_response(c,json.dumps({'medications':[],'negative_evidence_sentence_ids':[1]}))
        self.assertEqual(result['negative_evidence'][0]['text'],c['text'])
        self.assertEqual(result['medications'],[])

    def test_unknown_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'mode'):
            render_source('Take aspirin.','unsupported')
        with self.assertRaisesRegex(ValueError,'mode'):
            score([],[],'unsupported')

    def test_generated_quote_fields_are_not_accepted(self):
        payload=json.loads(response([1]));payload['medications'][0]['evidence']=['invented']
        with self.assertRaisesRegex(ValueError,'schema'):
            resolve_response(candidate('Take aspirin.'),json.dumps(payload))


if __name__=='__main__':
    unittest.main()
