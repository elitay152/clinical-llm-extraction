import copy
import json
import unittest

from scripts import medication_doses, name_spans, token_classifier
from scripts.label_medication_doses import validate_response
from tests.test_token_classifier import Tokenizer


def fixture(text='😀 aspirin 81 mg and metformin 500 mg.'):
    def span(value):
        start=text.index(value)
        return {'start':start,'end':start+len(value),'text':value}
    return {'text':text,'name_mentions':[span('aspirin'),span('metformin')],
            'name_label_protocol':name_spans.NAME_PROTOCOL,
            'dose_label_protocol':medication_doses.PROTOCOL,
            'dose_mentions':[span('81 mg'),span('500 mg' if '500 mg' in text else '500mg')],
            'dose_relations':[{'medication_index':0,'dose_index':0},{'medication_index':1,'dose_index':1}]}


class MedicationDoseTests(unittest.TestCase):
    def test_bio_alignment_preserves_unicode_and_subword_masking(self):
        record=fixture()
        encoded, words=medication_doses.encode_record(record,Tokenizer())
        expected=medication_doses.word_labels(record)
        self.assertEqual(expected,[0,1,3,4,0,1,3,4,0])
        self.assertEqual(encoded['labels'].count(-100),3)
        predicted=[0 if word is None else expected[word] for word in words]
        payload=medication_doses.decode_predictions(words,predicted,len(expected))
        output=medication_doses.output(record['text'],payload)
        self.assertEqual(output['medication_names'],record['name_mentions'])
        self.assertEqual(output['dose_spans'],record['dose_mentions'])
        self.assertEqual(output['medications'][1]['doses'],[record['dose_mentions'][1]])
        self.assertEqual(output['unlinked_doses'],[])
        self.assertEqual(medication_doses.score([record],[payload])['medication_dose_pairs']['f1'],1)

    def test_reference_offsets_overlap_and_relations_are_checked(self):
        for change in ('wrong_text','overlap','wrong_link','duplicate_link'):
            record=copy.deepcopy(fixture())
            if change=='wrong_text': record['dose_mentions'][0]['text']='82 mg'
            if change=='overlap': record['dose_mentions'][0]=record['name_mentions'][0]
            if change=='wrong_link': record['dose_relations'][0]['medication_index']=2
            if change=='duplicate_link': record['dose_relations'].append(record['dose_relations'][0])
            with self.assertRaises(ValueError,msg=change):
                medication_doses.reference_spans(record)

    def test_source_selections_keep_name_labels_and_copy_exact_amounts(self):
        record=fixture('aspirin 81 mg and metformin 500mg.')
        value={'reviewed_medication_indices':[0,1],'coverage_reviewed':True,
               'doses':[{'medication_index':0,'dose_index':0},
                        {'medication_index':1,'dose_index':1}]}
        result=validate_response(record,json.dumps(value))
        self.assertEqual(result['name_mentions'],record['name_mentions'])
        self.assertEqual([s['text'] for s in result['dose_mentions']],['81 mg','500mg'])
        value['reviewed_medication_indices']=[0]
        with self.assertRaises(ValueError): validate_response(record,json.dumps(value))

    def test_invented_candidates_unknown_names_and_duplicate_links_are_rejected(self):
        record=fixture()
        base={'reviewed_medication_indices':[0,1],'coverage_reviewed':True}
        for item in [{'medication_index':0,'dose_index':5}, {'medication_index':2,'dose_index':0},
                     {'medication_index':True,'dose_index':0}, {'medication_index':0,'dose':'80 mg'}]:
            with self.assertRaises(ValueError): validate_response(record,json.dumps(dict(base,doses=[item])))
        with self.assertRaises(ValueError):
            validate_response(record,json.dumps(dict(base,doses=[{'medication_index':0,'dose_index':0}]*2)))

    def test_candidate_units_preserve_weight_normalization_and_exclude_rates(self):
        text='1 mg/kg, 50 mg/m2, 500mg, two tablets; 10 mg/ml; 2 g/day; 5 mg/hour.'
        self.assertEqual([s['text'] for s in medication_doses.dose_candidates(text)],
                         ['1 mg/kg','50 mg/m2','500mg','two tablets'])
        marked,names,candidates=medication_doses.marked_source(fixture())
        self.assertIn('⟦MED_0:aspirin⟧',marked)
        self.assertIn('⟦DOSE_1:500 mg⟧',marked)

    def test_linker_abstains_on_shared_lists_and_cross_sentence_assignments(self):
        for text,names,dose,expected in [
            ('aspirin and warfarin 5 mg.', ['aspirin','warfarin'], '5 mg', []),
            ('aspirin stopped. 5 mg', ['aspirin'], '5 mg', []),
            ('aspirin 5 mg warfarin', ['aspirin','warfarin'], '5 mg', []),
            ('5 mg of warfarin', ['warfarin'], '5 mg', [{'medication_index':0,'dose_index':0}])]:
            def span(value):
                start=text.index(value)
                return {'start':start,'end':start+len(value),'text':value}
            self.assertEqual(medication_doses.link_doses(text,[span(n) for n in names],[span(dose)]),expected)

    def test_explicit_dose_introductions_and_routes_link_the_correct_name(self):
        for text in ['warfarin at a dose of 5 mg', 'warfarin in the dose of 5 mg',
                     'warfarin in the dose 5 mg', 'warfarin at a maximal dosage of 5 mg',
                     'warfarin, dose reduced to 5 mg', 'warfarin dosage was 5 mg',
                     'warfarin orally at a dose of 5 mg', '5 mg of oral warfarin',
                     '5 mg daily oral warfarin', '5 mg intravenous warfarin',
                     'warfarin at a dose of 5 mg was not started']:
            def span(value):
                start=text.index(value)
                return {'start':start,'end':start+len(value),'text':value}
            names,doses=[span('warfarin')],[span('5 mg')]
            expected=[{'medication_index':0,'dose_index':0}]
            self.assertEqual(medication_doses.link_doses(text,names,doses),expected,text)
            self.assertEqual(medication_doses.link_doses(text,names,doses,rule='conservative-local-rule-v1'),[],text)

    def test_expanded_connectors_do_not_cross_unrelated_clauses(self):
        for text in ['warfarin was stopped. Start 5 mg', 'warfarin was stopped; 5 mg',
                     'warfarin is unrelated to 5 mg', 'warfarin\n5 mg']:
            def span(value):
                start=text.index(value)
                return {'start':start,'end':start+len(value),'text':value}
            self.assertEqual(medication_doses.link_doses(text,[span('warfarin')],[span('5 mg')]),[],text)
        with self.assertRaises(ValueError):
            medication_doses.link_doses('',[],[],rule='unknown')

    def test_rule_comparison_changes_links_without_changing_entity_scores(self):
        record=fixture('aspirin at a dose of 81 mg and metformin 500 mg.')
        payload=dict(name_spans.reference_payload(record),dose_spans=medication_doses.reference_spans(record))
        before=medication_doses.score([record],[payload],linking_rule='conservative-local-rule-v1')
        after=medication_doses.score([record],[payload])
        self.assertEqual(before['names'],after['names'])
        self.assertEqual(before['doses'],after['doses'])
        self.assertEqual(before['medication_dose_pairs']['fn'],1)
        self.assertEqual(after['medication_dose_pairs']['f1'],1)

    def test_unlinked_doses_and_missing_attributes_remain_explicit(self):
        record=fixture()
        payload={'medication_name_spans':[[2,2],[6,6]],'dose_spans':[]}
        output=medication_doses.output(record['text'],payload)
        self.assertTrue(all(m['doses']==[] for m in output['medications']))
        self.assertEqual(medication_doses.score([record],[payload])['doses']['fn'],2)
        for label in [-1,5,True]:
            with self.assertRaises(ValueError): medication_doses.decode_predictions([0],[label],1)

    def test_five_class_weights_and_synthetic_negatives(self):
        weights=token_classifier.loss_weights({i:i+1 for i in range(5)},num_labels=5)
        self.assertAlmostEqual(sum(weights),5)
        for row in medication_doses.synthetic_negatives():
            self.assertTrue(all(label==0 for label in medication_doses.word_labels(row)))


if __name__=='__main__':
    unittest.main()
