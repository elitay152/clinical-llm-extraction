import json
import unittest

from scripts.name_spans import render_source, reference_payload, resolve_response, score, synthetic_negatives
from scripts.train_distillation import encode_record
from tests.test_distillation import Tokenizer, fixture


class NameSpanTests(unittest.TestCase):
    def test_unicode_punctuation_and_repeated_mentions(self):
        text = '😀 aspirin, then aspirin.'
        record = {'text': text, 'medications': [{'name': 'aspirin', 'evidence': [
            {'start': 0, 'end': len(text), 'text': text}]}]}
        payload = reference_payload(record)
        self.assertEqual(payload, {'medication_name_spans': [[2, 2], [5, 5]]})
        spans = resolve_response(record, json.dumps(payload))
        self.assertEqual([s['text'] for s in spans], ['aspirin', 'aspirin'])
        self.assertNotEqual(spans[0]['start'], spans[1]['start'])
        self.assertIn('2:aspirin', render_source(text))

    def test_only_evidence_attested_occurrences_become_targets(self):
        text = 'aspirin and aspirin'
        record = {'text': text, 'medications': [{'name': 'aspirin', 'evidence': [
            {'start': 12, 'end': 19, 'text': 'aspirin'}]}]}
        self.assertEqual(reference_payload(record)['medication_name_spans'], [[3, 3]])

    def test_substrings_and_unaligned_names_are_not_silently_changed(self):
        text = 'metformin'
        record = {'text': text, 'medications': [{'name': 'formin', 'evidence': [
            {'start': 0, 'end': len(text), 'text': text}]}]}
        with self.assertRaises(ValueError):
            reference_payload(record)

    def test_ranges_and_schema_are_strict(self):
        for pairs in [[[0, 1]], [[1, 99]], [[2, 1]], [[True, 1]], [[1.0, 1]],
                      [[1]], [[1, 1], [1, 1]], 'aspirin']:
            with self.subTest(pairs=pairs), self.assertRaises(ValueError):
                resolve_response({'text': 'Take aspirin.'}, json.dumps({'medication_name_spans': pairs}))
        with self.assertRaises(ValueError):
            resolve_response({'text': 'aspirin'}, '{"medication_name_spans": [], "dose": "5mg"}')

    def test_empty_output_and_invalid_predictions_score_differently(self):
        record, _ = fixture()
        good = {'id': record['id'], 'response': json.dumps(reference_payload(record))}
        self.assertEqual(score([record], [good])['exact_name_spans']['f1'], 1)
        bad = dict(good, response='{}')
        self.assertEqual(score([record], [bad])['schema_and_spans_valid'], 0)
        self.assertEqual(score([record], [bad])['exact_name_spans']['fn'], 1)
        negative = dict(record, medications=[])
        empty = dict(good, response='{"medication_name_spans": []}')
        self.assertEqual(score([negative], [empty])['exact_document_agreement'], 1)
        self.assertEqual(score([negative], [bad])['exact_document_agreement'], 0)
        with self.assertRaises(ValueError):
            score([record], [good, good])

    def test_synthetic_negatives_are_train_only_and_not_teacher_labels(self):
        for record in synthetic_negatives():
            self.assertEqual(record['split'], 'train')
            self.assertEqual(record['label_origin'], 'synthetic')
            self.assertEqual(reference_payload(record), {'medication_name_spans': []})
            self.assertIsNotNone(encode_record(record, Tokenizer(), 10000, 'Select names.', 'name_spans'))
        record = dict(synthetic_negatives()[0], text='Take aspirin.')
        with self.assertRaises(ValueError):
            encode_record(record, Tokenizer(), 10000, 'Select names.', 'name_spans')
        with self.assertRaises(ValueError):
            encode_record(synthetic_negatives()[0], Tokenizer(), 10000, 'Select names.', 'quotes')

    def test_training_masks_prompt_and_deduplicates_status_records(self):
        record, _ = fixture()
        record['medications'] *= 2
        self.assertEqual(len(reference_payload(record)['medication_name_spans']), 1)
        encoded = encode_record(record, Tokenizer(), 10000, 'Select names.', 'name_spans')
        boundary = len(encoded['prompt_ids'])
        self.assertTrue(all(v == -100 for v in encoded['labels'][:boundary]))
        answer = bytes(encoded['labels'][boundary:]).decode()
        self.assertIn('medication_name_spans', answer)
        self.assertNotIn('daily', answer)


if __name__ == '__main__':
    unittest.main()
