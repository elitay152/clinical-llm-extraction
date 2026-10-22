import json
from pathlib import Path
import tempfile
import unittest
from scripts.train_distillation import encode_record, read_splits
from scripts.evaluate_distillation import score
from scripts.pseudo_label import validate_teacher_response


class Tokenizer:
    eos_token = '<end>'
    def apply_chat_template(self, messages, **kwargs):
        return messages[0]['content'] + messages[1]['content']
    def __call__(self, text, **kwargs):
        return {'input_ids': list(text.encode())}


def fixture():
    text = 'Take aspirin daily.'
    candidate = {'id': 'test', 'text': text, 'input_sha256': 'fixture', 'split': 'train', 'source_group': 'group'}
    response = json.dumps({'medications': [{'name': 'aspirin', 'strength': None, 'dose': None,
                          'route': None, 'frequency': 'daily', 'status': 'current', 'evidence': [text]}],
                           'negative_evidence': []})
    return validate_teacher_response(candidate, response), response


class DistillationTests(unittest.TestCase):
    def test_training_masks_prompt_and_rejects_validation(self):
        record, _ = fixture()
        encoded = encode_record(record, Tokenizer(), 10000)
        self.assertTrue(all(n == -100 for n in encoded['labels'][:len(encoded['prompt_ids'])]))
        self.assertTrue(all(n >= 0 for n in encoded['labels'][len(encoded['prompt_ids']):]))
        self.assertIsNone(encode_record(record, Tokenizer(), 10))
        record['split'] = 'validation'
        with self.assertRaises(ValueError):
            encode_record(record, Tokenizer(), 10000)

    def test_split_overlap_rejected(self):
        record, _ = fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / 'train.jsonl').write_text(json.dumps(record))
            record = dict(record, id='validation', split='validation')
            (path / 'validation.jsonl').write_text(json.dumps(record))
            with self.assertRaises(ValueError):
                read_splits(path)

    def test_teacher_agreement_counts_invalid_predictions_as_misses(self):
        record, response = fixture()
        exact = score([record], [{'id': record['id'], 'response': response}])
        self.assertEqual(exact['exact_medication_record_f1'], 1)
        invalid = score([record], [{'id': record['id'], 'response': '{}'}])
        self.assertEqual(invalid['exact_medication_record_recall'], 0)
        self.assertEqual(invalid['schema_and_evidence_valid'], 0)
        with self.assertRaises(ValueError):
            score([], [])
