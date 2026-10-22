import json
import unittest
from scripts.prepare_data import text_hash
from scripts.train_pilot import build_examples


class FakeTokenizer:
    eos_token = '<end>'

    def apply_chat_template(self, messages, **kwargs):
        return messages[0]['content'] + '\nASSISTANT:'

    def __call__(self, text, **kwargs):
        return {'input_ids': list(text.encode())}


class TrainingDataTests(unittest.TestCase):
    def row(self, text, name='aspirin', idx=1):
        return {'idx': idx, 'full_note': text, 'summary': json.dumps({'treatments': [{'name': name}]})}

    def test_reserved_duplicates_and_unsupported_labels_excluded(self):
        rows = [self.row('Reserved aspirin.'), self.row('Take aspirin.'),
                self.row('Take  aspirin.', idx=2), self.row('No medication mentioned.', name='invented')]
        examples, counts = build_examples(rows, {text_hash('Reserved aspirin.')}, FakeTokenizer(), 1000)
        self.assertEqual(len(examples), 1)
        self.assertEqual(counts, {'reserved': 1, 'duplicates': 1, 'unusable': 1, 'too_long': 0})
        example = examples[0]
        self.assertEqual(example['silver_target'], ['aspirin'])
        self.assertTrue(all(n == -100 for n in example['labels'][:len(example['prompt_ids'])]))
        self.assertTrue(all(n != -100 for n in example['labels'][len(example['prompt_ids']):]))
        self.assertEqual(len(example['input_ids']), len(example['labels']))

    def test_overlength_examples_not_truncated(self):
        examples, counts = build_examples([self.row('Take aspirin.')], set(), FakeTokenizer(), 2)
        self.assertEqual(examples, [])
        self.assertEqual(counts['too_long'], 1)


if __name__ == '__main__':
    unittest.main()
