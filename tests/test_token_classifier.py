import unittest

from scripts.token_classifier import word_labels, encode_record, decode_predictions
from scripts.name_spans import resolve_response
from tests.test_distillation import fixture
import json


class Encoded(dict):
    def __init__(self, words):
        self.words = words
        super().__init__(input_ids=list(range(len(words))), attention_mask=[1]*len(words))
    def word_ids(self):
        return self.words


class Tokenizer:
    def __call__(self, words, **kwargs):
        ids = [None]
        for index, word in enumerate(words):
            ids.extend([index]*(2 if word == 'aspirin' else 1))
        return Encoded(ids+[None])


class TokenClassifierTests(unittest.TestCase):
    def test_subwords_and_specials_are_masked(self):
        record, _ = fixture()
        encoded, words = encode_record(record, Tokenizer())
        self.assertEqual(words, [None,0,1,1,2,3,None])
        self.assertEqual(encoded['labels'], [-100,0,1,-100,0,0,-100])
        self.assertNotIn('labels', encode_record(record, Tokenizer(), labeled=False)[0])

    def test_decoding_copies_exact_unicode_and_whitespace(self):
        record = {'text': '😀 oral  phosphorus supplements.'}
        words = [None,0,1,2,2,3,4,None]
        payload = decode_predictions(words, [0,0,1,2,0,2,0,0], 5)
        self.assertEqual(payload, {'medication_name_spans': [[2,4]]})
        spans = resolve_response(record, json.dumps(payload))
        self.assertEqual(spans, [{'start':2,'end':30,'text':'oral  phosphorus supplements'}])

    def test_orphan_i_and_adjacent_b_are_deterministic(self):
        self.assertEqual(decode_predictions([0,1,2,3,4],[0,2,2,1,0],5),
                         {'medication_name_spans':[[2,3],[4,4]]})
        self.assertEqual(decode_predictions([0,1],[1,1],2),
                         {'medication_name_spans':[[1,1],[2,2]]})
        self.assertEqual(decode_predictions([0,1],[0,0],2), {'medication_name_spans':[]})

    def test_missing_tokens_unknown_labels_and_truncation_are_rejected(self):
        record, _ = fixture()
        with self.assertRaises(ValueError):
            encode_record(record, Tokenizer(), max_length=2)
        for words, predictions, count in [([0],[0,1],1), ([0,2],[1,0],3), ([0],[3],1)]:
            with self.assertRaises(ValueError):
                decode_predictions(words,predictions,count)
        class OmittingTokenizer:
            def __call__(self, words, **kwargs):
                return Encoded([None,0,None])
        with self.assertRaises(ValueError):
            encode_record(record,OmittingTokenizer())

    def test_multiword_targets_and_overlaps(self):
        text = 'oral phosphorus supplements and aspirin'
        evidence = [{'start':0,'end':len(text),'text':text}]
        record = {'text':text,'medications':[{'name':'oral phosphorus supplements','evidence':evidence},
                                           {'name':'aspirin','evidence':evidence}]}
        self.assertEqual(word_labels(record), [1,2,2,0,1])
        record['medications'].append({'name':'phosphorus','evidence':evidence})
        with self.assertRaises(ValueError):
            word_labels(record)


if __name__ == '__main__':
    unittest.main()
