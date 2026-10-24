import copy
import unittest

from scripts import medication_relations as relations
from scripts.train_relation_classifier import score, validate_config
from tests.test_medication_regimens import fixture


class RelationTests(unittest.TestCase):
    def test_reference_candidates_preserve_all_positive_links(self):
        record = fixture()
        payload = relations.reference_payload(record, candidates=True)
        rows = relations.pairs(record['text'], payload)
        targets = relations.targets(record, payload, rows)
        self.assertEqual(sum(t == 1 for t in targets), 6)
        self.assertTrue(all(len(r['layout']) == len(relations.LAYOUT) for r in rows))
        self.assertEqual(score([record], [payload], [relations.rule_links(record['text'], payload)])['complete_records']['f1'], 1)

    def test_ambiguous_attribute_is_not_a_negative(self):
        record = fixture()
        record['route_relations'] = [r for r in record['route_relations'] if r['route_index'] != 0]
        payload = relations.reference_payload(record, candidates=True)
        rows = relations.pairs(record['text'], payload)
        targets = relations.targets(record, payload, rows)
        self.assertEqual([t for r, t in zip(rows, targets) if r['kind'] == 1 and r['attribute_index'] == 0], [None, None])

    def test_decode_supports_shared_links_and_abstention(self):
        record = fixture(); payload = relations.reference_payload(record)
        rows = relations.pairs(record['text'], payload)
        probabilities = [float(r['kind'] == 2 and r['attribute_index'] == 0) for r in rows]
        links = relations.decode(rows, probabilities)
        self.assertEqual(len(links['frequency']), 2)
        self.assertEqual(links['dose'], [])
        self.assertEqual(payload, relations.reference_payload(record))
        for bad in ([float('nan')] * len(rows), [-1] * len(rows), []):
            with self.assertRaises(ValueError): relations.decode(rows, bad)

    def test_pair_limit_and_missing_positive_fail(self):
        record = fixture(); payload = relations.reference_payload(record)
        with self.assertRaises(ValueError): relations.pairs(record['text'], payload, limit=1)
        payload = copy.deepcopy(payload); payload['dose_spans'] = []
        with self.assertRaises(ValueError): relations.targets(record, payload, relations.pairs(record['text'], payload))

    def test_empty_entities_have_no_relations(self):
        payload = dict(medication_name_spans=[], dose_spans=[], route_spans=[], frequency_spans=[])
        self.assertEqual(relations.pairs('No medications.', payload), [])
        self.assertEqual(relations.decode([], []), dict(dose=[], route=[], frequency=[]))

    def test_scoring_detects_wrong_and_missing_links(self):
        record = fixture(); payload = relations.reference_payload(record)
        links = relations.rule_links(record['text'], payload)
        links['dose'][0]['medication_index'] = 1
        links['frequency'] = []
        result = score([record], [payload], [links])
        self.assertEqual((result['dose']['tp'], result['dose']['fp'], result['dose']['fn']), (1, 1, 1))
        self.assertEqual(result['frequency']['fn'], 2)
        self.assertEqual(result['complete_records']['tp'], 0)

    def test_configuration_rejects_invalid_training_parameters(self):
        from pathlib import Path
        import json
        config = json.loads(Path('configs/relation-classifier.json').read_text())
        validate_config(config)
        for key, value in [('epochs', 0), ('batch_size', True), ('threshold', 1), ('dropout', -1), ('learning_rate', 0)]:
            with self.assertRaises(ValueError): validate_config(dict(config, **{key: value}))
