import unittest

from scripts.calibrate_relation_thresholds import accepted, decode, group_folds, pair_metrics, select_threshold


class RelationThresholdTests(unittest.TestCase):
    def test_groups_stay_together_and_assignment_ignores_record_order(self):
        records = [dict(id=str(i), source_group=str(i//2)) for i in range(12)]
        folds = group_folds(records, count=3)
        self.assertEqual(set(folds), {0, 1, 2})
        self.assertTrue(all(folds[i] == folds[i+1] for i in range(0, 12, 2)))
        self.assertEqual(dict(zip((r['id'] for r in records), folds)),
                         dict(zip((r['id'] for r in reversed(records)), group_folds(list(reversed(records)), count=3))))
        for count in (1, 7, True):
            with self.assertRaises(ValueError): group_folds(records, count=count)

    def test_selection_enforces_rule_false_positive_and_precision_constraints(self):
        result = select_threshold([1, 1, 0], [.95, .85, .8], [True, False, False])
        self.assertEqual(result['selected']['threshold'], .95)
        self.assertEqual(result['selected']['metrics']['fp'], 0)
        self.assertEqual(result['selected']['metrics']['tp'], 1)

    def test_abstain_all_when_no_numeric_threshold_meets_constraint(self):
        result = select_threshold([1, 0], [.8, .995], [True, False])
        self.assertIsNone(result['selected']['threshold'])
        self.assertEqual(result['selected']['metrics']['fn'], 1)
        self.assertEqual(result['selected']['metrics']['fp'], 0)

    def test_field_thresholds_allow_shared_links_and_preserve_unlinked_fields(self):
        rows = [dict(kind=0, medication_index=0, attribute_index=0),
                dict(kind=1, medication_index=0, attribute_index=0),
                dict(kind=2, medication_index=0, attribute_index=0),
                dict(kind=2, medication_index=1, attribute_index=0)]
        links = decode(rows, [.7, 1., .95, .95], dict(dose=.9, route=None, frequency=.9))
        self.assertEqual(links['dose'], [])
        self.assertEqual(links['route'], [])
        self.assertEqual(len(links['frequency']), 2)
        self.assertEqual(links['frequency'][1]['medication_index'], 1)
        self.assertFalse(accepted(1., None))
        self.assertTrue(accepted(.9, .9))

    def test_invalid_scores_thresholds_and_lengths_are_rejected(self):
        for probability, threshold in ((float('nan'), .9), (-.1, .9), (1.1, .9), (.5, float('nan')), (.5, 1.1)):
            with self.assertRaises(ValueError): accepted(probability, threshold)
        with self.assertRaises(ValueError): decode([], [], dict(dose=.9))
        with self.assertRaises(ValueError): select_threshold([], [], [])
        with self.assertRaises(ValueError): select_threshold([1], [], [True])
        with self.assertRaises(ValueError): pair_metrics([1, 0], [True])
        with self.assertRaises(ValueError): pair_metrics([None], [False])
