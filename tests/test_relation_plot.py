import copy
import unittest

from scripts.plot_relation_comparison import aggregates, APPROACHES


class RelationPlotTests(unittest.TestCase):
    def test_export_reads_only_aggregate_counts_in_fixed_order(self):
        report=dict(notes=20,results={name:dict(complete_records=dict(f1=.5),dose=dict(fp=1),route=dict(fp=2),frequency=dict(fp=3)) for name in reversed(APPROACHES)})
        report['results']['rules']['complete_records']['f1']=.7
        self.assertEqual(aggregates(report),[(70.,6),(50.,6),(50.,6),(50.,6)])
        for invalid in (0,None):
            modified=copy.deepcopy(report);modified['notes']=invalid
            with self.assertRaises(ValueError): aggregates(modified)
        modified=copy.deepcopy(report);modified['results']['rules']['complete_records']['f1']=float('nan')
        with self.assertRaises(ValueError): aggregates(modified)
