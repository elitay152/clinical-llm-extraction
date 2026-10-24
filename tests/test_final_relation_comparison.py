import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import final_relation_comparison as final, medication_relations as relations
from scripts import expand_relation_training as expansion
from tests.test_relation_expansion import source_fixture, teacher


class FinalComparisonTests(unittest.TestCase):
    def test_frozen_inputs_and_plan_changes_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); source=source_fixture(root)
            encoder=root/'encoder'; encoder.mkdir(); (encoder/'model.safetensors').write_bytes(b'encoder')
            (encoder/'run.json').write_text(json.dumps(dict(training_ids=['0'],source_groups=['0'])))
            runs=[]
            for name in ('original','expanded'):
                run=root/name; run.mkdir(); (run/'model.safetensors').write_bytes(name.encode())
                (run/'run.json').write_text(json.dumps(dict(encoder_sha256=relations.fingerprint(encoder/'model.safetensors'),
                    feature_code_sha256=relations.fingerprint(Path(relations.__file__)),train_sha256=relations.fingerprint(source/'train.jsonl'),config=dict(threshold=.5))))
                runs.append(run)
            calibration=root/'calibration';calibration.mkdir()
            (calibration/'thresholds.json').write_text(json.dumps(dict(thresholds=dict(dose=.9,route=None,frequency=.9))))
            (calibration/'plan.json').write_text(json.dumps(dict(inputs={'model.safetensors':relations.fingerprint(runs[1]/'model.safetensors')})))
            output=root/'final'
            result=final.freeze(source,encoder,*runs,calibration,output,source)
            self.assertEqual(result['test_notes'],1)
            self.assertEqual(final.checked(output)['approaches'],['rules','original','expanded','abstention'])
            with patch.object(expansion,'hosted_response',side_effect=teacher) as call:
                final.label(output);final.label(output)
                self.assertEqual(call.call_count,3)
            rows=[json.loads(s) for s in (output/'annotations.jsonl').read_text().splitlines()]
            self.assertEqual(len(final.references(final.checked(output),rows)),1)
            (encoder/'model.safetensors').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'inputs changed'): final.checked(output)
            (output/'plan.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'plan changed'): final.checked(output)

    def test_missing_failed_and_tampered_references_block_scoring(self):
        with tempfile.TemporaryDirectory() as directory:
            source=source_fixture(Path(directory))
            candidate=next(r for r in json.loads((source/'manifest.json').read_text())['candidates'] if r['split']=='test')
            plan=dict(candidates=[candidate])
            with patch.object(expansion,'hosted_response',side_effect=teacher):
                row=expansion.annotate_record(candidate,dict(model='fixture',base_url='https://example.test',max_tokens=100,max_attempts=3))
            self.assertEqual(final.references(plan,[row])[0]['split'],'test')
            for rows in ([],[row,row],[dict(row,record=None)]):
                with self.assertRaises(ValueError): final.references(plan,rows)
            row['record']['frequency_relations']=[]
            with self.assertRaisesRegex(ValueError,'Stored reference'): final.references(plan,[row])
