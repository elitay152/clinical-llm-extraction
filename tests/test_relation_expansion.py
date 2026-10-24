import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import expand_relation_training as expansion
from scripts import label_medication_doses as attributes, label_medication_names as names
from scripts.train_distillation import read_splits


def source_fixture(root):
    source = root/'source'; source.mkdir()
    candidates = []; records = []
    for i, split in enumerate(('train', 'development', 'test', 'train', 'train')):
        text = f'Case {i}: Take aspirin 81 mg orally daily.'
        candidate = dict(id=str(i), text=text, split=split, source_group=str(i), input_sha256=hashlib.sha256(text.encode()).hexdigest())
        candidates.append(candidate)
        record = names.validate_response(candidate, json.dumps(dict(mentions=[dict(name='aspirin', context=text)], coverage_reviewed=True)))
        record = attributes.validate_response(record, json.dumps(dict(reviewed_medication_indices=[0], doses=[dict(medication_index=0, dose_index=0)], coverage_reviewed=True)))
        record = attributes.annotation_spec('regimen')[-1](record, json.dumps(dict(reviewed_medication_indices=[0], routes=[dict(medication_index=0, route_index=0)], frequencies=[dict(medication_index=0, frequency_index=0)], coverage_reviewed=True)))
        records.append(record)
    for split in ('train', 'development', 'test'):
        (source/f'{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records[:2] if r['split'] == split))
    manifest = dict(protocol='frozen-three-way-v1', test_locked=True, candidates=candidates,
                    frozen_label_hashes={f'{s}.jsonl': attributes.digest(source/f'{s}.jsonl') for s in ('train','development','test')})
    expansion.write_json(source/'manifest.json', manifest)
    (source/'manifest.sha256').write_text(attributes.digest(source/'manifest.json'))
    return source


def teacher(candidate, model, url, limit, system_prompt):
    if system_prompt == names.PROMPT.read_text():
        response = dict(mentions=[dict(name='aspirin', context=candidate['text'])], coverage_reviewed=True)
    elif system_prompt == attributes.PROMPT.read_text():
        response = dict(reviewed_medication_indices=[0], doses=[dict(medication_index=0, dose_index=0)], coverage_reviewed=True)
    else:
        response = dict(reviewed_medication_indices=[0], routes=[dict(medication_index=0, route_index=0)], frequencies=[dict(medication_index=0, frequency_index=0)], coverage_reviewed=True)
    return json.dumps(response), dict(returned_model='fixture')


class RelationExpansionTests(unittest.TestCase):
    def test_annotation_resume_and_frozen_evaluation_preservation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = source_fixture(root); selected = root/'selected'; output = root/'output'
            expansion.prepare(source, selected, 2)
            self.assertEqual({r['id'] for r in expansion.checked(selected)['candidates']}, {'3', '4'})
            with patch.object(expansion, 'hosted_response', side_effect=teacher) as call:
                expansion.annotate(selected)
                expansion.annotate(selected)
                self.assertEqual(call.call_count, 6)
            result = expansion.collect(selected, output)
            self.assertEqual(result['training_notes'], 3)
            for split in ('development', 'test'):
                self.assertEqual((source/f'{split}.jsonl').read_bytes(), (output/f'{split}.jsonl').read_bytes())
            self.assertEqual(len(read_splits(output)[0]), 3)
            rows = [json.loads(s) for s in (selected/'annotations.jsonl').read_text().splitlines()]
            rows[0]['record']['route_relations'] = []
            (selected/'annotations.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with self.assertRaisesRegex(ValueError, 'validated teacher'):
                expansion.collect(selected, root/'tampered')

    def test_tampered_selection_or_base_labels_fail_before_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = source_fixture(root); selected = root/'selected'
            expansion.prepare(source, selected, 1)
            plan = json.loads((selected/'selection.json').read_text())
            manifest = json.loads((source/'manifest.json').read_text())
            plan['candidates'] = [manifest['candidates'][1]]
            expansion.write_json(selected/'selection.json', plan)
            (selected/'selection.sha256').write_text(attributes.digest(selected/'selection.json'))
            with self.assertRaisesRegex(ValueError, 'training-only'):
                expansion.checked(selected)
            (source/'train.jsonl').write_text('')
            with self.assertRaisesRegex(ValueError, 'Source snapshot'):
                expansion.checked(selected)

    def test_priority_is_deterministic_and_count_is_bounded(self):
        candidates = [dict(id='simple', text='aspirin 81 mg', source_group='a'),
                      dict(id='complex', text='aspirin (acetylsalicylic acid) and metformin 500 mg; oral cavity.', source_group='b')]
        self.assertEqual(expansion.select_targeted(candidates, 1)[0]['id'], 'complex')
        self.assertEqual(expansion.select_targeted(candidates, 2), expansion.select_targeted(list(reversed(candidates)), 2))
        for count in (0, 3, True):
            with self.assertRaises(ValueError): expansion.select_targeted(candidates, count)

    def test_failed_source_validation_is_excluded_and_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = source_fixture(root); selected = root/'selected'; output = root/'output'
            expansion.prepare(source, selected, 2)
            def sometimes_invalid(candidate, *args, **kwargs):
                if 'Case 4:' in candidate['text']:
                    return '{"mentions":[{"name":"invented","context":"absent"}],"coverage_reviewed":true}', {}
                return teacher(candidate, *args, **kwargs)
            with patch.object(expansion, 'hosted_response', side_effect=sometimes_invalid):
                expansion.annotate(selected)
            report = expansion.collect(selected, output)
            self.assertEqual(report['additions'], 1)
            self.assertEqual([r['id'] for r in report['rejected']], ['4'])
            self.assertEqual(len(read_splits(output)[0]), 2)
