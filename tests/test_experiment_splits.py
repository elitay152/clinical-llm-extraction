import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts.prepare_experiment import group_notes, build_manifest
from scripts.pseudo_label import load_manifest, label, collect
from scripts.train_distillation import read_splits
from scripts.evaluate_distillation import saved_prompt


def manifest():
    rows = [{'idx': i, 'full_note': ' '.join(f'case{i}_word{j}' for j in range(50)) + f'. Takes drug{i} 5 mg daily.'} for i in range(15)]
    return build_manifest(rows, [rows[0]['full_note'], rows[1]['full_note']], 6, 2, 2, 1800, 42)


def save(root, value):
    path = root/'manifest.json'
    path.write_text(json.dumps(value))
    (root/'manifest.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest())


class ExperimentSplitTests(unittest.TestCase):
    def test_near_duplicate_grouping_is_transitive(self):
        groups = group_notes(['a b c d e f g h', 'a b c d e f g h i', 'b c d e f g h i', 'unrelated tokens belong to another entirely different text'])
        self.assertEqual(groups[0], groups[1])
        self.assertEqual(groups[1], groups[2])
        self.assertNotEqual(groups[0], groups[3])

    def test_split_is_reproducible_and_excludes_seen_sources(self):
        value = manifest()
        self.assertEqual(value, manifest())
        rows = value['candidates']
        self.assertEqual([sum(r['split'] == name for r in rows) for name in ['train','development','test']], [6,2,2])
        self.assertFalse({'acn:0','acn:1'} & {r['id'] for r in rows})
        self.assertEqual(len({r['source_group'] for r in rows}), len(rows))

    def test_locked_test_or_implicit_split_never_calls_teacher(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); save(root, manifest())
            for split in [None, 'test', 'validation']:
                with patch('scripts.pseudo_label.hosted_response') as call:
                    with self.assertRaisesRegex(ValueError, 'locked'):
                        label(root,'model','https://api.example.org','hosted',None,1,20,split)
                    call.assert_not_called()

    def test_checksum_detects_manifest_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); value=manifest();save(root,value)
            load_manifest(root)
            value['candidates'][0]['split']='test'
            (root/'manifest.json').write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError,'changed'):
                load_manifest(root)

    def test_training_cannot_relabel_test_rows_as_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);value=manifest();save(root,value)
            candidate=next(r for r in value['candidates'] if r['split']=='test')
            forged=dict(candidate, split='train', label_origin='teacher_generated')
            (root/'train.jsonl').write_text(json.dumps(forged))
            with self.assertRaisesRegex(ValueError,'membership'):
                read_splits(root)

    def test_saved_prompt_has_integrity_check(self):
        prompt='Return the medication JSON.'
        report={'system_prompt':prompt, 'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest()}
        self.assertEqual(saved_prompt(report),prompt)
        report['system_prompt']='Changed prompt'
        with self.assertRaisesRegex(ValueError,'checksum'):
            saved_prompt(report)

    def test_collect_refuses_test_responses(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);value=manifest();save(root,value)
            candidate=next(r for r in value['candidates'] if r['split']=='test')
            (root/'responses.jsonl').write_text(json.dumps({'id':candidate['id'],'input_sha256':candidate['input_sha256'],'response':'{}','teacher':{}}))
            with self.assertRaisesRegex(ValueError,'Locked test'):
                collect(root)


if __name__ == '__main__':
    unittest.main()
