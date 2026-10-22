import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from scripts.pseudo_label import validate_teacher_response, quote_span, collect, PROMPT_VERSION, SYSTEM, select_excerpt


def candidate():
    text = '😀 She takes aspirin 5 mg by mouth daily.'
    return {'id': 'fixture', 'text': text, 'split': 'train', 'source_group': 'fixture-group',
            'input_sha256': hashlib.sha256(text.encode()).hexdigest()}


def answer():
    return {'medications': [{'name': 'aspirin', 'strength': None, 'dose': '5 mg', 'route': 'by mouth',
            'frequency': 'daily', 'status': 'current', 'evidence': [candidate()['text']]}], 'negative_evidence': []}


class PseudoLabelTests(unittest.TestCase):
    def test_evidence_offsets_are_unicode_and_origin_is_not_human(self):
        value = validate_teacher_response(candidate(), json.dumps(answer()))
        self.assertEqual(value['label_origin'], 'teacher_generated')
        self.assertNotIn('annotation_status', value)
        self.assertEqual(value['medications'][0]['evidence'][0]['end'], len(candidate()['text']))
        self.assertEqual(quote_span(candidate()['text'], 'aspirin')['start'], 12)

    def test_hallucinated_attributes_and_names_rejected(self):
        for key, value in [('dose', '100 mg'), ('route', 'intravenous'), ('name', 'invented')]:
            with self.subTest(key=key):
                a = answer(); a['medications'][0][key] = value
                with self.assertRaises(ValueError):
                    validate_teacher_response(candidate(), json.dumps(a))

    def test_dosage_excerpt_uses_source_text_and_records_its_offset(self):
        text = 'Surgical history. ' * 100 + 'The patient takes aspirin 5 mg daily. Follow-up planned.'
        excerpt, start = select_excerpt(text, 900, 'dosage')
        self.assertIn('aspirin 5 mg daily', excerpt)
        self.assertEqual(text[start:start + len(excerpt)], excerpt)
        self.assertGreater(start, 0)
        self.assertIsNone(select_excerpt('No dosage information.', 900, 'dosage'))
        self.assertEqual(select_excerpt(text, 200, 'prefix'), (text[:200], 0))

    def test_ambiguous_evidence_rejected(self):
        with self.assertRaises(ValueError):
            quote_span('aspirin then aspirin', 'aspirin')

    def test_unverified_negative_rejected(self):
        with self.assertRaises(ValueError):
            validate_teacher_response(candidate(), json.dumps({'medications': [], 'negative_evidence': []}))
        c = candidate(); c['text'] = 'No medications.'
        result = validate_teacher_response(c, json.dumps({'medications': [], 'negative_evidence': ['No medications.']}))
        self.assertEqual(result['medications'], [])
        self.assertEqual(result['label_origin'], 'teacher_generated')

    def test_collect_separates_train_validation_and_rejects_bad_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); c = candidate(); v = dict(c, id='validation', split='validation', source_group='other')
            bad = dict(c, id='bad')
            (root / 'manifest.json').write_text(json.dumps({'candidates': [c, v, bad], 'prompt_version': PROMPT_VERSION, 'system_prompt': SYSTEM}))
            rows = [{'id': r['id'], 'input_sha256': r['input_sha256'], 'response': json.dumps(answer()) if r['id'] != 'bad' else '{}', 'teacher': {'model': 'fixture'}} for r in [c, v, bad]]
            (root / 'responses.jsonl').write_text('\n'.join(json.dumps(row) for row in rows))
            report = collect(root)
            self.assertEqual(report['accepted'], 2)
            self.assertEqual(report['rejected'], 1)
            self.assertEqual(json.loads((root / 'train.jsonl').read_text())['id'], 'fixture')
            self.assertEqual(json.loads((root / 'validation.jsonl').read_text())['id'], 'validation')
            rows[0]['input_sha256'] = 'changed'
            (root / 'responses.jsonl').write_text('\n'.join(json.dumps(row) for row in rows))
            with self.assertRaises(ValueError):
                collect(root)


if __name__ == '__main__':
    unittest.main()
