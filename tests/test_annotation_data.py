import copy
import unittest
from scripts.annotation_data import validate_document, validate_workspace, FORMAT
from scripts.prepare_annotation_pilot import canonical_text


def document():
    text = '😀 Take aspirin 5 mg daily.'
    start = text.index('aspirin')
    return {'id': 'example', 'text': text, 'annotation_status': 'draft', 'medications': [
        {'name': 'aspirin', 'strength': None, 'dose': '5 mg', 'route': None,
         'frequency': 'daily', 'status': 'current',
         'evidence': [{'start': start, 'end': len(text), 'text': text[start:]}]}]}


def workspace():
    return {'format': FORMAT, 'role': 'development_pilot', 'protocol_version': '1',
            'documents': [document()], 'metadata': {'example': {'annotator': 'test'}},
            'provenance': {'example': {'source': 'fixture'}}}


class AnnotationTests(unittest.TestCase):
    def test_unicode_evidence_uses_codepoints(self):
        doc = document()
        self.assertEqual(doc['medications'][0]['evidence'][0]['start'], 7)
        self.assertIs(validate_document(doc), doc)

    def test_invalid_offsets_and_unsupported_name_fail(self):
        for change in ('offset', 'boolean', 'name'):
            with self.subTest(change=change):
                doc = document()
                if change == 'offset':
                    doc['medications'][0]['evidence'][0]['start'] += 1
                elif change == 'boolean':
                    doc['medications'][0]['evidence'][0]['start'] = True
                else:
                    doc['medications'][0]['name'] = 'invented'
                with self.assertRaises(ValueError):
                    validate_document(doc)

    def test_reviewed_negative_requires_explicit_confirmations(self):
        w = workspace()
        w['documents'][0].update(annotation_status='reviewed', medications=[])
        with self.assertRaises(ValueError):
            validate_workspace(w)
        w['metadata']['example']['review_confirmed'] = True
        with self.assertRaises(ValueError):
            validate_workspace(w)
        w['metadata']['example']['negative_confirmed'] = True
        validate_workspace(w)

    def test_import_cannot_change_source_or_provenance(self):
        original = workspace()
        for field in ('text', 'provenance', 'membership'):
            altered = copy.deepcopy(original)
            if field == 'text':
                altered['documents'][0]['text'] += ' Changed.'
            elif field == 'provenance':
                altered['provenance']['example']['source'] = 'changed'
            else:
                altered['documents'] *= 2
            with self.assertRaises(ValueError):
                validate_workspace(altered, original)

    def test_canonicalization_is_explicit_and_stable(self):
        text = 'a\\\\nb\\nc\r\nd'
        self.assertEqual(canonical_text(text), 'a\nb\nc\nd')
        self.assertEqual(canonical_text(canonical_text(text)), canonical_text(text))


if __name__ == '__main__':
    unittest.main()
