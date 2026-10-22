import json
import unittest
from scripts.prepare_data import audit, parse_summary, text_hash


class DataAuditTests(unittest.TestCase):
    def test_repairs_only_whitespace_outside_strings(self):
        raw = '{\\n"treatments": [{"name": "A", "details": "line\\nnext"}]\\n}'
        parsed, repaired = parse_summary(raw)
        self.assertTrue(repaired)
        self.assertEqual(parsed['treatments'][0]['details'], 'line\nnext')

    def test_literal_newline_in_value_is_reported_as_repair(self):
        parsed, repaired = parse_summary('{"name": "line\nnext"}')
        self.assertEqual(parsed['name'], 'line\nnext')
        self.assertTrue(repaired)

    def test_valid_json_remains_unchanged(self):
        value = {'treatments': [{'name': 'A "quoted" drug'}]}
        parsed, repaired = parse_summary(json.dumps(value))
        self.assertEqual(parsed, value)
        self.assertFalse(repaired)

    def test_invalid_json_is_not_silently_accepted(self):
        with self.assertRaises(json.JSONDecodeError):
            parse_summary('{"treatments": [broken]}')

    def test_audit_counts_failures_placeholders_and_duplicates(self):
        rows = [
            {'note': 'short', 'full_note': 'long note', 'summary': '{"treatments": [{"name": "None", "dosage": null}, {"name": "Drug", "dosage": "5 mg"}]}'},
            {'note': 'long  note', 'full_note': 'long  note', 'summary': 'bad JSON'},
        ]
        result = audit(rows)
        self.assertEqual(result['counts']['summary_parse_failures'], 1)
        self.assertEqual(result['populated_treatment_fields'], {'name': 1, 'dosage': 1})
        self.assertEqual(result['duplicate_full_note_rows'], 1)
        self.assertEqual(result['counts']['note_shorter_than_full_note'], 1)
        self.assertEqual(text_hash(' a\n b '), text_hash('a b'))


if __name__ == '__main__':
    unittest.main()
