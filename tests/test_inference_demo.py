import http.client
import json
import threading
import unittest
from http.server import HTTPServer
from unittest.mock import patch

from scripts.predict_medication_names import MedicationNameExtractor
from scripts.serve_demo import make_handler


class FakeExtractor:
    def predict(self, text):
        if not isinstance(text, str) or not text.strip():
            raise ValueError('Provide nonempty note text')
        return {'medication_names': [], 'elapsed_seconds': 0.01}


class InferenceTests(unittest.TestCase):
    def test_public_output_uses_exact_unicode_offsets(self):
        extractor = MedicationNameExtractor.__new__(MedicationNameExtractor)
        with patch.object(extractor, 'predict_payload', return_value=(
                {'medication_name_spans': [[2, 2], [4, 4]]}, 0.1)):
            text = '😀 aspirin and aspirin.'
            result = extractor.predict(text)
        self.assertEqual(result['medication_names'], [
            {'start': 2, 'end': 9, 'text': 'aspirin'},
            {'start': 14, 'end': 21, 'text': 'aspirin'}])
        for span in result['medication_names']:
            self.assertEqual(text[span['start']:span['end']], span['text'])
        self.assertTrue(result['end_exclusive'])

    def setUp(self):
        self.server = HTTPServer(('127.0.0.1', 0), make_handler(FakeExtractor()))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.client = http.client.HTTPConnection(*self.server.server_address)

    def tearDown(self):
        self.client.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def request(self, method, path, body=None, headers=None):
        self.client.request(method, path, body, headers or {})
        response = self.client.getresponse()
        return response.status, response.read()

    def test_demo_and_prediction(self):
        status, body = self.request('GET', '/')
        self.assertEqual(status, 200)
        self.assertIn(b'Medication extractor', body)
        status, body = self.request('POST', '/predict', json.dumps({'text': 'No medication.'}))
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)['medication_names'], [])

    def test_input_errors_and_cross_origin_requests(self):
        for body in ['{', '[]', '{"text":null}', '{"text":""}', '{"text":"a","extra":1}']:
            self.assertEqual(self.request('POST', '/predict', body)[0], 400)
        self.assertEqual(self.request('POST', '/predict', 'x'*65537)[0], 400)
        self.assertEqual(self.request('POST', '/predict', '{"text":"a"}',
                                      {'Origin': 'https://unrelated.example'})[0], 403)
        self.assertEqual(self.request('GET', '/missing')[0], 404)
