import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from scripts.pseudo_label import hosted_response
from scripts.run_distillation_pilot import run


class HostedTeacherTests(unittest.TestCase):
    def test_transport_uses_json_mode_and_rejects_truncated_output(self):
        result = {'model': 'fixture-version', 'choices': [{'finish_reason': 'stop', 'message': {'content': '{}'}}]}
        with patch.dict(os.environ, {'TEACHER_API_KEY': 'synthetic-test-key'}):
            with patch('urllib.request.urlopen', return_value=io.BytesIO(json.dumps(result).encode())) as request:
                response, metadata = hosted_response({'text': 'fixture'}, 'fixture-model', 'https://api.example.org/v1', 20)
                body = json.loads(request.call_args.args[0].data)
                self.assertEqual(body['response_format'], {'type': 'json_object'})
                self.assertEqual(metadata['returned_model'], 'fixture-version')
                self.assertEqual(response, '{}')
            result['choices'][0]['finish_reason'] = 'length'
            with patch('urllib.request.urlopen', return_value=io.BytesIO(json.dumps(result).encode())):
                with self.assertRaises(ValueError):
                    hosted_response({'text': 'fixture'}, 'fixture-model', 'https://api.example.org/v1', 20)

    def test_provider_error_body_is_not_exposed(self):
        error = HTTPError('https://api.example.org', 401, 'Unauthorized', {}, io.BytesIO(b'sensitive provider body'))
        with patch.dict(os.environ, {'TEACHER_API_KEY': 'synthetic-test-key'}):
            with patch('urllib.request.urlopen', side_effect=error):
                with self.assertRaisesRegex(RuntimeError, '^Teacher API returned HTTP 401; response body suppressed$'):
                    hosted_response({'text': 'fixture'}, 'model', 'https://api.example.org', 20)

    def test_missing_credential_prevents_api_calls(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            path = Path(directory)
            with patch('scripts.run_distillation_pilot.label') as label:
                with self.assertRaisesRegex(ValueError, 'TEACHER_API_KEY'):
                    run(path / 'silver', path / 'out', path / 'evaluation', 'model', 'https://api.example.org', 2)
                label.assert_not_called()

    def test_resuming_does_not_exceed_total_pilot_request_budget(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {'TEACHER_API_KEY': 'fixture'}):
            path = Path(directory)
            candidates = [{'id': str(i), 'split': 'train' if i < 3 else 'validation'} for i in range(4)]
            (path / 'manifest.json').write_text(json.dumps({'candidates': candidates}))
            (path / 'responses.jsonl').write_text('\n'.join(json.dumps({'id': str(i)}) for i in range(3)))
            with patch('scripts.run_distillation_pilot.label') as label, patch('scripts.run_distillation_pilot.collect', return_value={'accepted_by_split': {'train': 3, 'validation': 1}}), patch('subprocess.run') as process:
                run(path, path / 'out', path / 'comparison', 'model', 'https://api.example.org', 2)
                self.assertEqual(label.call_count, 1)
                self.assertEqual(label.call_args.args[-3:], (1, 1200, 'validation'))
                self.assertEqual(process.call_count, 2)


if __name__ == '__main__':
    unittest.main()
