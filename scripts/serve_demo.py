"""Serve a local medication-name highlighting demo."""
import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.predict_medication_names import MedicationNameExtractor

MAX_BODY = 65536


def make_handler(extractor):
    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, body, content_type='application/json; charset=utf-8'):
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path != '/':
                self.reply(404, b'{"error":"Not found"}')
                return
            self.reply(200, (ROOT/'demo/index.html').read_bytes(), 'text/html; charset=utf-8')

        def do_POST(self):
            if self.path != '/predict':
                self.reply(404, b'{"error":"Not found"}')
                return
            # The browser must originate from this local demo, including forwarded ports.
            origin = self.headers.get('Origin')
            if origin and origin not in (f'http://{self.headers.get("Host")}', f'https://{self.headers.get("Host")}'):
                self.reply(403, b'{"error":"Origin not allowed"}')
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= MAX_BODY:
                    raise ValueError('Request must contain at most 64 KiB')
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict) or set(payload) != {'text'}:
                    raise ValueError('Request needs a text field only')
                result = extractor.predict(payload['text'])
            except (ValueError, TypeError) as error:
                self.reply(400, json.dumps({'error': str(error)}).encode())
                return
            self.reply(200, json.dumps(result, ensure_ascii=False).encode())

        def log_message(self, format, *args):
            pass
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    extractor = MedicationNameExtractor(args.run)
    server = HTTPServer(('127.0.0.1', args.port), make_handler(extractor))
    print(f'Demo: http://127.0.0.1:{server.server_port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
