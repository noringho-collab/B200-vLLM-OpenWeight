"""Protocol tests only; do not validate actual GPU inference."""
from contextlib import contextmanager, redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import io
import json
from pathlib import Path
import threading
import unittest
from urllib.error import HTTPError

spec = importlib.util.spec_from_file_location('smoke', Path(__file__).parents[1] / 'scripts/smoke.py')
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


@contextmanager
def server(model='qwen3-8b', content='안녕하세요!', status=200):
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            calls.append(self.path)
            if self.headers.get('Authorization') != 'Bearer test-key':
                self.send_response(401)
                self.end_headers()
                return
            self.send_response(status)
            self.end_headers()
            self.wfile.write(json.dumps({'data': [{'id': model}]} if self.path == '/v1/models' else {}).encode())

        def do_POST(self):
            calls.append(self.path)
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            if body.get('chat_template_kwargs') != {'enable_thinking': False}:
                self.send_response(400)
                self.end_headers()
                return
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({'choices': [{'message': {'content': content}}]}).encode())

    httpd = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{httpd.server_port}', calls
    finally:
        httpd.shutdown()
        thread.join()
        httpd.server_close()


class SmokeTests(unittest.TestCase):
    def test_real_request_sequence(self):
        with server() as (url, calls), redirect_stdout(io.StringIO()) as output:
            smoke.check(url, 'qwen3-8b', 'test-key')
            self.assertEqual(calls, ['/health', '/v1/models', '/v1/chat/completions'])
            self.assertIn('PASS', output.getvalue())

    def test_wrong_model_prevents_generation(self):
        with server(model='other') as (url, calls):
            with self.assertRaisesRegex(ValueError, 'model missing'):
                smoke.check(url, 'qwen3-8b', 'test-key')
            self.assertNotIn('/v1/chat/completions', calls)

    def test_empty_content_fails(self):
        with server(content=' ') as (url, _), self.assertRaisesRegex(ValueError, 'Empty'):
            smoke.check(url, 'qwen3-8b', 'test-key')

    def test_non_korean_content_fails(self):
        with server(content='Hello') as (url, _), self.assertRaisesRegex(ValueError, 'Korean'):
            smoke.check(url, 'qwen3-8b', 'test-key')

    def test_authentication_failure(self):
        with server() as (url, _), self.assertRaises(HTTPError) as error:
            smoke.check(url, 'qwen3-8b', 'wrong-key')
        self.assertEqual(error.exception.code, 401)

    def test_health_failure(self):
        with server(status=503) as (url, calls), self.assertRaises(HTTPError):
            smoke.check(url, 'qwen3-8b', 'test-key')
        self.assertEqual(calls, ['/health'])


if __name__ == '__main__':
    unittest.main()
