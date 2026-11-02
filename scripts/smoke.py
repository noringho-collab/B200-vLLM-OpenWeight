"""Check health, model registration, and actual Korean chat generation."""
import json
import os
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def check(base_url, model, api_key):
    def request(path, payload=None):
        data = None if payload is None else json.dumps(payload).encode()
        headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}
        with urlopen(Request(base_url.rstrip('/') + path, data=data, headers=headers), timeout=180) as response:
            body = response.read()
            return json.loads(body) if body else None

    request('/health')
    models = request('/v1/models')
    if model not in [entry['id'] for entry in models['data']]:
        raise ValueError('Expected served model missing from /v1/models')
    result = request('/v1/chat/completions', {
        'model': model,
        'messages': [{'role': 'user', 'content': '한국어로 짧게 인사해 주세요.'}],
        'max_tokens': 128, 'temperature': 0,
        'chat_template_kwargs': {'enable_thinking': False},
    })
    text = result['choices'][0]['message']['content']
    if not isinstance(text, str) or not text.strip():
        raise ValueError('Empty chat response')
    if not any('\uac00' <= char <= '\ud7a3' for char in text):
        raise ValueError('Chat response did not contain Korean text')
    print('PASS: health, model registration, Korean chat generation')
    print(text)


if __name__ == '__main__':
    try:
        key = os.environ.get('VLLM_API_KEY', '')
        if not key:
            raise ValueError('Set VLLM_API_KEY before testing')
        check(os.environ.get('API_BASE_URL', 'http://127.0.0.1:8000'),
              os.environ.get('SERVED_MODEL_NAME', 'qwen3-8b'), key)
    except (HTTPError, URLError, ValueError, KeyError, IndexError, TypeError) as error:
        # Do not dump headers, environment, or server error bodies containing secrets.
        print(f'FAIL: {type(error).__name__}: {error}', file=sys.stderr)
        sys.exit(1)
