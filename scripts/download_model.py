"""Download weights and record the immutable revision actually selected."""
import json
from pathlib import Path
import sys
from huggingface_hub import HfApi, snapshot_download

model_id, revision, record_path = sys.argv[1:]
info = HfApi().model_info(model_id, revision=revision)
snapshot = snapshot_download(
    repo_id=model_id, revision=info.sha,
    allow_patterns=['*.json', '*.safetensors', '*.model', '*.txt', '*.jinja', 'LICENSE*', 'README*'],
)
root = Path(snapshot)
if not (root / 'config.json').is_file() or not list(root.glob('*.safetensors')):
    raise SystemExit('Expected config.json and safetensors weights; unsupported/incomplete snapshot.')
record = {'model_id': model_id, 'revision': info.sha, 'snapshot': str(root)}
target = Path(record_path)
temporary = target.with_suffix('.tmp')
temporary.write_text(json.dumps(record, indent=2) + '\n')
temporary.replace(target)
print(json.dumps(record, indent=2))
