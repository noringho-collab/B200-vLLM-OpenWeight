# B200-vLLM-OpenWeight

MSP의 기존 컨테이너에서 **sudo·Docker 없이** Hugging Face 모델을 vLLM으로 서빙합니다. 28개 컨테이너의 전체 솔루션이 아닌 LLM API 1개를 먼저 검증합니다.

## 대상과 기본 설정

사용자 제공 정보: Python 3.12.3, Linux x86_64, NVIDIA B200 1장(183,359 MiB), 드라이버 580.95.05, `nproc` 8. 기존 NVIDIA PyTorch는 `2.10.0a0…nv25.12` / CUDA 13.1이며 GPU 인식과 capability `(10, 0)`을 확인했습니다. 실제 CPU quota·RAM·영구 저장 공간은 미확인입니다.

| 항목 | 첫 검증 설정 |
|---|---|
| 모델 | Qwen/Qwen3-8B, BF16, 양자화 없음 |
| 런타임 | 별도 .venv, vLLM 0.11.0 / PyTorch 2.8.0 / CUDA 12.8 휠 |
| GPU | 1장, tensor parallel 1 |
| GPU 메모리 예산 | 총 메모리의 40%, 약 72GiB |
| 컨텍스트 / 동시 시퀀스 | 8,192토큰 / 4 |
| API | OpenAI 호환, 기본 127.0.0.1:8000, API 키 필수 |

8B로 설치와 한국어 응답부터 검증한 뒤 32B 등으로 확장합니다. 메모리 비율은 다른 GPU 프로세스를 격리하지 않습니다. 기존 Python GPU 작업이 관찰되었으므로 현재 사용량을 확인하고 작업 소유자와 조율하세요. 스크립트는 기존 작업을 종료하지 않습니다.

**실제 B200에서 설치·다운로드·추론은 아직 미검증입니다.** 개발용 클라우드에는 GPU가 없고 Hugging Face 접근도 403으로 차단되어 있습니다. 이 저장소가 준비되었다는 사실과 대상 장비의 서빙 성공은 별개입니다.

## 1. 준비

이미 있는 체크아웃을 사용해도 됩니다. 별도의 Git worktree는 필요하지 않습니다.

```bash
git clone https://github.com/noringho-collab/B200-vLLM-OpenWeight.git
cd B200-vLLM-OpenWeight
python3 scripts/preflight.py
```

패키지와 모델 캐시용 여유 공간은 최소 50GiB 정도를 권장하며 실제 설치 조합에 따라 더 필요합니다. 영구 볼륨 경로와 컨테이너 재생성 시 보존 범위를 MSP에 확인하세요. 현재 보이는 CPU 8개는 전체 솔루션의 32코어 요구사항과 차이가 있습니다.

처음에만 아래를 실행합니다. `.env`가 이미 있으면 덮어쓰지 말고 기존 설정을 보존하세요.

```bash
cp -n .env.example .env
chmod 600 .env
python3 - <<'PY'
from pathlib import Path
import secrets
p = Path('.env')
s = p.read_text()
if '\nVLLM_API_KEY=\n' not in s:
    raise SystemExit('Existing API key/config detected; preserve it.')
p.write_text(s.replace('\nVLLM_API_KEY=\n', '\nVLLM_API_KEY=' + secrets.token_hex(32) + '\n'))
PY
```

`.env`는 셸 코드로 읽으므로 신뢰하는 내용만 넣습니다. 영구 저장소가 있다면 `HF_HOME`을 해당 경로로 지정하세요. 기본 모델은 공개 모델로 HF 토큰을 전제로 하지 않습니다. 제한 모델은 약관 승인과 접근 가능한 `HF_TOKEN`이 필요합니다. 키·토큰·모델·환경변수 덤프는 GitHub에 올리지 않습니다.

## 2. 설치

```bash
bash scripts/install.sh
```

기존 NVIDIA PyTorch를 수정하지 않고 `.venv`에 공식 CUDA 12.8 PyTorch 휠과 vLLM을 설치합니다. sudo·apt·Docker·소스 빌드를 사용하지 않습니다. 기존 `.venv`가 스크립트 소유가 아니면 중단합니다. `venv/ensurepip`가 없으면 MSP에 Python 3.12 가상환경 지원을 요청해야 합니다.

스크립트는 `pip check`, B200 장치 확인, BF16 GPU 행렬 연산을 실행합니다. 실패하면 서버를 띄우기 전에 오류를 진단하세요. 직접 의존성은 `requirements-serving.txt`, 전이 의존성까지 고정한 Python 3.12용 해석 결과는 `requirements-serving.lock`입니다. 설치는 lock을 사용하며 실제 설치 버전은 `.runtime/installed-packages.txt`에 기록됩니다. 의존성 해석 성공은 GPU 런타임 호환성 검증과 별개입니다.

이 버전 조합은 공식 CUDA 12.8 휠을 사용하기 위한 **초기 호환성 검증 기준**입니다. 최신 버전 또는 운영 보안 검토 완료 버전이 아닙니다. 외부 운영 전 지원 버전의 보안 수정·드라이버 요구사항을 검토하고 별도 환경에서 업그레이드 및 회귀 검증하세요. 기존 NVIDIA PyTorch에 `--no-deps`로 vLLM을 결합하지 마세요.

## 3. 모델 다운로드

```bash
bash scripts/download.sh
```

Hub 리비전을 실제 커밋 SHA로 해석해 safetensors 가중치와 설정·토크나이저를 받습니다. 모델 ID, SHA, snapshot 경로는 `.runtime/model.json`에 기록됩니다. 서버는 이 로컬 snapshot을 사용합니다. 다운로드를 다시 실행하면 `MODEL_REVISION=main`의 새 커밋을 선택할 수 있습니다. 재현하려면 `.env`의 `MODEL_REVISION`을 기록된 전체 SHA로 고정하세요.

네트워크에는 `pypi.org`, `files.pythonhosted.org`, `download-r2.pytorch.org`, `huggingface.co` 및 다운로드 리다이렉트의 Hugging Face CDN/Xet 호스트 접근이 필요할 수 있습니다. 공식 PyTorch 인덱스에 게시된 Python 3.12용 CUDA 휠 URL과 SHA256을 고정하고, 나머지는 PyPI에서 받습니다. 차단 시 실제 오류의 목적지를 MSP에 허용 요청하세요. TLS 검증을 끄거나 프록시를 우회하지 마세요. 개발 클라우드의 차단 여부는 실제 MSP 네트워크 상태를 뜻하지 않습니다.

## 4. 서버와 실제 응답 확인

터미널 A:

```bash
bash scripts/serve.sh
```

첫 기동은 모델 로딩과 커널 준비 시간이 필요합니다. 초기 검증에서는 `--enforce-eager`로 CUDA graph를 생략했습니다. 메모리 부족 시 기존 사용량부터 확인하세요.

터미널 B에서 같은 저장소로 이동해:

```bash
bash scripts/smoke.sh
```

`/health`, `/v1/models`의 모델 등록, `/v1/chat/completions`의 **비어 있지 않은 한국어 생성**이 모두 통과해야 성공입니다. Qwen3는 `enable_thinking=false`로 검증합니다. 포트가 열리거나 PID가 있는 것만으로 성공이 아닙니다. 실패 시 종료 코드는 0이 아닙니다. 종료는 서버 터미널의 Ctrl+C입니다.

클라이언트 설정: base URL `http://127.0.0.1:8000/v1`, model `qwen3-8b`, API key는 로컬 `.env`의 값입니다. Qwen3 비추론 모드 요청에는 `chat_template_kwargs={"enable_thinking": false}`를 전달합니다.

포그라운드 검증 성공 후 백그라운드 실행이 필요하면:

```bash
mkdir -p .runtime
(umask 077; nohup bash scripts/serve.sh > .runtime/server.log 2>&1 < /dev/null & echo $! > .runtime/server.pid)
bash scripts/smoke.sh
```

로그는 `.runtime/server.log`입니다. `nohup`은 컨테이너 재생성·호스트 재부팅 후 자동 복구를 제공하지 않습니다. 운영용 재시작 정책과 영구 저장소는 MSP의 지원이 필요합니다.

다른 컨테이너나 외부에서 접속하려면 MSP의 포트 매핑/내부 네트워크 또는 인증된 TLS 프록시가 필요합니다. 그 경로가 준비된 뒤 `.env`에 `HOST=0.0.0.0`을 설정하세요. 바인딩 변경만으로 호스트 포트가 개방되지는 않습니다. 내부 분산 통신 포트도 신뢰 네트워크 안에 두어야 합니다.

## 검증과 확장

개발용 검사:

```bash
python3 -m unittest discover -s tests -v
bash -n scripts/common.sh scripts/install.sh scripts/download.sh scripts/serve.sh scripts/smoke.sh
```

장비 결과는 [검증 기록](docs/validation.md)에 비밀값을 제거한 요약만 추가합니다. 실제 smoke 통과 후 문서 질의 품질·동시성·지연 시간을 실측하고 모델 크기와 컨텍스트를 하나씩 조정합니다. 이 API는 Studio/Document Parse 등 나머지 컨테이너의 배포를 대신하지 않습니다.

근거: [vLLM CUDA 문서](https://github.com/vllm-project/vllm/blob/v0.11.0/docs/getting_started/installation/gpu/cuda.inc.md), [vLLM 배포 메타데이터](https://pypi.org/project/vllm/0.11.0/), [모델 카드](https://huggingface.co/Qwen/Qwen3-8B).
