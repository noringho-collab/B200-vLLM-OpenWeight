# 검증 기록

사용자 제공: B200 183,359 MiB, 드라이버 580.95.05, Python 3.12.3, x86_64, nproc 8. 기존 PyTorch CUDA available True, capability (10, 0). 현재 GPU 여유량은 재확인해야 합니다.

## 개발 머신 검사

- Python 3.12에서 모의 HTTP 테스트 6개 통과: 정상 요청 순서, 인증 거부, 잘못된 모델, 빈 응답, 비한국어 응답, health 오류.
- 셸 구문 검사, Python 컴파일, Git 공백 검사 통과.
- 공식 SHA256 고정 CUDA 12.8 휠 + PyPI만으로 binary-only 의존성 해석 성공. 결과는 `requirements-serving.lock`에 보관.
- `preflight.py`는 개발 머신의 NVIDIA GPU 부재를 탐지하고 종료 코드 1로 중단. 대상 장비의 GPU 오류를 뜻하지 않음.
- GitHub Actions에는 CPU 검사만 등록. 원격 CI 실행 결과와 GPU 설치/추론은 아직 미확인.

## 실제 B200 실행 기록 — 미실행

개발 머신에는 GPU가 없습니다. 아래는 대상 MSP 장비에서 실행 후 채울 양식입니다.

| 항목 | 결과 |
|---|---|
| 날짜 / Git 커밋 | 미실행 |
| preflight / CPU quota / RAM / 디스크 / shm | 미확인 |
| install.sh / pip check | 미실행 |
| 설치된 torch / CUDA / vLLM 버전 | 미확인 |
| GPU BF16 행렬 연산 | 미실행 |
| 모델 ID / 커밋 SHA | 미확인 |
| 서버 기동 / health / 모델 등록 | 미실행 |
| 한국어 chat 생성 | 미실행 |
| 종료 후 재시작 및 smoke | 미실행 |
| GPU 메모리 / 동시성 / 지연 시간 | 미측정 |

`.runtime/installed-packages.txt`와 `model.json`은 로컬 보관합니다. GitHub에는 필요한 버전·SHA·오류 요약만 선별하고 키·토큰·사용자 프롬프트는 올리지 마세요.
