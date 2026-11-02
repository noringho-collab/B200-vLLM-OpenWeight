"""Read-only host checks. Does not enumerate environment/credential values."""
import argparse
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu-check', action='store_true')
    args = parser.parse_args()
    print('Platform:', platform.system(), platform.machine(), 'Python:', platform.python_version())
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        raise SystemExit('This baseline requires Linux x86_64.')
    if sys.version_info[:2] != (3, 12):
        raise SystemExit('Use Python 3.12: this baseline pins CPython 3.12 CUDA wheels.')
    print('CPU affinity:', len(os.sched_getaffinity(0)))
    for name in ('cpu.max', 'memory.max', 'cpu/cpu.cfs_quota_us', 'cpu/cpu.cfs_period_us'):
        path = Path('/sys/fs/cgroup') / name
        if path.is_file():
            print('cgroup', name, path.read_text().strip())
    for path in (Path.cwd(), Path('/dev/shm')):
        if path.exists():
            usage = shutil.disk_usage(path)
            print(f'{path}: free={usage.free / 2**30:.1f} GiB total={usage.total / 2**30:.1f} GiB')
    if not shutil.which('nvidia-smi'):
        raise SystemExit('nvidia-smi absent: GPU host setup must be provided by MSP.')
    subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu',
                    '--format=csv'], check=True)
    if args.gpu_check:
        import torch
        import vllm
        print('torch:', torch.__version__, 'CUDA runtime:', torch.version.cuda, 'vLLM:', vllm.__version__)
        if not torch.cuda.is_available():
            raise SystemExit('CUDA unavailable in serving venv.')
        capability = torch.cuda.get_device_capability()
        print('Selected GPU:', torch.cuda.get_device_name(), 'capability:', capability)
        if capability != (10, 0):
            raise SystemExit('Expected B200 (10, 0); verify GPU selection.')
        x = torch.randn(1024, 1024, device='cuda', dtype=torch.bfloat16)
        result = x @ x
        torch.cuda.synchronize()
        if not torch.isfinite(result).all().item():
            raise SystemExit('GPU BF16 operation produced nonfinite output.')
        print('GPU BF16 matmul: PASS')


if __name__ == '__main__':
    main()
