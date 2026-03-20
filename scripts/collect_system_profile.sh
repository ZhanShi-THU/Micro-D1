#!/usr/bin/env bash

set -u

PROJECT_ROOT="$(pwd)"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-microvqa}"

section() {
  echo
  echo "============================================================"
  echo "$1"
  echo "============================================================"
}

run_cmd() {
  local label="$1"
  shift

  echo
  echo "[${label}]"
  if command -v "$1" >/dev/null 2>&1; then
    "$@"
  else
    echo "Command not found: $1"
  fi
}

section "SYSTEM PROFILE"
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "Hostname: $(hostname 2>/dev/null || echo unavailable)"
echo "Project root: ${PROJECT_ROOT}"
echo "Conda env target: ${CONDA_ENV_NAME}"

section "GPU SUMMARY"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,driver_version --format=csv

  echo
  echo "[Full nvidia-smi]"
  nvidia-smi

  echo
  echo "[GPU topology]"
  nvidia-smi topo -m 2>/dev/null || echo "Topology output unavailable on this machine."
else
  echo "nvidia-smi not found. This machine may not have NVIDIA GPUs available."
fi

section "CPU AND MEMORY"
run_cmd "lscpu" lscpu
run_cmd "free -h" free -h

section "DISK"
run_cmd "df -h" df -h

echo
echo "[Current directory size]"
du -sh "${PROJECT_ROOT}" 2>/dev/null || echo "Unable to measure ${PROJECT_ROOT}"

if [ -d "/data1" ]; then
  echo
  echo "[/data1 size]"
  du -sh /data1 2>/dev/null || echo "Unable to measure /data1"
fi

section "SOFTWARE"
run_cmd "uname -a" uname -a

if command -v conda >/dev/null 2>&1; then
  echo
  echo "[conda env list]"
  conda env list 2>/dev/null || echo "Unable to list conda environments."

  echo
  echo "[PyTorch and CUDA check via conda env: ${CONDA_ENV_NAME}]"
  conda run -n "${CONDA_ENV_NAME}" python -c "
import platform
try:
    import torch
    print('python', platform.python_version())
    print('torch', torch.__version__)
    print('cuda', torch.version.cuda)
    print('cuda_available', torch.cuda.is_available())
    print('device_count', torch.cuda.device_count())
    if torch.cuda.is_available():
        print('bf16', torch.cuda.is_bf16_supported())
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f'device_{i}_name', props.name)
            print(f'device_{i}_total_memory_gb', round(props.total_memory / 1024**3, 2))
    else:
        print('bf16', False)
except Exception as exc:
    print('python', platform.python_version())
    print('torch_check_error', repr(exc))
"
else
  echo "conda not found. Skipping PyTorch environment check."
fi

section "QUICK SUMMARY TEMPLATE"
echo "Please send back these items:"
echo "1. GPU model and count"
echo "2. Per-GPU total memory"
echo "3. Current GPU usage snapshot"
echo "4. CPU core count"
echo "5. System RAM"
echo "6. Free disk space on the data/training volume"
echo "7. torch/cuda/bf16 availability"
