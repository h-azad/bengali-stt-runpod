#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python3}"

if [[ "${SKIP_APT:-0}" != 1 ]]; then
  if [[ $EUID -eq 0 ]]; then
    apt-get update
    apt-get install -y git curl bzip2 ffmpeg python3-venv libsndfile1
  else
    echo 'Run as root on RunPod, or install system dependencies and set SKIP_APT=1.' >&2
    exit 1
  fi
fi

"$PYTHON" - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit('Use Python 3.10 or newer; Python 3.11/3.12 recommended.')
try:
    import torch
except ImportError:
    raise SystemExit('Use a RunPod PyTorch template with CUDA 12.8 and cuDNN 9.')
print('Python:', sys.version.split()[0])
print('PyTorch CUDA:', torch.version.cuda)
print('cuDNN:', torch.backends.cudnn.version())
if not torch.cuda.is_available():
    raise SystemExit('PyTorch cannot access the GPU. Check the Pod/template first.')
if not (torch.version.cuda or '').startswith('12.'):
    raise SystemExit('This setup uses CUDA 12 wheels. Select a CUDA 12.8 template.')
if (torch.backends.cudnn.version() or 0) < 90000:
    raise SystemExit('Select a template with cuDNN 9.')
print('GPU:', torch.cuda.get_device_name(0))
PY

"$PYTHON" -m venv --system-site-packages .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-stt.txt
.venv/bin/python -m pip install \
  'sherpa-onnx==2.0.1+cuda12.cudnn9.onnxruntime1.28.2' \
  -f https://k2-fsa.github.io/sherpa/onnx/cuda.html

"$PYTHON" -m venv gradio-venv
gradio-venv/bin/python -m pip install --upgrade pip
gradio-venv/bin/python -m pip install -r requirements-ui.txt

# Fetch an exact upstream commit, matching release v2.0.1.
UPSTREAM_COMMIT=af2d8d81dbfa61c4cea4c568da78f3982edca005
if [[ ! -d upstream/.git ]]; then
  git init upstream
  git -C upstream remote add origin https://github.com/k2-fsa/sherpa-onnx.git
fi
if [[ "$(git -C upstream rev-parse HEAD 2>/dev/null || true)" != "$UPSTREAM_COMMIT" ]]; then
  git -C upstream fetch --depth 1 origin "$UPSTREAM_COMMIT"
  git -C upstream checkout --detach "$UPSTREAM_COMMIT"
fi

MODEL_NAME=sherpa-onnx-streaming-zipformer-bn-vosk-2026-02-09
mkdir -p models .cache .logs
if [[ ! -d "models/$MODEL_NAME" ]]; then
  curl --fail --location --retry 3 \
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/$MODEL_NAME.tar.bz2" \
    -o ".cache/$MODEL_NAME.tar.bz2"
  tar -xjf ".cache/$MODEL_NAME.tar.bz2" -C models
fi
for file in encoder.onnx decoder.onnx joiner.onnx tokens.txt; do
  test -s "models/$MODEL_NAME/$file" || { echo "Missing model file: $file" >&2; exit 1; }
done
.venv/bin/python -m pip freeze > .logs/stt-packages.txt
gradio-venv/bin/python -m pip freeze > .logs/ui-packages.txt
echo 'Setup complete. Start with: bash start.sh'

