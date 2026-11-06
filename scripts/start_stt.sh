#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/upstream"
MODEL="$ROOT/models/sherpa-onnx-streaming-zipformer-bn-vosk-2026-02-09"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
exec "$ROOT/.venv/bin/python" -u "$ROOT/scripts/run_stt.py" \
  --encoder="$MODEL/encoder.onnx" \
  --decoder="$MODEL/decoder.onnx" \
  --joiner="$MODEL/joiner.onnx" \
  --tokens="$MODEL/tokens.txt" \
  --provider=cuda \
  --sample-rate=16000 \
  --decoding-method=greedy_search \
  --num-threads=2 \
  --nn-pool-size=1 \
  --max-batch-size="${MAX_BATCH_SIZE:-16}" \
  --max-wait-ms="${MAX_WAIT_MS:-5}" \
  --max-active-connections="${MAX_CONNECTIONS:-100}" \
  --port="${STT_PORT:-6006}"

