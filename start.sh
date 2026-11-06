#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi
export STT_PORT="${STT_PORT:-6006}"
export GRADIO_PORT="${GRADIO_PORT:-7860}"
export STT_URL="ws://127.0.0.1:$STT_PORT"
for file in .venv/bin/python gradio-venv/bin/python upstream/python-api-examples/streaming_server.py; do
  [[ -f "$file" ]] || { echo 'Setup missing. Run: bash setup.sh' >&2; exit 1; }
done
# Refuse to start against another process already using either port.
.venv/bin/python - "$STT_PORT" "$GRADIO_PORT" <<'PY'
import socket, sys
ports = [int(p) for p in sys.argv[1:]]
if len(set(ports)) != len(ports):
    raise SystemExit('STT and Gradio ports must differ.')
for port in ports:
    with socket.socket() as sock:
        try:
            sock.bind(('0.0.0.0', port))
        except OSError as exc:
            raise SystemExit(f'Port {port} unavailable: {exc}')
PY

mkdir -p .logs
stt_pid=''
ui_pid=''
cleanup() {
  trap - EXIT INT TERM
  [[ -z "$ui_pid" ]] || kill "$ui_pid" 2>/dev/null || true
  [[ -z "$stt_pid" ]] || kill "$stt_pid" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

bash scripts/start_stt.sh > .logs/stt.log 2>&1 &
stt_pid=$!
echo "Starting CUDA STT; log: $ROOT/.logs/stt.log"
ready=0
for ((i=0; i<180; i++)); do
  if ! kill -0 "$stt_pid" 2>/dev/null; then
    tail -n 60 .logs/stt.log
    exit 1
  fi
  if .venv/bin/python - "$STT_PORT" <<'PY'
import socket, sys
try:
    with socket.create_connection(('127.0.0.1', int(sys.argv[1])), timeout=1):
        pass
except OSError:
    raise SystemExit(1)
PY
  then
    ready=1
    break
  fi
  sleep 1
done
if [[ $ready != 1 ]]; then
  tail -n 60 .logs/stt.log
  echo 'STT did not become ready in 180 attempts.' >&2
  exit 1
fi

gradio-venv/bin/python -u gradio_app.py > .logs/gradio.log 2>&1 &
ui_pid=$!
echo "STT listening on $STT_PORT; starting Gradio on $GRADIO_PORT."
echo "Gradio log: $ROOT/.logs/gradio.log"
echo 'Open the RunPod HTTPS proxy link for the Gradio port once its log shows startup.'
echo 'Keep this terminal open. Ctrl+C stops both processes.'
set +e
wait -n "$stt_pid" "$ui_pid"
status=$?
set -e
echo 'A service exited; stopping both. Recent logs:'
tail -n 30 .logs/stt.log .logs/gradio.log
exit "$status"

