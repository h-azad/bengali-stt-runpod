# Bengali Zipformer GPU + Gradio on RunPod

A Git-ready test project for an RTX A5000 or another CUDA-capable RunPod GPU.
Uses the streaming Bengali Vosk Zipformer model, Sherpa-ONNX CUDA and a
record/upload Gradio interface. Microphone capture is submitted after recording
stops; this interface does not provide live browser microphone transcription.

## First deployment

1. Select a RunPod PyTorch template with CUDA 12.8, cuDNN 9, and Python 3.11/3.12.
2. Before installing, expose HTTP port **7860**. Port **6006** is optional unless
   you want external WebSocket clients. Gradio uses 6006 internally.
3. No persistent storage is required for testing. Pod updates/resets can wipe
   this checkout; rerun the clone and setup afterward.

Once these project files have been pushed to your own Git repository:

```bash
cd /workspace
git clone https://github.com/YOUR_USERNAME/bengali-stt-runpod.git
cd bengali-stt-runpod
bash setup.sh
bash start.sh
```

The GitHub URL above is a placeholder, not an existing repository.
You can also unzip the provided project on the Pod and run the same scripts.
`setup.sh` runs apt as root and downloads the model, exact upstream source
commit and CUDA wheel. It creates separate STT and Gradio environments.
Repeated setup reuses complete model files and an existing source checkout.
Run setup again after an interruption; if model extraction was interrupted,
move its incomplete model directory out of `models/` before rerunning setup.

Open `https://YOUR_POD_ID-7860.proxy.runpod.net` and allow microphone access.
Record, stop recording, and press **Transcribe**.

## Create your Git repository

Create an empty GitHub repository, then run from the extracted project folder:

```bash
git init -b main
git add .
git commit -m "Add Bengali STT RunPod test setup"
git remote add origin https://github.com/YOUR_USERNAME/bengali-stt-runpod.git
git push -u origin main
```

Authenticate Git using SSH or your credential manager for private repositories.
Never put access tokens in this repository. Models, upstream checkout,
virtual environments, logs, recordings and `.env` are excluded by `.gitignore`.

## Configuration

```bash
cp .env.example .env
```

Edit `.env` before running `start.sh`. Optional `GRADIO_PASSWORD` enables login
with username `admin`. Defaults: STT 6006, Gradio 7860, batch size 16, batch wait
5 ms. The 100-connection setting is an admission limit, not benchmarked capacity.
`start.sh` starts both services, waits for the STT socket, and stops both if
either service exits or you press Ctrl+C. Keep the terminal open for testing.
It does not automatically restart after a Pod reboot.

## Logs and GPU checks

```bash
tail -f .logs/stt.log .logs/gradio.log
nvidia-smi
```

The server selects `--provider=cuda`, imports torch first to load CUDA libraries,
and uses the exact CUDA-enabled wheel. A successful transcript does not alone
prove GPU execution: check provider warnings and GPU memory/activity during a
test. GPU process visibility can vary in containers.

## Test without Gradio

With `start.sh` already running, open another terminal in the repository:

```bash
.venv/bin/python upstream/python-api-examples/online-websocket-client-decode-file.py \
  --server-addr 127.0.0.1 --server-port 6006 \
  models/sherpa-onnx-streaming-zipformer-bn-vosk-2026-02-09/test_wavs/0.wav
```

The WebSocket server expects mono 16 kHz normalized float32 samples, not encoded
audio or RTP packets. Gradio converts/resamples uploads before sending them.
The UI reports elapsed wall time, including transfer and optional real-time
pacing, not a pure GPU benchmark.

## Validation and limits

Shell/Python syntax and a local mock-WebSocket transcription flow were checked.
GPU installation and RunPod deployment still require testing on your Pod.
Python packages other than the CUDA wheel/Gradio/WebSockets use bounded ranges;
setup records resolved versions under `.logs/`. Do not change the PyTorch
template without checking CUDA/cuDNN compatibility. This is a test deployment,
not an authenticated production call gateway. Public STT port 6006 has no auth.

Upstream source: https://github.com/k2-fsa/sherpa-onnx (release v2.0.1).
Model: https://huggingface.co/alphacep/vosk-model-small-streaming-bn .
Upstream software and model retain their respective licenses.
