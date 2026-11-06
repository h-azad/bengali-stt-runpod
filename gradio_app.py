"""Record/upload Bengali audio and transcribe using an existing Sherpa server.

Install: pip install gradio numpy scipy 'websockets>=13,<16'
Run: python gradio_app.py
Expose HTTP port 7860 in RunPod. STT server must already run on port 6006.
Microphone recordings are submitted after recording stops, not live capture.
"""

import asyncio
import json
import math
import os
import time

import gradio as gr
import numpy as np
from scipy.signal import resample_poly
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosedOK

STT_URL = os.getenv("STT_URL", "ws://127.0.0.1:6006")


def prepare_audio(audio):
    rate, data = audio
    data = np.asarray(data)
    if rate <= 0 or not data.size:
        raise ValueError("Audio is empty or has an invalid sample rate.")
    if np.issubdtype(data.dtype, np.integer):
        limits = np.iinfo(data.dtype)
        midpoint = (int(limits.max) + int(limits.min) + 1) / 2
        scale = (int(limits.max) - int(limits.min) + 1) / 2
        data = (data.astype(np.float32) - midpoint) / scale
    else:
        data = data.astype(np.float32)
    if data.ndim == 2:
        data = data.mean(axis=1)
    if data.ndim != 1:
        raise ValueError("Expected mono or stereo audio.")
    if rate != 16000:
        divisor = math.gcd(int(rate), 16000)
        data = resample_poly(data, 16000 // divisor, int(rate) // divisor)
    return np.clip(np.nan_to_num(data), -1, 1).astype("<f4")


async def transcribe(audio, realtime):
    if audio is None:
        raise gr.Error("Record or upload audio first.")
    try:
        samples = prepare_audio(audio)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc

    duration = len(samples) / 16000
    started = time.perf_counter()
    segments = {}

    def transcript():
        return "\n".join(segments[k] for k in sorted(segments) if segments[k])

    yield "", f"Connecting · Audio: {duration:.1f}s"
    try:
        async with connect(
            STT_URL, open_timeout=15, ping_interval=20,
            ping_timeout=60, max_size=2**20,
        ) as socket:
            async def send_audio():
                for offset in range(0, len(samples), 1600):
                    frame = samples[offset:offset + 1600]
                    await socket.send(frame.tobytes())
                    await asyncio.sleep(len(frame) / 16000 if realtime else 0)
                await socket.send("Done")

            sender = asyncio.create_task(send_audio())
            try:
                while True:
                    try:
                        raw = await asyncio.wait_for(socket.recv(), timeout=60)
                    except ConnectionClosedOK:
                        break
                    result = json.loads(raw)
                    segments[int(result.get("segment", 0))] = str(
                        result.get("text", "")
                    ).strip()
                    elapsed = time.perf_counter() - started
                    yield transcript(), f"Processing · Audio: {duration:.1f}s · Elapsed: {elapsed:.1f}s"
                await sender
            finally:
                if not sender.done():
                    sender.cancel()
                await asyncio.gather(sender, return_exceptions=True)
    except asyncio.TimeoutError as exc:
        raise gr.Error("No STT response for 60 seconds. Check server logs.") from exc
    except Exception as exc:
        raise gr.Error(f"STT request failed: {exc}") from exc

    elapsed = time.perf_counter() - started
    yield transcript(), (
        f"Completed · Audio: {duration:.1f}s · Elapsed: {elapsed:.1f}s · "
        f"Wall-time/audio ratio: {elapsed / duration:.2f} "
        "(includes transfer and optional pacing)"
    )


def build_app():
    with gr.Blocks(title="Bengali STT Test") as demo:
        gr.Markdown(
            "# Bengali Speech-to-Text\n"
            "Record or upload audio, then click **Transcribe**. "
            "Microphone audio is submitted after recording stops."
        )
        audio = gr.Audio(sources=["microphone", "upload"], type="numpy", label="Audio")
        realtime = gr.Checkbox(
            value=False, label="Send at real-time speed",
            info="Simulates a live call using the completed recording.",
        )
        with gr.Row():
            submit = gr.Button("Transcribe", variant="primary")
            clear = gr.ClearButton()
        text = gr.Textbox(label="Bengali transcript", lines=8)
        status = gr.Textbox(label="Status", interactive=False)
        submit.click(
            transcribe, inputs=[audio, realtime], outputs=[text, status],
            concurrency_limit=2,
        )
        clear.add([audio, text, status])
    return demo


if __name__ == "__main__":
    password = os.getenv("GRADIO_PASSWORD")
    build_app().queue(max_size=10).launch(
        server_name="0.0.0.0", server_port=int(os.getenv("GRADIO_PORT", "7860")),
        share=False, auth=("admin", password) if password else None,
    )
