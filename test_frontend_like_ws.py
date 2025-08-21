#!/usr/bin/env python3
"""
Frontend-like WebSocket test client (no flags)

Mimics the browser frontend behavior with hardcoded settings:
- Connects to ws://127.0.0.1:8001/ws/transcribe
- Streams audio_for_test.mp3 as 16kHz mono PCM16 in ~1s frames
- Responds to text-level ping/pong
- Sends end_session after streaming
- Waits for provisional and final transcript files under logs/

Run:
  python test_frontend_like_ws.py
"""

import asyncio
import glob
import json
import os
import sys
from typing import Optional

import websockets
from pydub import AudioSegment


# Hardcoded settings to simulate the real frontend
AUDIO_PATH = "audio_for_test.mp3"
WS_URL = "ws://127.0.0.1:8001/ws/transcribe"
SAMPLE_RATE = 16000
FRAME_SEC = 1.0       # ~1s frames like typical MediaRecorder cadence
REALTIME = True       # sleep between frames to emulate realtime mic streaming


def load_audio_as_pcm16(audio_path: str, target_sr: int = SAMPLE_RATE) -> bytes:
    seg = AudioSegment.from_file(audio_path)
    seg = seg.set_frame_rate(target_sr).set_channels(1).set_sample_width(2)
    return seg.raw_data


def chunk_bytes(pcm_bytes: bytes, sample_rate: int = SAMPLE_RATE, frame_sec: float = FRAME_SEC):
    frame_size = int(sample_rate * 2 * frame_sec)  # int16 mono
    for i in range(0, len(pcm_bytes), frame_size):
        yield pcm_bytes[i : i + frame_size]


async def wait_for_file(path_pattern: str, timeout_sec: float) -> Optional[str]:
    deadline = asyncio.get_event_loop().time() + timeout_sec
    while asyncio.get_event_loop().time() < deadline:
        matches = glob.glob(path_pattern)
        if matches:
            # Return the most recent match
            matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return matches[0]
        await asyncio.sleep(0.5)
    return None


async def listener(ws, session_info: dict, session_ended_event: asyncio.Event):
    try:
        async for message in ws:
            if isinstance(message, bytes):
                # Backend does not send binary payloads to client currently
                continue
            try:
                payload = json.loads(message)
            except Exception:
                print(f"[server] {message}")
                continue

            mtype = payload.get("type")
            if mtype == "connected":
                session_info["session_id"] = payload.get("session_id")
                print(f"✅ Connected. session_id={session_info['session_id']}")
            elif mtype == "ping":
                await ws.send(json.dumps({"type": "pong"}))
            elif mtype == "session_ended":
                print("🛑 Session ended signaled by server.")
                session_ended_event.set()
            elif mtype == "error":
                print(f"❌ Server error: {payload.get('message')}")
            else:
                # Other messages can be logged noisily if needed
                pass
    except websockets.exceptions.ConnectionClosed:
        print("🔌 WebSocket closed by server.")


async def run_test():
    # Load audio and convert to PCM16 mono 16kHz
    print(f"🎵 Loading audio: {AUDIO_PATH}")
    pcm = load_audio_as_pcm16(AUDIO_PATH, target_sr=SAMPLE_RATE)
    total_seconds = len(pcm) / (SAMPLE_RATE * 2)
    print(f"   Duration: {total_seconds:.1f}s @ 16kHz mono PCM16")

    # Connect
    print(f"📡 Connecting to {WS_URL}")
    async with websockets.connect(WS_URL, ping_interval=None, ping_timeout=None, max_size=None) as ws:
        session_info = {}
        session_ended_event = asyncio.Event()

        # Start listener
        listen_task = asyncio.create_task(listener(ws, session_info, session_ended_event))

        # Wait briefly for 'connected' handshake
        await asyncio.sleep(0.2)

        # Stream audio frames
        sent_sec = 0.0
        for frame in chunk_bytes(pcm, SAMPLE_RATE, FRAME_SEC):
            if not frame:
                continue
            await ws.send(frame)
            sent_sec += len(frame) / (SAMPLE_RATE * 2)
            if REALTIME:
                await asyncio.sleep(FRAME_SEC)

        print(f"📤 Finished streaming audio ({sent_sec:.1f}s). Signaling end_session…")
        await ws.send(json.dumps({"type": "end_session"}))

        # Wait for session_ended signal or a short timeout
        try:
            await asyncio.wait_for(session_ended_event.wait(), timeout=10)
        except asyncio.TimeoutError:
            print("⚠️ No session_ended signal within 10s; proceeding to close.")

        # Give server a moment to flush macros
        await asyncio.sleep(1.0)

        # Close client (server persists transcript on disconnect)
        await ws.close()
        await listen_task

    session_id = session_info.get("session_id")
    if not session_id:
        print("❌ No session_id received; cannot locate transcript files.")
        sys.exit(1)

    # Check provisional file
    prov_path = os.path.join("logs", f"provisional_{session_id}.txt")
    if os.path.exists(prov_path):
        size_kb = os.path.getsize(prov_path) / 1024.0
        print(f"📝 Provisional file: {prov_path} ({size_kb:.1f} KB)")
    else:
        print("⚠️ Provisional file not found (may be short audio or fast finish).")

    # Wait for final transcript
    print("⌛ Waiting for final transcript file…")
    pattern = os.path.join("logs", f"transcript_*_{session_id}.txt")
    final_path = await wait_for_file(pattern, timeout_sec=120)
    if not final_path:
        print("❌ Final transcript not found within timeout.")
        sys.exit(2)

    size_kb = os.path.getsize(final_path) / 1024.0
    print(f"💾 Final transcript: {final_path} ({size_kb:.1f} KB)")


def main():
    if not os.path.exists(AUDIO_PATH):
        print(f"❌ Audio file not found: {AUDIO_PATH}")
        sys.exit(3)
    asyncio.run(run_test())


if __name__ == "__main__":
    main()


