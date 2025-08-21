#!/usr/bin/env python3
"""
Minimal WebSocket Transcription Server

A simplified server for testing WAV file transcription without database dependencies.
This server provides only the WebSocket transcription functionality.
"""

import asyncio
import json
import uuid
import os
from datetime import datetime
from typing import Dict, Any, Optional
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import uvicorn
from starlette.websockets import WebSocketState
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
import multiprocessing as mp

# Import only the config - transcription service loaded lazily
from config import get_settings

"""
Pass 1 process worker: isolate Faster-Whisper in a separate process to prevent
main server kills under memory pressure. The worker loads the model once and
serves window transcription requests.
"""

_RT_MODEL = None


def _rt_worker_init(model_name: str, device: str, cpu_threads: int, num_workers: int):
    global _RT_MODEL
    try:
        from faster_whisper import WhisperModel as FWModel
        import os as _os

        # Threading safeguards inside worker as well
        _os.environ.setdefault("OMP_NUM_THREADS", "1")
        _os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        _os.environ.setdefault("MKL_NUM_THREADS", "1")
        _os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
        _os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
        _os.environ.setdefault("CT2_USE_MMAP", "1")  # mmap model weights
        _os.environ.setdefault("CT2_CACHE_CAPACITY", "1")  # tiny runtime cache
        kwargs = dict(
            device=device,
            compute_type="int8",
        )
        if device == "cpu":
            kwargs.update(
                {
                    "cpu_threads": max(1, int(cpu_threads)),
                    "num_workers": max(1, int(num_workers)),
                }
            )
        _RT_MODEL = FWModel(model_name, **kwargs)
    except Exception as e:
        # Fail hard in worker; parent will recreate if needed
        raise RuntimeError(f"Worker init failed: {e}")


def _rt_transcribe_bytes(
    audio_bytes: bytes, sample_rate: int, prompt: str
) -> tuple[str, list]:
    global _RT_MODEL
    if _RT_MODEL is None:
        raise RuntimeError("Worker model not initialized")
    import numpy as _np

    # Convert to float32 mono
    audio_array = (
        _np.frombuffer(audio_bytes, dtype=_np.int16).astype(_np.float32) / 32767.0
    )
    if audio_array.size == 0:
        return "", []
    # Energy gate
    energy = float(_np.sqrt(_np.mean(audio_array**2)))
    if energy < 0.005:
        return "", []
    # Decode with low-latency settings
    fw_params = {
        "language": "en",
        "task": "transcribe",
        "beam_size": 1,
        "best_of": 1,
        "vad_filter": False,
        "vad_parameters": {"min_silence_duration_ms": 600, "speech_pad_ms": 150},
        "temperature": 0.0,
        "initial_prompt": prompt,
        "condition_on_previous_text": False,
        "word_timestamps": False,
        "no_speech_threshold": 0.6,
        "compression_ratio_threshold": 2.3,
        "log_prob_threshold": -0.5,
    }
    segments, _info = _RT_MODEL.transcribe(audio_array, **fw_params)
    parts = []
    for seg in segments:
        parts.append(seg.text.strip())
    return (" ".join(parts).strip(), [])


def _rt_compute_vad_cut_points(
    audio_bytes: bytes, sample_rate: int, min_silence_ms: int
) -> list[int]:
    import webrtcvad

    frame_ms = 20
    bytes_per_frame = int(sample_rate * (frame_ms / 1000.0)) * 2
    frames = [
        audio_bytes[i : i + bytes_per_frame]
        for i in range(0, len(audio_bytes), bytes_per_frame)
    ]
    vad = webrtcvad.Vad(1)
    speech = [
        (len(fr) >= bytes_per_frame) and vad.is_speech(fr, sample_rate) for fr in frames
    ]
    min_frames = max(1, int(min_silence_ms / frame_ms))
    cut_points = []
    run_start = None
    for idx, is_speech in enumerate(speech + [True]):
        if not is_speech and run_start is None:
            run_start = idx
        elif (is_speech or idx == len(speech)) and run_start is not None:
            length = idx - run_start
            if length >= min_frames:
                mid = run_start + length // 2
                cut_sample = mid * bytes_per_frame // 2
                cut_points.append(cut_sample)
            run_start = None
    return cut_points


def _rt_build_windows_from_raw(
    audio_bytes: bytes,
    sample_rate: int,
    win_s: float,
    overlap_s: float,
    cut_points_samples: list[int],
    pad_ms: int,
) -> list[bytes]:
    bytes_per_sec = sample_rate * 2
    win_bytes = int(win_s * bytes_per_sec)
    pad_bytes = int((pad_ms / 1000.0) * bytes_per_sec)
    n = len(audio_bytes)
    cut_points_b = [cp * 2 for cp in cut_points_samples]
    windows = []
    start = 0
    while start < n:
        target_end = min(start + win_bytes, n)
        search_radius = int(1.0 * bytes_per_sec)
        nearest = None
        best = None
        for cp in cut_points_b:
            if abs(cp - target_end) <= search_radius:
                d = abs(cp - target_end)
                if nearest is None or d < best:
                    nearest, best = cp, d
        end = target_end
        if nearest is not None:
            end = max(0, min(n, nearest + pad_bytes))
        if end <= start:
            end = min(n, start + win_bytes)
        windows.append(audio_bytes[start:end])
        if end >= n:
            break
        start = max(0, end - int(overlap_s * bytes_per_sec))
    return windows


def _rt_process_macro_block(
    audio_bytes: bytes,
    sample_rate: int,
    win_s: float,
    overlap_s: float,
    min_silence_ms: int,
    pad_ms: int,
    prompt: str,
) -> str:
    # Build windows entirely in the worker, then transcribe each
    cuts = _rt_compute_vad_cut_points(audio_bytes, sample_rate, min_silence_ms)
    windows = _rt_build_windows_from_raw(
        audio_bytes, sample_rate, win_s, overlap_s, cuts, pad_ms
    )
    stitched = []
    for w in windows:
        text, _ = _rt_transcribe_bytes(w, sample_rate, prompt)
        if text:
            stitched.append(text)
    return " ".join(stitched).strip()


class MinimalTranscriptionManager:
    """Minimal WebSocket transcription manager without database."""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.transcription_service = None  # Lazy loaded for Pass 2 only
        self.session_transcripts: Dict[str, list[str]] = {}
        self.saved_sessions: set[str] = set()
        # Buffer-first additions
        self.session_audio_bytes: Dict[str, bytearray] = {}
        self.session_segments: Dict[str, list[Dict[str, Any]]] = {}
        # Rolling macro-block buffers and tasks
        self.session_macro_bytes: Dict[str, bytearray] = {}
        self.session_tasks: Dict[str, set[asyncio.Task]] = {}
        self.session_finalizing: Dict[str, bool] = {}
        # Provisional file appends state
        self.session_block_count: Dict[str, int] = {}
        self.session_file_locks: Dict[str, asyncio.Lock] = {}
        # Spill full-session audio to disk to reduce memory pressure
        self.session_audio_paths: Dict[str, str] = {}

        # Load configuration
        settings = get_settings()
        ws_config = settings.websocket_settings

        self.sample_rate = ws_config.sample_rate
        self.chunk_duration = ws_config.chunk_duration
        self.heartbeat_interval = ws_config.heartbeat_interval
        self.chunk_timeout = ws_config.chunk_timeout
        # Select live model; enforce base.en on CPU for stability (overrides env)
        self.whisper_model = ws_config.whisper_model_realtime
        try:
            asr_device = settings.asr_settings.device
        except Exception:
            asr_device = "cpu"
        if asr_device.lower() == "cpu" and self.whisper_model != "base.en":
            self.whisper_model = "base.en"

        # Pass 1 macro params (seconds) - larger windows to reduce IPC churn
        self.macro_seconds = 12.0
        # Pass 1 window configuration (used in _process_macro_block)
        self.pass1_window_sec = 12.0
        self.pass1_overlap_sec = 1.0
        # Backpressure: bound in-memory backlog per session
        self.max_in_memory_backlog_sec = 20.0
        # Pass 2 refinement window parameters (unchanged)
        self.win_min = 20.0
        self.win_max = 24.0
        self.overlap_sec = 1.0
        # Guard to avoid overlapping Pass 1 workers per session
        self.session_pass1_running: Dict[str, bool] = {}
        # Create isolated Pass 1 worker process (single worker)
        self._rt_device = asr_device.lower()
        # Use physical cores sensibly for CPU
        import os

        phy = os.cpu_count() or 2
        cpu_threads = (
            max(1, min(phy, 4)) if self._rt_device == "cpu" else 2
        )  # 2-4 threads
        num_workers = 1  # keep 1; extra workers duplicate model memory
        self.proc_pool: Optional[ProcessPoolExecutor] = None
        try:
            try:
                mp.set_start_method("spawn", force=True)
            except RuntimeError:
                pass

            ctx = mp.get_context("spawn")  # hard-pin spawn for the pool
            self.proc_pool = ProcessPoolExecutor(
                max_workers=1,
                mp_context=ctx,  # important
                initializer=_rt_worker_init,
                initargs=(
                    self.whisper_model,
                    self._rt_device,
                    cpu_threads,
                    num_workers,
                ),
            )
        except Exception as e:
            print(f"⚠️ Failed to start Pass1 worker process: {e}")

        print(f"📊 Loaded WebSocket Configuration:")
        print(f"   Sample Rate: {self.sample_rate} Hz")
        print(f"   Chunk Duration: {self.chunk_duration} seconds")
        print(f"   Chunk Timeout: {self.chunk_timeout} seconds")
        print(f"   Heartbeat Interval: {self.heartbeat_interval} seconds")
        print(f"   Live model: {self.whisper_model}")
        print(
            f"   Pass1 window: {self.pass1_window_sec}s, overlap: {self.pass1_overlap_sec}s, trigger: {self.macro_seconds}s"
        )

        # RSS memory monitor for debugging
        try:
            import psutil

            self._ps = psutil.Process(os.getpid())

            async def _mem_watch():
                while True:
                    rss = self._ps.memory_info().rss / (1024 * 1024)
                    print(f"🧠 RSS(main): {rss:.0f} MB")
                    await asyncio.sleep(5)

            asyncio.create_task(_mem_watch())
        except Exception:
            pass

    async def connect(self, websocket: WebSocket, session_id: str) -> bool:
        """Connect a WebSocket client."""
        await websocket.accept()
        self.active_connections[session_id] = websocket
        self.session_transcripts[session_id] = []
        # Initialize buffers and tasks for this session
        self.session_audio_bytes[session_id] = bytearray()
        self.session_segments[session_id] = []
        self.session_macro_bytes[session_id] = bytearray()
        self.session_tasks[session_id] = set()
        self.session_finalizing[session_id] = False
        self.session_block_count[session_id] = 0
        self.session_file_locks[session_id] = asyncio.Lock()
        # Maintain tail words for cross-block deduplication
        if not hasattr(self, "session_dedup_tail_words"):
            self.session_dedup_tail_words: Dict[str, list[str]] = {}
        self.session_dedup_tail_words[session_id] = []
        # Initialize pass1 running flag
        self.session_pass1_running[session_id] = False
        # Prepare per-session audio spill file
        os.makedirs("logs", exist_ok=True)
        audio_path = os.path.join("logs", f"session_audio_{session_id}.pcm")
        try:
            with open(audio_path, "wb") as _f:
                pass
        except Exception as e:
            print(f"⚠️ Could not initialize session audio file: {e}")
        self.session_audio_paths[session_id] = audio_path

        # Send connection confirmation
        await self.send_message(session_id, {
            "type": "connected",
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "sample_rate": self.sample_rate,
                "chunk_duration": self.chunk_duration
            }
        })

        print(f"✅ WebSocket connected: {session_id}")
        return True

    async def disconnect(self, session_id: str):
        """Disconnect a WebSocket client."""
        if session_id in self.active_connections:
            del self.active_connections[session_id]
        # Wait for any background tasks to finish
        await self._wait_session_tasks(session_id)
        # Persist transcript on disconnect if not already saved
        await self._persist_transcript(session_id)
        print(f"🔌 WebSocket disconnected: {session_id}")

    async def send_message(self, session_id: str, message: Dict[str, Any]):
        """Send a message to a WebSocket client."""
        if session_id not in self.active_connections:
            return
        ws = self.active_connections[session_id]
        if ws.application_state != WebSocketState.CONNECTED:
            return
        try:
            await ws.send_text(json.dumps(message))
        except Exception as e:
            print(f"❌ Failed to send message to {session_id}: {e}")

    async def process_audio_data(self, session_id: str, audio_data: bytes):
        """Buffer incoming audio data; launch macro-block Pass 1 in background."""
        try:
            # Spill full-session audio to disk instead of keeping in memory
            audio_path = self.session_audio_paths.get(session_id)
            if audio_path:
                try:
                    with open(audio_path, "ab") as af:
                        af.write(audio_data)
                except Exception as werr:
                    print(f"⚠️ Failed to append to session audio file: {werr}")
            mbuf = self.session_macro_bytes.setdefault(session_id, bytearray())
            mbuf.extend(audio_data)
            # Backpressure: bound RAM usage by dropping oldest bytes from macro buffer
            bytes_per_sec = self.sample_rate * 2
            max_bytes = int(self.max_in_memory_backlog_sec * bytes_per_sec)
            if len(mbuf) > max_bytes:
                # just drop the oldest bytes from the in-memory macro buffer
                # (they're already written to session_audio file above)
                excess = len(mbuf) - max_bytes
                del mbuf[:excess]  # free RAM, no duplicate write
            # If macro buffer exceeds threshold duration, process it in background
            samples = len(mbuf) // 2  # int16
            duration = samples / float(self.sample_rate)
            if (
                duration >= self.macro_seconds
                and not self.session_finalizing.get(session_id, False)
                and not self.session_pass1_running.get(session_id, False)
            ):
                macro_blob = bytes(mbuf)
                self.session_macro_bytes[session_id] = bytearray()
                self.session_pass1_running[session_id] = True
                task = asyncio.create_task(
                    self._process_macro_block(session_id, macro_blob)
                )
                self.session_tasks[session_id].add(task)
                task.add_done_callback(
                    lambda t: self.session_tasks[session_id].discard(t)
                )
        except Exception as e:
            print(f"❌ Audio buffering error for {session_id}: {e}")
            await self.send_message(
                session_id,
                {"type": "error", "message": f"Audio buffering failed: {str(e)}"},
            )

    async def handle_text_message(self, session_id: str, message: Dict[str, Any]):
        """Handle text control messages."""
        msg_type = message.get("type")

        if msg_type == "ping":
            await self.send_message(session_id, {"type": "pong"})
        elif msg_type == "pong":
            return
        elif msg_type == "end_session":
            await self.send_message(
                session_id,
                {"type": "session_ended", "timestamp": datetime.now().isoformat()},
            )
            # Flip finalizing flag and process any remaining macro buffer
            self.session_finalizing[session_id] = True
            remainder = self.session_macro_bytes.get(session_id, bytearray())
            if remainder and not self.session_pass1_running.get(session_id, False):
                macro_blob = bytes(remainder)
                self.session_macro_bytes[session_id] = bytearray()
                self.session_pass1_running[session_id] = True
                task = asyncio.create_task(
                    self._process_macro_block(session_id, macro_blob)
                )
                self.session_tasks[session_id].add(task)
                task.add_done_callback(
                    lambda t: self.session_tasks[session_id].discard(t)
                )

            # Concurrency guard: Wait for macro flush to complete, then run final pass
            # This prevents CPU contention and reduces latency
            await self._wait_session_tasks(session_id)

            # Now run Pass 2 refinement on full audio (single worker per session)
            task2 = asyncio.create_task(self._process_full_session_refined(session_id))
            self.session_tasks[session_id].add(task2)
            task2.add_done_callback(lambda t: self.session_tasks[session_id].discard(t))
            # We do not close here; disconnect() will persist after tasks
        elif msg_type == "config":
            await self.send_message(
                session_id,
                {
                    "type": "config_updated",
                    "config": {
                        "sample_rate": self.sample_rate,
                        "chunk_duration": self.chunk_duration,
                    },
                },
            )
        else:
            print(f"⚠️ Unknown message type: {msg_type}")

    async def _wait_session_tasks(self, session_id: str):
        tasks = list(self.session_tasks.get(session_id, set()))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    # -------- Pass 1: rolling macro-block with VAD segmentation --------
    def _get_provisional_path(self, session_id: str) -> str:
        os.makedirs("logs", exist_ok=True)
        return os.path.join("logs", f"provisional_{session_id}.txt")

    async def _process_macro_block(self, session_id: str, audio_bytes: bytes):
        """Process macro block entirely in worker to avoid main process memory spikes."""
        try:
            loop = asyncio.get_running_loop()
            prompt = self._soc_prompt("")
            stitched_text = ""
            if self.proc_pool is not None:
                try:
                    stitched_text = await loop.run_in_executor(
                        self.proc_pool,
                        _rt_process_macro_block,
                        audio_bytes,
                        self.sample_rate,
                        self.pass1_window_sec,
                        self.pass1_overlap_sec,
                        600,  # min_silence_ms
                        150,  # pad_ms
                        prompt,
                    )
                except BrokenProcessPool:
                    print("⚠️ Pass1 worker crashed; recreating…")
                    try:
                        self.proc_pool.shutdown(cancel_futures=True)
                    except Exception:
                        pass
                    try:
                        ctx = mp.get_context("spawn")
                        cpu_threads = (
                            max(1, min(os.cpu_count() or 2, 4))
                            if self._rt_device == "cpu"
                            else 2
                        )
                        num_workers = 1
                        self.proc_pool = ProcessPoolExecutor(
                            max_workers=1,
                            mp_context=ctx,
                            initializer=_rt_worker_init,
                            initargs=(
                                self.whisper_model,
                                self._rt_device,
                                cpu_threads,
                                num_workers,
                            ),
                        )
                    except Exception as e:
                        print(f"❌ Could not recreate Pass1 worker: {e}")
                        self.proc_pool = None

            if stitched_text:
                block_joined = self._clean_domain_text(stitched_text)
                import re as _re

                tail_words = self.session_dedup_tail_words.get(session_id, [])
                if tail_words:
                    tail = " ".join(tail_words)
                    if block_joined.lower().startswith(tail):
                        block_joined = block_joined[len(tail) :].lstrip()
                self.session_dedup_tail_words[session_id] = _re.findall(
                    r"\w+", block_joined.lower()
                )[-20:]
                self.session_block_count[session_id] += 1
                header = f"=== Pass1 Block {self.session_block_count[session_id]} @ {datetime.now().isoformat(timespec='seconds')} ===\n"
                to_write = header + block_joined + "\n\n"
                try:
                    async with self.session_file_locks[session_id]:
                        with open(
                            self._get_provisional_path(session_id),
                            "a",
                            encoding="utf-8",
                        ) as pf:
                            pf.write(to_write)
                    print(
                        f"📝 Appended provisional block {self.session_block_count[session_id]} to {self._get_provisional_path(session_id)}"
                    )
                except Exception as werr:
                    print(
                        f"⚠️ Failed to append provisional transcript for {session_id}: {werr}"
                    )
        except Exception as e:
            print(f"❌ Macro block processing error for {session_id}: {e}")
        finally:
            self.session_pass1_running[session_id] = False

    def _vad_segment_pcm(self, audio_bytes: bytes, sample_rate: int) -> list[bytes]:
        import webrtcvad

        # Slightly less aggressive to avoid clipping quiet speech
        vad = webrtcvad.Vad(1)
        frame_ms = 20
        bytes_per_frame = int(sample_rate * (frame_ms / 1000.0)) * 2
        frames = [
            audio_bytes[i : i + bytes_per_frame]
            for i in range(0, len(audio_bytes), bytes_per_frame)
        ]
        voiced_chunks: list[bytes] = []
        cur = bytearray()

        def frame_is_speech(b: bytes) -> bool:
            if len(b) < bytes_per_frame:
                return False
            return vad.is_speech(b, sample_rate)

        for fr in frames:
            if frame_is_speech(fr):
                cur.extend(fr)
            else:
                if cur:
                    voiced_chunks.append(bytes(cur))
                    cur = bytearray()
        if cur:
            voiced_chunks.append(bytes(cur))
        return voiced_chunks

    def _compute_vad_cut_points(
        self,
        audio_bytes: bytes,
        sample_rate: int,
        min_silence_ms: int = 600,
    ) -> list[int]:
        """Return sample indices of long silences to use as preferred cut points.
        We do not remove silence; we only mark where to cut windows.
        """
        import webrtcvad

        vad = webrtcvad.Vad(1)
        frame_ms = 20
        bytes_per_frame = int(sample_rate * (frame_ms / 1000.0)) * 2
        frames = [
            audio_bytes[i : i + bytes_per_frame]
            for i in range(0, len(audio_bytes), bytes_per_frame)
        ]
        speech_flags: list[bool] = []
        for fr in frames:
            if len(fr) < bytes_per_frame:
                speech_flags.append(False)
            else:
                speech_flags.append(vad.is_speech(fr, sample_rate))

        # Find non-speech runs >= min_silence_ms
        min_frames = max(1, int(min_silence_ms / frame_ms))
        cut_points: list[int] = []
        run_start = None
        for idx, is_speech in enumerate(speech_flags + [True]):  # sentinel
            if not is_speech and run_start is None:
                run_start = idx
            elif (is_speech or idx == len(speech_flags)) and run_start is not None:
                length = idx - run_start
                if length >= min_frames:
                    # Choose midpoint of silence run as preferred cut
                    mid_frame = run_start + length // 2
                    cut_sample = mid_frame * bytes_per_frame // 2  # samples (int16)
                    cut_points.append(cut_sample)
                run_start = None
        return cut_points

    def _build_windows_from_raw(
        self,
        audio_bytes: bytes,
        sample_rate: int,
        win_s: float,
        overlap_s: float,
        cut_points_samples: list[int],
        pad_ms: int = 150,
    ) -> list[bytes]:
        """Build fixed-length windows from RAW PCM, snapping boundaries to nearby VAD cut points.
        Keep internal silences. Returns list of window bytes.
        """
        bytes_per_sec = sample_rate * 2
        win_bytes = int(win_s * bytes_per_sec)
        hop_bytes = int((win_s - overlap_s) * bytes_per_sec)
        pad_bytes = int((pad_ms / 1000.0) * bytes_per_sec)

        windows: list[bytes] = []
        start = 0
        n = len(audio_bytes)
        # Convert cut points from samples to bytes
        cut_points_b = [cp * 2 for cp in cut_points_samples]

        while start < n:
            target_end = min(start + win_bytes, n)
            # Try to snap target_end to nearest cut point within ±1s
            search_radius = int(1.0 * bytes_per_sec)
            nearest = None
            nearest_dist = None
            for cp in cut_points_b:
                if abs(cp - target_end) <= search_radius:
                    d = abs(cp - target_end)
                    if nearest is None or d < nearest_dist:
                        nearest = cp
                        nearest_dist = d
            end = target_end
            if nearest is not None:
                end = max(0, min(n, nearest))
                # Apply small pad
                end = max(0, min(n, end + pad_bytes))

            if end <= start:
                end = min(n, start + win_bytes)

            windows.append(audio_bytes[start:end])
            if end >= n:
                break
            # Advance by hop
            start = max(0, end - int(overlap_s * bytes_per_sec))

        return windows

    def _merge_windows_by_timestamps(
        self,
        prev_text: str,
        prev_segs: list[Dict[str, Any]],
        curr_text: str,
        curr_segs: list[Dict[str, Any]],
        overlap_s: float,
    ) -> tuple[str, list[Dict[str, Any]]]:
        """Merge two consecutive windows keeping the later window's words in the overlap.
        Trims words from prev that fall into the last overlap_s seconds.
        """
        # Compute prev window duration
        prev_dur = 0.0
        for s in prev_segs:
            prev_dur = max(prev_dur, float(s.get("end", 0.0)))

        # Filter prev words/segments to drop words starting in the overlap tail
        def filter_segment(seg: Dict[str, Any]) -> Dict[str, Any]:
            words = seg.get("words") or []
            kept_words = [
                w
                for w in words
                if float(w.get("start", 0.0)) < max(0.0, prev_dur - overlap_s)
            ]
            if words and kept_words != words:
                # Adjust text by joining kept words
                seg = dict(seg)
                seg["words"] = kept_words
                seg["text"] = " ".join(
                    w.get("word", "").strip() for w in kept_words
                ).strip()
                # Adjust end
                if kept_words:
                    seg["end"] = float(kept_words[-1].get("end", seg.get("start", 0.0)))
            return seg

        trimmed_prev_segs = [filter_segment(s) for s in prev_segs]
        # Merge texts
        trimmed_prev_text = " ".join(
            s.get("text", "").strip() for s in trimmed_prev_segs if s.get("text")
        ).strip()
        merged_text = (
            (trimmed_prev_text + " " + (curr_text or "")).strip()
            if curr_text
            else trimmed_prev_text
        )
        merged_text = self._clean_domain_text(merged_text)

        merged_segs = trimmed_prev_segs + curr_segs
        return merged_text, merged_segs

    def _build_windows(
        self, segments: list[bytes], min_s: float, max_s: float, overlap_s: float
    ) -> list[bytes]:
        windows: list[bytes] = []
        cur = bytearray()
        cur_dur = 0.0
        bytes_per_sec = self.sample_rate * 2
        for seg in segments:
            seg_dur = len(seg) / bytes_per_sec
            if cur_dur + seg_dur <= max_s:
                cur.extend(seg)
                cur_dur += seg_dur
                if cur_dur >= min_s:
                    # finalize window with overlap
                    windows.append(bytes(cur))
                    # keep tail overlap
                    keep = int(overlap_s * bytes_per_sec)
                    cur = bytearray(cur[-keep:]) if keep < len(cur) else bytearray(cur)
                    cur_dur = len(cur) / bytes_per_sec
            else:
                if cur:
                    windows.append(bytes(cur))
                cur = bytearray(seg)
                cur_dur = seg_dur
        if cur_dur >= min_s / 2 and cur:  # last partial
            windows.append(bytes(cur))
        return windows

    def _soc_prompt(self, prior: str) -> str:
        base = (
            "This is a Start of Care (SOC) visit for Home Health. The transcript will contain a conversation\n"
            "between a nurse and a patient. Use U.S. medical English. Maintain accuracy for medications,\n"
            "dosages, numbers, and OASIS-E terminology.\n\n"
            "Key terms to bias: OASIS-E (M1030, M1040, M1050, M1060, M1400, M1800, M1860); sections: medication reconciliation,\n"
            "past medical history, ADLs/IADLs, functional status, cognitive assessment, fall risk, PHQ-2/PHQ-9, skin/wound,\n"
            "nutrition, pain rating scale, oxygen use, assistive devices. Clinical vocab: edema, orthopnea, dyspnea,\n"
            "gait, ambulation, vital signs (blood pressure, heart rate, respiratory rate, oxygen saturation, temperature).\n"
            "Abbreviations: BID, TID, QID, PRN, mg, mL, L, %, HbA1c, CHF, COPD, DM2, HTN. Common meds: metformin, lisinopril,\n"
            "furosemide, levothyroxine, insulin glargine, warfarin, apixaban, atorvastatin.\n\n"
            "Formatting: Expand numbers clearly (e.g., 120/80, 7.5 mg). Use correct punctuation. Do not hallucinate.\n"
            "Never insert YouTube-style phrases like 'like, share and subscribe' or unrelated boilerplate.\n"
            "Prefer clinical phrasing: 'Patient reports', 'Nurse asks', 'Vitals', 'Medication reconciliation'.\n"
        )
        return (base + ("\nContext: " + prior if prior else "")).strip()

    def _clean_domain_text(self, text: str) -> str:
        """Domain-specific cleaning for SOC provisional text: normalize meds, filter boilerplate, fix common ASR slips."""
        import re

        t = text.strip()
        # Remove common non-clinical boilerplate/hallucinations
        junk_phrases = [
            "thank you for watching",
            "please like, share and subscribe",
            "subscribe for more",
            "hit the like button",
        ]
        for jp in junk_phrases:
            if jp in t.lower():
                t = t.lower().replace(jp, " ")

        # Normalize medical terms
        t = self._normalize_medical_terms(t)

        # Safer numeric format fixes - only replace " over " when flanked by digits
        t = re.sub(r"(\d+)\s+over\s+(\d+)", r"\1/\2", t)

        # Remove known video boilerplate phrases commonly hallucinated
        junk_phrases2 = [
            "this video is not meant to replace proper medical",
            "thanks for watching",
        ]
        tl = t.lower()
        for jp in junk_phrases2:
            if jp in tl:
                tl = tl.replace(jp, " ")
        t = tl

        # Light de-dup of speaker tags artifacts
        replacements = {
            "Patient number.": "Patient: No.",
            "Nurse,": "Nurse:",
            "Patient.": "Patient:",
        }
        for k, v in replacements.items():
            t = t.replace(k, v)

        # Collapse whitespace
        t = " ".join(t.split())
        return t

    def _transcribe_window(
        self, window_bytes: bytes, prompt: str, pass2: bool
    ) -> tuple[str, list[Dict[str, Any]]]:
        # Lazy load transcription service for Pass 2 only
        if self.transcription_service is None:
            from transcription import TranscriptionService  # lazy module import

            self.transcription_service = TranscriptionService()

        # Direct array transcription - no temp files
        audio_array = (
            np.frombuffer(window_bytes, dtype=np.int16).astype(np.float32) / 32767.0
        )

        # Validate audio data
        if len(audio_array) == 0:
            print(f"⚠️ Empty audio window, skipping transcription")
            return "", []

        # Check for sufficient audio energy (lowered threshold since VAD is enabled)
        audio_energy = np.sqrt(np.mean(audio_array**2))
        if audio_energy < 0.005:  # Lowered from 0.001 to avoid dropping quiet speech
            print(f"⚠️ Silent audio window (energy: {audio_energy:.6f}), skipping")
            return "", []

        # Pass settings - Optimized for CPU performance
        if pass2:
            # Final pass: CPU-optimized for stability
            params = dict(
                temperature=[0.0],  # single pass
                best_of=1,
                patience=0.8,  # was 1.1
                beam_size=4,  # was 8
                word_timestamps=True,
                vad_filter=True,
                vad_parameters={
                    "min_silence_duration_ms": 600,
                    "speech_pad_ms": 150,
                },
                no_speech_threshold=0.6,
                compression_ratio_threshold=2.3,
                logprob_threshold=-0.5,
            )
        else:
            # Live pass: Optimized for speed (beam_size=2 for lower latency)
            params = dict(
                temperature=0.0,  # Single temperature for speed
                best_of=1,
                patience=0.5,
                beam_size=1,  # Greedy decoding for low CPU
                word_timestamps=False,  # Disable in Pass 1 to reduce memory/CPU
                vad_filter=False,
                vad_parameters={
                    "min_silence_duration_ms": 600,
                    "speech_pad_ms": 150,
                },
                no_speech_threshold=0.6,
                compression_ratio_threshold=2.3,
                logprob_threshold=-0.5,
            )

        result = self.transcription_service.transcribe_array(
            audio_array,
            sr=self.sample_rate,
            model_name=self.whisper_model,
            language="en",
            task="transcribe",
            initial_prompt=prompt,
            condition_on_previous_text=pass2,  # Disable for Pass 1 to reduce drift
            **params,
        )
        text = (result.get("text") or "").strip()
        segs = result.get("segments") or []
        return text, segs

    # -------- Pass 2: refinement on full audio after stop --------
    async def _process_full_session_refined(self, session_id: str):
        audio_path = self.session_audio_paths.get(session_id)
        if (
            not audio_path
            or not os.path.exists(audio_path)
            or os.path.getsize(audio_path) == 0
        ):
            return
        # Read full session audio from disk for Pass 2 (safer memory profile)
        try:
            with open(audio_path, "rb") as f:
                raw_bytes = f.read()
        except Exception as e:
            print(f"❌ Failed to read session audio file for {session_id}: {e}")
            return
        # Re-segment full audio via VAD
        segments_pcm = await asyncio.to_thread(
            self._vad_segment_pcm, raw_bytes, self.sample_rate
        )
        windows = self._build_windows(
            segments_pcm, self.win_min, self.win_max, self.overlap_sec
        )
        # Final pass: Use last 1-2 sentences as context (≤ 600 chars)
        prior_ctx = " ".join(self.session_transcripts.get(session_id, []))[-600:]
        prompt = self._soc_prompt(prior_ctx)
        refined_texts: list[str] = []
        refined_segments: list[Dict[str, Any]] = []
        for w in windows:
            t, s = await asyncio.to_thread(self._transcribe_window, w, prompt, True)
            if t:
                refined_texts.append(self._normalize_medical_terms(t))
            if s:
                refined_segments.extend(s)
        if refined_texts:
            # Replace provisional transcript with refined one
            self.session_transcripts[session_id] = [
                self._join_with_dedup(refined_texts)
            ]
        if refined_segments:
            self.session_segments[session_id] = refined_segments
        # Cleanup large raw audio file after refinement completes
        try:
            os.remove(audio_path)
        except Exception:
            pass

    def _join_with_dedup(self, texts: list[str]) -> str:
        out: list[str] = []
        prev_tail_words = []
        import re

        for t in texts:
            tokens = re.findall(r"\w+", t)
            if prev_tail_words and t.lower().startswith(" ".join(prev_tail_words)):
                # trim duplicated prefix
                trim_len = len(" ".join(prev_tail_words))
                t = t[trim_len:].lstrip()
            out.append(t)
            prev_tail_words = re.findall(r"\w+", t.lower())[-20:]
        return " ".join(out).strip()

    def _normalize_medical_terms(self, text: str) -> str:
        mapping = {
            "listenopryl": "lisinopril",
            "lisnopril": "lisinopril",
            "metaprolol": "metoprolol",
            "insulin glaging": "insulin glargine",
            "insulin-glaging": "insulin glargine",
            "atovastatin": "atorvastatin",
            "acetominophen": "acetaminophen",
        }
        out = text
        for wrong, right in mapping.items():
            out = out.replace(wrong, right)
            out = out.replace(wrong.capitalize(), right.capitalize())
        return out

    async def _persist_transcript(self, session_id: str):
        """Persist the full transcript for a session to logs/ as a txt file."""
        if session_id in self.saved_sessions:
            return
        texts = self.session_transcripts.get(session_id, [])
        if not texts:
            return
        # Ensure logs directory exists
        os.makedirs("logs", exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        output_path = os.path.join("logs", f"transcript_{timestamp}_{session_id}.txt")
        json_path = os.path.join("logs", f"transcript_{timestamp}_{session_id}.json")
        srt_path = os.path.join("logs", f"transcript_{timestamp}_{session_id}.srt")
        content = [
            "Transcript Recording",
            "===================",
            f"Date: {datetime.now().isoformat(timespec='seconds')}",
            f"Session ID: {session_id}",
            "",
            "Transcript:",
            "-----------",
            " ".join(texts).strip(),
            "",
            "---",
            "Generated by MinimalTranscriptionServer",
        ]
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("\n".join(content))
            # Save segments JSON and SRT if available
            import json as _json

            segments = self.session_segments.get(session_id, [])
            if segments:
                with open(json_path, "w", encoding="utf-8") as jf:
                    _json.dump(
                        {
                            "session_id": session_id,
                            "created_at": datetime.now().isoformat(timespec="seconds"),
                            "model": self.whisper_model,
                            "segments": segments,
                        },
                        jf,
                        ensure_ascii=False,
                        indent=2,
                    )
                with open(srt_path, "w", encoding="utf-8") as sfh:
                    sfh.write(self._segments_to_srt(segments))
            self.saved_sessions.add(session_id)
            print(f"💾 Transcript saved to: {output_path}")
        except Exception as e:
            print(f"❌ Failed to save transcript for {session_id}: {e}")

    def _segments_to_srt(self, segments: list[Dict[str, Any]]) -> str:
        def fmt(t: float) -> str:
            if t < 0:
                t = 0
            h = int(t // 3600)
            m = int((t % 3600) // 60)
            s = int(t % 60)
            ms = int((t - int(t)) * 1000)
            return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

        lines = []
        for i, seg in enumerate(segments, 1):
            start = fmt(seg.get("start", 0.0))
            end = fmt(seg.get("end", 0.0))
            text = seg.get("text", "").strip()
            lines.append(str(i))
            lines.append(f"{start} --> {end}")
            lines.append(text)
            lines.append("")
        return "\n".join(lines)


# Create FastAPI app
app = FastAPI(title="Minimal WebSocket Transcription Server")

# Transcription manager is created on startup to avoid initializing during worker import
transcription_manager: Optional[MinimalTranscriptionManager] = None


@app.on_event("startup")
async def _startup_init_manager():
    global transcription_manager
    if transcription_manager is None:
        transcription_manager = MinimalTranscriptionManager()


@app.on_event("shutdown")
async def _shutdown_cleanup():
    global transcription_manager
    if transcription_manager and transcription_manager.proc_pool:
        try:
            transcription_manager.proc_pool.shutdown(wait=True, cancel_futures=True)
        except Exception:
            pass


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return JSONResponse({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "minimal_transcription_server"
    })


@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    """WebSocket endpoint for transcription."""
    session_id = str(uuid.uuid4())

    try:
        if transcription_manager is None:
            await websocket.close()
            return
        # Connect
        await transcription_manager.connect(websocket, session_id)

        while True:
            try:
                # Check WebSocket state
                if websocket.client_state.name == "DISCONNECTED":
                    print(f"🔌 Client disconnected (state check): {session_id}")
                    break

                # Receive message with timeout
                try:
                    message = await asyncio.wait_for(
                        websocket.receive(), timeout=transcription_manager.chunk_timeout
                    )
                except asyncio.TimeoutError:
                    # Send ping to keep connection alive
                    await transcription_manager.send_message(
                        session_id, {"type": "ping"}
                    )
                    continue

                # Check for disconnect message
                if message.get("type") == "websocket.disconnect":
                    print(f"🔌 Disconnect message received: {session_id}")
                    break

                if "bytes" in message:
                    # Handle audio data
                    audio_data = message["bytes"]
                    await transcription_manager.process_audio_data(session_id, audio_data)

                elif "text" in message:
                    # Handle text messages
                    try:
                        text_message = json.loads(message["text"])
                        await transcription_manager.handle_text_message(session_id, text_message)
                    except json.JSONDecodeError:
                        print(f"⚠️ Invalid JSON from {session_id}")

            except WebSocketDisconnect:
                print(f"🔌 WebSocket client disconnected: {session_id}")
                break

            except Exception as e:
                error_msg = str(e)

                # Check for disconnect-related errors
                if ("disconnect" in error_msg.lower() or 
                    "receive" in error_msg.lower() or
                    ("connection" in error_msg.lower() and "closed" in error_msg.lower())):
                    print(f"🔌 Connection closed: {session_id}")
                    break

                print(f"❌ Error processing message: {error_msg}")
                try:
                    await transcription_manager.send_message(session_id, {
                        "type": "error",
                        "message": f"Processing error: {error_msg}"
                    })
                except:
                    print(f"⚠️ Could not send error message to {session_id}")
                    break

    except Exception as e:
        print(f"❌ WebSocket error for {session_id}: {e}")

    finally:
        await transcription_manager.disconnect(session_id)


if __name__ == "__main__":
    print("🚀 Starting Minimal WebSocket Transcription Server")
    print("=" * 60)
    print("📡 WebSocket endpoint: ws://localhost:8001/ws/transcribe")
    print("🏥 Health check: http://localhost:8001/health")
    print("=" * 60)
    
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8001,
        log_level="info"
    )
