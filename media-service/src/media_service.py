"""
Media Service: RTP endpoint for handling audio streams.
Implements VAD, barge-in, and AI integration.
"""

import asyncio
import logging
import os
import socket
from pathlib import Path
from typing import Optional
import numpy as np
from aiohttp import web
import json
import io
import re
from collections import deque

from .rtp_handler import RTPHandler
from .audio_codec import ulaw_to_linear, linear_to_ulaw, alaw_to_linear
from .vad import EnergyVAD
from .ai_providers import create_asr_client, create_llm_client, create_tts_client, ASRClient, LLMClient, TTSClient

def _resolve_path_candidates(p: str) -> list[Path]:
    raw = (p or "").strip().strip('"').strip("'")
    if not raw:
        return []

    path = Path(raw)
    if path.is_absolute():
        return [path]

    try:
        this_file = Path(__file__).resolve()
        media_service_dir = this_file.parent.parent  # .../media-service
        repo_root = media_service_dir.parent
    except Exception:
        media_service_dir = None
        repo_root = None

    candidates: list[Path] = []
    candidates.append(Path.cwd() / path)
    if media_service_dir:
        candidates.append(media_service_dir / path)
    if repo_root:
        candidates.append(repo_root / path)
    return candidates


def get_system_prompt() -> str:
    """
    Resolve the system prompt from env.

    Priority:
    - SYSTEM_PROMPT_FILE (file contents)
    - SYSTEM_PROMPT (plain string)
    - default prompt
    """
    prompt_file = os.getenv("SYSTEM_PROMPT_FILE")
    response_language = (os.getenv("RESPONSE_LANGUAGE") or "").strip()

    def _apply_language(p: str) -> str:
        if not response_language:
            return p
        # Keep this as a small, deterministic suffix so the prompt file can remain clean.
        return (
            f"{p.strip()}\n\n"
            "Output language:\n"
            f"- Always respond in {response_language}.\n"
            f"- Use natural, idiomatic, grammatically correct {response_language}.\n"
            f"- Do not invent words; avoid malformed or pseudo-{response_language}.\n"
            "- Keep responses coherent and answer the user's question.\n"
            "- Avoid English except for proper nouns / technical terms.\n"
        ).strip()
    if prompt_file:
        for candidate in _resolve_path_candidates(prompt_file):
            try:
                if candidate.exists() and candidate.is_file():
                    text = candidate.read_text(encoding="utf-8")
                    if text.strip():
                        logger.info("Loaded system prompt from file", {"path": str(candidate)})
                        return _apply_language(text)
            except Exception:
                continue

        raise RuntimeError(
            f"SYSTEM_PROMPT_FILE was set but could not be read: {prompt_file} "
            f"(tried: {[str(p) for p in _resolve_path_candidates(prompt_file)]})"
        )

    prompt = os.getenv("SYSTEM_PROMPT")
    if prompt and prompt.strip():
        return _apply_language(prompt)

    return _apply_language("You are a helpful assistant.")


def _load_dotenv_if_present() -> None:
    """
    Load environment variables from .env files (if present).

    We try, in order:
    - repo root `.env` (parent of `media-service/`)
    - `media-service/.env`

    This lets you run from either repo root or the media-service directory.
    """
    try:
        from dotenv import load_dotenv
    except Exception:
        # Optional at runtime; required if you want .env loading.
        return

    try:
        this_file = Path(__file__).resolve()
        media_service_dir = this_file.parent.parent  # .../media-service
        repo_root = media_service_dir.parent

        candidates = [
            repo_root / ".env",
            media_service_dir / ".env",
        ]

        loaded_any = False
        for p in candidates:
            if p.exists() and p.is_file():
                # Do not override already-set env vars (process env wins).
                load_dotenv(dotenv_path=str(p), override=False)
                loaded_any = True

        # We intentionally don't log which vars were loaded to avoid leaking secrets.
        if loaded_any:
            logging.getLogger(__name__).info(".env loaded (process env takes precedence)")
    except Exception:
        # Never block service startup because of dotenv.
        return


_load_dotenv_if_present()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


class MediaSession:
    """Manages a single media session (one call)."""
    
    def __init__(self, session_id: str, local_port: int, remote_addr: tuple, 
                 asr_client: ASRClient, llm_client: LLMClient, tts_client: TTSClient,
                 mode: str = "full"):
        """
        Initialize media session.
        
        Args:
            session_id: Unique session identifier
            local_port: Local UDP port for RTP
            remote_addr: Remote address (host, port) for sending RTP
            asr_client: ASR client instance
            llm_client: LLM client instance
            tts_client: TTS client instance
            mode: Operation mode ("echo", "tts_only", "full")
        """
        self.session_id = session_id
        self.local_port = local_port
        self.remote_addr = remote_addr
        self.mode = mode
        
        # Audio processing
        self.sample_rate = 8000
        self.frame_duration_ms = 20
        self.samples_per_frame = (self.sample_rate * self.frame_duration_ms) // 1000
        
        # RTP handler
        self.rtp_handler = RTPHandler(ssrc=hash(session_id) & 0xFFFFFFFF, 
                                      sample_rate=self.sample_rate,
                                      frame_duration_ms=self.frame_duration_ms)
        
        # VAD
        self.vad = EnergyVAD(sample_rate=self.sample_rate, 
                            frame_duration_ms=self.frame_duration_ms)
        
        # AI clients
        self.asr_client = asr_client
        self.llm_client = llm_client
        self.tts_client = tts_client
        
        # State
        self.socket: Optional[socket.socket] = None
        self.running = False
        self.asr_stream_id: Optional[str] = None
        self.tts_active = False
        self.conversation_history: list[dict] = []
        self._receive_task: Optional[asyncio.Task] = None
        self._rtp_ready = asyncio.Event()
        self._greeting_task: Optional[asyncio.Task] = None
        self._greeting_sent = False
        self._was_speech = False
        self._keepalive_task: Optional[asyncio.Task] = None
        self._response_worker_task: Optional[asyncio.Task] = None
        self._response_queue: "asyncio.Queue[str]" = asyncio.Queue()
        self._barge_in_speech_frames = 0
        self._no_rtp_silence_ms = 0
        self._speech_frames = 0
        self._silence_frames = 0
        self._barge_in_buffer: "deque[bytes]" = deque()
        
        # System prompt
        system_prompt = get_system_prompt()
        self.conversation_history.append({"role": "system", "content": system_prompt})
    
    async def start(self):
        """Start the media session."""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.socket.bind(('0.0.0.0', self.local_port))
            bound_addr = self.socket.getsockname()
            logger.info("Socket bound successfully", {
                "session_id": self.session_id,
                "local_port": self.local_port,
                "bound_addr": bound_addr
            })
        except Exception as e:
            logger.error("Failed to bind socket", {
                "session_id": self.session_id,
                "local_port": self.local_port,
                "error": str(e),
                "error_type": type(e).__name__
            })
            raise
        
        self.socket.setblocking(False)
        self.running = True
        
        logger.info("Media session started", {
            "session_id": self.session_id,
            "local_port": self.local_port,
            "remote_addr": self.remote_addr,
            "mode": self.mode,
            "socket_fileno": self.socket.fileno()
        })
        
        # Start ASR stream if not in echo mode
        if self.mode != "echo":
            self.asr_stream_id = await self.asr_client.start_stream()

        # Start response worker in "full" mode so the RTP receive loop never blocks on LLM/TTS.
        if self.mode == "full":
            self._response_worker_task = asyncio.create_task(self._response_worker_loop())
        
        # Start receive loop (store task reference to prevent garbage collection)
        try:
            self._receive_task = asyncio.create_task(self._receive_loop())
            logger.info("Receive loop task created", {
                "session_id": self.session_id,
                "task_done": self._receive_task.done(),
                "task_name": self._receive_task.get_name() if hasattr(self._receive_task, 'get_name') else "unknown"
            })
        except Exception as e:
            logger.error("Failed to create receive loop task", {
                "session_id": self.session_id,
                "error": str(e),
                "error_type": type(e).__name__
            })
            raise
        
        # Send initial silence packet to establish RTP session with Asterisk
        # This helps trigger RTP flow in both directions
        if self.mode == "echo":
            # For echo mode, send a silence packet to establish RTP
            await asyncio.sleep(0.1)  # Wait a moment for remote address to be learned
            silence = np.zeros(self.samples_per_frame, dtype=np.int16)
            try:
                await self._send_audio_frame(silence)
                logger.info("Sent initial silence packet to establish RTP", {
                    "session_id": self.session_id,
                    "remote_addr": self.remote_addr
                })
            except Exception as e:
                import traceback
                error_details = {
                    "session_id": self.session_id,
                    "remote_addr": self.remote_addr,
                    "socket_available": self.socket is not None,
                    "error": str(e),
                    "error_type": type(e).__name__,
                }
                logger.warning(f"Could not send initial silence packet: {json.dumps(error_details)}")
                logger.debug(f"Traceback: {traceback.format_exc()}")
        
        # Send initial greeting if TTS-only or full mode.
        # IMPORTANT: only speak once RTP address is known, otherwise we may send to a placeholder.
        if self.mode in ["tts_only", "full"]:
            self._greeting_task = asyncio.create_task(self._send_greeting_when_rtp_ready())

        # RTP keepalive: some endpoints hang up if they don't receive RTP for ~30s.
        # We send silent PCMU frames when we're otherwise idle.
        keepalive = (os.getenv("RTP_KEEPALIVE", "1") or "").strip().lower() not in {"0", "false", "no", "off"}
        if keepalive and self.mode != "echo":
            self._keepalive_task = asyncio.create_task(self._rtp_keepalive_loop())

    async def _rtp_keepalive_loop(self):
        # Wait for RTP to be ready so we know where to send.
        try:
            await asyncio.wait_for(self._rtp_ready.wait(), timeout=15.0)
        except asyncio.TimeoutError:
            return

        silence = np.zeros(self.samples_per_frame, dtype=np.int16)
        while self.running:
            try:
                # Only send silence when we're not actively speaking.
                if not self.tts_active:
                    await self._send_audio_frame(silence)
                await asyncio.sleep(self.frame_duration_ms / 1000.0)
            except asyncio.CancelledError:
                break
            except Exception:
                # Never crash the session because of keepalive.
                await asyncio.sleep(0.1)
    
    async def _send_greeting_when_rtp_ready(self):
        try:
            await asyncio.wait_for(self._rtp_ready.wait(), timeout=15.0)
        except asyncio.TimeoutError:
            logger.warning("RTP not ready; skipping greeting", {
                "session_id": self.session_id,
                "remote_addr": self.remote_addr,
            })
            return

        if not self.running or self._greeting_sent:
            return

        self._greeting_sent = True
        await self._send_greeting()

    async def _send_greeting(self):
        """Send initial greeting via TTS."""
        greeting = os.getenv(
            "GREETING_TEXT",
            "Hello! I'm ready to help. How can I assist you today?",
        )
        await self._synthesize_and_send(greeting)
    
    async def _receive_loop(self):
        """Main receive loop for RTP packets."""
        try:
            loop = asyncio.get_event_loop()
            
            logger.info("Receive loop started", {
                "session_id": self.session_id,
                "local_port": self.local_port,
                "remote_addr": self.remote_addr,
                "mode": self.mode,
                "socket_bound": self.socket is not None,
                "socket_fileno": self.socket.fileno() if self.socket else None
            })
        except Exception as e:
            logger.error("Error starting receive loop", {
                "session_id": self.session_id,
                "error": str(e),
                "error_type": type(e).__name__
            })
            raise
        
        packet_count = 0
        last_packet_time = asyncio.get_event_loop().time()
        last_no_rtp_log_time = last_packet_time
        recv_timeout_s = float(os.getenv("RTP_RECV_TIMEOUT_S", "1.0") or "1.0")
        try:
            no_rtp_warn_after_s = float(os.getenv("RTP_NO_RTP_WARN_AFTER_S", "15"))
        except ValueError:
            no_rtp_warn_after_s = 15.0
        try:
            no_rtp_warn_every_s = float(os.getenv("RTP_NO_RTP_WARN_EVERY_S", "15"))
        except ValueError:
            no_rtp_warn_every_s = 15.0
        no_rtp_warn_after_s = max(1.0, no_rtp_warn_after_s)
        no_rtp_warn_every_s = max(1.0, no_rtp_warn_every_s)
        
        while self.running:
            try:
                # Receive RTP packet with timeout to allow periodic status checks
                try:
                    data, addr = await asyncio.wait_for(
                        loop.sock_recvfrom(self.socket, 1500),
                        timeout=recv_timeout_s
                    )
                    self._no_rtp_silence_ms = 0
                    now = asyncio.get_event_loop().time()
                    # If we were in a long RTP gap, log resumption once.
                    gap_s = now - last_packet_time
                    if packet_count > 0 and gap_s >= no_rtp_warn_after_s:
                        logger.info("RTP resumed after %.1fs (session_id=%s)", gap_s, self.session_id)
                    last_packet_time = now
                except asyncio.TimeoutError:
                    # No RTP received during the timeout window.
                    await self._handle_no_rtp(timeout_s=recv_timeout_s)
                    current_time = asyncio.get_event_loop().time()
                    since_last_packet = current_time - last_packet_time
                    if self._rtp_ready.is_set() and since_last_packet >= no_rtp_warn_after_s:
                        if current_time - last_no_rtp_log_time >= no_rtp_warn_every_s:
                            logger.warning(
                                "No RTP packets received for %.1fs (session_id=%s, local_port=%s, remote_addr=%s). "
                                "This is often normal during silence with VAD-enabled endpoints.",
                                since_last_packet,
                                self.session_id,
                                self.local_port,
                                self.remote_addr,
                            )
                            last_no_rtp_log_time = current_time
                    continue
                
                packet_count += 1
                
                if packet_count == 1:
                    logger.info("First RTP packet received", {
                        "session_id": self.session_id,
                        "from_addr": addr,
                        "packet_size": len(data)
                    })
                    self._rtp_ready.set()
                
                # Update remote address if changed
                if addr != self.remote_addr:
                    logger.info("Remote address updated", {
                        "session_id": self.session_id,
                        "old_addr": self.remote_addr,
                        "new_addr": addr
                    })
                    self.remote_addr = addr
                    self._rtp_ready.set()
                
                # Parse RTP packet
                result = self.rtp_handler.parse_packet(data)
                if result is None:
                    if packet_count <= 5:
                        logger.debug("RTP packet dropped (invalid or out of order)", {
                            "session_id": self.session_id,
                            "packet_count": packet_count
                        })
                    continue
                
                sequence, pt, payload = result
                
                if packet_count <= 5:
                    logger.debug("RTP packet parsed successfully", {
                        "session_id": self.session_id,
                        "sequence": sequence,
                        "pt": pt,
                        "payload_size": len(payload)
                    })

                # Process based on mode
                if self.mode == "echo":
                    # Echo mode: bounce the RTP payload bytes directly (PCMU) to avoid
                    # any codec conversion issues that manifest as white noise.
                    if packet_count <= 10:
                        logger.info("Echoing RTP payload", {
                            "session_id": self.session_id,
                            "payload_bytes": len(payload),
                            "packet_count": packet_count,
                            "remote_addr": self.remote_addr
                        })
                    try:
                        loop = asyncio.get_event_loop()
                        rtp_packet = self.rtp_handler.create_packet(payload, marker=0)
                        await loop.sock_sendto(self.socket, rtp_packet, self.remote_addr)
                    except Exception as e:
                        logger.error("Error echoing RTP payload", {
                            "session_id": self.session_id,
                            "error": str(e),
                            "error_type": type(e).__name__
                        })
                else:
                    # Decode to PCM16 for VAD/ASR pipeline based on RTP payload type.
                    # 0 = PCMU (ulaw), 8 = PCMA (alaw)
                    if pt == 0:
                        pcm_audio = ulaw_to_linear(payload)
                    elif pt == 8:
                        pcm_audio = alaw_to_linear(payload)
                    else:
                        if packet_count <= 20:
                            logger.warning("Unsupported RTP payload type; dropping packet", {
                                "session_id": self.session_id,
                                "pt": pt,
                                "sequence": sequence,
                                "payload_bytes": len(payload),
                            })
                        continue
                    await self._process_audio_frame(pcm_audio)
                    
            except asyncio.CancelledError:
                logger.info("Receive loop cancelled", {"session_id": self.session_id})
                break
            except Exception as e:
                logger.error("Error in receive loop", {
                    "session_id": self.session_id,
                    "error": str(e),
                    "error_type": type(e).__name__
                })
                await asyncio.sleep(0.01)
        
        logger.info("Receive loop ended", {
            "session_id": self.session_id,
            "total_packets": packet_count
        })

    async def _handle_no_rtp(self, *, timeout_s: float) -> None:
        """
        Some endpoints stop sending RTP during silence.
        If we previously detected speech, treat "no RTP" as silence and finalize ASR.
        """
        if self.mode in {"echo", "tts_only"}:
            return
        if self.tts_active:
            return
        if not self.asr_stream_id:
            return
        if not self._was_speech:
            return

        # Accumulate "silence" time based on missing RTP windows.
        self._no_rtp_silence_ms += int(timeout_s * 1000)
        try:
            finalize_ms = int(os.getenv("NO_RTP_FINALIZE_MS", "1500"))
        except ValueError:
            finalize_ms = 1500
        finalize_ms = max(100, finalize_ms)

        if self._no_rtp_silence_ms < finalize_ms:
            return

        # Avoid chunking: require a minimum amount of speech before finalizing.
        try:
            min_speech_frames = int(os.getenv("MIN_SPEECH_FRAMES", "10"))
        except ValueError:
            min_speech_frames = 10
        min_speech_frames = max(1, min_speech_frames)
        if self._speech_frames < min_speech_frames:
            self._no_rtp_silence_ms = 0
            self._was_speech = False
            self._speech_frames = 0
            self._silence_frames = 0
            return

        logger.info(
            "Finalizing ASR due to no RTP (session_id=%s, no_rtp_silence_ms=%s, finalize_ms=%s)",
            self.session_id,
            self._no_rtp_silence_ms,
            finalize_ms,
        )

        self._no_rtp_silence_ms = 0

        try:
            final_text = await self.asr_client.get_final(self.asr_stream_id)
        except Exception as e:
            logger.error("ASR get_final failed (no RTP finalize)", {
                "session_id": self.session_id,
                "error": str(e),
                "error_type": type(e).__name__,
            })
            final_text = None

        if final_text:
            logger.info("ASR final result (session_id=%s): %s", self.session_id, final_text)
            await self._enqueue_user_message(final_text)

        # Reset stream + VAD after utterance.
        try:
            await self.asr_client.stop_stream(self.asr_stream_id)
        except Exception:
            pass
        self.asr_stream_id = await self.asr_client.start_stream()
        try:
            self.vad.reset()
        except Exception:
            pass
        self._was_speech = False
        self._speech_frames = 0
        self._silence_frames = 0
    
    async def _process_audio_frame(self, pcm_audio: np.ndarray):
        """Process audio frame with VAD and AI."""
        if self.mode == "tts_only":
            # In this mode we only speak the greeting and ignore input audio.
            return

        # Run VAD
        is_speech = self.vad.process_frame(pcm_audio)

        # While TTS is playing, do NOT feed audio to ASR (echo/loopback can pollute transcription).
        # We only listen for "barge-in" (user starts speaking) and then stop TTS + reset ASR.
        if self.tts_active:
            # Keep a short buffer of inbound audio so we don't miss the beginning of barge-in speech.
            try:
                buffer_ms = int(os.getenv("BARGE_IN_BUFFER_MS", "600"))
            except ValueError:
                buffer_ms = 600
            buffer_ms = max(0, buffer_ms)
            max_frames = max(0, int(buffer_ms / self.frame_duration_ms))
            if max_frames > 0:
                self._barge_in_buffer.append(pcm_audio.astype(np.dtype("<i2"), copy=False).tobytes())
                while len(self._barge_in_buffer) > max_frames:
                    self._barge_in_buffer.popleft()

            if is_speech:
                self._barge_in_speech_frames += 1
            else:
                self._barge_in_speech_frames = 0

            try:
                required = int(os.getenv("BARGE_IN_FRAMES", "5"))
            except ValueError:
                required = 5
            required = max(1, required)

            if is_speech and self._barge_in_speech_frames >= required:
                logger.info("Barge-in detected, stopping TTS", {"session_id": self.session_id})
                self.tts_active = False
                self._barge_in_speech_frames = 0
                self._was_speech = False
                try:
                    self.vad.reset()
                except Exception:
                    pass

                # Ensure ASR stream exists, then flush buffered audio so STT catches the beginning.
                if not self.asr_stream_id:
                    self.asr_stream_id = await self.asr_client.start_stream()

                if self.asr_stream_id and self._barge_in_buffer:
                    for chunk in self._barge_in_buffer:
                        await self.asr_client.send_audio(self.asr_stream_id, chunk)
                    self._barge_in_buffer.clear()
                    self._was_speech = True
                    self._speech_frames += 1
                    self._silence_frames = 0
            return
        else:
            self._barge_in_speech_frames = 0
            self._barge_in_buffer.clear()
        
        if is_speech:
            # Speech detected
            # Send audio to ASR
            if self.asr_stream_id:
                await self.asr_client.send_audio(self.asr_stream_id, pcm_audio.tobytes())
                self._was_speech = True
                self._speech_frames += 1
                self._silence_frames = 0
        else:
            if self._was_speech:
                self._silence_frames += 1

            # Speech ended: finalize only after a sustained silence gap (avoid chunking on micro-pauses).
            if self._was_speech and self.asr_stream_id and not self.tts_active:
                try:
                    silence_finalize_ms = int(os.getenv("SILENCE_FINALIZE_MS", "1200"))
                except ValueError:
                    silence_finalize_ms = 1200
                silence_finalize_ms = max(100, silence_finalize_ms)
                silence_finalize_frames = max(1, int(silence_finalize_ms / self.frame_duration_ms))

                try:
                    min_speech_frames = int(os.getenv("MIN_SPEECH_FRAMES", "10"))
                except ValueError:
                    min_speech_frames = 10
                min_speech_frames = max(1, min_speech_frames)

                if self._speech_frames < min_speech_frames:
                    # Too short; treat as noise and reset.
                    self._was_speech = False
                    self._speech_frames = 0
                    self._silence_frames = 0
                elif self._silence_frames >= silence_finalize_frames:
                    try:
                        final_text = await self.asr_client.get_final(self.asr_stream_id)
                    except Exception as e:
                        logger.error("ASR get_final failed", {
                            "session_id": self.session_id,
                            "error": str(e),
                            "error_type": type(e).__name__,
                        })
                        final_text = None

                    if final_text:
                        logger.info("ASR final result (session_id=%s): %s", self.session_id, final_text)
                        await self._enqueue_user_message(final_text)

                    # Always reset stream after an utterance ends to avoid growing buffers.
                    try:
                        await self.asr_client.stop_stream(self.asr_stream_id)
                    except Exception:
                        pass
                    self.asr_stream_id = await self.asr_client.start_stream()
                    try:
                        self.vad.reset()
                    except Exception:
                        pass
                    self._speech_frames = 0
                    self._silence_frames = 0

            # If we're in silence but haven't finalized yet, keep _was_speech=True so we can finalize later.
            if not self._was_speech:
                self._speech_frames = 0
                self._silence_frames = 0

            # No speech, check for partial results (if supported by provider)
            if self.asr_stream_id and not self.tts_active:
                partial_text = await self.asr_client.get_partial(self.asr_stream_id)
                if partial_text:
                    logger.debug(f"ASR partial result", {
                        "session_id": self.session_id,
                        "text": partial_text
                    })
    
    async def _handle_user_message(self, text: str):
        """Handle user message: send to LLM and synthesize response."""
        # Add to conversation history
        self.conversation_history.append({"role": "user", "content": text})
        
        # Get LLM response
        try:
            response = await self.llm_client.generate(text, self.conversation_history)
            logger.info(f"LLM response", {
                "session_id": self.session_id,
                "response": response
            })

            response = self._truncate_spoken_response(response)
            
            # Add to conversation history
            self.conversation_history.append({"role": "assistant", "content": response})
            
            # Synthesize and send TTS
            await self._synthesize_and_send(response)
            
        except Exception as e:
            logger.error(f"Error handling user message", {
                "session_id": self.session_id,
                "error": str(e)
            })

    async def _enqueue_user_message(self, text: str) -> None:
        """
        Enqueue a finalized user utterance for response generation.

        Critical: the RTP receive loop must not block on LLM/TTS, otherwise we stop
        consuming RTP and the call "goes deaf" after the first exchange.
        """
        if self.mode != "full":
            return
        s = (text or "").strip()
        if not s:
            return
        if not self.running:
            return

        try:
            max_pending = int(os.getenv("MAX_PENDING_UTTERANCES", "3"))
        except ValueError:
            max_pending = 3
        max_pending = max(1, max_pending)

        if self._response_queue.qsize() >= max_pending:
            logger.warning("Dropping user utterance: response queue full", {
                "session_id": self.session_id,
                "max_pending": max_pending,
            })
            return

        await self._response_queue.put(s)

    async def _response_worker_loop(self) -> None:
        """Sequentially generate responses (LLM) and speak them (TTS)."""
        while self.running:
            try:
                text = await self._response_queue.get()
            except asyncio.CancelledError:
                break

            try:
                if text and self.running:
                    await self._handle_user_message(text)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Response worker error", {
                    "session_id": self.session_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                })
            finally:
                try:
                    self._response_queue.task_done()
                except Exception:
                    pass

    def _truncate_spoken_response(self, text: str) -> str:
        """
        Hard limit what we send to TTS so audio stays short.

        Env vars:
        - MAX_SPOKEN_SENTENCES (default: 1)
        - MAX_SPOKEN_CHARS (default: 160)
        """
        s = (text or "").strip()
        if not s:
            return ""

        try:
            max_sentences = int(os.getenv("MAX_SPOKEN_SENTENCES", "1"))
        except ValueError:
            max_sentences = 1
        try:
            max_chars = int(os.getenv("MAX_SPOKEN_CHARS", "160"))
        except ValueError:
            max_chars = 160

        max_sentences = max(1, max_sentences)
        max_chars = max(20, max_chars)

        # Split into sentences while keeping punctuation.
        parts = re.split(r"(?<=[\.\!\?…])\s+", s)
        parts = [p.strip() for p in parts if p and p.strip()]
        if parts:
            s2 = " ".join(parts[:max_sentences]).strip()
        else:
            s2 = s

        if len(s2) > max_chars:
            s2 = s2[:max_chars].rstrip()
            # Avoid cutting in the middle of a word if we can.
            s2 = re.sub(r"\s+\S*$", "", s2).rstrip()
            if s2 and s2[-1] not in ".!?…":
                s2 += "…"
        return s2
    
    async def _synthesize_and_send(self, text: str):
        """Synthesize text to speech and send audio."""
        self.tts_active = True
        logger.info(f"Starting TTS synthesis", {
            "session_id": self.session_id,
            "text": text
        })
        
        try:
            started_at = asyncio.get_event_loop().time()
            first_audio_sent = False
            async for audio_chunk in self.tts_client.synthesize_stream(text):
                if not self.tts_active:
                    # Barge-in occurred, stop sending
                    break
                await self._send_audio_frame(audio_chunk)
                if not first_audio_sent:
                    first_audio_sent = True
                    ttf_audio_ms = int((asyncio.get_event_loop().time() - started_at) * 1000)
                    logger.info("TTS time-to-first-audio", {
                        "session_id": self.session_id,
                        "ttf_audio_ms": ttf_audio_ms
                    })
                # Pace RTP output to real-time frames (20ms)
                await asyncio.sleep(self.frame_duration_ms / 1000.0)
            
            self.tts_active = False
            logger.info(f"TTS synthesis completed", {
                "session_id": self.session_id
            })
        except Exception as e:
            import traceback
            # NOTE: logging format is message-only, so embed context/traceback in the message.
            logger.error(
                f"TTS synthesis failed (session_id={self.session_id}, error_type={type(e).__name__}, error={e})"
            )
            logger.error(f"TTS synthesis traceback (session_id={self.session_id}): {traceback.format_exc()}")
            self.tts_active = False
    
    async def _send_audio_frame(self, pcm_audio: "np.ndarray | bytes | bytearray | memoryview"):
        """Send a single 20ms PCM16 frame as RTP packet."""
        if not self.running:
            return
        if not self.socket:
            logger.warning("Cannot send audio: socket not available", {
                "session_id": self.session_id
            })
            return
        if self.socket.fileno() < 0:
            # Socket already closed (can happen during hangup/cleanup races).
            return
        
        if not self.remote_addr:
            logger.warning("Cannot send audio: remote address not set", {
                "session_id": self.session_id
            })
            return
        
        # Accept either bytes (PCM16 little-endian) or numpy int16 samples.
        if isinstance(pcm_audio, (bytes, bytearray, memoryview)):
            pcm_audio = np.frombuffer(pcm_audio, dtype=np.dtype("<i2"))
        elif not isinstance(pcm_audio, np.ndarray):
            raise TypeError(f"Unsupported pcm_audio type: {type(pcm_audio).__name__}")

        # Ensure correct dtype
        if pcm_audio.dtype != np.int16:
            pcm_audio = pcm_audio.astype(np.int16, copy=False)

        # Ensure exact 20ms frame length (pad or trim defensively)
        if pcm_audio.size < self.samples_per_frame:
            pcm_audio = np.pad(pcm_audio, (0, self.samples_per_frame - pcm_audio.size))
        elif pcm_audio.size > self.samples_per_frame:
            pcm_audio = pcm_audio[: self.samples_per_frame]
        
        # Encode PCM16 to ulaw
        ulaw_data = linear_to_ulaw(pcm_audio)
        
        # Create RTP packet
        rtp_packet = self.rtp_handler.create_packet(ulaw_data, marker=0)
        
        # Send via socket
        try:
            loop = asyncio.get_event_loop()
            await loop.sock_sendto(self.socket, rtp_packet, self.remote_addr)
        except OSError as e:
            # Ignore "Bad file descriptor" if teardown raced with send.
            if getattr(e, "errno", None) == 9:
                return
            import traceback
            error_details = {
                "session_id": self.session_id,
                "remote_addr": self.remote_addr,
                "remote_addr_type": type(self.remote_addr).__name__ if self.remote_addr else None,
                "error": str(e),
                "error_type": type(e).__name__,
            }
            logger.error(f"Error sending RTP packet: {json.dumps(error_details)}")
            # NOTE: logging is configured to only print the message, so embed traceback in the message.
            logger.error(f"Traceback: {traceback.format_exc()}")
            return
        except Exception as e:
            import traceback
            error_details = {
                "session_id": self.session_id,
                "remote_addr": self.remote_addr,
                "remote_addr_type": type(self.remote_addr).__name__ if self.remote_addr else None,
                "error": str(e),
                "error_type": type(e).__name__,
            }
            logger.error(f"Error sending RTP packet: {json.dumps(error_details)}")
            # NOTE: logging is configured to only print the message, so embed traceback in the message.
            logger.error(f"Traceback: {traceback.format_exc()}")
            return
    
    async def stop(self):
        """Stop the media session."""
        self.running = False
        self.tts_active = False
        self._rtp_ready.set()

        if self._greeting_task and not self._greeting_task.done():
            self._greeting_task.cancel()
            try:
                await self._greeting_task
            except asyncio.CancelledError:
                pass

        if self._keepalive_task and not self._keepalive_task.done():
            self._keepalive_task.cancel()
            try:
                await self._keepalive_task
            except asyncio.CancelledError:
                pass

        if self._response_worker_task and not self._response_worker_task.done():
            self._response_worker_task.cancel()
            try:
                await self._response_worker_task
            except asyncio.CancelledError:
                pass
        
        # Cancel receive task
        if self._receive_task and not self._receive_task.done():
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
        
        if self.asr_stream_id:
            await self.asr_client.stop_stream(self.asr_stream_id)
        
        if self.socket:
            self.socket.close()
            self.socket = None
        
        logger.info(f"Media session stopped", {
            "session_id": self.session_id
        })


class MediaService:
    """Main media service that manages multiple sessions."""
    
    def __init__(self, host: str = "127.0.0.1", port_base: int = 40000):
        """
        Initialize media service.
        
        Args:
            host: Host to bind to
            port_base: Base port for sessions
        """
        self.host = host
        self.port_base = port_base
        self.sessions: dict[str, MediaSession] = {}
        self.next_port = port_base
    
    async def create_session(self, session_id: str, remote_addr: tuple, 
                            mode: str = "full") -> int:
        """
        Create a new media session.
        
        Args:
            session_id: Unique session identifier
            remote_addr: Remote address (host, port)
            mode: Operation mode ("echo", "tts_only", "full")
        
        Returns:
            Local port for the session
        """
        local_port = self.next_port
        self.next_port += 1
        
        # Create AI clients
        asr_provider = os.getenv("ASR_PROVIDER", "mock")
        llm_provider = os.getenv("LLM_PROVIDER", "mock")
        tts_provider = os.getenv("TTS_PROVIDER", "mock")
        
        asr_client = create_asr_client(asr_provider, sample_rate=8000)
        system_prompt = get_system_prompt()
        llm_client = create_llm_client(
            llm_provider,
            system_prompt=system_prompt,
        )
        tts_client = create_tts_client(
            tts_provider,
            sample_rate=8000,
            frame_duration_ms=20,
        )
        
        # Create session
        session = MediaSession(
            session_id=session_id,
            local_port=local_port,
            remote_addr=remote_addr,
            asr_client=asr_client,
            llm_client=llm_client,
            tts_client=tts_client,
            mode=mode
        )
        
        self.sessions[session_id] = session
        await session.start()
        
        logger.info(f"Session created", {
            "session_id": session_id,
            "local_port": local_port,
            "remote_addr": remote_addr,
            "mode": mode
        })
        
        return local_port
    
    async def update_session_rtp_address(self, session_id: str, remote_host: str, remote_port: int):
        """Update the RTP remote address for a session."""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            session.remote_addr = (remote_host, remote_port)
            session._rtp_ready.set()
            logger.info("Updated session RTP address", {
                "session_id": session_id,
                "remote_addr": session.remote_addr
            })
            return True
        return False
    
    async def stop_session(self, session_id: str):
        """Stop a media session."""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            await session.stop()
            del self.sessions[session_id]
            logger.info(f"Session stopped", {
                "session_id": session_id
            })


async def create_session_handler(request: web.Request, service: MediaService):
    """HTTP handler to create a new media session."""
    try:
        data = await request.json()
        session_id = data.get("session_id")
        remote_host = data.get("remote_host", "127.0.0.1")
        remote_port = data.get("remote_port", 10000)
        mode = data.get("mode", "full")
        
        if not session_id:
            return web.json_response({"error": "session_id required"}, status=400)
        
        local_port = await service.create_session(
            session_id=session_id,
            remote_addr=(remote_host, remote_port),
            mode=mode
        )
        
        return web.json_response({
            "session_id": session_id,
            "local_port": local_port,
            "status": "created"
        })
    except Exception as e:
        logger.error(f"Error creating session", {"error": str(e)})
        return web.json_response({"error": str(e)}, status=500)


async def update_rtp_address_handler(request: web.Request, service: MediaService):
    """HTTP handler to update RTP address for a session."""
    try:
        session_id = request.match_info.get("session_id")
        if not session_id:
            return web.json_response({"error": "session_id required"}, status=400)
        
        data = await request.json()
        remote_host = data.get("remote_host")
        remote_port = data.get("remote_port")
        
        if not remote_host or not remote_port:
            return web.json_response({"error": "remote_host and remote_port required"}, status=400)
        
        # Ensure remote_port is an integer
        try:
            remote_port = int(remote_port)
        except (ValueError, TypeError):
            return web.json_response({"error": "remote_port must be a number"}, status=400)
        
        logger.info("Updating RTP address", {
            "session_id": session_id,
            "remote_host": remote_host,
            "remote_port": remote_port,
            "remote_port_type": type(remote_port).__name__
        })
        
        updated = await service.update_session_rtp_address(session_id, remote_host, remote_port)
        
        if updated:
            # Send initial silence packet to trigger RTP flow in both directions.
            if session_id in service.sessions:
                session = service.sessions[session_id]
                silence = np.zeros(session.samples_per_frame, dtype=np.int16)
                try:
                    await session._send_audio_frame(silence)
                    logger.info("Sent initial silence packet to trigger RTP", {
                        "session_id": session_id,
                        "remote_addr": session.remote_addr,
                        "mode": session.mode,
                    })
                except Exception as e:
                    import traceback
                    error_details = {
                        "session_id": session_id,
                        "remote_addr": session.remote_addr,
                        "socket_available": session.socket is not None,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    }
                    logger.warning(f"Could not send initial silence packet: {json.dumps(error_details)}")
                    logger.debug(f"Traceback: {traceback.format_exc()}")
            
            return web.json_response({
                "session_id": session_id,
                "status": "updated",
                "remote_addr": f"{remote_host}:{remote_port}"
            })
        else:
            return web.json_response({"error": "Session not found"}, status=404)
    except Exception as e:
        logger.error(f"Error updating RTP address", {"error": str(e)})
        return web.json_response({"error": str(e)}, status=500)


async def stop_session_handler(request: web.Request, service: MediaService):
    """HTTP handler to stop a media session."""
    try:
        data = await request.json()
        session_id = data.get("session_id")
        
        if not session_id:
            return web.json_response({"error": "session_id required"}, status=400)
        
        await service.stop_session(session_id)
        
        return web.json_response({
            "session_id": session_id,
            "status": "stopped"
        })
    except Exception as e:
        logger.error(f"Error stopping session", {"error": str(e)})
        return web.json_response({"error": str(e)}, status=500)


async def health_handler(request: web.Request):
    """Health check endpoint."""
    return web.json_response({"status": "healthy"})


async def stt_transcribe_handler(request: web.Request):
    """
    STT endpoint for quick testing without RTP.

    Request: multipart/form-data with:
      - file: WAV file (recommended)

    Response: { "text": "...", "provider": "...", "bytes": N }
    """
    try:
        if not request.content_type.startswith("multipart/"):
            return web.json_response({"error": "multipart/form-data required (field: file)"}, status=400)

        reader = await request.multipart()
        file_field = await reader.next()
        if file_field is None or file_field.name != "file":
            return web.json_response({"error": "missing multipart field 'file'"}, status=400)

        filename = (file_field.filename or "").lower()
        raw = await file_field.read(decode=False)
        if not raw:
            return web.json_response({"error": "empty file"}, status=400)

        # Decode WAV -> PCM16LE mono @ 8kHz (only WAV supported for now)
        try:
            from scipy.io import wavfile
            from scipy.signal import resample_poly
        except Exception:
            return web.json_response({"error": "scipy is required to decode wav uploads"}, status=500)

        if not (raw[:4] == b"RIFF" and raw[8:12] == b"WAVE") and not filename.endswith(".wav"):
            return web.json_response({"error": "only .wav is supported (send a WAV file)"}, status=400)

        rate, data = wavfile.read(io.BytesIO(raw))
        pcm = data
        if isinstance(pcm, np.ndarray) and pcm.ndim == 2:
            pcm = pcm.astype(np.float32).mean(axis=1)

        if isinstance(pcm, np.ndarray) and pcm.dtype != np.int16:
            if pcm.dtype == np.int32:
                pcm = np.clip(np.round(pcm.astype(np.float32) / 65536.0), -32768, 32767).astype(np.int16)
            elif pcm.dtype == np.float32 or pcm.dtype == np.float64:
                pcm = np.clip(np.round(pcm.astype(np.float32) * 32767.0), -32768, 32767).astype(np.int16)
            else:
                pcm = pcm.astype(np.int16, copy=False)

        rate = int(rate)
        if rate != 8000:
            # Resample in float domain, then back to PCM16.
            x = pcm.astype(np.float32) / 32768.0
            up = 8000
            down = rate
            import math
            g = math.gcd(up, down)
            up //= g
            down //= g
            y = resample_poly(x, up, down).astype(np.float32)
            pcm = np.clip(np.round(y * 32767.0), -32768, 32767).astype(np.int16)

        pcm_bytes = pcm.astype(np.dtype("<i2"), copy=False).tobytes()

        asr_provider = os.getenv("ASR_PROVIDER", "mock")
        asr_client = create_asr_client(asr_provider, sample_rate=8000)
        stream_id = await asr_client.start_stream()
        await asr_client.send_audio(stream_id, pcm_bytes)
        text = await asr_client.get_final(stream_id)
        await asr_client.stop_stream(stream_id)

        return web.json_response({
            "text": text or "",
            "provider": asr_provider,
            "bytes": len(pcm_bytes),
        })
    except Exception as e:
        logger.error("Error in STT transcribe handler", {"error": str(e), "error_type": type(e).__name__})
        return web.json_response({"error": str(e)}, status=500)


async def main():
    """Main entry point."""
    host = os.getenv("MEDIA_HOST", "127.0.0.1")
    port_base = int(os.getenv("MEDIA_PORT_BASE", "40000"))
    api_port = int(os.getenv("MEDIA_API_PORT", "5000"))
    mode = os.getenv("MEDIA_MODE", "full")  # "echo", "tts_only", "full"
    
    logger.info(f"Starting Media Service", {
        "host": host,
        "port_base": port_base,
        "api_port": api_port,
        "mode": mode
    })
    
    service = MediaService(host=host, port_base=port_base)
    
    # Create HTTP API server
    app = web.Application()
    app.router.add_post("/sessions", lambda r: create_session_handler(r, service))
    app.router.add_put("/sessions/{session_id}/rtp-address", lambda r: update_rtp_address_handler(r, service))
    app.router.add_delete("/sessions", lambda r: stop_session_handler(r, service))
    app.router.add_get("/health", health_handler)
    app.router.add_post("/stt/transcribe", stt_transcribe_handler)
    
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, api_port)
    await site.start()
    
    logger.info(f"Media Service HTTP API listening on {host}:{api_port}")
    logger.info("Media Service ready (sessions created via HTTP API)")
    
    # Keep running
    try:
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        logger.info("Shutting down Media Service")
        await runner.cleanup()
        # Stop all sessions
        for session_id in list(service.sessions.keys()):
            await service.stop_session(session_id)


if __name__ == "__main__":
    asyncio.run(main())
