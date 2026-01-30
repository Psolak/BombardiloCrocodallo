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

from .rtp_handler import RTPHandler
from .audio_codec import ulaw_to_linear, linear_to_ulaw
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
        
        # Send initial greeting if TTS-only or full mode
        if self.mode in ["tts_only", "full"]:
            await self._send_greeting()
    
    async def _send_greeting(self):
        """Send initial greeting via TTS."""
        greeting = "Hello! I'm ready to help. How can I assist you today?"
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
        no_packet_warnings = 0
        last_warning_time = asyncio.get_event_loop().time()
        
        while self.running:
            try:
                # Receive RTP packet with timeout to allow periodic status checks
                try:
                    data, addr = await asyncio.wait_for(
                        loop.sock_recvfrom(self.socket, 1500),
                        timeout=5.0
                    )
                    no_packet_warnings = 0  # Reset counter on successful receive
                except asyncio.TimeoutError:
                    # No packet received in 5 seconds - log periodically
                    current_time = asyncio.get_event_loop().time()
                    if current_time - last_warning_time >= 10.0:  # Warn every 10 seconds
                        logger.warning("No RTP packets received", {
                            "session_id": self.session_id,
                            "local_port": self.local_port,
                            "remote_addr": self.remote_addr,
                            "seconds_waiting": int(current_time - last_warning_time),
                            "packet_count": packet_count
                        })
                        last_warning_time = current_time
                    continue
                
                packet_count += 1
                
                if packet_count == 1:
                    logger.info("First RTP packet received", {
                        "session_id": self.session_id,
                        "from_addr": addr,
                        "packet_size": len(data)
                    })
                
                # Update remote address if changed
                if addr != self.remote_addr:
                    logger.info("Remote address updated", {
                        "session_id": self.session_id,
                        "old_addr": self.remote_addr,
                        "new_addr": addr
                    })
                    self.remote_addr = addr
                
                # Parse RTP packet
                result = self.rtp_handler.parse_packet(data)
                if result is None:
                    if packet_count <= 5:
                        logger.debug("RTP packet dropped (invalid or out of order)", {
                            "session_id": self.session_id,
                            "packet_count": packet_count
                        })
                    continue
                
                sequence, payload = result
                
                if packet_count <= 5:
                    logger.debug("RTP packet parsed successfully", {
                        "session_id": self.session_id,
                        "sequence": sequence,
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
                    # Decode ulaw to PCM16 for VAD/ASR pipeline
                    pcm_audio = ulaw_to_linear(payload)
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
    
    async def _process_audio_frame(self, pcm_audio: np.ndarray):
        """Process audio frame with VAD and AI."""
        # Run VAD
        is_speech = self.vad.process_frame(pcm_audio)
        
        if is_speech:
            # Speech detected
            if self.tts_active:
                # Barge-in: stop TTS and resume ASR
                logger.info(f"Barge-in detected, stopping TTS", {
                    "session_id": self.session_id
                })
                self.tts_active = False
                # Restart ASR stream
                if self.asr_stream_id:
                    await self.asr_client.stop_stream(self.asr_stream_id)
                self.asr_stream_id = await self.asr_client.start_stream()
            
            # Send audio to ASR
            if self.asr_stream_id:
                await self.asr_client.send_audio(self.asr_stream_id, pcm_audio.tobytes())
                
                # Check for final transcription
                final_text = await self.asr_client.get_final(self.asr_stream_id)
                if final_text:
                    logger.info(f"ASR final result", {
                        "session_id": self.session_id,
                        "text": final_text
                    })
                    await self._handle_user_message(final_text)
        else:
            # No speech, check for partial results
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
            
            # Add to conversation history
            self.conversation_history.append({"role": "assistant", "content": response})
            
            # Synthesize and send TTS
            await self._synthesize_and_send(response)
            
        except Exception as e:
            logger.error(f"Error handling user message", {
                "session_id": self.session_id,
                "error": str(e)
            })
    
    async def _synthesize_and_send(self, text: str):
        """Synthesize text to speech and send audio."""
        self.tts_active = True
        logger.info(f"Starting TTS synthesis", {
            "session_id": self.session_id,
            "text": text
        })
        
        try:
            async for audio_chunk in self.tts_client.synthesize_stream(text):
                if not self.tts_active:
                    # Barge-in occurred, stop sending
                    break
                await self._send_audio_frame(audio_chunk)
            
            self.tts_active = False
            logger.info(f"TTS synthesis completed", {
                "session_id": self.session_id
            })
        except Exception as e:
            logger.error(f"Error in TTS synthesis", {
                "session_id": self.session_id,
                "error": str(e)
            })
            self.tts_active = False
    
    async def _send_audio_frame(self, pcm_audio: np.ndarray):
        """Send audio frame as RTP packet."""
        if not self.socket:
            logger.warning("Cannot send audio: socket not available", {
                "session_id": self.session_id
            })
            return
        
        if not self.remote_addr:
            logger.warning("Cannot send audio: remote address not set", {
                "session_id": self.session_id
            })
            return
        
        # Ensure correct dtype and shape
        if pcm_audio.dtype != np.int16:
            pcm_audio = pcm_audio.astype(np.int16)
        
        # Encode PCM16 to ulaw
        ulaw_data = linear_to_ulaw(pcm_audio)
        
        # Create RTP packet
        rtp_packet = self.rtp_handler.create_packet(ulaw_data, marker=0)
        
        # Send via socket
        try:
            loop = asyncio.get_event_loop()
            await loop.sock_sendto(self.socket, rtp_packet, self.remote_addr)
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
        
        asr_client = create_asr_client(asr_provider)
        system_prompt = get_system_prompt()
        llm_client = create_llm_client(llm_provider, 
                                      system_prompt=system_prompt)
        tts_client = create_tts_client(tts_provider, sample_rate=8000)
        
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
            # Send initial silence packet to trigger RTP flow
            if session_id in service.sessions:
                session = service.sessions[session_id]
                if session.mode == "echo":
                    silence = np.zeros(session.samples_per_frame, dtype=np.int16)
                    try:
                        await session._send_audio_frame(silence)
                        logger.info("Sent initial silence packet to trigger RTP", {
                            "session_id": session_id,
                            "remote_addr": session.remote_addr
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
