"""
Abstract interfaces and implementations for ASR, LLM, and TTS providers.
"""

from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional
import asyncio
import json
import logging
import math
import os
import time
import io
import wave
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

try:
    import aiohttp
except Exception:  # pragma: no cover
    aiohttp = None

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

try:
    from scipy.signal import resample_poly
except Exception:  # pragma: no cover
    resample_poly = None

try:
    from scipy.io import wavfile
except Exception:  # pragma: no cover
    wavfile = None


# Abstract Interfaces

class ASRClient(ABC):
    """Abstract ASR (Automatic Speech Recognition) client."""
    
    @abstractmethod
    async def start_stream(self) -> str:
        """Start a new ASR stream. Returns stream ID."""
        pass
    
    @abstractmethod
    async def send_audio(self, stream_id: str, audio: bytes) -> None:
        """Send audio chunk to ASR stream."""
        pass
    
    @abstractmethod
    async def get_partial(self, stream_id: str) -> Optional[str]:
        """Get partial transcription (interim result)."""
        pass
    
    @abstractmethod
    async def get_final(self, stream_id: str) -> Optional[str]:
        """Get final transcription. Blocks until available."""
        pass
    
    @abstractmethod
    async def stop_stream(self, stream_id: str) -> None:
        """Stop ASR stream."""
        pass


class LLMClient(ABC):
    """Abstract LLM (Large Language Model) client."""
    
    @abstractmethod
    async def generate(self, user_message: str, conversation_history: list[dict]) -> str:
        """
        Generate response from LLM.
        
        Args:
            user_message: User's message
            conversation_history: List of {role: "user"|"assistant", content: str}
        
        Returns:
            LLM response text
        """
        pass


class TTSClient(ABC):
    """Abstract TTS (Text-to-Speech) client."""
    
    @abstractmethod
    async def synthesize_stream(self, text: str) -> AsyncIterator[bytes]:
        """
        Synthesize text to speech audio stream.
        
        Args:
            text: Text to synthesize
        
        Yields:
            Audio chunks (PCM16 little-endian, mono) at the requested `sample_rate`
        """
        pass


# Mock Implementations (for testing without external keys)

class MockASRClient(ASRClient):
    """Mock ASR client that returns predefined transcriptions."""
    
    def __init__(self):
        self.streams: dict[str, list[bytes]] = {}
        self.partial_results: dict[str, str] = {}
        self.final_results: dict[str, Optional[str]] = {}
        self.mock_responses = [
            "hello",
            "how are you",
            "what is the weather",
            "goodbye"
        ]
        self.response_index = 0
    
    async def start_stream(self) -> str:
        stream_id = f"mock_stream_{len(self.streams)}"
        self.streams[stream_id] = []
        self.partial_results[stream_id] = ""
        self.final_results[stream_id] = None
        return stream_id
    
    async def send_audio(self, stream_id: str, audio: bytes) -> None:
        if stream_id not in self.streams:
            return
        self.streams[stream_id].append(audio)
        # Simulate partial results
        if len(self.streams[stream_id]) > 10:  # After some audio
            self.partial_results[stream_id] = self.mock_responses[self.response_index % len(self.mock_responses)]
    
    async def get_partial(self, stream_id: str) -> Optional[str]:
        return self.partial_results.get(stream_id)
    
    async def get_final(self, stream_id: str) -> Optional[str]:
        # Simulate delay
        await asyncio.sleep(0.5)
        if stream_id in self.streams and len(self.streams[stream_id]) > 20:
            result = self.mock_responses[self.response_index % len(self.mock_responses)]
            self.response_index += 1
            self.final_results[stream_id] = result
            return result
        return None
    
    async def stop_stream(self, stream_id: str) -> None:
        if stream_id in self.streams:
            del self.streams[stream_id]
        self.partial_results.pop(stream_id, None)
        self.final_results.pop(stream_id, None)


class MockLLMClient(LLMClient):
    """Mock LLM client that returns predefined responses."""
    
    def __init__(self, system_prompt: str = "You are a helpful assistant."):
        self.system_prompt = system_prompt
        self.conversation_count = 0
    
    async def generate(self, user_message: str, conversation_history: list[dict]) -> str:
        self.conversation_count += 1
        # Simple mock responses
        user_lower = user_message.lower()
        if "hello" in user_lower or "hi" in user_lower:
            return "Hello! How can I help you today?"
        elif "weather" in user_lower:
            return "I'm sorry, I don't have access to weather information in this mock mode."
        elif "goodbye" in user_lower or "bye" in user_lower:
            return "Goodbye! Have a great day!"
        else:
            return f"I understand you said: {user_message}. This is a mock response (conversation #{self.conversation_count})."


def _mask_secret(value: Optional[str]) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return "[REDACTED]"
    return f"{value[:4]}…{value[-4:]}"


def _openai_compat_chat_completions_url(base_url: str) -> str:
    """
    Build an OpenAI-compatible chat completions endpoint from a base URL.

    Examples:
    - https://api.openai.com          -> https://api.openai.com/v1/chat/completions
    - https://api.openai.com/v1       -> https://api.openai.com/v1/chat/completions
    - http://localhost:1234/v1/       -> http://localhost:1234/v1/chat/completions
    """
    base = (base_url or "").strip()
    if not base:
        base = "https://api.openai.com"
    base = base.rstrip("/")
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def _normalize_messages(
    user_message: str,
    conversation_history: list[dict],
    system_prompt: str,
) -> list[dict]:
    """
    Normalize messages to OpenAI chat format.

    We accept that upstream may include a system message in `conversation_history`.
    We also accept that upstream may have already appended `user_message` to history.
    """
    messages: list[dict] = []

    # Prefer any explicit system messages in history; otherwise inject system_prompt.
    has_system = any(m.get("role") == "system" and m.get("content") for m in (conversation_history or []))
    if not has_system and system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    for m in conversation_history or []:
        role = m.get("role")
        content = m.get("content")
        if role not in {"system", "user", "assistant"}:
            continue
        if not isinstance(content, str) or not content.strip():
            continue
        messages.append({"role": role, "content": content})

    # Avoid duplicating the last user message if upstream already appended it.
    if isinstance(user_message, str) and user_message.strip():
        if not messages or messages[-1].get("role") != "user" or messages[-1].get("content") != user_message:
            messages.append({"role": "user", "content": user_message})

    return messages


class OpenAICompatLLMClient(LLMClient):
    """
    OpenAI-compatible LLM client (Chat Completions API).

    Controlled by env vars:
    - LLM_BASE_URL: base URL (default: https://api.openai.com)
    - LLM_API_KEY: API key (optional for some local servers)
    - LLM_MODEL: model name (default: gpt-4o-mini)
    - LLM_TIMEOUT_MS: request timeout in milliseconds (default: 30000)
    - LLM_MAX_TOKENS: cap output tokens (optional)
    """

    def __init__(
        self,
        system_prompt: str = "You are a helpful assistant.",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        self.system_prompt = system_prompt

        self.base_url = (base_url or os.getenv("LLM_BASE_URL") or "https://api.openai.com").strip()
        self.api_key = api_key if api_key is not None else os.getenv("LLM_API_KEY")
        self.model = (model or os.getenv("LLM_MODEL") or "gpt-4o-mini").strip()

        env_temp = os.getenv("LLM_TEMPERATURE")
        try:
            env_temperature = float(env_temp) if env_temp else None
        except ValueError:
            env_temperature = None
        self.temperature = temperature if temperature is not None else env_temperature

        env_timeout = os.getenv("LLM_TIMEOUT_MS")
        try:
            env_timeout_ms = int(env_timeout) if env_timeout else None
        except ValueError:
            env_timeout_ms = None
        self.timeout_ms = timeout_ms if timeout_ms is not None else (env_timeout_ms or 30000)

        env_max_tokens = os.getenv("LLM_MAX_TOKENS")
        try:
            env_max = int(env_max_tokens) if env_max_tokens else None
        except ValueError:
            env_max = None
        self.max_tokens = max_tokens if max_tokens is not None else env_max

        # Basic validation (no secrets in messages)
        try:
            parsed = urlparse(self.base_url)
            if parsed.scheme and parsed.scheme not in {"http", "https"}:
                raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")
        except Exception as e:
            raise ValueError(f"Invalid LLM_BASE_URL: {self.base_url}") from e

    async def generate(self, user_message: str, conversation_history: list[dict]) -> str:
        import aiohttp

        url = _openai_compat_chat_completions_url(self.base_url)
        messages = _normalize_messages(user_message, conversation_history, self.system_prompt)

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": messages,
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if isinstance(self.max_tokens, int) and self.max_tokens > 0:
            payload["max_tokens"] = self.max_tokens

        timeout_s = max(0.1, float(self.timeout_ms) / 1000.0)
        timeout = aiohttp.ClientTimeout(total=timeout_s)

        # NOTE: Do not log full headers/payload because they may contain secrets.
        logger.info(
            "LLM request (openai_compat)",
            {
                "base_url": self.base_url,
                "url": url,
                "model": self.model,
                "timeout_ms": self.timeout_ms,
                "temperature": self.temperature,
                "api_key": _mask_secret(self.api_key),
                "messages": len(messages),
            },
        )

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, headers=headers, json=payload) as resp:
                    text = await resp.text()
                    if resp.status < 200 or resp.status >= 300:
                        # Avoid dumping gigantic error pages into logs.
                        trimmed = text[:2000]
                        raise RuntimeError(
                            f"LLM request failed (openai_compat): HTTP {resp.status} from {url}. Body: {trimmed}"
                        )
        except asyncio.TimeoutError as e:
            raise RuntimeError(
                f"LLM request timed out (openai_compat) after {self.timeout_ms}ms to {url} (model={self.model})."
            ) from e
        except aiohttp.ClientConnectorError as e:
            raise RuntimeError(
                f"LLM connection failed (openai_compat) to {url} (model={self.model}). "
                "Check LLM_BASE_URL and network connectivity."
            ) from e
        except aiohttp.ClientResponseError as e:
            raise RuntimeError(
                f"LLM HTTP error (openai_compat) to {url} (model={self.model}): {e.status} {e.message}"
            ) from e
        except Exception as e:
            # Ensure secrets aren't accidentally included in exception chains.
            raise RuntimeError(
                f"LLM request error (openai_compat) to {url} (model={self.model}): {type(e).__name__}: {e}"
            ) from e

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"LLM response was not valid JSON (openai_compat) from {url}: {text[:500]}"
            ) from e

        try:
            # OpenAI chat completions format:
            # { choices: [ { message: { content: "...", role: "assistant" }, ... } ] }
            choices = data.get("choices") or []
            message = (choices[0] or {}).get("message") or {}
            content = message.get("content")
            if not isinstance(content, str):
                raise KeyError("choices[0].message.content")
            return content.strip()
        except Exception as e:
            raise RuntimeError(
                f"Unexpected LLM response shape (openai_compat) from {url}: {json.dumps(data)[:2000]}"
            ) from e


class MockTTSClient(TTSClient):
    """Mock TTS client that generates simple tone audio."""
    
    def __init__(self, sample_rate: int = 8000, frame_duration_ms: int = 20):
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
    
    async def synthesize_stream(self, text: str) -> AsyncIterator[bytes]:
        """Generate a simple tone as mock audio."""
        if np is None:
            raise RuntimeError("numpy is required for MockTTSClient")
        
        # Generate a simple beep tone (440 Hz for 1 second)
        duration = min(len(text) * 0.1, 3.0)  # Roughly 0.1s per character, max 3s
        samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, samples, endpoint=False)
        
        # Generate tone
        frequency = 440.0
        audio = np.sin(2 * np.pi * frequency * t)
        
        # Convert to int16
        audio_int16 = (audio * 32767 * 0.3).astype(np.int16)  # 30% volume
        
        # Yield in chunks (20ms frames)
        chunk_size = (self.sample_rate * self.frame_duration_ms) // 1000
        for i in range(0, len(audio_int16), chunk_size):
            chunk = audio_int16[i:i + chunk_size]
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
            yield chunk.tobytes()


class HttpTTSClient(TTSClient):
    """
    HTTP TTS adapter (BOM-16).

    Targets an OpenAI-style endpoint:
      POST {TTS_BASE_URL}/v1/audio/speech
      JSON: { model, voice, input, response_format }

    Supports:
    - response_format="pcm": raw PCM16 little-endian mono (often 24 kHz; resampled)
    - response_format="wav": WAV container (e.g. Orpheus-FastAPI); decoded and resampled
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: Optional[str],
        voice: str,
        timeout_ms: int,
        sample_rate: int = 8000,
        frame_duration_ms: int = 20,
        model: str = "tts-1",
        source_sample_rate: int = 24000,
        response_format: str = "pcm",
    ):
        if aiohttp is None:
            raise RuntimeError("aiohttp is required for HttpTTSClient")
        if np is None:
            raise RuntimeError("numpy is required for HttpTTSClient")
        if resample_poly is None:
            raise RuntimeError("scipy is required for HttpTTSClient")
        if wavfile is None:
            raise RuntimeError("scipy is required for HttpTTSClient (scipy.io.wavfile)")

        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.voice = voice
        self.timeout_ms = timeout_ms
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.model = model
        self.source_sample_rate = source_sample_rate
        self.response_format = response_format.strip().lower()

        self.samples_per_frame = (self.sample_rate * self.frame_duration_ms) // 1000

    def _speech_url(self) -> str:
        # Allow caller to pass either a base URL or a full speech endpoint.
        if self.base_url.endswith("/v1/audio/speech"):
            return self.base_url
        return f"{self.base_url}/v1/audio/speech"

    @staticmethod
    def _fade_in_out(pcm_i16: "np.ndarray", fade_samples: int) -> "np.ndarray":
        if fade_samples <= 0 or pcm_i16.size == 0:
            return pcm_i16
        fade = min(fade_samples, pcm_i16.size // 2)
        if fade <= 0:
            return pcm_i16

        x = pcm_i16.astype(np.float32)
        ramp = np.linspace(0.0, 1.0, fade, endpoint=False, dtype=np.float32)
        x[:fade] *= ramp
        x[-fade:] *= ramp[::-1]
        return np.clip(np.round(x), -32768, 32767).astype(np.int16)

    def _resample_to_target(self, pcm_bytes: bytes, *, source_sample_rate: Optional[int] = None) -> "np.ndarray":
        # Interpret response as little-endian PCM16 mono.
        src = np.frombuffer(pcm_bytes, dtype=np.dtype("<i2"))
        if src.size == 0:
            return src.astype(np.int16)

        src_rate = int(source_sample_rate or self.source_sample_rate)

        if src_rate == self.sample_rate:
            out = src.astype(np.int16, copy=False)
        else:
            # Resample in float domain for quality.
            x = src.astype(np.float32) / 32768.0
            up = self.sample_rate
            down = src_rate
            g = math.gcd(up, down)
            up //= g
            down //= g
            y = resample_poly(x, up, down).astype(np.float32)
            out = np.clip(np.round(y * 32767.0), -32768, 32767).astype(np.int16)

        # Apply a short fade to avoid clicks.
        fade_samples = max(1, self.samples_per_frame // 4)  # ~5ms at 20ms frames
        return self._fade_in_out(out, fade_samples=fade_samples)

    @staticmethod
    def _looks_like_wav(data: bytes) -> bool:
        return len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE"

    def _decode_wav_to_pcm16le(self, wav_bytes: bytes) -> tuple[int, bytes]:
        if wavfile is None:
            raise RuntimeError("scipy is required to decode wav responses")
        rate, data = wavfile.read(io.BytesIO(wav_bytes))

        # Normalize to mono PCM16
        if isinstance(data, np.ndarray):
            if data.ndim == 2:
                # Average channels to mono
                data = data.astype(np.float32).mean(axis=1)
            if data.dtype == np.int16:
                pcm = data
            elif data.dtype == np.int32:
                pcm = np.clip(np.round(data.astype(np.float32) / 65536.0), -32768, 32767).astype(np.int16)
            elif data.dtype == np.float32 or data.dtype == np.float64:
                pcm = np.clip(np.round(data.astype(np.float32) * 32767.0), -32768, 32767).astype(np.int16)
            else:
                pcm = data.astype(np.int16, copy=False)
        else:
            pcm = np.array([], dtype=np.int16)

        return int(rate), pcm.astype(np.dtype("<i2"), copy=False).tobytes()

    async def synthesize_stream(self, text: str) -> AsyncIterator[bytes]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        # Keep payload minimal and OpenAI-style.
        payload = {
            "model": self.model,
            "voice": self.voice,
            "input": text,
            "response_format": self.response_format,
        }

        timeout_s = max(1.0, self.timeout_ms / 1000.0)
        timeout = aiohttp.ClientTimeout(total=timeout_s)

        started_at = time.perf_counter()
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(self._speech_url(), json=payload, headers=headers) as resp:
                if resp.status >= 400:
                    body = await resp.text()
                    raise RuntimeError(f"TTS HTTP {resp.status}: {body[:500]}")
                raw = await resp.read()

        elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        logger.info("TTS HTTP response received", {
            "provider": "http",
            "elapsed_ms": elapsed_ms,
            "bytes": len(raw),
            "response_format": self.response_format,
        })

        # Orpheus-FastAPI currently returns WAV only. Auto-detect WAV too.
        if self.response_format == "wav" or self._looks_like_wav(raw):
            wav_rate, pcm_bytes = self._decode_wav_to_pcm16le(raw)
            pcm_i16 = self._resample_to_target(pcm_bytes, source_sample_rate=wav_rate)
        else:
            pcm_i16 = self._resample_to_target(raw)

        # Chunk into exact 20ms frames for RTP pacing upstream.
        spf = self.samples_per_frame
        for i in range(0, len(pcm_i16), spf):
            frame = pcm_i16[i:i + spf]
            if frame.size < spf:
                frame = np.pad(frame, (0, spf - frame.size))
            yield frame.astype(np.dtype("<i2"), copy=False).tobytes()


# ASR HTTP adapter

class HttpASRClient(ASRClient):
    """
    HTTP ASR adapter using an OpenAI Whisper-style transcription endpoint.

    Endpoint shape:
      POST {ASR_BASE_URL}/v1/audio/transcriptions
      multipart/form-data:
        - file: audio file (we send WAV PCM16 mono)
        - model: model name (e.g. whisper-1)
        - language (optional)
        - prompt (optional)

    Notes:
    - This client buffers audio in memory (no true streaming). `get_partial()` returns None.
    - `get_final()` performs a transcription request once, returning the transcript text.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: Optional[str],
        model: str = "whisper-1",
        timeout_ms: int = 30000,
        sample_rate: int = 8000,
        wav_sample_rate: int = 16000,
        pad_to_ms: int = 1000,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        min_audio_ms: int = 200,
    ):
        if aiohttp is None:
            raise RuntimeError("aiohttp is required for HttpASRClient")
        if np is None:
            raise RuntimeError("numpy is required for HttpASRClient")
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = api_key
        self.model = (model or "whisper-1").strip()
        self.timeout_ms = int(timeout_ms)
        self.sample_rate = int(sample_rate)
        self.wav_sample_rate = int(wav_sample_rate)
        self.pad_to_ms = max(0, int(pad_to_ms))
        self.language = (language or "").strip() or None
        self.prompt = (prompt or "").strip() or None
        self.min_audio_ms = max(0, int(min_audio_ms))

        self._streams: dict[str, bytearray] = {}
        self._final: dict[str, Optional[str]] = {}

    def _transcriptions_url(self) -> str:
        # Allow caller to pass either a base URL or a full endpoint.
        if self.base_url.endswith("/v1/audio/transcriptions"):
            return self.base_url
        return f"{self.base_url}/v1/audio/transcriptions"

    def _pcm16le_to_wav_bytes(self, pcm16le: bytes) -> bytes:
        # PCM16 mono, little-endian. Optionally resample to 16kHz for Whisper quality.
        pcm_bytes = pcm16le
        out_rate = self.wav_sample_rate or self.sample_rate

        if out_rate != self.sample_rate:
            if resample_poly is None:
                raise RuntimeError("scipy is required to resample audio for ASR")
            src = np.frombuffer(pcm_bytes, dtype=np.dtype("<i2"))
            if src.size:
                x = src.astype(np.float32) / 32768.0
                up = out_rate
                down = self.sample_rate
                g = math.gcd(up, down)
                up //= g
                down //= g
                y = resample_poly(x, up, down).astype(np.float32)
                out = np.clip(np.round(y * 32767.0), -32768, 32767).astype(np.int16)
                pcm_bytes = out.astype(np.dtype("<i2"), copy=False).tobytes()

        # whisper.cpp often expects ~>= 1s of audio; pad with silence if needed.
        if self.pad_to_ms > 0:
            min_samples = int(out_rate * (self.pad_to_ms / 1000.0))
            min_bytes = max(0, min_samples * 2)  # PCM16 mono
            if len(pcm_bytes) < min_bytes:
                pcm_bytes = pcm_bytes + (b"\x00" * (min_bytes - len(pcm_bytes)))

        bio = io.BytesIO()
        with wave.open(bio, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(out_rate)
            wf.writeframes(pcm_bytes)
        return bio.getvalue()

    def _min_audio_bytes(self) -> int:
        # 16-bit mono => 2 bytes/sample
        samples = int(self.sample_rate * (self.min_audio_ms / 1000.0))
        return max(0, samples * 2)

    async def start_stream(self) -> str:
        stream_id = f"http_asr_{len(self._streams)}_{int(time.time() * 1000)}"
        self._streams[stream_id] = bytearray()
        self._final[stream_id] = None
        return stream_id

    async def send_audio(self, stream_id: str, audio: bytes) -> None:
        buf = self._streams.get(stream_id)
        if buf is None:
            return
        if not audio:
            return
        buf.extend(audio)

    async def get_partial(self, stream_id: str) -> Optional[str]:
        return None

    async def get_final(self, stream_id: str) -> Optional[str]:
        cached = self._final.get(stream_id)
        if isinstance(cached, str) and cached.strip():
            return cached

        buf = self._streams.get(stream_id)
        if not buf:
            return None

        if len(buf) < self._min_audio_bytes():
            return None

        url = self._transcriptions_url()
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        wav_bytes = self._pcm16le_to_wav_bytes(bytes(buf))
        form = aiohttp.FormData()
        form.add_field("model", self.model)
        if self.language:
            form.add_field("language", self.language)
        if self.prompt:
            form.add_field("prompt", self.prompt)
        # Many servers accept response_format=json; keep default and parse robustly.
        form.add_field(
            "file",
            wav_bytes,
            filename="audio.wav",
            content_type="audio/wav",
        )

        timeout_s = max(0.1, float(self.timeout_ms) / 1000.0)
        timeout = aiohttp.ClientTimeout(total=timeout_s)

        started_at = time.perf_counter()
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, headers=headers, data=form) as resp:
                body = await resp.text()
                if resp.status < 200 or resp.status >= 300:
                    raise RuntimeError(f"ASR HTTP {resp.status} from {url}: {body[:2000]}")

        elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        logger.info(
            "ASR HTTP response received (provider=http, elapsed_ms=%s, bytes_in=%s, model=%s, url=%s)",
            elapsed_ms,
            len(buf),
            self.model,
            url,
        )

        # Expected OpenAI response: { "text": "..." }
        text: Optional[str] = None
        try:
            data = json.loads(body)
            t = data.get("text")
            if isinstance(t, str):
                text = t.strip()
        except Exception:
            # Some servers may respond plain-text; accept that as fallback.
            text = body.strip() if isinstance(body, str) else None

        self._final[stream_id] = text
        return text

    async def stop_stream(self, stream_id: str) -> None:
        self._streams.pop(stream_id, None)
        self._final.pop(stream_id, None)


# Factory function to create providers based on env vars

def create_asr_client(provider: str = "mock", **kwargs) -> ASRClient:
    """Create ASR client based on provider name."""
    if provider == "mock":
        return MockASRClient()
    elif provider == "http":
        base_url = os.getenv("ASR_BASE_URL", "").strip()
        if not base_url:
            raise RuntimeError("ASR_BASE_URL is required for ASR_PROVIDER=http")
        api_key = os.getenv("ASR_API_KEY")
        model = os.getenv("ASR_MODEL", "whisper-1")
        timeout_ms = int(os.getenv("ASR_TIMEOUT_MS", "30000"))
        language = os.getenv("ASR_LANGUAGE")
        prompt = os.getenv("ASR_PROMPT")
        min_audio_ms = int(os.getenv("ASR_MIN_AUDIO_MS", "200"))
        sample_rate = int(kwargs.get("sample_rate", os.getenv("ASR_SAMPLE_RATE", "8000")))
        wav_sample_rate = int(os.getenv("ASR_WAV_SAMPLE_RATE", "16000"))
        pad_to_ms = int(os.getenv("ASR_PAD_TO_MS", "1000"))
        return HttpASRClient(
            base_url=base_url,
            api_key=api_key,
            model=model,
            timeout_ms=timeout_ms,
            sample_rate=sample_rate,
            wav_sample_rate=wav_sample_rate,
            pad_to_ms=pad_to_ms,
            language=language,
            prompt=prompt,
            min_audio_ms=min_audio_ms,
        )
    # Add real providers here (e.g. deepgram, azure, etc.)
    else:
        logger.warning(f"Unknown ASR provider: {provider}, using mock")
        return MockASRClient()


def create_llm_client(provider: str = "mock", system_prompt: str = "You are a helpful assistant.", **kwargs) -> LLMClient:
    """Create LLM client based on provider name."""
    if provider == "mock":
        return MockLLMClient(system_prompt=system_prompt)
    elif provider == "openai_compat":
        return OpenAICompatLLMClient(system_prompt=system_prompt, **kwargs)
    # Add real providers here
    # elif provider == "openai":
    #     return OpenAILLMClient(system_prompt=system_prompt, **kwargs)
    else:
        logger.warning(f"Unknown LLM provider: {provider}, using mock")
        return MockLLMClient(system_prompt=system_prompt)


def create_tts_client(provider: str = "mock", sample_rate: int = 8000, **kwargs) -> TTSClient:
    """Create TTS client based on provider name."""
    if provider == "mock":
        return MockTTSClient(
            sample_rate=sample_rate,
            frame_duration_ms=int(kwargs.get("frame_duration_ms", 20)),
        )
    elif provider == "http":
        base_url = os.getenv("TTS_BASE_URL", "").strip()
        if not base_url:
            raise RuntimeError("TTS_BASE_URL is required for TTS_PROVIDER=http")
        api_key = os.getenv("TTS_API_KEY")
        voice = os.getenv("TTS_VOICE", "alloy")
        timeout_ms = int(os.getenv("TTS_TIMEOUT_MS", "30000"))
        model = os.getenv("TTS_MODEL", "tts-1")
        source_sr = int(os.getenv("TTS_SOURCE_SAMPLE_RATE", "24000"))
        response_format = os.getenv("TTS_RESPONSE_FORMAT", "pcm")
        return HttpTTSClient(
            base_url=base_url,
            api_key=api_key,
            voice=voice,
            timeout_ms=timeout_ms,
            sample_rate=sample_rate,
            frame_duration_ms=int(kwargs.get("frame_duration_ms", 20)),
            model=model,
            source_sample_rate=source_sr,
            response_format=response_format,
        )
    # Add real providers here
    # elif provider == "elevenlabs":
    #     return ElevenLabsTTSClient(sample_rate=sample_rate, **kwargs)
    else:
        logger.warning(f"Unknown TTS provider: {provider}, using mock")
        return MockTTSClient(
            sample_rate=sample_rate,
            frame_duration_ms=int(kwargs.get("frame_duration_ms", 20)),
        )
