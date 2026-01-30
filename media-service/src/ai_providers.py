"""
Abstract interfaces and implementations for ASR, LLM, and TTS providers.
"""

from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional
import asyncio
import logging
import math
import os
import time
import io

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


# Factory function to create providers based on env vars

def create_asr_client(provider: str = "mock", **kwargs) -> ASRClient:
    """Create ASR client based on provider name."""
    if provider == "mock":
        return MockASRClient()
    # Add real providers here
    # elif provider == "deepgram":
    #     return DeepgramASRClient(**kwargs)
    else:
        logger.warning(f"Unknown ASR provider: {provider}, using mock")
        return MockASRClient()


def create_llm_client(provider: str = "mock", system_prompt: str = "You are a helpful assistant.", **kwargs) -> LLMClient:
    """Create LLM client based on provider name."""
    if provider == "mock":
        return MockLLMClient(system_prompt=system_prompt)
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
