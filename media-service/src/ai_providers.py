"""
Abstract interfaces and implementations for ASR, LLM, and TTS providers.
"""

from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional
import asyncio
import json
import logging
import os
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


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
            Audio chunks (PCM16, 8kHz mono)
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
    """

    def __init__(
        self,
        system_prompt: str = "You are a helpful assistant.",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        temperature: Optional[float] = None,
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
    
    def __init__(self, sample_rate: int = 8000):
        self.sample_rate = sample_rate
    
    async def synthesize_stream(self, text: str) -> AsyncIterator[bytes]:
        """Generate a simple tone as mock audio."""
        import numpy as np
        
        # Generate a simple beep tone (440 Hz for 1 second)
        duration = min(len(text) * 0.1, 3.0)  # Roughly 0.1s per character, max 3s
        samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, samples)
        
        # Generate tone
        frequency = 440.0
        audio = np.sin(2 * np.pi * frequency * t)
        
        # Convert to int16
        audio_int16 = (audio * 32767 * 0.3).astype(np.int16)  # 30% volume
        
        # Yield in chunks (20ms frames)
        chunk_size = (self.sample_rate * 20) // 1000
        for i in range(0, len(audio_int16), chunk_size):
            chunk = audio_int16[i:i + chunk_size]
            yield chunk.tobytes()
            await asyncio.sleep(0.02)  # 20ms


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
        return MockTTSClient(sample_rate=sample_rate)
    # Add real providers here
    # elif provider == "elevenlabs":
    #     return ElevenLabsTTSClient(sample_rate=sample_rate, **kwargs)
    else:
        logger.warning(f"Unknown TTS provider: {provider}, using mock")
        return MockTTSClient(sample_rate=sample_rate)
