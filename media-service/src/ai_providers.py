"""
Abstract interfaces and implementations for ASR, LLM, and TTS providers.
"""

from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional
import asyncio
import logging

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
