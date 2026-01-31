"""
Audio codec utilities for PCMU (ulaw) encoding/decoding.
"""

import numpy as np
import audioop


def ulaw_to_linear(ulaw_data: bytes) -> np.ndarray:
    """
    Convert ulaw (PCMU) encoded bytes to linear PCM16.
    
    Args:
        ulaw_data: ulaw encoded audio bytes
    
    Returns:
        numpy array of int16 PCM samples
    """
    # Use Python's reference implementation to avoid codec mismatches (white noise).
    # audioop expects 16-bit linear width=2 (little-endian on common platforms).
    pcm_bytes = audioop.ulaw2lin(ulaw_data, 2)
    return np.frombuffer(pcm_bytes, dtype=np.dtype("<i2"))


def linear_to_ulaw(pcm_data: np.ndarray) -> bytes:
    """
    Convert linear PCM16 to ulaw (PCMU) encoded bytes.
    
    Args:
        pcm_data: numpy array of int16 PCM samples
    
    Returns:
        ulaw encoded audio bytes
    """
    pcm = np.asarray(pcm_data)
    if pcm.dtype != np.int16:
        pcm = pcm.astype(np.int16, copy=False)

    # audioop works on bytes; enforce little-endian PCM16.
    pcm_bytes = pcm.astype(np.dtype("<i2"), copy=False).tobytes()
    return audioop.lin2ulaw(pcm_bytes, 2)


def alaw_to_linear(alaw_data: bytes) -> np.ndarray:
    """
    Convert alaw (PCMA) encoded bytes to linear PCM16.
    """
    pcm_bytes = audioop.alaw2lin(alaw_data, 2)
    return np.frombuffer(pcm_bytes, dtype=np.dtype("<i2"))


def linear_to_alaw(pcm_data: np.ndarray) -> bytes:
    """
    Convert linear PCM16 to alaw (PCMA) encoded bytes.
    """
    pcm = np.asarray(pcm_data)
    if pcm.dtype != np.int16:
        pcm = pcm.astype(np.int16, copy=False)
    pcm_bytes = pcm.astype(np.dtype("<i2"), copy=False).tobytes()
    return audioop.lin2alaw(pcm_bytes, 2)
