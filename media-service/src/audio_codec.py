"""
Audio codec utilities for PCMU (ulaw) encoding/decoding.
"""

import numpy as np


def ulaw_to_linear(ulaw_data: bytes) -> np.ndarray:
    """
    Convert ulaw (PCMU) encoded bytes to linear PCM16.
    
    Args:
        ulaw_data: ulaw encoded audio bytes
    
    Returns:
        numpy array of int16 PCM samples
    """
    # ulaw is 8-bit, convert to numpy array
    ulaw_array = np.frombuffer(ulaw_data, dtype=np.uint8)
    
    # ulaw to linear conversion
    # Invert all bits
    ulaw_array = ~ulaw_array
    
    # Extract sign, exponent, and mantissa
    sign = (ulaw_array & 0x80) >> 7
    exponent = (ulaw_array & 0x70) >> 4
    mantissa = ulaw_array & 0x0F
    
    # Reconstruct linear value
    linear = ((mantissa << 1) + 33) << exponent
    linear = linear - 33
    
    # Apply sign
    linear = np.where(sign == 1, -linear, linear)
    
    # Scale to 16-bit range
    linear = linear.astype(np.int16)
    
    return linear


def linear_to_ulaw(pcm_data: np.ndarray) -> bytes:
    """
    Convert linear PCM16 to ulaw (PCMU) encoded bytes.
    
    Args:
        pcm_data: numpy array of int16 PCM samples
    
    Returns:
        ulaw encoded audio bytes
    """
    # Clamp to 16-bit range
    pcm_data = np.clip(pcm_data, -32768, 32767)
    
    # Get sign
    sign = (pcm_data < 0).astype(np.uint8)
    
    # Get absolute value
    abs_pcm = np.abs(pcm_data)
    
    # Add bias
    abs_pcm = abs_pcm + 33
    
    # Find exponent (power of 2)
    exponent = np.zeros_like(abs_pcm, dtype=np.uint8)
    for i in range(7, -1, -1):
        mask = abs_pcm >= (1 << (i + 4))
        exponent[mask] = i
    
    # Calculate mantissa
    mantissa = (abs_pcm >> (exponent + 1)) & 0x0F
    
    # Combine sign, exponent, mantissa
    ulaw = (sign << 7) | (exponent << 4) | mantissa
    
    # Invert all bits
    ulaw = ~ulaw
    
    return ulaw.astype(np.uint8).tobytes()
